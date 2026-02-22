import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


# ------------------------------
# Import helper (vendored MonarchAttention repo)
# ------------------------------

def _ensure_vendored_monarch_path() -> str:
    """Ensure `import ma` works by adding a vendored MonarchAttention repo to sys.path.

    Expected layout (recommended):

        comfyui_monarch_attention/third_party/monarch_attention/ma/...

    The directory we add to sys.path must directly contain the `ma/` package.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(here, "third_party", "monarch_attention"),
        os.path.join(here, "third_party", "monarch-attention"),
        # Common GitHub zip layout (nested repo folder names)
        os.path.join(here, "third_party", "monarch-attention-main"),
        os.path.join(here, "third_party", "monarch-attention-main", "monarch-attention-main"),
    ]

    for p in candidates:
        if os.path.isdir(os.path.join(p, "ma")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p

    raise ImportError(
        "MonarchAttention not found (vendored dependency missing).\n\n"
        "Expected one of these to exist and contain a `ma/` folder:\n"
        f"  - {os.path.join(here, 'third_party', 'monarch_attention')}\n"
        f"  - {os.path.join(here, 'third_party', 'monarch-attention')}\n"
        "If you downloaded a GitHub zip, copy the *inner* folder that contains `ma/` into one of the paths above.\n"
    )


def _import_monarch_attention():
    """Import MonarchAttention module from the vendored repo.

    Returns (MonarchAttention, PadType, available_impls)
    """
    try:
        from ma.monarch_attention import MonarchAttention, PadType  # type: ignore
        from ma import monarch_attention as ma_mod  # type: ignore
    except ModuleNotFoundError as e:
        # Only try vendored-path injection when the module isn't found.
        if getattr(e, "name", None) not in ("ma", "ma.monarch_attention"):
            raise
        _ensure_vendored_monarch_path()
        from ma.monarch_attention import MonarchAttention, PadType  # type: ignore
        from ma import monarch_attention as ma_mod  # type: ignore

    impls: list[str] = []
    try:
        impls = sorted(getattr(ma_mod, "_IMPLEMENTATIONS", {}).keys())
    except Exception:
        impls = []
    return MonarchAttention, PadType, impls


# ------------------------------
# Patch state
# ------------------------------


@dataclass
class _PatchConfig:
    impl: str
    block_size: int
    num_steps: int
    pad_type: str
    min_seq_len: int
    max_seq_len: int
    verbose: bool
    strict: bool
    print_debug: bool


_STATE: dict[str, Any] = {"last_status": "disabled"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _as_bool_mask(mask: torch.Tensor) -> Optional[torch.Tensor]:
    # Accept bool; accept {0,1} float/int.
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() or mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return mask != 0
    return None


# ------------------------------
# Model-level attention override (branch-specific)
# ------------------------------


def _should_use_monarch_self_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    cfg: _PatchConfig,
    skip_reshape: bool,
    attn_precision: Optional[torch.dtype],
) -> bool:
    # self-attn only
    try:
        if skip_reshape:
            # q,k,v are expected to be [B, H, N, Dh]
            if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
                return False
            if q.shape[1] != heads or k.shape[1] != heads or v.shape[1] != heads:
                return False
            # Dh must match
            if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
                return False
            nq = int(q.shape[2])
            nk = int(k.shape[2])
            nv = int(v.shape[2])
        else:
            # q,k,v are expected to be [B, N, H*Dh]
            if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
                return False
            nq = int(q.shape[1])
            nk = int(k.shape[1])
            nv = int(v.shape[1])
            if (q.shape[-1] % heads) != 0 or (k.shape[-1] % heads) != 0 or (v.shape[-1] % heads) != 0:
                return False

        if nq != nk or nk != nv:
            return False

        if nq < cfg.min_seq_len or nq > cfg.max_seq_len:
            return False

        # If caller forced FP32 attention, keep original path (Monarch impl may not match).
        if attn_precision == torch.float32:
            return False

        return True
    except Exception:
        return False


def _build_attention_override(cfg: _PatchConfig, fallback_override: Optional[Callable]) -> Callable:
    MonarchAttention, PadType, impls = _import_monarch_attention()

    impl = cfg.impl
    if impl == "auto":
        impl = "triton" if "triton" in impls else "torch"
    if impl not in impls:
        # Fall back safely
        impl = "torch"

    resolved_impl = impl
    pad_type = PadType.pre if cfg.pad_type == "pre" else PadType.post

    monarch = MonarchAttention(
        block_size=int(cfg.block_size),
        num_steps=int(cfg.num_steps),
        pad_type=pad_type,
        impl=resolved_impl,
    )

    stats = {"banner": False, "used": 0, "rejected": 0}

    def _dbg(msg: str):
        if cfg.print_debug:
            print(msg)

    def attention_override(func: Callable, *args, **kwargs):
        if cfg.print_debug and not stats["banner"]:
            stats["banner"] = True
            _dbg(
                "[MonarchAttention] override active | "
                f"impl={cfg.impl}->{resolved_impl} bs={cfg.block_size} steps={cfg.num_steps} "
                f"pad={cfg.pad_type} n=[{cfg.min_seq_len},{cfg.max_seq_len}] strict={cfg.strict}"
            )

        try:
            if len(args) < 4:
                raise RuntimeError(f"bad signature: len(args)={len(args)} (<4)")

            q, k, v, heads = args[0], args[1], args[2], args[3]
            # Preserve positional + alternate kwarg forms for mask.
            mask = kwargs.get("mask", None)
            if mask is None:
                mask = kwargs.get("attn_mask", None)
            if mask is None and len(args) >= 5:
                mask = args[4]
            attn_precision = kwargs.get("attn_precision", None)
            skip_reshape = bool(kwargs.get("skip_reshape", False))
            skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))

            if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
                raise RuntimeError(f"qkv not tensors: q={type(q)} k={type(k)} v={type(v)}")
            if not isinstance(heads, int) or heads <= 0:
                raise RuntimeError(f"invalid heads={heads!r}")

            if not _should_use_monarch_self_attn(q, k, v, heads, cfg, skip_reshape, attn_precision):
                raise RuntimeError(
                    "shape/gating rejected: "
                    f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
                    f"heads={heads} skip_reshape={skip_reshape} attn_precision={attn_precision}"
                )

            b = int(q.shape[0])

            if skip_reshape:
                # [B, H, N, Dh]
                q_bhnd = q
                k_bhnd = k
                v_bhnd = v
                n = int(q.shape[2])
                dim_head = int(q.shape[-1])
            else:
                # [B, N, H*Dh] -> [B, H, N, Dh]
                n = int(q.shape[1])
                dim = int(q.shape[-1])
                dim_head = dim // heads
                if dim_head * heads != dim:
                    raise RuntimeError(f"embed dim not divisible by heads: dim={dim} heads={heads}")
                q_bhnd = q.view(b, n, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                k_bhnd = k.view(b, n, heads, dim_head).permute(0, 2, 1, 3).contiguous()
                v_bhnd = v.view(b, n, heads, dim_head).permute(0, 2, 1, 3).contiguous()

            # Mask support (conservative): accept only boolean (or 0/1) key-padding masks [B, Nk]
            mask_bool = None
            if mask is not None:
                if not torch.is_tensor(mask):
                    raise RuntimeError(f"mask not tensor: {type(mask)}")

                # bool mask is typically [B, Nk] (or [1, Nk])
                if mask.dim() == 2 and mask.shape[1] == n and mask.shape[0] in (1, b):
                    if mask.shape[0] == 1 and b > 1:
                        mask = mask.expand(b, -1)
                    mask_bool = _as_bool_mask(mask)
                    if mask_bool is None:
                        raise RuntimeError(f"mask dtype unsupported: dtype={mask.dtype}")
                else:
                    raise RuntimeError(
                        "unsupported mask shape for MonarchAttention (expects [B,N] key-padding). "
                        f"mask.shape={tuple(mask.shape)} dtype={mask.dtype} n={n} b={b}"
                    )

            t0 = _now_ms() if cfg.verbose else 0
            out_bhnd = monarch(q_bhnd, k_bhnd, v_bhnd, attention_mask=mask_bool)
            out_bhnd = out_bhnd.to(dtype=v.dtype).contiguous()

            if skip_output_reshape:
                out = out_bhnd
            else:
                out = out_bhnd.permute(0, 2, 1, 3).reshape(b, -1, heads * dim_head)

            stats["used"] += 1
            if cfg.print_debug and stats["used"] <= 5:
                _dbg(f"[MonarchAttention] USED | n={n} heads={heads} mask={'yes' if mask is not None else 'no'}")

            if cfg.verbose:
                dt = _now_ms() - t0
                msg = (
                    f"monarch(model-override, {cfg.impl}->{resolved_impl}) used: n={n} h={heads} bs={cfg.block_size} "
                    f"steps={cfg.num_steps} pad={cfg.pad_type} | {dt}ms"
                )
                _STATE["last_status"] = msg
                _dbg(f"[MonarchAttention] {msg}")

            return out
        except Exception as e:
            stats["rejected"] += 1
            if cfg.print_debug and stats["rejected"] <= 20:
                _dbg(f"[MonarchAttention] REJECT | {e}")
            if cfg.strict:
                raise
            if fallback_override is not None:
                return fallback_override(func, *args, **kwargs)
            return func(*args, **kwargs)

    return attention_override


def _apply_model_override(model, cfg: _PatchConfig):
    """Return (model_clone, status) with optimized_attention_override set."""
    model_clone = model.clone()
    model_options = getattr(model_clone, "model_options", None)
    if model_options is None:
        # should not happen for a ComfyUI MODEL, but keep it robust
        model_clone.model_options = {}
        model_options = model_clone.model_options

    transformer_options = model_options.get("transformer_options")
    if transformer_options is None or not isinstance(transformer_options, dict):
        transformer_options = {}
        model_options["transformer_options"] = transformer_options

    prev_key = "_monarch_prev_optimized_attention_override"
    current = transformer_options.get("optimized_attention_override", None)
    # Only treat the current override as "previous" if it is NOT already ours.
    if not getattr(current, "_monarch_override", False):
        transformer_options[prev_key] = current

    fallback_override = transformer_options.get(prev_key, None)
    override = _build_attention_override(cfg, fallback_override)
    # Tag it so Disable doesn't nuke someone else's override.
    setattr(override, "_monarch_override", True)
    setattr(override, "_monarch_cfg", (cfg.impl, cfg.block_size, cfg.num_steps, cfg.pad_type, cfg.min_seq_len, cfg.max_seq_len))

    transformer_options["optimized_attention_override"] = override
    transformer_options["_monarch_attention_cfg"] = {
        "impl": cfg.impl,
        "block_size": cfg.block_size,
        "num_steps": cfg.num_steps,
        "pad_type": cfg.pad_type,
        "min_seq_len": cfg.min_seq_len,
        "max_seq_len": cfg.max_seq_len,
        "verbose": cfg.verbose,
    }

    status = (
        f"enabled (model override) | impl={cfg.impl} bs={cfg.block_size} steps={cfg.num_steps} "
        f"pad={cfg.pad_type} n=[{cfg.min_seq_len},{cfg.max_seq_len}]"
    )
    _STATE["last_status"] = status
    return model_clone, status


def _remove_model_override(model):
    """Return (model_clone, status) with optimized_attention_override restored/removed."""
    model_clone = model.clone()
    model_options = getattr(model_clone, "model_options", None)
    if model_options is None:
        model_clone.model_options = {}
        model_options = model_clone.model_options

    transformer_options = model_options.get("transformer_options")
    if transformer_options is None or not isinstance(transformer_options, dict):
        transformer_options = {}
        model_options["transformer_options"] = transformer_options

    prev_key = "_monarch_prev_optimized_attention_override"
    current = transformer_options.get("optimized_attention_override", None)
    # If the current override is not Monarch, do nothing (don't break other attention nodes).
    if not getattr(current, "_monarch_override", False):
        # But do clean up any stale keys we may have left behind.
        transformer_options.pop(prev_key, None)
        transformer_options.pop("_monarch_attention_cfg", None)
        status = "no monarch override present (no-op)"
        _STATE["last_status"] = status
        return model_clone, status

    prev = transformer_options.pop(prev_key, None)
    transformer_options.pop("_monarch_attention_cfg", None)

    if prev is None:
        # We were enabled without a prior override; just remove ours.
        transformer_options.pop("optimized_attention_override", None)
    else:
        transformer_options["optimized_attention_override"] = prev

    status = "disabled (model override removed)"
    _STATE["last_status"] = status
    return model_clone, status


# ------------------------------
# ComfyUI node classes
# ------------------------------


class MonarchAttentionEnable:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "impl": (["auto", "torch", "triton"], {"default": "auto"}),
                "block_size": ("INT", {"default": 128, "min": 16, "max": 8192, "step": 16}),
                "num_steps": ("INT", {"default": 2, "min": 1, "max": 64}),
                "pad_type": (["pre", "post"], {"default": "post"}),
                "min_seq_len": ("INT", {"default": 256, "min": 1, "max": 262144}),
                "max_seq_len": ("INT", {"default": 4096, "min": 1, "max": 262144}),
                "verbose": ("BOOLEAN", {"default": False}),
                "strict": ("BOOLEAN", {"default": True}),
                "print_debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply"
    CATEGORY = "attention/monarch"

    def apply(
        self,
        model,
        enable: bool,
        impl: str,
        block_size: int,
        num_steps: int,
        pad_type: str,
        min_seq_len: int,
        max_seq_len: int,
        verbose: bool,
        strict: bool,
        print_debug: bool,
    ):
        if not enable:
            model2, status = _remove_model_override(model)
            return (model2, status)

        cfg = _PatchConfig(
            impl=str(impl),
            block_size=int(block_size),
            num_steps=int(num_steps),
            pad_type=str(pad_type),
            min_seq_len=int(min_seq_len),
            max_seq_len=int(max_seq_len),
            verbose=bool(verbose),
            strict=bool(strict),
            print_debug=bool(print_debug),
        )
        model2, status = _apply_model_override(model, cfg)
        return (model2, status)


class MonarchAttentionDisable:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "apply"
    CATEGORY = "attention/monarch"

    def apply(self, model):
        model2, status = _remove_model_override(model)
        return (model2, status)


class MonarchAttentionStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get"
    CATEGORY = "attention/monarch"

    def get(self):
        return (str(_STATE.get("last_status", "disabled")),)


NODE_CLASS_MAPPINGS = {
    "MonarchAttentionEnable": MonarchAttentionEnable,
    "MonarchAttentionDisable": MonarchAttentionDisable,
    "MonarchAttentionStatus": MonarchAttentionStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MonarchAttentionEnable": "Enable MonarchAttention (self-attn)",
    "MonarchAttentionDisable": "Disable MonarchAttention",
    "MonarchAttentionStatus": "MonarchAttention Status",
}
