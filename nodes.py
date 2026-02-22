import os
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch


# ------------------------------
# Import helper (external MonarchAttention repo)
# ------------------------------

def _ensure_vendored_monarch_path() -> str:
    """Ensure `import ma` works by adding a vendored MonarchAttention repo to sys.path.

    This custom node expects the MonarchAttention repo to be bundled under:

        comfyui_monarch_attention/third_party/...

    The directory we add to sys.path must directly contain the `ma/` package.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # Preferred layout:
    #   comfyui_monarch_attention/third_party/monarch_attention/ma/...
    candidates = [
        os.path.join(here, "third_party", "monarch_attention"),
        os.path.join(here, "third_party", "monarch-attention"),
        # Common zip/github download layouts (nested repo folder names)
        os.path.join(here, "third_party", "monarch-attention-main"),
        os.path.join(here, "third_party", "monarch-attention-main", "monarch-attention-main"),
        os.path.join(here, "third_party", "monarch_attention_main"),
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
        # Only attempt vendored path injection when `ma` is missing.
        # If `ma` exists but errors during import, let that error surface.
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


_PATCH_STATE: dict[str, Any] = {
    "enabled": False,
    "orig": {},  # name -> callable
    "config": None,
    "last_status": "disabled",
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _as_bool_mask(mask: torch.Tensor) -> Optional[torch.Tensor]:
    # Accept bool; accept {0,1} float/int.
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() or mask.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return mask != 0
    return None


def _reshape_to_bhnd(x: torch.Tensor, heads: Optional[int]) -> Optional[Tuple[torch.Tensor, int, int]]:
    """Return (x_bhnd, batch, heads)."""
    if x.dim() == 4:
        b, h, n, d = x.shape
        return x, b, h
    if x.dim() == 3:
        bh, n, d = x.shape
        if not heads or heads <= 0 or (bh % heads) != 0:
            return None
        b = bh // heads
        return x.view(b, heads, n, d), b, heads
    return None


def _reshape_back(y_bhnd: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    if original.dim() == 4:
        return y_bhnd
    if original.dim() == 3:
        b, h, n, d = y_bhnd.shape
        return y_bhnd.reshape(b * h, n, d)
    return y_bhnd


def _should_use_monarch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cfg: _PatchConfig) -> bool:
    # Self-attention only
    if q.shape[-2] != k.shape[-2] or k.shape[-2] != v.shape[-2]:
        return False

    n = int(q.shape[-2])
    if n < cfg.min_seq_len or n > cfg.max_seq_len:
        return False

    # Head dim must be >0
    if q.shape[-1] <= 0:
        return False

    return True


def _build_wrapper(orig_fn: Callable, cfg: _PatchConfig) -> Callable:
    MonarchAttention, PadType, impls = _import_monarch_attention()

    impl = cfg.impl
    if impl == "auto":
        impl = "triton" if "triton" in impls else "torch"
    if impl not in impls:
        # Fall back safely
        impl = "torch"

    pad_type = PadType.pre if cfg.pad_type == "pre" else PadType.post

    monarch = MonarchAttention(
        block_size=int(cfg.block_size),
        num_steps=int(cfg.num_steps),
        pad_type=pad_type,
        impl=impl,
    )

    def wrapped(*args, **kwargs):
        # Try to map ComfyUI's attention signature(s) to (q, k, v, heads, mask)
        try:
            if len(args) < 3:
                return orig_fn(*args, **kwargs)

            q = args[0]
            k = args[1]
            v = args[2]

            if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
                return orig_fn(*args, **kwargs)

            heads = kwargs.get("heads", None)
            if heads is None and len(args) >= 4 and isinstance(args[3], int):
                heads = args[3]

            # Try common mask kwarg names
            mask = kwargs.get("mask", None)
            if mask is None:
                mask = kwargs.get("attn_mask", None)
            if mask is None and len(args) >= 5 and torch.is_tensor(args[4]):
                mask = args[4]

            q_r = _reshape_to_bhnd(q, heads)
            k_r = _reshape_to_bhnd(k, heads)
            v_r = _reshape_to_bhnd(v, heads)
            if not q_r or not k_r or not v_r:
                return orig_fn(*args, **kwargs)

            q_bhnd, b, h = q_r
            k_bhnd, b2, h2 = k_r
            v_bhnd, b3, h3 = v_r

            if b != b2 or b != b3 or h != h2 or h != h3:
                return orig_fn(*args, **kwargs)

            if not _should_use_monarch(q_bhnd, k_bhnd, v_bhnd, cfg):
                return orig_fn(*args, **kwargs)

            mask_bool = None
            if torch.is_tensor(mask):
                if mask.dim() == 2 and mask.shape[0] == b and mask.shape[1] == q_bhnd.shape[-2]:
                    mask_bool = _as_bool_mask(mask)
                else:
                    # Unsupported mask shape/type; fall back.
                    return orig_fn(*args, **kwargs)

            t0 = _now_ms() if cfg.verbose else 0
            out = monarch(q_bhnd, k_bhnd, v_bhnd, attention_mask=mask_bool)
            out = out.to(dtype=v.dtype)
            out = out.contiguous()
            out = _reshape_back(out, q)

            if cfg.verbose:
                dt = _now_ms() - t0
                _PATCH_STATE["last_status"] = (
                    f"monarch({cfg.impl}->{impl}) used: n={q_bhnd.shape[-2]} h={h} bs={cfg.block_size} "
                    f"steps={cfg.num_steps} pad={cfg.pad_type} | {dt}ms"
                )

            return out
        except Exception:
            # Any failure must be non-fatal: fall back to original attention.
            return orig_fn(*args, **kwargs)

    # Preserve a few useful attributes
    wrapped.__name__ = getattr(orig_fn, "__name__", "optimized_attention")
    wrapped.__doc__ = getattr(orig_fn, "__doc__", None)
    wrapped.__wrapped__ = orig_fn  # type: ignore
    return wrapped


def enable_monarch_attention(cfg: _PatchConfig) -> str:
    """Patch ComfyUI attention functions (global) to use MonarchAttention for self-attn."""
    # Import comfy lazily so this file can be syntax-checked outside ComfyUI.
    import importlib

    attn_mod = importlib.import_module("comfy.ldm.modules.attention")

    if _PATCH_STATE["enabled"]:
        # Update config by re-wrapping
        disable_monarch_attention()

    orig: dict[str, Callable] = {}

    # Patch the most common entry points (only those that exist)
    candidate_names = [
        "optimized_attention",
        "attention_basic",
        "attention_pytorch",
        "attention",  # some forks
    ]

    for name in candidate_names:
        fn = getattr(attn_mod, name, None)
        if callable(fn):
            orig[name] = fn
            setattr(attn_mod, name, _build_wrapper(fn, cfg))

    if not orig:
        _PATCH_STATE["enabled"] = False
        _PATCH_STATE["orig"] = {}
        _PATCH_STATE["config"] = None
        _PATCH_STATE["last_status"] = "No patch targets found in comfy.ldm.modules.attention"
        return _PATCH_STATE["last_status"]

    _PATCH_STATE["enabled"] = True
    _PATCH_STATE["orig"] = orig
    _PATCH_STATE["config"] = cfg
    _PATCH_STATE["last_status"] = (
        f"enabled (patched: {', '.join(sorted(orig.keys()))}) | impl={cfg.impl} bs={cfg.block_size} "
        f"steps={cfg.num_steps} pad={cfg.pad_type} n=[{cfg.min_seq_len},{cfg.max_seq_len}]"
    )
    return _PATCH_STATE["last_status"]


def disable_monarch_attention() -> str:
    """Restore original ComfyUI attention functions."""
    if not _PATCH_STATE["enabled"]:
        _PATCH_STATE["last_status"] = "disabled"
        return _PATCH_STATE["last_status"]

    import importlib

    attn_mod = importlib.import_module("comfy.ldm.modules.attention")
    orig: dict[str, Callable] = _PATCH_STATE.get("orig", {}) or {}

    for name, fn in orig.items():
        try:
            setattr(attn_mod, name, fn)
        except Exception:
            pass

    _PATCH_STATE["enabled"] = False
    _PATCH_STATE["orig"] = {}
    _PATCH_STATE["config"] = None
    _PATCH_STATE["last_status"] = "disabled"
    return _PATCH_STATE["last_status"]


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
    ):
        if enable:
            cfg = _PatchConfig(
                impl=str(impl),
                block_size=int(block_size),
                num_steps=int(num_steps),
                pad_type=str(pad_type),
                min_seq_len=int(min_seq_len),
                max_seq_len=int(max_seq_len),
                verbose=bool(verbose),
            )
            status = enable_monarch_attention(cfg)
        else:
            status = disable_monarch_attention()
        return (model, status)


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
        status = disable_monarch_attention()
        return (model, status)


class MonarchAttentionStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "get"
    CATEGORY = "attention/monarch"

    def get(self):
        return (str(_PATCH_STATE.get("last_status", "disabled")),)


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
