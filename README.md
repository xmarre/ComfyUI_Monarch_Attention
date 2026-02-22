# ComfyUI MonarchAttention node (model override)

This custom node enables **MonarchAttention** for **self-attention** only (where Q/K/V have the same sequence length) by setting a **model-level** `optimized_attention_override` in `model_options["transformer_options"]`.

That means it only affects the **MODEL branch** you connect it to (SDXL UNet branch, WAN branch, etc.), rather than patching ComfyUI globally.

## Install

1. Copy this folder to:

   `ComfyUI/custom_nodes/comfyui_monarch_attention/`

2. Vendor the MonarchAttention repo into this custom node (no env vars needed).

   Copy (or git clone) the MonarchAttention repo so that `ma/` ends up at one of:

   - `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch_attention/ma/`
   - `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch-attention/ma/`

3. Restart ComfyUI.

## Use

- Put **Enable MonarchAttention (self-attn)** on the **MODEL line** before it goes into your sampler.
- For multi-model workflows (e.g. WAN high/low models), place it on **each** model branch you want patched.
- If something breaks, use **Disable MonarchAttention** (or set `enable=false`) on that same model branch.

## Notes

- Only applies to self-attention (square attention). It falls back to the previous attention path for cross-attention or unsupported shapes.
- If the model already had an `optimized_attention_override` (e.g. SageAttention/FlashAttention), this node will **chain** it as a fallback and restore it when disabled.
- Mask support is conservative (only simple boolean key-padding masks). If a call uses an unsupported mask type/shape, it will fall back.
- `impl=auto` prefers `triton` if available; otherwise uses `torch`.
