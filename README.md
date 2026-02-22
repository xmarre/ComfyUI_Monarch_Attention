# ComfyUI MonarchAttention patch node

This custom node patches ComfyUI's global attention entry points to use **MonarchAttention** for **self-attention** only (where Q/K/V have the same sequence length). Cross-attention is left untouched.

## Install

1. Copy this folder to:

   `ComfyUI/custom_nodes/comfyui_monarch_attention/`

2. Vendor the MonarchAttention repo into this custom node (no env vars needed).

   The node expects a directory containing `ma/` at one of the following paths:

   - `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch_attention/ma/`
   - `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch-attention/ma/`

   The simplest way is to copy (or git clone) the MonarchAttention repo into:
   `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch_attention/`

   Make sure the *final* layout contains `ma/` directly under that folder.

3. Restart ComfyUI.

## Use

- Add **Enable MonarchAttention (self-attn)** before your sampler.
- If something breaks, add **Disable MonarchAttention** (or toggle enable=false) and rerun.

## Notes

- Only works for self-attention (square attention). It will automatically fall back to ComfyUI's original attention for cross-attention or unsupported tensor shapes.
- `impl=auto` prefers `triton` if the repo registers it; otherwise uses `torch`.
