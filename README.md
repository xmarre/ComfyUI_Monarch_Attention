# ComfyUI MonarchAttention patch node

This custom node patches ComfyUI's global attention entry points to use **MonarchAttention** for **self-attention** only (where Q/K/V have the same sequence length). Cross-attention is left untouched.

## Install

1. Copy this folder to:

   `ComfyUI/custom_nodes/comfyui_monarch_attention/`

2. Make the MonarchAttention repo importable (it must contain a top-level `ma/` directory):

   **Option A (recommended):** set an env var and restart ComfyUI

   - Windows (PowerShell):
     - `setx MONARCH_ATTENTION_PATH "D:\\path\\to\\monarch-attention-main\\monarch-attention-main"`
   - Linux:
     - `export MONARCH_ATTENTION_PATH=/path/to/monarch-attention-main/monarch-attention-main`

   **Option B:** copy the repo next to this node (no env var)

   - Place the repo directory at:
     - `ComfyUI/custom_nodes/comfyui_monarch_attention/third_party/monarch-attention-main/monarch-attention-main/`

3. Restart ComfyUI.

## Use

- Add **Enable MonarchAttention (self-attn)** before your sampler.
- If something breaks, add **Disable MonarchAttention** (or toggle enable=false) and rerun.

## Notes

- Only works for self-attention (square attention). It will automatically fall back to ComfyUI's original attention for cross-attention or unsupported tensor shapes.
- `impl=auto` prefers `triton` if the repo registers it; otherwise uses `torch`.
