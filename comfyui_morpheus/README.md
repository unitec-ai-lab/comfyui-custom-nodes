# ComfyUI-Morpheus v2.1

**Advanced Google Gemini Image Generation Custom Nodes for ComfyUI**

![Version](https://img.shields.io/badge/version-2.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-orange)

## What's New in v2.1

- **Gemini 3.1 Flash Image Preview** - New model `gemini-3.1-flash-image-preview` added as default
- **5 Gemini Models** - gemini-3.1-flash-image-preview, gemini-3-pro-image-preview, gemini-2.5-flash-image, gemini-2.5-pro, gemini-2.5-flash
- **Data-Driven Model Config** - Model capabilities managed via `MODELS_CONFIG`, no hardcoding

## Installation

### Method 1: Manual (Recommended)

1. Download or clone this repository
2. Copy the `comfyui_morpheus` folder into your ComfyUI custom nodes directory:

```
ComfyUI/
└── custom_nodes/
    └── comfyui_morpheus/      ← copy this folder here
        ├── __init__.py
        ├── nodes.py
        ├── gemini_api.py
        ├── pyproject.toml
        └── examples/
```

3. Install the Python dependency:

```bash
pip install google-genai
```

4. Restart ComfyUI — the nodes will appear in the **Morpheus** category.

### Method 2: ComfyUI Manager

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "Morpheus" in the Manager
3. Click Install

### Dependencies

- `google-genai>=0.2.0`
- `torch>=2.0.0` (already included with ComfyUI)
- `pillow>=10.0.0` (already included with ComfyUI)
- `numpy>=1.24.0` (already included with ComfyUI)

## API Key Setup

### Option 1: Environment Variable (Recommended)

```bash
# Windows
set GEMINI_API_KEY=your-api-key-here

# Linux / macOS
export GEMINI_API_KEY=your-api-key-here
```

### Option 2: Node Input

Enter your API key directly in the `api_key` field of the Morpheus · Gemini node.

Get your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

## Nodes

### 1. Morpheus · Batch Images

Collects multiple images and batches them to feed into the Gemini node.

**Inputs:**
- `image` — Primary image
- `image1` through `image12` — Additional images (optional)

**Outputs:**
- `IMAGE` — Batched preview of all images
- `ORIGINALS_LIST` (PYOBJECT) — List of original tensors for chaining

### 2. Morpheus · Gemini

Generates images using Google Gemini API with native aspect ratio and resolution control.

**Required Inputs:**
- `prompt` — Text description of the image
- `model` — Choose from:
  - `gemini-3.1-flash-image-preview` — **NEW** Gemini 3.1 Flash (default)
  - `gemini-3-pro-image-preview` — Gemini 3 Pro, high-quality with resolution control
  - `gemini-2.5-flash-image` — Gemini 2.5 Flash
  - `gemini-2.5-pro` — Text generation only
  - `gemini-2.5-flash` — Fast text generation only
- `batch_size` (1–8) — Number of images to generate
  - Gemini generates **one image per API call**; batch=4 makes 4 sequential calls
- `seed` — Random seed for reproducibility
- `aspect_ratio` — Native aspect ratio: 1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 21:9, 4:1, 1:4, 8:1, 1:8
  - Note: extreme ratios (4:1, 1:4, 8:1, 1:8) are only supported by `gemini-3.1-flash-image-preview`
- `resolution` — Output resolution: 512 (0.5K), 1K, 2K, 4K
  - `gemini-3.1-flash-image-preview`: 512, 1K, 2K, 4K
  - `gemini-3-pro-image-preview`: 1K, 2K, 4K
  - other models: resolution is ignored

**Optional Inputs:**
- `images` (IMAGE) — Input images for image-to-image generation
- `images_list` (PYOBJECT) — List of images from Batch Images node
- `system_prompt` — System-level instructions
- `api_key` — Gemini API key (if not set as env variable)
- `safety_filter` — BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE
- `top_p` / `max_tokens` — Parameters for text-only models

**Outputs:**
- `images` (IMAGE) — Generated images as batched tensor
- `images_list` (PYOBJECT) — List of individual tensors for downstream chaining
- `execution_log` (STRING) — Detailed log with timestamps and parameters

## Model Capabilities

| Model | Image Gen | Aspect Ratio | Resolution | Notes |
|-------|-----------|--------------|------------|-------|
| gemini-3.1-flash-image-preview | Yes | 14 ratios | 512/1K/2K/4K | Newest Flash, default |
| gemini-3-pro-image-preview | Yes | 10 ratios | 1K/2K/4K | High-quality, no 512 |
| gemini-2.5-flash-image | Yes | 10 ratios | — | Previous Flash |
| gemini-2.5-pro | No | — | — | Text only |
| gemini-2.5-flash | No | — | — | Text only |

**Aspect ratios — gemini-3.1-flash-image-preview (14):**
1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 21:9, 4:1, 1:4, 8:1, 1:8

**Aspect ratios — gemini-3-pro-image-preview and gemini-2.5-flash-image (10):**
1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 21:9

## Troubleshooting

**"API key not valid"** — Check that your API key is correct and set properly.

**"Rate limit exceeded"** — The node retries automatically. Try reducing `batch_size` or wait a moment.

**"No images generated"** — Check the `execution_log` output for details. Possible causes:
- Safety filter blocked the content
- Wrong model selected (use an image model, not text-only)
- API quota exceeded

## License

MIT License — see [LICENSE](LICENSE) file for details.
