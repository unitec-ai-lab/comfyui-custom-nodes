import torch
import numpy as np
import base64
import io
import random
import json
import urllib.request
import urllib.error
from PIL import Image


def tensor_to_base64(tensor):
    arr = (tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4", "4:5", "5:4", "21:9", "none"]
RESOLUTIONS   = ["1K", "2K", "4K", "512"]
SAFETY_LEVELS = ["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"]
AFTER_GEN     = ["randomize", "fixed"]

MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# Modelos que generan imagen
IMAGE_MODELS = {
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
}


class MorpheusGemini:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt":        ("STRING", {"multiline": True, "default": ""}),
                "model":         (MODELS,   {"default": "gemini-3-pro-image-preview"}),
                "batch_size":    ("INT",    {"default": 1, "min": 1, "max": 8}),
                "seed":          ("INT",    {"default": 42, "min": 0, "max": 0x7FFFFFFF}),
                "aspect_ratio":  (ASPECT_RATIOS, {"default": "9:16"}),
                "resolution":    (RESOLUTIONS,   {"default": "2K"}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key":       ("STRING", {"multiline": False, "default": ""}),
                "safety_filter": (SAFETY_LEVELS, {"default": "BLOCK_NONE"}),
                "top_p":         ("FLOAT",  {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens":    ("INT",    {"default": 2092, "min": 64, "max": 8192, "step": 1}),
            },
            "optional": {
                "images":      ("IMAGE",),
                "images_list": ("LIST",),
            }
        }

    RETURN_TYPES  = ("IMAGE", "LIST", "STRING")
    RETURN_NAMES  = ("images", "images_list", "execution_log")
    FUNCTION      = "run"
    CATEGORY      = "Morpheus"

    def run(self, prompt, model, batch_size, seed, aspect_ratio, resolution,
            system_prompt, api_key, safety_filter, top_p, max_tokens,
            images=None, images_list=None):

        log_lines = []

        if seed == 0:
            seed = random.randint(1, 0x7FFFFFFF)
        random.seed(seed)
        log_lines.append(f"seed: {seed} | model: {model}")

        parts = []

        if system_prompt and system_prompt.strip():
            parts.append({"text": system_prompt.strip()})

        if images is not None:
            for i in range(images.shape[0]):
                b64 = tensor_to_base64(images[i:i+1])
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
                log_lines.append(f"attached image {i+1} from batch")

        if images_list:
            for idx, img_t in enumerate(images_list):
                b64 = tensor_to_base64(img_t)
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
                log_lines.append(f"attached image from images_list[{idx}]")

        parts.append({"text": prompt})

        safety_settings = [
            {"category": c, "threshold": safety_filter}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        is_image_model = model in IMAGE_MODELS

        if is_image_model:
            gen_config = {
                "temperature": 1.0,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "response_modalities": ["IMAGE", "TEXT"],
            }
            image_gen_config = {}
            if aspect_ratio != "none":
                image_gen_config["aspect_ratio"] = aspect_ratio
            if resolution:
                image_gen_config["image_size"] = resolution
            if image_gen_config:
                gen_config["image_generation_config"] = image_gen_config
        else:
            gen_config = {
                "temperature": 1.0,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "response_modalities": ["TEXT"],
            }

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generation_config": gen_config,
            "safety_settings": safety_settings,
        }).encode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )

        out_tensors = []
        for b in range(batch_size):
            log_lines.append(f"calling Gemini API (batch {b+1}/{batch_size})...")
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    response = json.loads(resp.read().decode("utf-8"))

                candidates = response.get("candidates", [])
                found_image = False
                for candidate in candidates:
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inline_data" in part:
                            t = base64_to_tensor(part["inline_data"]["data"])
                            out_tensors.append(t)
                            log_lines.append(f"  -> image received ({t.shape[2]}x{t.shape[1]})")
                            found_image = True
                        elif "text" in part:
                            log_lines.append(f"  -> text: {part['text'][:120]}")

                if not found_image:
                    log_lines.append("  -> WARNING: no image in response")

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                log_lines.append(f"  -> HTTP {e.code}: {body[:300]}")
            except Exception as ex:
                log_lines.append(f"  -> ERROR: {str(ex)}")

        if not out_tensors:
            blank = torch.zeros(1, 512, 512, 3)
            out_tensors.append(blank)
            log_lines.append("returning blank image (no output from API)")

        batched = torch.cat(out_tensors, dim=0)
        return (batched, out_tensors, "\n".join(log_lines))


NODE_CLASS_MAPPINGS = {
    "MorpheusGemini": MorpheusGemini,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusGemini": "Morpheus · Gemini",
}
