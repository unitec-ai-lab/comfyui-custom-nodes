import sys
import traceback
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

from .gemini_api import GeminiAPIClient, MODELS_CONFIG

# -----------------------------
# Utils
# -----------------------------

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Convert any tensor format to BHWC (Batch, Height, Width, Channels)."""
    if x is None:
        raise ValueError("None image tensor")
    if x.ndim == 4 and x.shape[-1] in (1, 3, 4):  # Already BHWC
        return x
    if x.ndim == 4 and x.shape[1] in (1, 3, 4):   # BCHW -> BHWC
        return x.permute(0, 2, 3, 1).contiguous()
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):  # HWC -> BHWC
        return x.unsqueeze(0)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):   # CHW -> BHWC
        return x.permute(1, 2, 0).unsqueeze(0).contiguous()
    raise ValueError(f"Unsupported image shape: {tuple(x.shape)}")

def _pad_bhwc(bhwc: torch.Tensor, H: int, W: int, mode: str = "center") -> torch.Tensor:
    """Pad image tensor to specific dimensions."""
    b, h, w, c = bhwc.shape
    if h == H and w == W:
        return bhwc
    
    if mode == "center":
        top = (H - h) // 2
        bottom = H - h - top
        left = (W - w) // 2
        right = W - w - left
    else:
        top = 0
        left = 0
        bottom = H - h
        right = W - w
    
    bchw = bhwc.permute(0, 3, 1, 2).contiguous()
    bchw = F.pad(bchw, (left, right, top, bottom), value=0.0)
    return bchw.permute(0, 2, 3, 1).contiguous()

# -----------------------------
# Node 1: Morpheus · Batch Images
# -----------------------------

class MorpheusBatchImages:
    """
    Bridge node to collect multiple images and pass them as a batch to Morpheus Gemini node.
    Provides fixed connectors so each connection adds an image to the batch.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"image{i}": ("IMAGE",) for i in range(1, 13)}
        opt["image"] = ("IMAGE",)
        return {
            "required": {},
            "optional": opt,
        }

    RETURN_TYPES = ("IMAGE", "PYOBJECT")
    RETURN_NAMES = ("IMAGE", "ORIGINALS_LIST")
    FUNCTION = "run"
    CATEGORY = "Morpheus"
    OUTPUT_NODE = False

    def run(self, **kwargs):
        """Collect all input images into a batch and a list."""
        imgs: List[torch.Tensor] = []
        
        for k in sorted(kwargs.keys()):
            t = kwargs[k]
            if t is None: 
                continue
            try:
                t = _to_bhwc(t)
            except Exception:
                continue
            
            for i in range(t.shape[0]):
                imgs.append(t[i:i+1])

        if not imgs:
            return (torch.zeros((1, 64, 64, 4), dtype=torch.float32), [])

        max_h = max(x.shape[1] for x in imgs)
        max_w = max(x.shape[2] for x in imgs)
        preview = [_pad_bhwc(x, max_h, max_w, "center") for x in imgs]
        batch_preview = torch.cat(preview, dim=0)
        
        return (batch_preview, imgs)

# -----------------------------
# Node 2: Morpheus · Gemini
# -----------------------------

class MorpheusGemini:
    """
    Gemini image generation node with NATIVE aspect ratio and resolution support.
    Supports gemini-3.1-flash-image-preview, gemini-3-pro-image-preview, gemini-2.5-flash-image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (
                    list(MODELS_CONFIG.keys()),
                    {"default": "gemini-3.1-flash-image-preview"}
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "aspect_ratio": ([
                    "1:1", "16:9", "9:16", "3:2", "2:3",
                    "4:3", "3:4", "4:5", "5:4", "21:9",
                    "4:1", "1:4", "8:1", "1:8"
                ], {"default": "1:1"}),
                "resolution": (["512", "1K", "2K", "4K"], {"default": "1K"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "images_list": ("PYOBJECT",),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "safety_filter": ([
                    "BLOCK_NONE",
                    "BLOCK_ONLY_HIGH",
                    "BLOCK_MEDIUM_AND_ABOVE",
                    "BLOCK_LOW_AND_ABOVE"
                ], {"default": "BLOCK_NONE"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PYOBJECT", "STRING")
    RETURN_NAMES = ("images", "images_list", "execution_log")
    FUNCTION = "generate"
    CATEGORY = "Morpheus"
    OUTPUT_NODE = False
    
    CONTROL_AFTER_GENERATE = ["seed"]

    def generate(
        self, 
        prompt: str, 
        model: str, 
        batch_size: int, 
        seed: int,
        aspect_ratio: str, 
        resolution: str,
        images: Optional[torch.Tensor] = None, 
        images_list: Optional[list] = None,
        system_prompt: str = "", 
        api_key: str = "",
        safety_filter: str = "BLOCK_NONE",
        top_p: float = 0.95, 
        max_tokens: int = 2048
    ) -> Tuple[torch.Tensor, list, str]:
        
        try:
            client = GeminiAPIClient(
                api_key=api_key or None,
                enable_logging=True,
                log_level="detailed"
            )

            payload: List[torch.Tensor] = []
            
            if images_list and isinstance(images_list, (list, tuple)) and len(images_list) > 0:
                payload.extend(list(images_list))
            elif images is not None:
                t = _to_bhwc(images)
                for i in range(t.shape[0]):
                    payload.append(t[i:i+1])

            is_image_model = MODELS_CONFIG.get(model, {}).get("is_image_model", False)

            if is_image_model:
                all_imgs: List[torch.Tensor] = []
                all_logs: List[str] = []
                
                for i in range(max(1, int(batch_size))):
                    current_seed = int(seed) & 0x7fffffff
                    
                    out_imgs, exec_log = client.generate_image(
                        prompt=prompt,
                        model=model,
                        images=payload if payload else None,
                        seed=current_seed,
                        system_prompt=system_prompt if system_prompt else None,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution if MODELS_CONFIG.get(model, {}).get("supports_resolution") else None,
                        safety_filter=safety_filter,
                        number_of_images=1
                    )
                    
                    if not out_imgs:
                        all_imgs.append(torch.zeros((1, 64, 64, 4), dtype=torch.float32))
                    else:
                        all_imgs.extend(out_imgs)
                    
                    batch_log = f"\n{'='*60}\nBatch {i+1}/{batch_size} | Seed: {current_seed}\n{'='*60}\n{exec_log}"
                    all_logs.append(batch_log)
                
                if not all_imgs:
                    result_batch = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
                    result_list = [result_batch]
                else:
                    H0, W0 = all_imgs[0].shape[1], all_imgs[0].shape[2]
                    padded_imgs = [
                        x if (x.shape[1] == H0 and x.shape[2] == W0) 
                        else _pad_bhwc(x, H0, W0) 
                        for x in all_imgs
                    ]
                    result_batch = torch.cat(padded_imgs, dim=0)
                    result_list = all_imgs
                
                combined_log = "\n".join(all_logs)
                return (result_batch, result_list, combined_log)
            
            else:
                text_output = client.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt if system_prompt else None,
                    model=model,
                    images=payload if payload else None,
                    seed=seed,
                    top_p=top_p,
                    max_output_tokens=max_tokens
                )
                
                log_output = f"Text Generation Complete\nModel: {model}\nSeed: {seed}\n\n{text_output}"
                dummy_tensor = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
                return (dummy_tensor, [dummy_tensor], log_output)
        
        except Exception as e:
            error_msg = f"[MorpheusGemini] Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            dummy_tensor = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
            return (dummy_tensor, [dummy_tensor], error_msg)

# -----------------------------
# ComfyUI Registration
# -----------------------------

NODE_CLASS_MAPPINGS = {
    "MorpheusBatchImages": MorpheusBatchImages,
    "MorpheusGemini": MorpheusGemini,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusBatchImages": "Morpheus · Batch Images",
    "MorpheusGemini": "Morpheus · Gemini",
}
