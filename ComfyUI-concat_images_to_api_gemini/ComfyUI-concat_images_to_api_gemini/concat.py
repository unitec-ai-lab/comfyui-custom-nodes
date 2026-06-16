import math
from typing import List, Optional

import torch
import torch.nn.functional as F

# ---- helpers ----

def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    # ComfyUI images are typically [B,H,W,C] float32 0..1 (CPU)
    if x.dim() == 3:
        # [H,W,C] -> [1,C,H,W]
        H, W, C = x.shape
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
    elif x.dim() == 4:
        # [B,H,W,C] -> [B,C,H,W]
        x = x.permute(0, 3, 1, 2)
    else:
        raise ValueError("Unsupported tensor shape for IMAGE input")
    return x.contiguous()

def _ensure_bhwc(x: torch.Tensor) -> torch.Tensor:
    # [B,C,H,W] -> [B,H,W,C]
    return x.permute(0, 2, 3, 1).contiguous()

def _strip_or_composite_alpha(bchw: torch.Tensor, strip_alpha: bool) -> torch.Tensor:
    # bchw: [B,C,H,W]
    if bchw.shape[1] == 4:
        rgb = bchw[:, :3, :, :]
        a = bchw[:, 3:4, :, :]
        if strip_alpha:
            return rgb
        # composite on black and drop alpha
        return rgb * a
    return bchw

def _resize_letterbox_square(bchw: torch.Tensor, target: int, mode: str) -> torch.Tensor:
    # Resize each frame to fit inside target x target, preserving AR, pad with zeros
    b, c, h, w = bchw.shape
    if h == target and w == target:
        return bchw

    # scale
    scale = min(target / max(1, h), target / max(1, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # interpolation
    interp_map = {
        "nearest-exact": "nearest-exact",
        "nearest": "nearest",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "lanczos": "bicubic",  # torch doesn't expose lanczos in interpolate; use bicubic
    }
    interp = interp_map.get(mode, "bilinear")

    resized = F.interpolate(bchw, size=(new_h, new_w), mode=interp, align_corners=False if interp in ("bilinear","bicubic") else None)

    # pad to square
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return padded

def _flatten_image_inputs(images: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
    frames = []
    for img in images:
        if img is None:
            continue
        if not isinstance(img, torch.Tensor):
            # Some nodes might pass dicts; try to get "image" / "images"
            try:
                # comfy UI usually passes tensors; if not, ignore gracefully
                continue
            except Exception:
                continue
        t = img
        if t.dim() == 3:
            t = t.unsqueeze(0)  # [1,H,W,C]
        # Expect BHWC
        frames.append(t)
    if not frames:
        return []
    # cat along batch dimension (they can be different sizes; we'll resize later)
    return [f for f in frames]  # keep separate; we'll process per-batch


class ConcatImagesToAPIGeminiMPOnly:
    """
    Takes up to 8 IMAGE inputs, rescales ALL images to the same square size derived from megapixels,
    strips/composites alpha, and outputs a single IMAGE batch tensor [B,H,W,C].
    No preview output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "num_images": ("INT", {"default": 8, "min": 1, "max": 8}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 64.0, "step": 0.1}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "lanczos"],),
                "strip_alpha": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "build_batch"
    CATEGORY = "IF Gemini / Utils"

    def build_batch(
        self,
        image1,
        num_images: int = 8,
        megapixels: float = 4.0,
        upscale_method: str = "nearest-exact",
        strip_alpha: bool = False,
        image2=None, image3=None, image4=None, image5=None, image6=None, image7=None, image8=None,
    ):
        imgs = _flatten_image_inputs([image1, image2, image3, image4, image5, image6, image7, image8])
        if not imgs:
            # Return empty 1x1 black frame to avoid pipeline failure
            empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return (empty,)

        # limit
        # Flatten BHWC frames list
        bhwcs: List[torch.Tensor] = []
        for t in imgs:
            # t is [B,H,W,C]; split by batch
            if t.dim() != 4:
                # Try to coerce
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                else:
                    continue
            for b in range(t.shape[0]):
                bhwcs.append(t[b:b+1, ...])
        bhwcs = bhwcs[:num_images]

        # target side from MP
        target_side = max(64, int(round(math.sqrt(max(0.1, megapixels) * 1_000_000))))

        processed_bchw = []
        for bhwc in bhwcs:
            bchw = _ensure_bchw(bhwc)
            bchw = _strip_or_composite_alpha(bchw, strip_alpha)
            bchw = _resize_letterbox_square(bchw, target_side, upscale_method)
            processed_bchw.append(bchw)

        # Now all [1,3,H,W] the same size; cat on batch
        batch_bchw = torch.cat(processed_bchw, dim=0)
        batch_bhwc = _ensure_bhwc(batch_bchw)
        return (batch_bhwc,)


NODE_CLASS_MAPPINGS = {
    "ConcatImagesToAPIGeminiMPOnly": ConcatImagesToAPIGeminiMPOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatImagesToAPIGeminiMPOnly": "Concat Images → IF Gemini (MP Resize, no preview)",
}