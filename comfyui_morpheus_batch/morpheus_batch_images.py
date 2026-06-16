import torch
import torch.nn.functional as F


def pad_to_size(tensor, target_h, target_w):
    """Pad tensor (B,H,W,C) to target_h x target_w with zeros (center pad)."""
    b, h, w, c = tensor.shape
    if h == target_h and w == target_w:
        return tensor
    pad_top    = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left   = (target_w - w) // 2
    pad_right  = target_w - w - pad_left
    # F.pad works on BCHW
    t = tensor.permute(0, 3, 1, 2)  # B,C,H,W
    t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return t.permute(0, 2, 3, 1)    # B,H,W,C


class MorpheusBatchImages:

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 13):
            optional[f"image{i}"] = ("IMAGE",)
        optional["image"] = ("IMAGE",)
        return {
            "required": {},
            "optional": optional,
        }

    RETURN_TYPES  = ("IMAGE", "LIST")
    RETURN_NAMES  = ("IMAGE", "ORIGINALS_LIST")
    FUNCTION      = "batch_images"
    CATEGORY      = "Morpheus"

    def batch_images(self, **kwargs):
        keys = [f"image{i}" for i in range(1, 13)] + ["image"]
        images = [kwargs[k] for k in keys if k in kwargs and kwargs[k] is not None]

        if not images:
            raise ValueError("MorpheusBatchImages: at least one image input is required.")

        originals = list(images)

        # Encontrar el tamaño máximo entre todas las imágenes
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        # Hacer padding de todas al mismo tamaño
        padded = [pad_to_size(img, max_h, max_w) for img in images]

        batched = torch.cat(padded, dim=0)

        return (batched, originals)


NODE_CLASS_MAPPINGS = {
    "MorpheusBatchImages": MorpheusBatchImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusBatchImages": "MorpheusBatchImages",
}
