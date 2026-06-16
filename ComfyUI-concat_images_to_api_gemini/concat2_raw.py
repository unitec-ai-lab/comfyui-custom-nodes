import torch

class ConcatTwoImagesRaw:
    """
    Pass-through concat:
    - accepts two IMAGE inputs
    - returns a single IMAGE *list* containing the originals (no resize / no dtype change)
    Compatible con PreviewImage e con nodi che accettano liste (es. API Gemini).
    """
    @classmethod
    def INPUT_TYPES(cls):
        # IMAGE è un torch.FloatTensor (H, W, C) in [0,1] o una lista di tali tensors
        return {
            "required": {
                "image_a": ("IMAGE", ),
                "image_b": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "concat"
    CATEGORY = "IF Gemini / Utils"
    OUTPUT_IS_LIST = (True,)

    def _ensure_list(self, x):
        if x is None:
            return []
        # Alcuni nodi passano liste, altri un singolo tensor
        if isinstance(x, (list, tuple)):
            out = []
            for item in x:
                if item is None:
                    continue
                out.extend(self._ensure_list(item))
            return out
        # Deve essere un torch tensor (H, W, C) in [0,1]
        if isinstance(x, torch.Tensor):
            return [x]
        # In casi rari può arrivare numpy: convertiamo a tensor per evitare errori in PreviewImage
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                # Convertiamo a float32 [0,1] se sembra uint8
                arr = x
                if arr.dtype == np.uint8:
                    t = torch.from_numpy(arr).float() / 255.0
                else:
                    t = torch.from_numpy(arr).float()
                # sistemiamo dimensioni: (H, W, C) atteso da ComfyUI
                if t.ndim == 4 and t.shape[0] == 1:
                    t = t.squeeze(0)
                if t.ndim == 3 and t.shape[0] in (1,3) and t.shape[-1] not in (1,3):
                    # probabilmente è (C, H, W) -> portiamo a (H, W, C)
                    t = t.permute(1, 2, 0).contiguous()
                return [t]
        except Exception:
            pass
        # se è un tipo sconosciuto lo scartiamo per non rompere il grafo
        return []

    def concat(self, image_a, image_b):
        images = []
        images.extend(self._ensure_list(image_a))
        images.extend(self._ensure_list(image_b))
        # Rimuove eventuali None o duplicati accidentali
        images = [im for im in images if isinstance(im, torch.Tensor)]
        return (images,)


NODE_CLASS_MAPPINGS = {
    "ConcatTwoImagesRaw": ConcatTwoImagesRaw,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatTwoImagesRaw": "Concat Two Images (RAW, list output)",
}