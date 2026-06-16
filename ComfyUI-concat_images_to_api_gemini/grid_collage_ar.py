import math
import numpy as np
from PIL import Image
import torch


class GridCollageARCols:
    """
    Grid collage che mantiene l'Aspect Ratio di ogni immagine.
    - Determina QUANTE immagini usare solo dai pin collegati (image1..image32).
    - Colonne configurabili (cols).
    - Ogni cella è quadrata (cell_size x cell_size), con letterbox centrato (sfondo bianco).
    - Output: IMAGE torch.Tensor [1, H, W, C] in [0,1], compatibile con Preview/SaveImage.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # immagini opzionali dai pin image1..image32
        image_inputs = {f"image{i}": ("IMAGE",) for i in range(1, 33)}
        return {
            "required": {
                "cols": ("INT", {"default": 4, "min": 1, "max": 32}),
                "cell_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 16}),
            },
            "optional": image_inputs,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "do_collage"
    CATEGORY = "image/compose"

    # ---------------- utils ---------------- #

    def _tensor_to_pil(self, t):
        if not isinstance(t, torch.Tensor):
            raise TypeError("IMAGE input deve essere torch.Tensor")
        # t: [B,H,W,C] o [H,W,C]
        if t.ndim == 4:
            t = t[0]
        # usa solo RGB
        if t.shape[-1] >= 3:
            t = t[..., :3]
        arr = (t.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _pil_to_tensor(self, im):
        arr = np.array(im).astype(np.float32) / 255.0
        if arr.ndim == 2:  # grayscale safety
            arr = np.stack([arr] * 3, axis=-1)
        t = torch.from_numpy(arr)  # [H,W,C]
        return t.unsqueeze(0)      # [1,H,W,C]

    def _letterbox(self, im, cell_size, bg=(255, 255, 255)):
        w, h = im.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (cell_size, cell_size), bg)
        scale = min(cell_size / w, cell_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im_resized = im.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (cell_size, cell_size), bg)
        paste_x = (cell_size - new_w) // 2
        paste_y = (cell_size - new_h) // 2
        canvas.paste(im_resized, (paste_x, paste_y))
        return canvas

    # --------------- main ------------------ #

    def do_collage(self, cols, cell_size, **kwargs):
        # Raccoglie le immagini nell'ordine image1..image32
        pil_images = []
        for i in range(1, 33):
            key = f"image{i}"
            img = kwargs.get(key, None)
            if img is None:
                continue
            # se batch > 1, preleva il primo (Comfy di solito usa B=1 per LoadImage)
            if isinstance(img, torch.Tensor) and img.ndim == 4 and img.shape[0] > 1:
                img = img[0].unsqueeze(0)
            try:
                pil = self._tensor_to_pil(img)
            except Exception:
                # fallback nel caso un custom node passi direttamente PIL/numpy
                if isinstance(img, Image.Image):
                    pil = img.convert("RGB")
                else:
                    raise
            pil_images.append(pil)

        n = len(pil_images)
        if n == 0:
            # Nessuna immagine collegata -> tela bianca 1x1
            blank = Image.new("RGB", (cell_size, cell_size), (255, 255, 255))
            return (self._pil_to_tensor(blank),)

        cols = max(1, int(cols))
        rows = (n + cols - 1) // cols  # ceil

        grid_w = cols * cell_size
        grid_h = rows * cell_size
        grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

        for idx, pil in enumerate(pil_images):
            r = idx // cols
            c = idx % cols
            cell = self._letterbox(pil, cell_size)
            grid.paste(cell, (c * cell_size, r * cell_size))

        return (self._pil_to_tensor(grid),)
