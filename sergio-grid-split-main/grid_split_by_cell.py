"""
GridSplitByCell - ComfyUI Custom Node for Modular Grid Splitting

This node splits an image into cells based on cell dimensions (the "module"),
rather than specifying rows and columns directly.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_numpy(image: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """Convert image to numpy array, handling both numpy and torch tensors."""
    if HAS_TORCH and isinstance(image, torch.Tensor):
        return image.cpu().numpy()
    return np.asarray(image)


def to_tensor(array: np.ndarray) -> Union[np.ndarray, "torch.Tensor"]:
    """Convert numpy array to torch tensor if torch is available."""
    if HAS_TORCH:
        return torch.from_numpy(array)
    return array


class GridSplitByCell:
    """
    ComfyUI node that splits an image into cells based on cell dimensions.
    
    This provides a modular approach where you define the size of one cell
    and the node calculates how many cells fit in the image.
    
    Connect this to GridAnalyze to use detected cell dimensions, or
    manually specify the cell size.
    """
    
    CATEGORY = "image/grid"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "meta")
    FUNCTION = "split_by_cell"
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "cell_width": ("INT", {"default": 100, "min": 16, "max": 4096, "step": 1}),
                "cell_height": ("INT", {"default": 100, "min": 16, "max": 4096, "step": 1}),
                "inset_px": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "return_order": (["row_major", "column_major"],),
                "skip_partial_cells": (["yes", "no"],),
            }
        }
    
    def split_by_cell(
        self,
        image,
        cell_width: int,
        cell_height: int,
        inset_px: int,
        return_order: str,
        skip_partial_cells: str
    ) -> Tuple:
        """
        Split image into cells based on cell dimensions.
        
        Args:
            image: Input image
            cell_width: Width of each cell in pixels
            cell_height: Height of each cell in pixels
            inset_px: Pixel inset to apply to each crop
            return_order: 'row_major' or 'column_major'
            skip_partial_cells: Whether to skip cells that don't fit completely
            
        Returns:
            Tuple of (batch of cropped images, metadata string)
        """
        img_np = to_numpy(image)
        
        if len(img_np.shape) == 4:
            img_array = img_np[0]
        else:
            img_array = img_np
        
        height, width = img_array.shape[:2]
        
        skip_partial = skip_partial_cells == "yes"
        
        if skip_partial:
            cols = width // cell_width
            rows = height // cell_height
        else:
            cols = (width + cell_width - 1) // cell_width
            rows = (height + cell_height - 1) // cell_height
        
        if rows < 1 or cols < 1:
            raise ValueError(
                f"Cell size ({cell_width}x{cell_height}) is too large for "
                f"image ({width}x{height}). No cells would fit."
            )
        
        crops = self._crop_cells(
            img_array, rows, cols, cell_width, cell_height,
            inset_px, return_order, skip_partial
        )
        
        if len(crops) == 0:
            raise ValueError(
                "No valid crops produced. Check inset_px value - it may be too large."
            )
        
        batch = np.stack(crops, axis=0).astype(np.float32)
        batch_output = to_tensor(batch)
        
        remainder_x = width - (cols * cell_width)
        remainder_y = height - (rows * cell_height)
        
        meta = (
            f"Image size: {width}x{height}\n"
            f"Cell size: {cell_width}x{cell_height}\n"
            f"Grid: {rows} rows x {cols} cols\n"
            f"Total cells: {len(crops)}\n"
            f"Inset: {inset_px}px\n"
            f"Order: {return_order}\n"
            f"Remainder pixels: {remainder_x}px horizontal, {remainder_y}px vertical\n"
            f"Skip partial: {skip_partial_cells}"
        )
        
        return (batch_output, meta)
    
    def _crop_cells(
        self,
        img_array: np.ndarray,
        rows: int,
        cols: int,
        cell_width: int,
        cell_height: int,
        inset_px: int,
        return_order: str,
        skip_partial: bool
    ) -> List[np.ndarray]:
        """
        Crop the image into grid cells.
        
        Args:
            img_array: Input image array (H, W, C)
            rows: Number of rows
            cols: Number of columns
            cell_width: Width of each cell
            cell_height: Height of each cell
            inset_px: Pixel inset for each crop
            return_order: 'row_major' or 'column_major'
            skip_partial: Whether to skip partial cells
            
        Returns:
            List of cropped image arrays
        """
        height, width = img_array.shape[:2]
        
        crops = []
        
        if return_order == "row_major":
            indices = [(r, c) for r in range(rows) for c in range(cols)]
        else:
            indices = [(r, c) for c in range(cols) for r in range(rows)]
        
        for r, c in indices:
            x0 = c * cell_width
            y0 = r * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height
            
            if skip_partial and (x1 > width or y1 > height):
                continue
            
            x1 = min(x1, width)
            y1 = min(y1, height)
            
            x0_crop = x0 + inset_px
            y0_crop = y0 + inset_px
            x1_crop = x1 - inset_px
            y1_crop = y1 - inset_px
            
            crop_width = x1_crop - x0_crop
            crop_height = y1_crop - y0_crop
            
            if crop_width <= 0 or crop_height <= 0:
                continue
            if x0_crop < 0 or y0_crop < 0 or x1_crop > width or y1_crop > height:
                continue
            
            crop = img_array[y0_crop:y1_crop, x0_crop:x1_crop].copy()
            crops.append(crop)
        
        return crops


NODE_CLASS_MAPPINGS = {
    "GS_GridSplitByCell": GridSplitByCell
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GS_GridSplitByCell": "🔲 GS Grid Split By Cell"
}
