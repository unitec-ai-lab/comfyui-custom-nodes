"""
GridSplitAuto - ComfyUI Custom Node for Grid-Based Image Splitting

This node automatically detects or manually splits grid/collage images
into individual cells, outputting them as a batch of images.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Dict, Any, Union

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


class GridSplitAuto:
    """
    ComfyUI node that splits a grid/collage image into individual cells.
    
    Supports both automatic grid detection using gradient-based projection
    profiles and manual mode with user-specified dimensions.
    """
    
    CATEGORY = "image/grid"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "meta")
    FUNCTION = "split_grid"
    
    COMMON_GRIDS = [
        (2, 2), (3, 3), (4, 4),
        (2, 3), (3, 2),
        (2, 4), (4, 2),
        (2, 5), (5, 2),
        (3, 4), (4, 3),
        (3, 6), (6, 3),
    ]
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["auto", "manual"],),
                "rows": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "inset_px": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "min_cell_size": ("INT", {"default": 64, "min": 16, "max": 1024, "step": 1}),
                "return_order": (["row_major", "column_major"],),
            }
        }
    
    def split_grid(
        self,
        image,
        mode: str,
        rows: int,
        cols: int,
        inset_px: int,
        min_cell_size: int,
        return_order: str
    ) -> Tuple:
        """
        Main entry point for grid splitting.
        
        Args:
            image: Input image (torch tensor or numpy array) in (B, H, W, C) format from ComfyUI
            mode: 'auto' for automatic detection, 'manual' for user-specified grid
            rows: Number of rows (used in manual mode)
            cols: Number of columns (used in manual mode)
            inset_px: Pixel inset to apply to each crop to remove grid lines
            min_cell_size: Minimum allowed cell dimension
            return_order: 'row_major' or 'column_major' for output ordering
            
        Returns:
            Tuple of (batch of cropped images, metadata string)
        """
        img_np = to_numpy(image)
        
        if len(img_np.shape) == 4:
            img_array = img_np[0]
        else:
            img_array = img_np
        
        height, width = img_array.shape[:2]
        
        if mode == "auto":
            detected_rows, detected_cols, meta_info = self._auto_detect_grid(
                img_array, min_cell_size
            )
            if detected_rows is None or detected_cols is None:
                raise ValueError(
                    f"Grid detection failed: {meta_info}. "
                    "Please use manual mode with explicit rows and cols."
                )
            rows, cols = detected_rows, detected_cols
        else:
            meta_info = f"Manual mode: {rows}x{cols} grid specified by user"
        
        cell_width = width // cols
        cell_height = height // rows
        
        if cell_width < min_cell_size or cell_height < min_cell_size:
            raise ValueError(
                f"Cell size ({cell_width}x{cell_height}) is smaller than "
                f"min_cell_size ({min_cell_size}). Reduce grid dimensions or min_cell_size."
            )
        
        crops = self._crop_cells(
            img_array, rows, cols, inset_px, min_cell_size, return_order
        )
        
        if len(crops) == 0:
            raise ValueError(
                "No valid crops produced. Check inset_px value - it may be too large."
            )
        
        batch = np.stack(crops, axis=0).astype(np.float32)
        
        batch_output = to_tensor(batch)
        
        meta = (
            f"Mode: {mode}\n"
            f"Grid: {rows} rows x {cols} cols\n"
            f"Cells: {len(crops)}\n"
            f"Cell size: {cell_width}x{cell_height} (before inset)\n"
            f"Inset: {inset_px}px\n"
            f"Order: {return_order}\n"
            f"Detection info: {meta_info}"
        )
        
        return (batch_output, meta)
    
    def _auto_detect_grid(
        self,
        img_array: np.ndarray,
        min_cell_size: int
    ) -> Tuple[Optional[int], Optional[int], str]:
        """
        Automatically detect grid dimensions using gradient-based projection profiles.
        
        Args:
            img_array: Input image as numpy array (H, W, C)
            min_cell_size: Minimum allowed cell dimension
            
        Returns:
            Tuple of (rows, cols, detection_info_string)
        """
        height, width = img_array.shape[:2]
        
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array.astype(np.float32)
        
        gray = gray.astype(np.float32)
        
        grad_x = np.zeros_like(gray)
        grad_y = np.zeros_like(gray)
        
        grad_x[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) / 2.0
        grad_y[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :]) / 2.0
        
        v_proj = np.sum(grad_x, axis=0)
        h_proj = np.sum(grad_y, axis=1)
        
        window_size = max(3, min(width, height) // 100)
        v_proj_smooth = self._moving_average(v_proj, window_size)
        h_proj_smooth = self._moving_average(h_proj, window_size)
        
        v_lines = self._detect_divider_lines(v_proj_smooth, width, min_cell_size)
        h_lines = self._detect_divider_lines(h_proj_smooth, height, min_cell_size)
        
        if len(v_lines) > 0 and len(h_lines) > 0:
            cols = len(v_lines) + 1
            rows = len(h_lines) + 1
            
            cell_width = width // cols
            cell_height = height // rows
            
            if cell_width >= min_cell_size and cell_height >= min_cell_size:
                return (
                    rows, cols,
                    f"Detected {len(v_lines)} vertical, {len(h_lines)} horizontal dividers "
                    f"(confidence: high)"
                )
        
        best_score = -1
        best_grid = None
        
        for grid_rows, grid_cols in self.COMMON_GRIDS:
            cell_w = width // grid_cols
            cell_h = height // grid_rows
            
            if cell_w < min_cell_size or cell_h < min_cell_size:
                continue
            
            score = self._score_grid_candidate(
                gray, grid_rows, grid_cols, v_proj_smooth, h_proj_smooth
            )
            
            if score > best_score:
                best_score = score
                best_grid = (grid_rows, grid_cols)
        
        if best_grid is not None and best_score > 0.3:
            return (
                best_grid[0], best_grid[1],
                f"Fallback detection: {best_grid[0]}x{best_grid[1]} "
                f"(confidence: {best_score:.2f})"
            )
        
        return (
            None, None,
            "Could not detect grid structure. Low contrast or irregular layout."
        )
    
    def _moving_average(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing to a 1D array."""
        if window < 2:
            return arr
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')
    
    def _detect_divider_lines(
        self,
        projection: np.ndarray,
        dimension: int,
        min_cell_size: int
    ) -> List[int]:
        """
        Detect divider line positions from a projection profile.
        
        Args:
            projection: 1D array of gradient sums
            dimension: Total size (width or height)
            min_cell_size: Minimum cell dimension
            
        Returns:
            List of divider line positions
        """
        if len(projection) == 0:
            return []
        
        threshold = np.mean(projection) + 1.5 * np.std(projection)
        
        peaks = []
        for i in range(1, len(projection) - 1):
            if (projection[i] > threshold and
                projection[i] > projection[i-1] and
                projection[i] > projection[i+1]):
                peaks.append(i)
        
        if len(peaks) == 0:
            return []
        
        cluster_threshold = min_cell_size // 4
        clusters = []
        current_cluster = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peaks[i] - peaks[i-1] <= cluster_threshold:
                current_cluster.append(peaks[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [peaks[i]]
        clusters.append(current_cluster)
        
        dividers = [int(np.mean(cluster)) for cluster in clusters]
        
        margin = min_cell_size // 2
        dividers = [d for d in dividers if margin < d < dimension - margin]
        
        valid_dividers = []
        for i, d in enumerate(dividers):
            if i == 0:
                if d >= min_cell_size:
                    valid_dividers.append(d)
            else:
                if d - valid_dividers[-1] >= min_cell_size:
                    valid_dividers.append(d)
        
        return valid_dividers
    
    def _score_grid_candidate(
        self,
        gray: np.ndarray,
        rows: int,
        cols: int,
        v_proj: np.ndarray,
        h_proj: np.ndarray
    ) -> float:
        """
        Score a candidate grid layout based on discontinuity at theoretical borders.
        
        Args:
            gray: Grayscale image array
            rows: Candidate number of rows
            cols: Candidate number of columns
            v_proj: Vertical projection profile
            h_proj: Horizontal projection profile
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        height, width = gray.shape
        
        v_positions = [int(width * i / cols) for i in range(1, cols)]
        h_positions = [int(height * i / rows) for i in range(1, rows)]
        
        if len(v_proj) == 0 or len(h_proj) == 0:
            return 0.0
        
        v_mean = np.mean(v_proj)
        h_mean = np.mean(h_proj)
        
        v_score = 0.0
        if len(v_positions) > 0 and v_mean > 0:
            sample_width = max(1, width // (cols * 4))
            for pos in v_positions:
                start = max(0, pos - sample_width)
                end = min(len(v_proj), pos + sample_width + 1)
                region_max = np.max(v_proj[start:end])
                v_score += region_max / v_mean
            v_score /= len(v_positions)
        
        h_score = 0.0
        if len(h_positions) > 0 and h_mean > 0:
            sample_height = max(1, height // (rows * 4))
            for pos in h_positions:
                start = max(0, pos - sample_height)
                end = min(len(h_proj), pos + sample_height + 1)
                region_max = np.max(h_proj[start:end])
                h_score += region_max / h_mean
            h_score /= len(h_positions)
        
        if len(v_positions) == 0:
            combined = h_score
        elif len(h_positions) == 0:
            combined = v_score
        else:
            combined = (v_score + h_score) / 2.0
        
        normalized = min(1.0, combined / 3.0)
        
        return normalized
    
    def _crop_cells(
        self,
        img_array: np.ndarray,
        rows: int,
        cols: int,
        inset_px: int,
        min_cell_size: int,
        return_order: str
    ) -> List[np.ndarray]:
        """
        Crop the image into grid cells.
        
        Args:
            img_array: Input image array (H, W, C)
            rows: Number of rows
            cols: Number of columns
            inset_px: Pixel inset for each crop
            min_cell_size: Minimum cell dimension
            return_order: 'row_major' or 'column_major'
            
        Returns:
            List of cropped image arrays
        """
        height, width = img_array.shape[:2]
        cell_width = width // cols
        cell_height = height // rows
        
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
            
            x0 += inset_px
            y0 += inset_px
            x1 -= inset_px
            y1 -= inset_px
            
            crop_width = x1 - x0
            crop_height = y1 - y0
            
            if crop_width < min_cell_size or crop_height < min_cell_size:
                continue
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                continue
            
            crop = img_array[y0:y1, x0:x1].copy()
            crops.append(crop)
        
        return crops


NODE_CLASS_MAPPINGS = {
    "GS_GridSplitAuto": GridSplitAuto
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GS_GridSplitAuto": "🔲 GS Grid Split Auto"
}
