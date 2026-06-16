"""
GridAnalyze - ComfyUI Custom Node for Grid Analysis and Preview

This node analyzes an image to detect grid cell dimensions and outputs
a preview image with the detected first cell highlighted.
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Dict, Any, Union, Optional

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


class GridAnalyze:
    """
    ComfyUI node that analyzes a grid image and outputs:
    - A preview image with the detected first cell highlighted
    - Detected cell dimensions (width and height)
    - Number of rows and columns detected
    
    Use this to verify grid detection before splitting.
    """
    
    CATEGORY = "image/grid"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("preview", "cell_width", "cell_height", "rows", "cols", "meta")
    FUNCTION = "analyze_grid"
    
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
                "detection_mode": (["auto", "by_cell_count", "by_cell_size"],),
                "rows_hint": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "cols_hint": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "cell_width_hint": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "cell_height_hint": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "preview_line_width": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }
    
    def analyze_grid(
        self,
        image,
        detection_mode: str,
        rows_hint: int,
        cols_hint: int,
        cell_width_hint: int,
        cell_height_hint: int,
        preview_line_width: int
    ) -> Tuple:
        """
        Analyze grid and create preview with first cell highlighted.
        
        Args:
            image: Input image
            detection_mode: 'auto', 'by_cell_count', or 'by_cell_size'
            rows_hint: Number of rows (used in by_cell_count mode)
            cols_hint: Number of columns (used in by_cell_count mode)
            cell_width_hint: Cell width in pixels (used in by_cell_size mode)
            cell_height_hint: Cell height in pixels (used in by_cell_size mode)
            preview_line_width: Width of the preview rectangle lines
            
        Returns:
            Tuple of (preview_image, cell_width, cell_height, rows, cols, meta)
        """
        img_np = to_numpy(image)
        
        if len(img_np.shape) == 4:
            img_array = img_np[0]
        else:
            img_array = img_np
        
        height, width = img_array.shape[:2]
        
        if detection_mode == "auto":
            rows, cols, cell_width, cell_height, meta_info = self._auto_detect(img_array)
        elif detection_mode == "by_cell_count":
            rows, cols = rows_hint, cols_hint
            cell_width = width // cols
            cell_height = height // rows
            meta_info = f"By cell count: {rows}x{cols} specified"
        else:
            if cell_width_hint <= 0 or cell_height_hint <= 0:
                raise ValueError("cell_width_hint and cell_height_hint must be > 0 in by_cell_size mode")
            cell_width = cell_width_hint
            cell_height = cell_height_hint
            cols = width // cell_width
            rows = height // cell_height
            meta_info = f"By cell size: {cell_width}x{cell_height}px"
        
        if rows < 1 or cols < 1:
            raise ValueError(f"Invalid grid: {rows} rows x {cols} cols")
        
        preview = self._create_preview(
            img_array, rows, cols, cell_width, cell_height, preview_line_width
        )
        
        preview_batch = np.expand_dims(preview, axis=0).astype(np.float32)
        preview_output = to_tensor(preview_batch)
        
        meta = (
            f"Mode: {detection_mode}\n"
            f"Image size: {width}x{height}\n"
            f"Detected grid: {rows} rows x {cols} cols\n"
            f"Cell size: {cell_width}x{cell_height}px\n"
            f"Total cells: {rows * cols}\n"
            f"Info: {meta_info}"
        )
        
        return (preview_output, cell_width, cell_height, rows, cols, meta)
    
    def _auto_detect(self, img_array: np.ndarray) -> Tuple[int, int, int, int, str]:
        """
        Auto-detect grid dimensions using gradient analysis.
        
        Returns:
            Tuple of (rows, cols, cell_width, cell_height, meta_info)
        """
        height, width = img_array.shape[:2]
        min_cell_size = 64
        
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
                    rows, cols, cell_width, cell_height,
                    f"Detected dividers: {len(v_lines)}V x {len(h_lines)}H (high confidence)"
                )
        
        best_score = -1
        best_grid = None
        
        for grid_rows, grid_cols in self.COMMON_GRIDS:
            cell_w = width // grid_cols
            cell_h = height // grid_rows
            
            if cell_w < min_cell_size or cell_h < min_cell_size:
                continue
            
            score = self._score_grid_candidate(
                gray, grid_rows, grid_cols, v_proj_smooth, h_proj_smooth, width, height
            )
            
            if score > best_score:
                best_score = score
                best_grid = (grid_rows, grid_cols)
        
        if best_grid is not None:
            rows, cols = best_grid
            cell_width = width // cols
            cell_height = height // rows
            confidence = "high" if best_score > 0.5 else "medium" if best_score > 0.3 else "low"
            return (
                rows, cols, cell_width, cell_height,
                f"Fallback: {rows}x{cols} (confidence: {confidence}, score: {best_score:.2f})"
            )
        
        rows, cols = 2, 2
        cell_width = width // 2
        cell_height = height // 2
        return (rows, cols, cell_width, cell_height, "Default 2x2 (detection failed)")
    
    def _create_preview(
        self,
        img_array: np.ndarray,
        rows: int,
        cols: int,
        cell_width: int,
        cell_height: int,
        line_width: int
    ) -> np.ndarray:
        """
        Create a preview image with grid overlay.
        
        Draws:
        - First cell in green (the "module")
        - All other grid lines in semi-transparent red
        """
        if img_array.max() <= 1.0:
            img_uint8 = (img_array * 255).astype(np.uint8)
        else:
            img_uint8 = img_array.astype(np.uint8)
        
        pil_img = Image.fromarray(img_uint8)
        draw = ImageDraw.Draw(pil_img, 'RGBA')
        
        height, width = img_array.shape[:2]
        
        for c in range(1, cols):
            x = c * cell_width
            draw.line([(x, 0), (x, height)], fill=(255, 100, 100, 180), width=line_width)
        
        for r in range(1, rows):
            y = r * cell_height
            draw.line([(0, y), (width, y)], fill=(255, 100, 100, 180), width=line_width)
        
        x0, y0 = 0, 0
        x1, y1 = cell_width, cell_height
        
        draw.rectangle(
            [(x0 + line_width, y0 + line_width), 
             (x1 - line_width, y1 - line_width)],
            outline=(0, 255, 0, 255),
            width=line_width * 2
        )
        
        preview = np.array(pil_img).astype(np.float32) / 255.0
        
        if preview.shape[2] == 4:
            preview = preview[:, :, :3]
        
        return preview
    
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
    ) -> list:
        """Detect divider line positions from a projection profile."""
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
            elif len(valid_dividers) > 0:
                if d - valid_dividers[-1] >= min_cell_size:
                    valid_dividers.append(d)
            else:
                if d >= min_cell_size:
                    valid_dividers.append(d)
        
        return valid_dividers
    
    def _score_grid_candidate(
        self,
        gray: np.ndarray,
        rows: int,
        cols: int,
        v_proj: np.ndarray,
        h_proj: np.ndarray,
        width: int,
        height: int
    ) -> float:
        """Score a candidate grid layout."""
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


NODE_CLASS_MAPPINGS = {
    "GS_GridAnalyze": GridAnalyze
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GS_GridAnalyze": "🔲 GS Grid Analyze"
}
