"""
ComfyUI Grid Split - Custom nodes for grid-based image splitting

This package provides nodes for automatically detecting and splitting
grid/collage images into individual cells.

Nodes:
- GridSplitAuto: Original auto/manual grid splitting
- GridAnalyze: Preview detected grid with visual overlay
- GridSplitByCell: Split by cell dimensions (modular approach)
- ImageSelector: Interactive image selection from batch
"""

from .grid_split_auto import NODE_CLASS_MAPPINGS as AUTO_MAPPINGS
from .grid_split_auto import NODE_DISPLAY_NAME_MAPPINGS as AUTO_DISPLAY
from .grid_analyze import NODE_CLASS_MAPPINGS as ANALYZE_MAPPINGS
from .grid_analyze import NODE_DISPLAY_NAME_MAPPINGS as ANALYZE_DISPLAY
from .grid_split_by_cell import NODE_CLASS_MAPPINGS as CELL_MAPPINGS
from .grid_split_by_cell import NODE_DISPLAY_NAME_MAPPINGS as CELL_DISPLAY
from .image_selector import NODE_CLASS_MAPPINGS as SELECTOR_MAPPINGS
from .image_selector import NODE_DISPLAY_NAME_MAPPINGS as SELECTOR_DISPLAY

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(AUTO_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ANALYZE_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CELL_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SELECTOR_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(AUTO_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(ANALYZE_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(CELL_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(SELECTOR_DISPLAY)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"
