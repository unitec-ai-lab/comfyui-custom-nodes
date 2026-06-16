from .concat import ConcatImagesToAPIGeminiMPOnly
from .concat2_raw import ConcatTwoImagesRaw
from .grid_collage_ar import GridCollageARCols

# Map internal node names to their classes so ComfyUI can discover them.
NODE_CLASS_MAPPINGS = {
    "ConcatImagesToAPIGeminiMPOnly": ConcatImagesToAPIGeminiMPOnly,
    "ConcatTwoImagesRaw": ConcatTwoImagesRaw,
    "GridCollageARCols": GridCollageARCols,
}

# Optional: user-friendly display names in the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConcatImagesToAPIGeminiMPOnly": "Concat Images → Gemini (MP only)",
    "ConcatTwoImagesRaw": "Concat Two Images (RAW passthrough)",
    "GridCollageARCols": "Grid Collage AR Cols",
}
