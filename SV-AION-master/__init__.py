"""
ComfyUI_AION - AionTheos Portrait Master Suite

Three custom nodes for ComfyUI:

1. AION - Gemini Portrait Master Generator
   Generates realistic face portraits using Google Gemini API
   (3.1 Pro / 3 Flash) and Nano Banana Pro.
   - 10 optional image inputs for facial feature references
   - 33 face attribute selects with external JSON configurations
   - 3-step flow (with images): Analyze -> Compose -> Generate
   - 2-step flow (without images): Compose -> Generate
   - Full logging and AION JSON output for reproducibility

2. AION - Portrait Master Prompter
   Generates optimized text prompts for Flux2 Klein and zimage.
   - Two outputs: zimage_prompt (keyword-rich) and flux_prompt (natural language)
   - Local mode (no API): builds prompts from combo selections
   - Gemini mode (optional): uses Gemini 3.0 Flash for richer composition
   - Same 33 face attribute selects, no image inputs

3. AION - Portrait Master Fusion
   Merges individual facial-part reference images into one coherent portrait.
   - Direct image-to-image fusion via Nano Banana Pro
   - No intermediate analysis step — images go straight to generation
   - 10 image inputs (eyes, nose, lips, hair, etc.)
   - Optional 33 combo attribute modifiers
   - Generic fusion prompt handles all combinations
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

try:
    from .aion_server import setup_routes
    import server
    setup_routes(server.PromptServer.instance)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
