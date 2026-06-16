"""
ComfyUI Morpheus NanoBanana Mask - v25_fix
Custom nodes for batch image processing with Google Gemini 2.5 Flash API integration
"""

from .python.batch_v25_fix import MorpheusBatchImagesCropV25Fix
from .python.mask_v25_fix import MorpheusNanoBananaMaskGeminiV25Fix
from .python.editing_prompt import MorpheusImageEditingPrompt

NODE_CLASS_MAPPINGS = {
    "MorpheusBatchImagesCropV25Fix": MorpheusBatchImagesCropV25Fix,
    "MorpheusNanoBananaMaskGeminiV25Fix": MorpheusNanoBananaMaskGeminiV25Fix,
    "MorpheusImageEditingPrompt": MorpheusImageEditingPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusBatchImagesCropV25Fix": "Morpheus · Batch Images + crop image",
    "MorpheusNanoBananaMaskGeminiV25Fix": "Morpheus · NanoBanana Mask",
    "MorpheusImageEditingPrompt": "Morpheus · Image Editing Prompt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
