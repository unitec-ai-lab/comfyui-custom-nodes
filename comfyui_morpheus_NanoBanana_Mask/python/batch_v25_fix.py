"""
Morpheus Batch Images + Crop Image (Analyzer) - Simplified
Collects images and provides analysis for the composer node
"""

import torch
from google import genai
from google.genai import types
from PIL import Image
import numpy as np
import io


class MorpheusBatchImagesCropV25Fix:
    """
    Node 1: Batch Images + Crop Image
    Collects 1-6 images and provides simple analysis
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_0_subject": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "context": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "*", "STRING")
    RETURN_NAMES = ("crop_image", "images_list", "analysis_text")
    FUNCTION = "analyze_batch"
    CATEGORY = "Morpheus"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def analyze_batch(self, image_0_subject, api_key, image_1=None, image_2=None, 
                     image_3=None, image_4=None, image_5=None, context=""):
        """
        Analyze images and return:
        - crop_image: passthrough of image_0_subject (unchanged)
        - images_list: list of reference images 1-5 (unchanged)
        - analysis_text: Gemini description of images
        """
        
        # Build images list
        all_images = [image_0_subject]
        reference_images = []
        
        for img in [image_1, image_2, image_3, image_4, image_5]:
            if img is not None:
                all_images.append(img)
                reference_images.append(img)
        
        # Convert to PIL for Gemini API
        pil_images = []
        for img_tensor in all_images:
            pil_img = self.tensor_to_pil(img_tensor)
            pil_images.append(pil_img)
        
        # Build analysis prompt - optimized for brevity with grouped ELEMENTS
        analysis_prompt = "Analyze images for composition. Be VERY brief.\n\n"
        analysis_prompt += "Image 0 (SUBJECT): Describe the main subject in 1-2 sentences. Key visual features only.\n"
        
        if len(pil_images) > 1:
            analysis_prompt += f"\nImages 1-{len(pil_images)-1} (ELEMENTS): Describe ALL reference elements together in ONE brief paragraph (maximum 3 sentences total). Don't list them separately. Identify the common type and key shared features.\n"
        
        if context:
            analysis_prompt += f"\nContext: {context}\n"
        
        analysis_prompt += "\nKeep it short and focused."
        
        # Call Gemini API for analysis
        analysis_text = ""
        client = None
        try:
            client = genai.Client(api_key=api_key)
            
            # Build contents: images first, then prompt (SDK accepts PIL Images directly)
            contents = pil_images + [analysis_prompt]
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            
            analysis_text = response.text
            print(f"[Batch Analyzer] Analyzed {len(pil_images)} images successfully")
            
        except Exception as e:
            error_msg = f"Error analyzing images: {str(e)}"
            print(f"[Batch Analyzer] {error_msg}")
            analysis_text = f"Analysis failed: {error_msg}\n\nUsing {len(pil_images)} images for composition."
        finally:
            # Always close the client to release HTTP connections
            if client is not None:
                try:
                    if hasattr(client, 'close'):
                        client.close()
                    else:
                        # Fallback for older SDK versions without close() method
                        del client
                except:
                    pass
        
        # Return unchanged images + analysis
        return (image_0_subject, reference_images, analysis_text)


NODE_CLASS_MAPPINGS = {
    "MorpheusBatchImagesCropV25Fix": MorpheusBatchImagesCropV25Fix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusBatchImagesCropV25Fix": "Morpheus · Batch Images + crop image"
}
