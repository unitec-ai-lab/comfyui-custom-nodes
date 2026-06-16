"""
Morpheus Image Editing Prompt - Simplified
Collects images and generates editing-style prompt template
"""

import torch
from google import genai
from google.genai import types
from PIL import Image
import numpy as np
import io


class MorpheusImageEditingPrompt:
    """
    Node: Image Editing Prompt
    Collects 1-6 images and generates editing-style prompt for composition
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_0_subject": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "prompt_template": ("STRING", {
                    "multiline": True, 
                    "default": "Using the SUBJECT from the first image, integrate the ELEMENTS from the reference images as follows: {action}. Match the style, lighting, and perspective of the original subject."
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "action": ("STRING", {"default": "wear the eyewear naturally with perfect fitting", "multiline": True}),
                "context": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "*", "STRING", "STRING")
    RETURN_NAMES = ("crop_image", "images_list", "analysis_text", "editing_prompt")
    FUNCTION = "build_editing_prompt"
    CATEGORY = "Morpheus"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def build_editing_prompt(self, image_0_subject, api_key, prompt_template, 
                            image_1=None, image_2=None, image_3=None, 
                            image_4=None, image_5=None, action="", context=""):
        """
        Build editing-style prompt and return:
        - crop_image: passthrough of image_0_subject (unchanged)
        - images_list: list of reference images 1-5 (unchanged)
        - editing_prompt: Formatted prompt based on template
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
        
        # Get brief analysis from Gemini for context
        analysis_text = ""
        client = None
        try:
            client = genai.Client(api_key=api_key)
            
            # Simple analysis prompt
            analysis_prompt = "Briefly describe: Image 0 (SUBJECT) and remaining images (ELEMENTS). Be concise (2-3 sentences total)."
            
            # Build contents: images first, then prompt (SDK accepts PIL Images directly)
            contents = pil_images + [analysis_prompt]
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            
            analysis_text = response.text
            print(f"[Image Editing Prompt] Analyzed {len(pil_images)} images successfully")
            
        except Exception as e:
            error_msg = f"Error analyzing images: {str(e)}"
            print(f"[Image Editing Prompt] {error_msg}")
            analysis_text = f"Subject image with {len(reference_images)} reference elements."
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
        
        # Build the editing prompt using template
        # Replace {action} placeholder with actual action
        editing_prompt = prompt_template.replace("{action}", action)
        
        # Add context if provided
        if context:
            editing_prompt += f"\n\nAdditional context: {context}"
        
        print(f"[Image Editing Prompt] Generated editing prompt ({len(editing_prompt)} chars)")
        print(f"[Image Editing Prompt] Analysis text ({len(analysis_text)} chars)")
        
        # Return unchanged images + analysis_text + editing_prompt
        return (image_0_subject, reference_images, analysis_text, editing_prompt)


NODE_CLASS_MAPPINGS = {
    "MorpheusImageEditingPrompt": MorpheusImageEditingPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusImageEditingPrompt": "Morpheus · Image Editing Prompt"
}
