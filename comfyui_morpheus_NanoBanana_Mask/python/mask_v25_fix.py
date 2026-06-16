"""
Morpheus NanoBanana Mask (Composer) - Simplified
Uses analysis + user prompt to generate images with Gemini Nano Banana
"""

import torch
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
import random
from datetime import datetime


class MorpheusNanoBananaMaskGeminiV25Fix:
    """
    Node 2: NanoBanana Mask (Composer)
    Generates images using Gemini 2.5 Flash Image
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_image": ("IMAGE",),
                "images_list": ("*",),
                "analysis_text": ("STRING", {"multiline": True}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Generate a professional image incorporating the elements"}),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_after_generate": (["fixed", "randomize", "increment", "decrement"], {"default": "randomize"}),
            },
            "optional": {
                "aspect_ratio": (["auto", "original", "1:1", "3:4", "4:3", "9:16", "16:9"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "Morpheus"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI IMAGE tensor to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI IMAGE tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if len(np_image.shape) == 2:
            np_image = np.stack([np_image] * 3, axis=-1)
        elif np_image.shape[2] == 4:
            np_image = np_image[:, :, :3]
        tensor = torch.from_numpy(np_image)[None,]
        return tensor
    
    def create_error_image(self, error_message, width=1024, height=1024):
        """Create red error image with message"""
        img = Image.new('RGB', (width, height), color=(180, 40, 40))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        title = "GENERATION ERROR"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((width - title_width) // 2, 40), title, fill=(255, 255, 255), font=font)
        
        # Word wrap error message
        max_width = width - 80
        lines = []
        words = error_message.split()
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font_small)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = 120
        for line in lines[:20]:
            draw.text((40, y_offset), line, fill=(255, 255, 200), font=font_small)
            y_offset += 30
        
        return img
    
    def save_error_image(self, error_image):
        """Save error image to errors folder"""
        try:
            errors_dir = os.path.join(os.path.dirname(__file__), "..", "errors")
            os.makedirs(errors_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_{timestamp}.png"
            filepath = os.path.join(errors_dir, filename)
            
            error_image.save(filepath)
            print(f"[NanoBanana Mask] Error image saved: {filepath}")
            
        except Exception as e:
            print(f"[NanoBanana Mask] Failed to save error image: {e}")
    
    def build_info_text(self, pil_images, combined_prompt, aspect_ratio, seed, seed_suffix, 
                        latency_ms, status, finish_reason, error_message=None):
        """Build detailed info text about the generation request"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        info_lines = [
            "=" * 60,
            "Morpheus NanoBanana Mask - Generation Info",
            "=" * 60,
            f"Timestamp: {timestamp}",
            f"Model: gemini-2.5-flash-image",
            f"Aspect Ratio: {aspect_ratio}",
            f"Seed: {seed} (variation: {seed_suffix})",
            f"Status: {status}",
            f"Latency: {latency_ms:.2f} ms",
            "",
            f"--- Images Sent ({len(pil_images)} total) ---"
        ]
        
        for i, img in enumerate(pil_images):
            width, height = img.size
            ratio = f"{width}:{height}"
            if width == height:
                ratio = "1:1"
            elif abs(width/height - 4/3) < 0.1:
                ratio = "4:3"
            elif abs(width/height - 3/4) < 0.1:
                ratio = "3:4"
            elif abs(width/height - 16/9) < 0.1:
                ratio = "16:9"
            elif abs(width/height - 9/16) < 0.1:
                ratio = "9:16"
            
            info_lines.append(f"Image {i}: {width}x{height} pixels ({ratio}), mode={img.mode}")
        
        # Build config info
        if aspect_ratio in ["auto", "original"]:
            config_str = '{"response_modalities": ["IMAGE"]}'
        else:
            config_str = f'{{"response_modalities": ["IMAGE"], "image_config": {{"aspect_ratio": "{aspect_ratio}"}}}}'
        
        info_lines.extend([
            "",
            f"--- Prompt Sent ({len(combined_prompt)} characters) ---",
            combined_prompt,
            "",
            "--- Request Metadata ---",
            "{",
            f'  "model": "gemini-2.5-flash-image",',
            f'  "images_count": {len(pil_images)},',
            f'  "aspect_ratio_setting": "{aspect_ratio}",',
            f'  "config": {config_str},',
            f'  "prompt_length": {len(combined_prompt)}',
            "}",
            "",
            "--- Response Info ---",
            f"Status: {status}",
            f"Latency: {latency_ms:.2f} ms",
            f"Finish Reason: {finish_reason}"
        ])
        
        if error_message:
            info_lines.extend([
                "",
                "--- Error Details ---",
                error_message
            ])
        
        info_lines.append("=" * 60)
        
        return "\n".join(info_lines)
    
    def generate_image(self, crop_image, images_list, analysis_text, user_prompt, 
                       api_key, seed, control_after_generate, aspect_ratio="auto"):
        """
        Generate image using Gemini 2.5 Flash Image
        """
        
        try:
            # Convert all images to PIL
            crop_pil = self.tensor_to_pil(crop_image)
            reference_pils = []
            
            # Add reference images
            for img_tensor in images_list:
                if img_tensor is not None:
                    reference_pils.append(self.tensor_to_pil(img_tensor))
            
            # Order images based on aspect_ratio setting
            # When aspect_ratio="original", crop_image must be LAST (Gemini uses last image for aspect ratio)
            if aspect_ratio == "original":
                pil_images = reference_pils + [crop_pil]
            else:
                pil_images = [crop_pil] + reference_pils
            
            # Generate seed-based variation suffix to ensure different outputs
            # Use seed to generate a consistent but unique variation prompt
            random.seed(seed)
            variation_words = [
                "with unique style", "with distinct details", "with creative interpretation",
                "with subtle variations", "with artistic flair", "with refined details",
                "with enhanced composition", "with natural flow", "with dynamic energy",
                "with balanced aesthetics", "with cohesive elements", "with harmonious blend"
            ]
            seed_suffix = random.choice(variation_words)
            
            # Build combined prompt with seed variation
            combined_prompt = f"""# Image Analysis:
{analysis_text}

# Task:
{user_prompt} {seed_suffix}
"""
            
            # Build contents: images first, then prompt (SDK accepts PIL Images directly)
            contents = pil_images + [combined_prompt]
            
            # Log detailed info about request
            start_time = datetime.now()
            print(f"[NanoBanana Mask] === Generation Request ===")
            print(f"[NanoBanana Mask] Seed: {seed} (variation: {seed_suffix})")
            print(f"[NanoBanana Mask] Images: {len(pil_images)} total")
            for i, img in enumerate(pil_images):
                print(f"[NanoBanana Mask]   Image {i}: {img.size[0]}x{img.size[1]} pixels, mode={img.mode}")
            print(f"[NanoBanana Mask] Prompt length: {len(combined_prompt)} characters")
            print(f"[NanoBanana Mask] Prompt preview: {combined_prompt[:200]}...")
            print(f"[NanoBanana Mask] Aspect ratio: {aspect_ratio}")
            
            # Call Gemini 2.5 Flash Image API
            client = genai.Client(api_key=api_key)
            
            try:
                # Build config based on aspect_ratio setting
                # "auto" and "original" don't pass aspect_ratio parameter (use default/last image behavior)
                if aspect_ratio in ["auto", "original"]:
                    config = types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    )
                else:
                    # Use image_config dict with aspect_ratio parameter (e.g., "16:9", "1:1")
                    config = types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config={"aspect_ratio": aspect_ratio}
                    )
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=contents,
                    config=config
                )
            finally:
                # Always close the client to release HTTP connections
                try:
                    if hasattr(client, 'close'):
                        client.close()
                    else:
                        # Fallback for older SDK versions without close() method
                        del client
                except:
                    pass
            
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000
            
            # Detailed logging of API response
            print(f"[NanoBanana Mask] Response received in {latency_ms:.0f}ms")
            
            # Check if response has candidates (API may reject request)
            if not response.candidates or len(response.candidates) == 0:
                error_details = f"API returned no candidates. "
                if hasattr(response, 'prompt_feedback'):
                    error_details += f"Feedback: {response.prompt_feedback}"
                print(f"[NanoBanana Mask] ERROR: {error_details}")
                raise ValueError(error_details)
            
            candidate = response.candidates[0]
            
            # Log finish reason for debugging
            finish_reason = str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else "UNKNOWN"
            print(f"[NanoBanana Mask] Finish reason: {finish_reason}")
            
            # Check for safety blocks or other issues
            if finish_reason not in ["STOP", "FINISH_REASON_STOP", "1"]:
                print(f"[NanoBanana Mask] WARNING: Unusual finish reason: {finish_reason}")
            
            # Extract generated image
            output_image = None
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        img_bytes = part.inline_data.data
                        output_image = Image.open(io.BytesIO(img_bytes))
                        break
            
            if output_image is None:
                error_msg = f"No image in response. Finish reason: {finish_reason}"
                if hasattr(candidate, 'safety_ratings'):
                    error_msg += f", Safety: {candidate.safety_ratings}"
                raise ValueError(error_msg)
            
            print(f"[NanoBanana Mask] Image generated successfully: {output_image.size}")
            
            # Build info text
            info_text = self.build_info_text(
                pil_images=pil_images,
                combined_prompt=combined_prompt,
                aspect_ratio=aspect_ratio,
                seed=seed,
                seed_suffix=seed_suffix,
                latency_ms=latency_ms,
                status="success",
                finish_reason=finish_reason,
                error_message=None
            )
            
            # Convert to tensor and return
            return (self.pil_to_tensor(output_image), info_text)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"[NanoBanana Mask] {error_msg}")
            
            # Create and save error image
            error_image = self.create_error_image(error_msg)
            self.save_error_image(error_image)
            
            # Build info text for error case
            info_text = self.build_info_text(
                pil_images=pil_images if 'pil_images' in locals() else [],
                combined_prompt=combined_prompt if 'combined_prompt' in locals() else "",
                aspect_ratio=aspect_ratio,
                seed=seed,
                seed_suffix=seed_suffix if 'seed_suffix' in locals() else "N/A",
                latency_ms=0,
                status="error",
                finish_reason="ERROR",
                error_message=error_msg
            )
            
            # Return error image as tensor with info
            return (self.pil_to_tensor(error_image), info_text)


NODE_CLASS_MAPPINGS = {
    "MorpheusNanoBananaMaskGeminiV25Fix": MorpheusNanoBananaMaskGeminiV25Fix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MorpheusNanoBananaMaskGeminiV25Fix": "Morpheus · NanoBanana Mask"
}
