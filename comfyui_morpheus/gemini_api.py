import os
import time
import base64
import traceback
from typing import Optional, List, Callable, Tuple
from datetime import datetime
from google import genai
from google.genai import types
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
RETRYABLE_ERRORS = ['rate limit', 'timeout', '429', '503', '500', 'connection']
CRITICAL_ERRORS = ["API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED"]
QUOTA_ERRORS = ["RESOURCE_EXHAUSTED", "quota"]

# Supported models and their capabilities
# is_image_model: True = generates images, False = text only
# resolution_options: list of valid image_size values for this model (empty = not supported)
MODELS_CONFIG = {
    "gemini-3.1-flash-image-preview": {
        "is_image_model": True,
        "supports_aspect_ratio": True,
        "supports_resolution": True,
        "resolution_options": ["512", "1K", "2K", "4K"],
    },
    "gemini-3-pro-image-preview": {
        "is_image_model": True,
        "supports_aspect_ratio": True,
        "supports_resolution": True,
        "resolution_options": ["1K", "2K", "4K"],
    },
    "gemini-2.5-flash-image": {
        "is_image_model": True,
        "supports_aspect_ratio": True,
        "supports_resolution": False,
        "resolution_options": [],
    },
    "gemini-2.5-pro": {
        "is_image_model": False,
        "supports_aspect_ratio": False,
        "supports_resolution": False,
        "resolution_options": [],
    },
    "gemini-2.5-flash": {
        "is_image_model": False,
        "supports_aspect_ratio": False,
        "supports_resolution": False,
        "resolution_options": [],
    },
}

# Safety filter mapping from UI strings to SDK enums
SAFETY_FILTER_MAP = {
    "BLOCK_NONE": types.HarmBlockThreshold.BLOCK_NONE,
    "BLOCK_ONLY_HIGH": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "BLOCK_MEDIUM_AND_ABOVE": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "BLOCK_LOW_AND_ABOVE": types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}


class GeminiAPIClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        max_retries: int = DEFAULT_MAX_RETRIES, 
        retry_delay: float = DEFAULT_RETRY_DELAY,
        enable_logging: bool = True,
        log_level: str = "detailed"
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY env variable not set")
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_logging = enable_logging
        self.log_level = log_level
        self.execution_log = []
        
        try:
            self.client = genai.Client(api_key=self.api_key)
            self._log("INFO", "Gemini client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini client: {str(e)}")
    
    def _log(self, level: str, message: str):
        if not self.enable_logging:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        if self.log_level == "debug":
            print(log_entry)
        self.execution_log.append(log_entry)
    
    def get_execution_log(self) -> str:
        return "\n".join(self.execution_log)
    
    def clear_log(self):
        self.execution_log = []
    
    def _retry_on_failure(self, func: Callable, *args, **kwargs):
        """Retry a function call with exponential backoff."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                if any(x in error_str for x in RETRYABLE_ERRORS):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        self._log("WARN", f"Attempt {attempt + 1}/{self.max_retries} failed, retrying in {wait_time}s... Error: {str(e)[:100]}")
                        time.sleep(wait_time)
                        continue
                
                self._log("ERROR", f"API request failed: {str(e)}")
                raise e
        
        if last_error:
            raise last_error
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a torch tensor (BHWC or HWC) to PIL Image (RGB)."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
        
        np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if np_image.shape[-1] == 1:
            return Image.fromarray(np_image.squeeze(-1), mode='L')
        elif np_image.shape[-1] == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif np_image.shape[-1] == 4:
            return Image.fromarray(np_image[:, :, :3], mode='RGB')
        else:
            raise ValueError(f"Unsupported image shape: {np_image.shape}")
    
    def bytes_to_tensor(self, image_bytes: bytes) -> torch.Tensor:
        """Convert image bytes to BHWC tensor with 4 channels (RGBA)."""
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        np_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_array)
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def generate_image(
        self,
        prompt: str,
        model: str = "gemini-3.1-flash-image-preview",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        safety_filter: str = "BLOCK_NONE",
        number_of_images: int = 1
    ) -> Tuple[List[torch.Tensor], str]:
        """
        Generate images using Gemini API.
        - Uses client.models.generate_content() (new SDK API)
        - Gemini generates ONE image per call; batch loops with seed offset
        """
        start_time = time.time()
        self.clear_log()
        
        self._log("INFO", f"Starting image generation | model: {model}")
        self._log("INFO", f"aspect_ratio={aspect_ratio} | resolution={resolution} | seed={seed}")
        
        model_config = MODELS_CONFIG.get(model, {})
        
        # Build contents list
        contents = []
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        contents.append(full_prompt)
        self._log("DEBUG", f"Prompt: {prompt[:100]}...")
        
        if images:
            self._log("INFO", f"Adding {len(images)} input images")
            for idx, img_tensor in enumerate(images):
                pil_image = self.tensor_to_pil(img_tensor)
                contents.append(pil_image)
                self._log("DEBUG", f"Input image {idx+1}: {pil_image.size}")
        
        # Build ImageConfig (aspect ratio / resolution)
        image_config_params = {}
        if aspect_ratio and model_config.get("supports_aspect_ratio"):
            image_config_params["aspect_ratio"] = aspect_ratio
            self._log("INFO", f"Native aspect ratio: {aspect_ratio}")
        if resolution and model_config.get("supports_resolution"):
            resolution_options = model_config.get("resolution_options", [])
            if resolution in resolution_options:
                image_config_params["image_size"] = resolution
                self._log("INFO", f"Resolution: {resolution}")
            else:
                fallback = "1K"
                self._log("WARN", f"Resolution '{resolution}' not supported by {model}. Supported: {resolution_options}. Falling back to {fallback}")
                image_config_params["image_size"] = fallback
        
        # Build GenerateContentConfig
        # IMPORTANT: Must include "TEXT" alongside "IMAGE" — per official docs:
        # "Silent failures (HTTP 200 with no images) are almost always a responseModalities misconfiguration"
        config_params = {
            "response_modalities": ["TEXT", "IMAGE"],
        }
        if image_config_params:
            config_params["image_config"] = types.ImageConfig(**image_config_params)
        if seed is not None:
            config_params["seed"] = seed
            self._log("DEBUG", f"Seed: {seed}")
        if safety_filter:
            threshold_enum = SAFETY_FILTER_MAP.get(safety_filter, types.HarmBlockThreshold.BLOCK_NONE)
            config_params["safety_settings"] = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=threshold_enum),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=threshold_enum),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=threshold_enum),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=threshold_enum),
            ]
            self._log("DEBUG", f"Safety filter: {safety_filter}")
        
        output_images = []
        num_to_generate = max(1, min(number_of_images, 8))
        self._log("INFO", f"Generating {num_to_generate} image(s) (one API call each)")
        
        for i in range(num_to_generate):
            try:
                # Deterministic seed offset for each image in batch
                if seed is not None:
                    current_seed = (seed + i) & 0x7fffffff
                    config_params["seed"] = current_seed
                    self._log("DEBUG", f"Seed for image {i+1}: {current_seed}")
                
                generation_config = types.GenerateContentConfig(**config_params)
                
                self._log("DEBUG", f"Sending request {i+1}/{num_to_generate}...")
                response = self._retry_on_failure(
                    self.client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=generation_config
                )
                
                # Parse response.parts for inline image data
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            try:
                                image_data = getattr(part.inline_data, 'data', None) or part.inline_data.get('data', '')
                                if image_data:
                                    image_bytes = base64.b64decode(image_data) if isinstance(image_data, str) else image_data
                                    tensor = self.bytes_to_tensor(image_bytes)
                                    output_images.append(tensor)
                                    self._log("INFO", f"Image {len(output_images)} decoded: {tensor.shape}")
                            except Exception as e:
                                self._log("ERROR", f"Error decoding image {i+1}: {e}")
                                traceback.print_exc()
            except Exception as e:
                self._log("ERROR", f"Error on request {i+1}: {e}")
                traceback.print_exc()
        
        execution_time = (time.time() - start_time) * 1000
        
        if not output_images:
            self._log("WARN", "No images generated, returning blank tensor")
            output_images = [torch.zeros((1, 64, 64, 4))]
        
        self._log("INFO", f"Done in {execution_time:.0f}ms — {len(output_images)} image(s) generated")
        return output_images, self.get_execution_log()
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        images: Optional[List[torch.Tensor]] = None,
        seed: Optional[int] = None,
        top_p: float = 0.95,
        max_output_tokens: int = 8192
    ) -> str:
        """Generate text using Gemini API (for text-only models)."""
        start_time = time.time()
        self.clear_log()
        self._log("INFO", f"Starting text generation | model: {model}")
        
        contents = []
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        contents.append(full_prompt)
        
        if images:
            self._log("INFO", f"Adding {len(images)} images")
            for img_tensor in images:
                contents.append(self.tensor_to_pil(img_tensor))
        
        config = types.GenerateContentConfig(
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        if seed is not None:
            config.seed = seed
        
        response = self._retry_on_failure(
            self.client.models.generate_content,
            model=model,
            contents=contents,
            config=config
        )
        
        execution_time = (time.time() - start_time) * 1000
        self._log("INFO", f"Text generation done in {execution_time:.0f}ms")
        return response.text if response.text else ""
