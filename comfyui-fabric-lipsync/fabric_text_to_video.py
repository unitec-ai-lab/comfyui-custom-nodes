import os
import tempfile
import time

import numpy as np
import requests as http_requests
from PIL import Image


def _make_video_output(path: str):
    """Wrap video path as ComfyUI VIDEO type for SaveVideo compatibility."""
    try:
        from comfy_api.input_impl import VideoFromFile
        return VideoFromFile(path)
    except ImportError:
        pass
    try:
        from comfy_api.latest._input_impl import VideoFromFile
        return VideoFromFile(path)
    except (ImportError, AttributeError):
        pass
    return path


class FabricTextToVideo:
    """Generates a talking-head video from an image and text using VEED Fabric 1.0 via fal.ai."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, welcome to my video!",
                    },
                ),
                "fal_api_key": ("STRING", {"default": ""}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
            },
            "optional": {
                "voice_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    CATEGORY = "VEED Fabric 1.0"
    FUNCTION = "execute"

    def execute(self, image, text, fal_api_key, resolution, voice_description=""):
        import fal_client
        import folder_paths

        if not fal_api_key.strip():
            raise ValueError(
                "fal.ai API key is required. Get one at https://fal.ai/dashboard/keys"
            )

        if not text.strip():
            raise ValueError("Text input cannot be empty.")

        os.environ["FAL_KEY"] = fal_api_key.strip()

        # Convert IMAGE tensor (B, H, W, C) to temp PNG and upload
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        img_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            pil_img.save(img_tmp.name)
            img_tmp.close()
            image_url = fal_client.upload_file(img_tmp.name)
        finally:
            os.unlink(img_tmp.name)

        # Build API arguments
        arguments = {
            "image_url": image_url,
            "text": text.strip(),
            "resolution": resolution,
        }
        if voice_description and voice_description.strip():
            arguments["voice_description"] = voice_description.strip()

        # Call fal.ai text-to-video endpoint
        result = fal_client.subscribe(
            "veed/fabric-1.0/text",
            arguments=arguments,
        )

        # Download and save video to ComfyUI output
        video_url = result["video"]["url"]
        response = http_requests.get(video_url)
        response.raise_for_status()

        output_dir = folder_paths.get_output_directory()
        filename = f"fabric_text_to_video_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(response.content)

        return (_make_video_output(output_path),)
