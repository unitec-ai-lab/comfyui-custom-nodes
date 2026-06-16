import os
import tempfile
import time

import numpy as np
import requests as http_requests
import soundfile as sf
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


class FabricLipsync:
    """Generates a lipsync video from an image and audio using VEED Fabric 1.0 via fal.ai."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "fal_api_key": ("STRING", {"default": ""}),
                "resolution": (["720p", "480p"], {"default": "720p"}),
            },
            "optional": {
                "fast_mode": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_NODE = True
    CATEGORY = "VEED Fabric 1.0"
    FUNCTION = "execute"

    def execute(self, image, audio, fal_api_key, resolution, fast_mode=False):
        import fal_client
        import folder_paths

        if not fal_api_key.strip():
            raise ValueError(
                "fal.ai API key is required. Get one at https://fal.ai/dashboard/keys"
            )

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

        # Convert AUDIO tensor to temp WAV and upload
        waveform = audio["waveform"].squeeze().cpu().numpy()
        sample_rate = audio["sample_rate"]
        audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(audio_tmp.name, waveform, sample_rate)
            audio_tmp.close()
            audio_url = fal_client.upload_file(audio_tmp.name)
        finally:
            os.unlink(audio_tmp.name)

        # Call fal.ai API
        model_id = "veed/fabric-1.0/fast" if fast_mode else "veed/fabric-1.0"
        result = fal_client.subscribe(
            model_id,
            arguments={
                "image_url": image_url,
                "audio_url": audio_url,
                "resolution": resolution,
            },
        )

        # Download and save video to ComfyUI output
        video_url = result["video"]["url"]
        response = http_requests.get(video_url)
        response.raise_for_status()

        output_dir = folder_paths.get_output_directory()
        filename = f"fabric_lipsync_{int(time.time())}.mp4"
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            f.write(response.content)

        return (_make_video_output(output_path),)
