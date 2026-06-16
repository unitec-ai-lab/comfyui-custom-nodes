import math
import torch


class AudioDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "extra_seconds": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("duration",)
    FUNCTION = "calculate"
    CATEGORY = "audio"

    def calculate(self, audio, extra_seconds):
        waveform = audio["waveform"]      # shape: (1, 1, samples)
        sample_rate = audio["sample_rate"]
        num_samples = waveform.shape[-1]
        duration_seconds = num_samples / sample_rate
        return (math.ceil(duration_seconds) + extra_seconds,)


NODE_CLASS_MAPPINGS = {
    "AudioDuration": AudioDuration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioDuration": "Audio Duration (INT)",
}
