import tempfile
import torch
import soundfile as sf
from elevenlabs import ElevenLabs


class ElevenLabsTextToSpeech:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":                             ("STRING",  {"multiline": True, "default": "text to speech"}),
                "api_key":                          ("STRING",  {"default": ""}),
                "voice_id":                         ("STRING",  {"default": ""}),
                "model_id":                         (["eleven_v3", "eleven_multilingual_v2", "eleven_monolingual_v1", "eleven_turbo_v2"], {"default": "eleven_v3"}),
                "seed":                             ("INT",     {"default": 123, "min": 0, "max": 0x7FFFFFFF}),
                "apply_text_normalization":         (["auto", "on", "off"], {"default": "auto"}),
                "voice_settings_stability":         ("FLOAT",   {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "voice_settings_use_speaker_boost": ("BOOLEAN", {"default": True}),
                "voice_settings_similarity_boost":  ("FLOAT",   {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
                "voice_settings_style":             ("FLOAT",   {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "voice_settings_speed":             ("FLOAT",   {"default": 1.00, "min": 0.5, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "generate"
    CATEGORY      = "ElevenLabs"

    def generate(self, text, api_key, voice_id, model_id, seed,
                 apply_text_normalization,
                 voice_settings_stability, voice_settings_use_speaker_boost,
                 voice_settings_similarity_boost, voice_settings_style,
                 voice_settings_speed):

        client = ElevenLabs(api_key=api_key)

        voice_settings = {
            "stability":         voice_settings_stability,
            "similarity_boost":  voice_settings_similarity_boost,
            "style":             voice_settings_style,
            "use_speaker_boost": voice_settings_use_speaker_boost,
            "speed":             voice_settings_speed,
        }

        audio_gen = client.text_to_speech.convert(
            voice_id                 = voice_id,
            output_format            = "mp3_44100_128",
            text                     = text,
            model_id                 = model_id,
            seed                     = seed if seed > 0 else None,
            apply_text_normalization = apply_text_normalization,
            voice_settings           = voice_settings,
        )

        audio_bytes = b"".join(audio_gen)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        data, sample_rate = sf.read(tmp_path, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0).float()

        return ({"waveform": tensor, "sample_rate": int(sample_rate)},)


class ElevenLabsTextToEffect:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":             ("STRING", {"multiline": True, "default": "motorcycle engine revving"}),
                "api_key":          ("STRING", {"default": ""}),
                "duration_seconds": ("FLOAT",  {"default": 5.0, "min": 0.5, "max": 22.0, "step": 0.5}),
                "prompt_influence": ("FLOAT",  {"default": 0.3, "min": 0.0, "max": 1.0,  "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "generate"
    CATEGORY      = "ElevenLabs"

    def generate(self, text, api_key, duration_seconds, prompt_influence):
        client    = ElevenLabs(api_key=api_key)
        audio_gen = client.text_to_sound_effects.convert(
            text             = text,
            duration_seconds = duration_seconds,
            prompt_influence = prompt_influence,
        )
        audio_bytes = b"".join(audio_gen)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        data, sample_rate = sf.read(tmp_path, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0).float()

        return ({"waveform": tensor, "sample_rate": int(sample_rate)},)


NODE_CLASS_MAPPINGS = {
    "ElevenlabsTextToSpeech": ElevenLabsTextToSpeech,
    "ElevenLabsTextToEffect": ElevenLabsTextToEffect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenlabsTextToSpeech": "ElevenlabsTextToSpeech",
    "ElevenLabsTextToEffect": "ElevenLabs Text to Effect",
}
