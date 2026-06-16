from .elevenlabs_text_to_effect import ElevenlabsTextToEffect
from .elevenlabs_text_to_speech import ElevenlabsTextToSpeech
from .elevenlabs_voice_clone import ElevenlabsInstantVoiceClone

NODE_CLASS_MAPPINGS = {
    "ElevenlabsTextToSpeech": ElevenlabsTextToSpeech,
    "ElevenlabsTextToEffect": ElevenlabsTextToEffect,
    "ElevenlabsInstantVoiceClone": ElevenlabsInstantVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Elevenlabs text to speech": "Create your speech with Elevenlabs API",
    "Elevenlabs text to effect": "Create your effect with Elevenlabs API",
    "ElevenlabsInstantVoiceClone": "ElevenLabs Instant Voice Clone 🎤",
}
