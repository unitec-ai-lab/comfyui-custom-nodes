"""
ElevenLabs Instant Voice Clone node for ComfyUI.
Sends an AUDIO tensor to ElevenLabs IVC API and returns the cloned voice_id.
The voice_id can then be used directly in the Text-to-Speech node.
"""

import io
import os
import subprocess
import tempfile
import numpy as np
import requests
import torch


def _find_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg",
              "/opt/homebrew/bin/ffmpeg", "/usr/local/opt/ffmpeg/bin/ffmpeg",
              "/opt/conda/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    return None


def audio_tensor_to_mp3(audio: dict) -> bytes:
    """Convert ComfyUI AUDIO dict → MP3 bytes. Accepts any input format via ffmpeg."""
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    channels = waveform.shape[0]

    # 1. ffmpeg PCM → MP3 (handles any sample rate / channel layout)
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        try:
            pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
            pcm_bytes = pcm.T.flatten().tobytes()
            cmd = [ffmpeg, "-y",
                   "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
                   "-i", "pipe:0", "-vn", "-ar", "44100", "-ac", "2",
                   "-b:a", "192k", "-f", "mp3", "pipe:1"]
            res = subprocess.run(cmd, input=pcm_bytes, capture_output=True, timeout=30)
            if res.returncode == 0 and res.stdout:
                return res.stdout
        except Exception as e:
            print(f"[IVC] ffmpeg failed ({e}), trying torchaudio...")

    # 2. torchaudio MP3
    try:
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, waveform.cpu(), sample_rate, format="mp3")
        return buf.getvalue()
    except Exception as e:
        print(f"[IVC] torchaudio MP3 failed ({e}), trying WAV...")

    # 3. Python wave fallback
    import wave as wavemod
    pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
    pcm_bytes = pcm.T.flatten().tobytes()
    buf = io.BytesIO()
    with wavemod.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


class ElevenlabsInstantVoiceClone:
    """
    Instant Voice Clone via ElevenLabs API.
    Accepts a ComfyUI AUDIO tensor (from Load Audio node).
    Returns the cloned voice_id string — connect to the Text-to-Speech node.
    """

    CATEGORY = "Elevenlabs API integration"
    FUNCTION = "clone_voice"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_id",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "ElevenLabs API key from elevenlabs.io/app/settings/api-keys",
                    },
                ),
                "audio": ("AUDIO",),
                "voice_name": (
                    "STRING",
                    {
                        "default": "My Cloned Voice",
                        "tooltip": "Name to assign to the cloned voice in your ElevenLabs account",
                    },
                ),
            },
            "optional": {
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional description for the cloned voice",
                    },
                ),
                "remove_background_noise": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Apply noise removal to the sample before cloning",
                    },
                ),
            },
        }

    def clone_voice(
        self,
        api_key: str,
        audio: dict,
        voice_name: str = "My Cloned Voice",
        description: str = "",
        remove_background_noise: bool = False,
    ):
        if not api_key.strip():
            raise ValueError("ElevenLabs API key is required.")
        if not voice_name.strip():
            raise ValueError("voice_name cannot be empty.")

        print(f"[IVC] Encoding audio sample...")
        mp3_bytes = audio_tensor_to_mp3(audio)
        print(f"[IVC] Sample size: {len(mp3_bytes) / 1024:.1f} KB")

        files = [("files", ("sample.mp3", mp3_bytes, "audio/mpeg"))]
        data = {
            "name": voice_name.strip(),
            "description": description.strip(),
            "remove_background_noise": str(remove_background_noise).lower(),
        }

        print(f"[IVC] Submitting IVC request to ElevenLabs...")
        r = requests.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers={"xi-api-key": api_key.strip()},
            data=data,
            files=files,
            timeout=120,
        )

        if r.status_code not in (200, 201):
            raise RuntimeError(
                f"ElevenLabs IVC failed {r.status_code}: {r.text[:400]}"
            )

        voice_id = r.json().get("voice_id", "")
        if not voice_id:
            raise RuntimeError(f"No voice_id in response: {r.text[:200]}")

        print(f"[IVC] ✅ Voice cloned! voice_id = {voice_id}")
        print(f"[IVC] Use this voice_id in the TTS node to speak with your cloned voice.")
        return (voice_id,)
