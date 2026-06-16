import os
import time
import tempfile
import logging
import threading
import queue

import requests
import yaml
import soundfile as sf
import folder_paths

from comfy_api.latest._input_impl.video_types import VideoFromFile
from comfy.comfy_types import IO
import comfy.model_management

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
config_dir = os.path.join(folder_paths.base_path, "config")
os.makedirs(config_dir, exist_ok=True)


def _get_fal_key():
    try:
        path = os.path.join(config_dir, "fal_config.yml")
        with open(path, "r") as f:
            return yaml.safe_load(f).get("FAL_KEY", "")
    except Exception:
        return os.environ.get("FAL_KEY", "")


def _save_fal_key(key):
    path = os.path.join(config_dir, "fal_config.yml")
    with open(path, "w") as f:
        yaml.dump({"FAL_KEY": key}, f)


# ── Node ──────────────────────────────────────────────────────────────────
class FalLipsyncV3:
    """Lip-sync a video to an audio track using fal.ai sync-lipsync v3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO,),
                "audio": ("AUDIO",),
                "sync_mode": (
                    ["cut_off", "loop", "bounce", "silence", "remap"],
                    {"default": "cut_off"},
                ),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 600, "min": 60, "max": 3600}),
            },
        }

    RETURN_TYPES = (IO.VIDEO,)
    RETURN_NAMES = ("video",)
    FUNCTION = "lipsync"
    CATEGORY = "audio/lipsync"

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _video_to_path(video):
        if hasattr(video, "save_to"):
            tmp = os.path.join(tempfile.mkdtemp(), "input_video.mp4")
            video.save_to(tmp)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                return tmp
            raise ValueError("Failed to save VideoFromFile to disk")
        if isinstance(video, str) and os.path.exists(video):
            return video
        raise ValueError(f"Cannot resolve VIDEO object: {type(video)}")

    @staticmethod
    def _audio_to_path(audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        wv_np = waveform.squeeze(0).cpu().numpy()
        if wv_np.ndim == 2:
            wv_np = wv_np.mean(axis=0) if wv_np.shape[0] > 1 else wv_np[0]
        tmp = os.path.join(tempfile.mkdtemp(), "input_audio.wav")
        sf.write(tmp, wv_np, sample_rate)
        return tmp

    # ── main ──────────────────────────────────────────────────────────────
    def lipsync(self, video, audio, sync_mode, api_key="", timeout=600):
        import fal_client

        # API key
        key = api_key.strip() if api_key else ""
        if key:
            _save_fal_key(key)
            os.environ["FAL_KEY"] = key
        else:
            key = _get_fal_key()
            if key:
                os.environ["FAL_KEY"] = key
        if not os.environ.get("FAL_KEY"):
            raise ValueError(
                "FAL API key not found. Provide it in the node or in config/fal_config.yml"
            )

        t0 = time.time()

        # Prepare files
        video_path = self._video_to_path(video)
        audio_path = self._audio_to_path(audio)
        print(f"[FalLipsync] Video: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB  |  Audio: {os.path.getsize(audio_path) / 1024:.0f} KB")

        print(f"[FalLipsync] Uploading video...")
        video_url = fal_client.upload_file(video_path)
        print(f"[FalLipsync] Video uploaded ({time.time() - t0:.1f}s)")

        print(f"[FalLipsync] Uploading audio...")
        audio_url = fal_client.upload_file(audio_path)
        print(f"[FalLipsync] Audio uploaded ({time.time() - t0:.1f}s)")

        arguments = {
            "video_url": video_url,
            "audio_url": audio_url,
            "sync_mode": sync_mode,
        }

        print(f"[FalLipsync] Calling fal-ai/sync-lipsync/v3  sync_mode={sync_mode}")
        t2 = time.time()

        # Run with interrupt check
        result_q = queue.Queue()
        error_q = queue.Queue()

        def _on_queue_update(update):
            try:
                if isinstance(update, fal_client.Queued):
                    print(f"[FalLipsync] Queued (position={update.position})")
                elif isinstance(update, fal_client.InProgress):
                    if hasattr(update, "logs"):
                        for log in update.logs:
                            msg = log["message"] if isinstance(log, dict) else str(log)
                            print(f"[FalLipsync] {msg}")
                elif isinstance(update, fal_client.Completed):
                    print(f"[FalLipsync] Completed!")
                else:
                    print(f"[FalLipsync] Status: {update}")
            except Exception as e:
                print(f"[FalLipsync] Log error (non-fatal): {e}")

        def _run():
            try:
                res = fal_client.subscribe(
                    "fal-ai/sync-lipsync/v3",
                    arguments=arguments,
                    with_logs=True,
                    on_queue_update=_on_queue_update,
                )
                result_q.put(res)
            except Exception as e:
                print(f"[FalLipsync] ERROR: {type(e).__name__}: {e}")
                error_q.put(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while t.is_alive():
            elapsed = time.time() - t2
            if elapsed > timeout:
                raise TimeoutError(f"[FalLipsync] Timed out after {timeout}s")
            if comfy.model_management.processing_interrupted():
                raise comfy.model_management.InterruptProcessingException(
                    "Processing interrupted"
                )
            time.sleep(1.0)

        if not error_q.empty():
            raise error_q.get()
        if result_q.empty():
            raise RuntimeError("[FalLipsync] No result returned from fal.ai")

        result = result_q.get()
        print(f"[FalLipsync] fal.ai responded in {time.time() - t2:.1f}s")
        result_video_url = result["video"]["url"]

        # Download
        out_dir = os.path.join(folder_paths.get_output_directory(), "videos_fal_lipsync")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"lipsync_{int(time.time())}.mp4")

        print(f"[FalLipsync] Downloading result video...")
        resp = requests.get(result_video_url, timeout=120)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)

        print(f"[FalLipsync] Done! Saved to {out_path} — total: {time.time() - t0:.1f}s")
        return (VideoFromFile(out_path),)


# ── Register ──────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "FalLipsyncV3": FalLipsyncV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalLipsyncV3": "FAL Lipsync v3",
}
