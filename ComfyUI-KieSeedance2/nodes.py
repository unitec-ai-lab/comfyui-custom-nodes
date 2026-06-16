"""
Kie.ai Seedance 2.0 — ComfyUI Nodes
=====================================
Video generation via https://api.kie.ai using Bytedance Seedance 2.0.

All nodes output native ComfyUI VIDEO type — compatible with Save Video node.
All inputs are real ComfyUI types (IMAGE, AUDIO) — no raw URLs.
Images/audio are uploaded to Kie's temp storage automatically.

Nodes:
  KieSeedance2_TextToVideo      — Text prompt → VIDEO
  KieSeedance2_ImageToVideo     — IMAGE (start frame + optional end frame) → VIDEO
  KieSeedance2_ReferenceToVideo — Reference IMAGEs + AUDIO → VIDEO
  KieSeedance2_ApiKey           — Store API key once, wire to all nodes
"""

import io
import json
import os
import tempfile
import time

import numpy as np
import requests
import torch
from PIL import Image

try:
    import folder_paths
except ImportError:
    class folder_paths:
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.expanduser("~"), "comfyui_output")

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "https://api.kie.ai"
UPLOAD_BASE = "https://api.kie.ai"
CREATE_TASK = f"{API_BASE}/api/v1/jobs/createTask"
RECORD_INFO = f"{API_BASE}/api/v1/jobs/recordInfo"
FILE_UPLOAD = f"{UPLOAD_BASE}/api/file-stream-upload"
MODEL_NAME = "bytedance/seedance-2"
POLL_INTERVAL = 10
MAX_WAIT = 900

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]
RESOLUTIONS = ["480p", "720p"]

# ── Helpers: Auth ─────────────────────────────────────────────────────────────

def _get_api_key(api_key_input):
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()
    env_key = os.environ.get("KIE_API_KEY", "")
    if env_key:
        return env_key
    raise RuntimeError(
        "No Kie.ai API key. Paste it in the api_key field or set KIE_API_KEY env var. "
        "Get yours at https://kie.ai/api-key"
    )


def _auth_header(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def _json_headers(api_key):
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# ── Helpers: File Upload ──────────────────────────────────────────────────────

def _upload_image(api_key, image_tensor):
    """Upload a ComfyUI IMAGE tensor [B,H,W,3] to Kie temp storage → URL."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    print(f"[Kie Upload] Uploading image ({buf.getbuffer().nbytes // 1024} KB)...")
    resp = requests.post(
        FILE_UPLOAD,
        headers=_auth_header(api_key),
        files={"file": ("image.png", buf, "image/png")},
        data={"uploadPath": "comfyui"},
        timeout=120,
    )
    return _parse_upload_response(resp)


def _upload_audio(api_key, audio_dict):
    """Upload a ComfyUI AUDIO dict {waveform: Tensor[B,C,T], sample_rate: int} → URL."""
    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]

    if waveform.dim() == 3:
        waveform = waveform[0]  # [C, T]
    if waveform.shape[0] > 2:
        waveform = waveform[:2]  # max stereo

    # Convert to WAV bytes
    import wave
    buf = io.BytesIO()
    channels = waveform.shape[0]
    samples = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype(np.int16)
    # Interleave channels
    interleaved = np.empty(channels * samples.shape[1], dtype=np.int16)
    for c in range(channels):
        interleaved[c::channels] = samples[c]
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())
    buf.seek(0)
    print(f"[Kie Upload] Uploading audio ({buf.getbuffer().nbytes // 1024} KB)...")
    resp = requests.post(
        FILE_UPLOAD,
        headers=_auth_header(api_key),
        files={"file": ("audio.wav", buf, "audio/wav")},
        data={"uploadPath": "comfyui"},
        timeout=120,
    )
    return _parse_upload_response(resp)


def _parse_upload_response(resp):
    if resp.status_code == 401:
        raise RuntimeError("[Kie Upload] Auth failed — check API key.")
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success") and body.get("code", 0) != 200:
        raise RuntimeError(f"[Kie Upload] Failed: {body.get('msg', body)}")
    data = body.get("data", {})
    url = data.get("downloadUrl") or data.get("fileUrl") or ""
    if not url:
        raise RuntimeError(f"[Kie Upload] No URL in response: {body}")
    print(f"[Kie Upload] OK → {url[:80]}...")
    return url


# ── Helpers: Task API ─────────────────────────────────────────────────────────

def _check_response(resp):
    if resp.status_code == 401:
        raise RuntimeError("[Kie] Auth failed — check your API key.")
    if resp.status_code == 402:
        raise RuntimeError("[Kie] Insufficient credits — top up at kie.ai")
    if resp.status_code == 429:
        raise RuntimeError("[Kie] Rate limited — wait and retry.")
    resp.raise_for_status()
    body = resp.json()
    code = body.get("code", 0)
    if code != 200:
        raise RuntimeError(f"[Kie] API error {code}: {body.get('msg', 'unknown')}")
    return body


def _create_task(api_key, input_params):
    payload = {"model": MODEL_NAME, "input": input_params}
    resp = requests.post(CREATE_TASK, headers=_json_headers(api_key), json=payload, timeout=60)
    body = _check_response(resp)
    task_id = body.get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"[Kie] No taskId in response: {body}")
    print(f"[Kie Seedance2] Task created: {task_id}")
    return task_id


def _poll_task(api_key, task_id):
    deadline = time.time() + MAX_WAIT
    headers = _json_headers(api_key)
    while time.time() < deadline:
        resp = requests.get(RECORD_INFO, headers=headers,
                            params={"taskId": task_id}, timeout=30)
        body = _check_response(resp)
        data = body.get("data", {})
        state = data.get("state", "unknown")
        progress = data.get("progress", 0)
        print(f"[Kie Seedance2] {task_id} → {state} ({progress}%)")

        if state == "success":
            return data
        if state == "fail":
            raise RuntimeError(
                f"[Kie Seedance2] Task failed: {data.get('failCode', '')} "
                f"— {data.get('failMsg', 'unknown')}"
            )
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(f"[Kie Seedance2] Timeout after {MAX_WAIT}s for task {task_id}")


def _extract_video_url(data):
    result_str = data.get("resultJson", "")
    if not result_str:
        raise RuntimeError("[Kie Seedance2] No resultJson in completed task")
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    urls = result.get("resultUrls", [])
    if urls:
        return urls[0]
    raise RuntimeError(f"[Kie Seedance2] No video URL in result: {result}")


def _extract_last_frame_url(data):
    result_str = data.get("resultJson", "")
    if not result_str:
        return None
    result = json.loads(result_str) if isinstance(result_str, str) else result_str
    urls = result.get("lastFrameUrl", [])
    return urls[0] if urls else None


# ── Helpers: Video download → ComfyUI VIDEO type ─────────────────────────────

def _download_video(video_url, prefix="seedance2"):
    """Download video to ComfyUI output dir and return local path."""
    out_dir = os.path.join(folder_paths.get_output_directory(), "kie_seedance2")
    os.makedirs(out_dir, exist_ok=True)
    n = 1
    fp = os.path.join(out_dir, f"{prefix}_{n:05d}.mp4")
    while os.path.exists(fp):
        n += 1
        fp = os.path.join(out_dir, f"{prefix}_{n:05d}.mp4")

    print(f"[Kie Seedance2] Downloading video...")
    r = requests.get(video_url, stream=True, timeout=300)
    r.raise_for_status()
    with open(fp, "wb") as fh:
        for chunk in r.iter_content(8192):
            if chunk:
                fh.write(chunk)
    print(f"[Kie Seedance2] Saved → {fp}")
    return fp


def _make_video_output(path):
    """Wrap a local .mp4 path as ComfyUI VIDEO type for Save Video compatibility."""
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
    # Fallback — return raw path (won't connect to Save Video, but won't crash)
    print("[Kie Seedance2] WARNING: VideoFromFile not found. "
          "Update ComfyUI for native Save Video support.")
    return path


def _download_image_as_tensor(url):
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        print(f"[Kie Seedance2] Download image failed: {e}")
        return torch.zeros(1, 64, 64, 3)


# ── Nodes ─────────────────────────────────────────────────────────────────────

class KieSeedance2_ApiKey:
    """Store your Kie.ai API key once and wire it to other Seedance 2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "api_key": ("STRING", {
                "multiline": False, "default": "",
                "tooltip": "Your Kie.ai API key — get one at https://kie.ai/api-key",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = "Kie.ai/Seedance 2.0"

    def run(self, api_key):
        return (_get_api_key(api_key),)


class KieSeedance2_TextToVideo:
    """Generate a video purely from a text prompt. No input images needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic aerial shot of a futuristic city at dusk, volumetric lighting",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (RESOLUTIONS, {"default": "720p"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 15, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "task_id")
    FUNCTION = "run"
    CATEGORY = "Kie.ai/Seedance 2.0"

    def run(self, prompt, aspect_ratio, resolution, duration, generate_audio, api_key=""):
        api_key = _get_api_key(api_key)
        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "generate_audio": generate_audio,
            "web_search": False,
        }
        print("[Kie Seedance2 T2V] Submitting...")
        task_id = _create_task(api_key, input_params)
        result = _poll_task(api_key, task_id)
        video_url = _extract_video_url(result)
        local_path = _download_video(video_url, "t2v")
        return (_make_video_output(local_path), task_id)


class KieSeedance2_ImageToVideo:
    """
    Animate a starting image into a video.
    Connect an IMAGE as the start frame. Optionally connect an end frame
    IMAGE to control where the animation ends.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The scene comes to life with cinematic motion and natural movement",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "adaptive"}),
                "resolution": (RESOLUTIONS, {"default": "720p"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 15, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "end_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("video", "last_frame", "task_id")
    FUNCTION = "run"
    CATEGORY = "Kie.ai/Seedance 2.0"

    def run(self, image, prompt, aspect_ratio, resolution, duration,
            generate_audio, api_key="", end_image=None):
        api_key = _get_api_key(api_key)

        # Upload start image
        print("[Kie Seedance2 I2V] Uploading start image...")
        first_url = _upload_image(api_key, image)

        input_params = {
            "prompt": prompt,
            "first_frame_url": first_url,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "generate_audio": generate_audio,
            "return_last_frame": True,
            "web_search": False,
        }

        # Upload end image if provided
        if end_image is not None:
            print("[Kie Seedance2 I2V] Uploading end image...")
            last_url = _upload_image(api_key, end_image)
            input_params["last_frame_url"] = last_url

        print("[Kie Seedance2 I2V] Submitting task...")
        task_id = _create_task(api_key, input_params)
        result = _poll_task(api_key, task_id)
        video_url = _extract_video_url(result)
        local_path = _download_video(video_url, "i2v")

        # Get last frame if available
        last_frame = torch.zeros(1, 64, 64, 3)
        lf_url = _extract_last_frame_url(result)
        if lf_url:
            last_frame = _download_image_as_tensor(lf_url)

        return (_make_video_output(local_path), last_frame, task_id)


class KieSeedance2_ReferenceToVideo:
    """
    Generate a video using reference images and/or audio.
    Connect up to 9 reference IMAGEs and optional AUDIO.
    The model will incorporate the visual style and elements from your references.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A dynamic scene combining the reference elements with cinematic motion",
                }),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "resolution": (RESOLUTIONS, {"default": "720p"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 15, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "ref_image_4": ("IMAGE",),
                "ref_image_5": ("IMAGE",),
                "ref_image_6": ("IMAGE",),
                "ref_image_7": ("IMAGE",),
                "ref_image_8": ("IMAGE",),
                "ref_image_9": ("IMAGE",),
                "ref_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "task_id")
    FUNCTION = "run"
    CATEGORY = "Kie.ai/Seedance 2.0"

    def run(self, prompt, aspect_ratio, resolution, duration, generate_audio,
            api_key="",
            ref_image_1=None, ref_image_2=None, ref_image_3=None,
            ref_image_4=None, ref_image_5=None, ref_image_6=None,
            ref_image_7=None, ref_image_8=None, ref_image_9=None,
            ref_audio=None):
        api_key = _get_api_key(api_key)

        # Upload all connected reference images
        image_slots = [ref_image_1, ref_image_2, ref_image_3,
                       ref_image_4, ref_image_5, ref_image_6,
                       ref_image_7, ref_image_8, ref_image_9]
        image_urls = []
        for i, img in enumerate(image_slots, 1):
            if img is not None:
                print(f"[Kie Seedance2 Ref] Uploading ref image {i}...")
                image_urls.append(_upload_image(api_key, img))

        # Upload audio if connected
        audio_urls = []
        if ref_audio is not None:
            print("[Kie Seedance2 Ref] Uploading ref audio...")
            audio_urls.append(_upload_audio(api_key, ref_audio))

        if not image_urls and not audio_urls:
            raise ValueError("Connect at least one reference image or audio input.")

        input_params = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "generate_audio": generate_audio,
            "web_search": False,
        }
        if image_urls:
            input_params["reference_image_urls"] = image_urls
        if audio_urls:
            input_params["reference_audio_urls"] = audio_urls

        print(f"[Kie Seedance2 Ref] Submitting ({len(image_urls)} images, "
              f"{len(audio_urls)} audio)...")
        task_id = _create_task(api_key, input_params)
        result = _poll_task(api_key, task_id)
        video_url = _extract_video_url(result)
        local_path = _download_video(video_url, "ref")
        return (_make_video_output(local_path), task_id)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "KieSeedance2_ApiKey": KieSeedance2_ApiKey,
    "KieSeedance2_TextToVideo": KieSeedance2_TextToVideo,
    "KieSeedance2_ImageToVideo": KieSeedance2_ImageToVideo,
    "KieSeedance2_ReferenceToVideo": KieSeedance2_ReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KieSeedance2_ApiKey": "🔑 Kie Seedance 2.0 API Key",
    "KieSeedance2_TextToVideo": "🎬 Kie Seedance 2.0 Text-to-Video",
    "KieSeedance2_ImageToVideo": "🖼️ Kie Seedance 2.0 Image-to-Video",
    "KieSeedance2_ReferenceToVideo": "🎭 Kie Seedance 2.0 Reference-to-Video",
}
