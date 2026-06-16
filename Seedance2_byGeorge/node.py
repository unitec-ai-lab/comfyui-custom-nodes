import requests
import time
import random
import json
import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO

REPLICATE_BASE  = "https://api.replicate.com/v1"
REPLICATE_MODEL = "bytedance/seedance-2.0"
DEBUG_FILE      = "C:/Users/Georg/Documents/seedance2_debug.txt"

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
RESOLUTIONS   = ["720p", "1080p"]
AFTER_GEN     = ["randomize", "fixed"]


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


def log_debug(msg):
    print(msg)
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def tensor_to_bytes(tensor, max_size=1024):
    t = tensor.squeeze(0).cpu()
    if t.shape[-1] == 4:
        t = t[..., :3]
    img_np = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np, "RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        log_debug("[Seedance2] Imagen reducida a {}x{}".format(img.width, img.height))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


def audio_tensor_to_bytes(audio):
    """Convierte tensor de audio ComfyUI a bytes WAV."""
    try:
        import soundfile as sf

        # ComfyUI LoadAudio devuelve dict con waveform y sample_rate
        if isinstance(audio, dict):
            # Buscar waveform sin usar 'or' con tensores
            waveform = None
            for key in ["waveform", "samples", "audio", "data"]:
                if key in audio and audio[key] is not None:
                    waveform = audio[key]
                    break
            sample_rate = int(audio.get("sample_rate", 44100))
        else:
            waveform = audio
            sample_rate = 44100

        if waveform is None:
            raise ValueError("No se encontró waveform en el audio")

        # Convertir a numpy
        if hasattr(waveform, "cpu"):
            arr = waveform.detach().cpu().numpy()
        else:
            arr = np.array(waveform)

        log_debug("[Seedance2] Audio shape original: {} dtype: {}".format(arr.shape, arr.dtype))

        # ComfyUI waveform shape: (batch, channels, samples) o (channels, samples) o (samples,)
        if arr.ndim == 3:
            arr = arr[0]  # quitar batch -> (channels, samples)
        if arr.ndim == 2:
            arr = arr.T   # (channels, samples) -> (samples, channels)
        # arr ahora es (samples,) o (samples, channels)

        log_debug("[Seedance2] Audio shape para sf.write: {}".format(arr.shape))

        buf = BytesIO()
        sf.write(buf, arr, sample_rate, format="WAV")
        buf.seek(0)
        return buf.getvalue(), "audio/wav"

    except ImportError:
        raise RuntimeError("soundfile no instalado. Corre: pip install soundfile")


def upload_file_to_replicate(file_bytes, api_key, mime_type="image/jpeg", filename="input.jpg"):
    log_debug("[Seedance2] Subiendo archivo ({} bytes, {})...".format(len(file_bytes), mime_type))
    resp = requests.post(
        "{}/files".format(REPLICATE_BASE),
        headers={"Authorization": "Bearer {}".format(api_key)},
        files={"content": (filename, BytesIO(file_bytes), mime_type)},
        timeout=60
    )
    log_debug("[Seedance2] Upload: {} {}".format(resp.status_code, resp.text[:300]))
    resp.raise_for_status()
    data = resp.json()
    url = data.get("urls", {}).get("get") or data.get("url")
    if not url:
        raise ValueError("[Seedance2] No URL en respuesta: {}".format(data))
    log_debug("[Seedance2] Subido OK: {}".format(url))
    return url


def poll_prediction(prediction_id, api_key, timeout=600):
    url = "{}/predictions/{}".format(REPLICATE_BASE, prediction_id)
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    elapsed = 0
    interval = 5
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        log_debug("[Seedance2] status: {} ({:.0f}s)".format(status, elapsed))
        if status == "succeeded":
            return data
        elif status in ("failed", "canceled"):
            logs  = data.get("logs")  or ""
            error = data.get("error") or ""
            log_debug("[Seedance2] FALLO — error: {} logs: {}".format(error, logs[-500:]))
            raise RuntimeError("[Seedance2] {} — error: {} | logs: {}".format(
                status, error, logs[-300:]))
    raise TimeoutError("[Seedance2] Timeout ({:.0f}s)".format(timeout))


def download_video(url, api_key):
    log_debug("[Seedance2] Descargando video: {}".format(url))
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    output_dir = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output")
    os.makedirs(output_dir, exist_ok=True)
    filename = "seedance2_{}.mp4".format(int(time.time()))
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(r.content)
    log_debug("[Seedance2] Video guardado: {}".format(filepath))
    return filepath


class Seedance2ByGeorge:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":       ("STRING",  {"multiline": False, "default": "r8_..."}),
                "prompt":        ("STRING",  {"multiline": True,  "default": ""}),
                "aspect_ratio":  (ASPECT_RATIOS, {"default": "16:9"}),
                "resolution":    (RESOLUTIONS,   {"default": "720p"}),
                "duration":      ("FLOAT",   {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed":          ("INT",     {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "control_after_generate": (AFTER_GEN, {"default": "randomize"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "images_list":      ("LIST",),
                "reference_audio":  ("AUDIO",),
            }
        }

    RETURN_TYPES  = ("VIDEO", "STRING")
    RETURN_NAMES  = ("VIDEO", "video_url")
    FUNCTION      = "generate"
    CATEGORY      = "Seedance by George"
    OUTPUT_NODE   = True

    def generate(self, api_key, prompt, aspect_ratio, resolution, duration,
                 generate_audio, seed, control_after_generate,
                 reference_images=None, images_list=None, reference_audio=None):

        with open(DEBUG_FILE, "w") as f:
            f.write("=== Seedance 2.0 by George Debug ===\n")

        if control_after_generate == "randomize":
            seed = random.randint(0, 0x7FFFFFFF)

        # ── Subir imágenes de referencia ─────────────────────────────────────
        ref_image_urls = []

        if images_list and isinstance(images_list, (list, tuple)) and len(images_list) > 0:
            log_debug("[Seedance2] images_list: {} imágenes".format(len(images_list)))
            for idx, img_t in enumerate(images_list):
                if img_t is None:
                    continue
                try:
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    img_bytes = tensor_to_bytes(img_t)
                    img_url = upload_file_to_replicate(img_bytes, api_key, "image/jpeg", "image_{}.jpg".format(idx))
                    ref_image_urls.append(img_url)
                except Exception as e:
                    log_debug("[Seedance2] Error images_list[{}]: {}".format(idx, str(e)))

        elif reference_images is not None:
            log_debug("[Seedance2] reference_images: {} frames".format(reference_images.shape[0]))
            for i in range(reference_images.shape[0]):
                try:
                    img_bytes = tensor_to_bytes(reference_images[i].unsqueeze(0))
                    img_url = upload_file_to_replicate(img_bytes, api_key, "image/jpeg", "image_{}.jpg".format(i))
                    ref_image_urls.append(img_url)
                except Exception as e:
                    log_debug("[Seedance2] Error reference_images[{}]: {}".format(i, str(e)))

        # ── Subir audio de referencia ─────────────────────────────────────────
        ref_audio_urls = []

        if reference_audio is not None:
            try:
                audio_bytes, mime_type = audio_tensor_to_bytes(reference_audio)
                audio_url = upload_file_to_replicate(audio_bytes, api_key, mime_type, "audio.wav")
                ref_audio_urls.append(audio_url)
                log_debug("[Seedance2] Audio subido OK")
            except Exception as e:
                log_debug("[Seedance2] Error subiendo audio: {}".format(str(e)))

        # ── Agregar referencias al prompt ────────────────────────────────────
        full_prompt = prompt
        if ref_image_urls:
            full_prompt += " [Image1]"
        if ref_audio_urls:
            full_prompt += " [Audio1]"

        # ── Payload ──────────────────────────────────────────────────────────
        payload_input = {
            "prompt":           full_prompt,
            "aspect_ratio":     aspect_ratio,
            "resolution":       resolution,
            "duration":         int(round(duration)),
            "generate_audio":   generate_audio,
            "seed":             seed,
            "reference_images": ref_image_urls,
            "reference_audios": ref_audio_urls,
            "reference_videos": [],
        }

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type":  "application/json",
        }

        url = "{}/models/{}/predictions".format(REPLICATE_BASE, REPLICATE_MODEL)
        log_debug("[Seedance2] Enviando prediccion seed={}".format(seed))
        log_debug("[Seedance2] Payload: {}".format(json.dumps(payload_input, default=str)[:600]))

        max_retries = 3
        data = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                log_debug("[Seedance2] Intento {}/{}".format(attempt, max_retries))
                resp = requests.post(url, json={"input": payload_input}, headers=headers, timeout=30)
                log_debug("[Seedance2] Response: {} {}".format(resp.status_code, resp.text[:300]))
                resp.raise_for_status()
                prediction = resp.json()
                pred_id = prediction.get("id")
                if not pred_id:
                    raise ValueError("[Seedance2] Sin ID: {}".format(prediction))
                data = poll_prediction(pred_id, api_key)
                break
            except Exception as e:
                last_error = e
                log_debug("[Seedance2] Intento {} fallo: {}".format(attempt, str(e)[:200]))
                if attempt < max_retries:
                    log_debug("[Seedance2] Reintentando en 5s...")
                    time.sleep(5)

        if data is None:
            raise RuntimeError("[Seedance2] Fallaron los {} intentos. Ultimo error: {}".format(
                max_retries, str(last_error)[:300]))

        output = data.get("output")
        if isinstance(output, list):
            video_url = output[0]
        else:
            video_url = output

        filepath = download_video(video_url, api_key)
        log_debug("[Seedance2] OK — seed: {} | archivo: {}".format(seed, filepath))

        return (_make_video_output(filepath), video_url)


NODE_CLASS_MAPPINGS = {
    "Seedance2ByGeorge": Seedance2ByGeorge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedance2ByGeorge": "Seedance 2.0 by George",
}
