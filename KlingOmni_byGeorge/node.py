import requests
import time
import random
import json
import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO


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

REPLICATE_BASE = "https://api.replicate.com/v1"
REPLICATE_MODEL = "kwaivgi/kling-v3-omni-video"
DEBUG_FILE = "C:/Users/Georg/Documents/kling_omni_debug.txt"

MODES           = ["pro", "standard"]
ASPECT_RATIOS   = ["9:16", "16:9", "1:1"]
RESOLUTIONS     = ["1080p", "720p"]
REF_TYPES       = ["feature", "motion", "face"]
AFTER_GEN       = ["randomize", "fixed"]


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
        log_debug("[Kling Omni] Imagen reducida a {}x{}".format(img.width, img.height))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


def upload_image_to_replicate(image_bytes, api_key):
    log_debug("[Kling Omni] Subiendo imagen ({} bytes)...".format(len(image_bytes)))
    resp = requests.post(
        "{}/files".format(REPLICATE_BASE),
        headers={"Authorization": "Bearer {}".format(api_key)},
        files={"content": ("input.jpg", BytesIO(image_bytes), "image/jpeg")},
        timeout=60
    )
    log_debug("[Kling Omni] Upload: {} {}".format(resp.status_code, resp.text[:300]))
    resp.raise_for_status()
    data = resp.json()
    url = data.get("urls", {}).get("get") or data.get("url")
    if not url:
        raise ValueError("[Kling Omni] No URL en respuesta: {}".format(data))
    log_debug("[Kling Omni] Imagen subida OK: {}".format(url))
    return url


def poll_prediction(prediction_id, api_key, timeout=700):
    """Kling tarda hasta ~550s segun el JSON, usamos 700s de timeout."""
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
        log_debug("[Kling Omni] status: {} ({:.0f}s)".format(status, elapsed))
        if status == "succeeded":
            return data
        elif status in ("failed", "canceled"):
            logs  = data.get("logs")  or ""
            error = data.get("error") or ""
            log_debug("[Kling Omni] FALLO — error: {} logs: {}".format(error, logs[-500:]))
            raise RuntimeError("[Kling Omni] {} — error: {} | logs: {}".format(
                status, error, logs[-300:]))
    raise TimeoutError("[Kling Omni] Timeout esperando Replicate ({}s)".format(timeout))


def download_video(url, api_key):
    """Descarga el video y lo guarda en la carpeta output de ComfyUI."""
    log_debug("[Kling Omni] Descargando video: {}".format(url))
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    # Guardar en carpeta output de ComfyUI
    output_dir = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "output")
    os.makedirs(output_dir, exist_ok=True)
    filename = "kling_omni_{}.mp4".format(int(time.time()))
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        f.write(r.content)

    log_debug("[Kling Omni] Video guardado: {}".format(filepath))
    return filepath, url


class KlingOmniByGeorge:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":       ("STRING",  {"multiline": False, "default": "r8_..."}),
                "prompt":        ("STRING",  {"multiline": True,  "default": ""}),
                "model_name":    (["kling-v3-omni"], {"default": "kling-v3-omni"}),
                "mode":          (MODES,         {"default": "pro"}),
                "aspect_ratio":  (ASPECT_RATIOS, {"default": "9:16"}),
                "duration":      ("FLOAT", {"default": 5.0, "min": 3.0, "max": 15.0, "step": 1.0}),
                "resolution":    (RESOLUTIONS,   {"default": "1080p"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "keep_original_sound": ("BOOLEAN", {"default": True}),
                "video_reference_type": (REF_TYPES, {"default": "feature"}),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "control_after_generate": (AFTER_GEN, {"default": "randomize"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "images_list":      ("LIST",),
            }
        }

    RETURN_TYPES  = ("VIDEO", "STRING")
    RETURN_NAMES  = ("VIDEO", "video_url")
    FUNCTION      = "generate"
    CATEGORY      = "Kling by George"
    OUTPUT_NODE   = True

    def generate(self, api_key, prompt, model_name, mode, aspect_ratio, duration,
                 resolution, generate_audio, keep_original_sound, video_reference_type,
                 seed, control_after_generate,
                 reference_images=None, images_list=None):

        with open(DEBUG_FILE, "w") as f:
            f.write("=== Kling Omni by George Debug ===\n")

        if control_after_generate == "randomize":
            seed = random.randint(0, 0x7FFFFFFF)

        # ── Subir imágenes de referencia ─────────────────────────────────────
        ref_urls = []

        if images_list and isinstance(images_list, (list, tuple)) and len(images_list) > 0:
            log_debug("[Kling Omni] images_list: {} imágenes".format(len(images_list)))
            for idx, img_t in enumerate(images_list):
                if img_t is None:
                    continue
                try:
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    img_bytes = tensor_to_bytes(img_t)
                    img_url = upload_image_to_replicate(img_bytes, api_key)
                    ref_urls.append(img_url)
                except Exception as e:
                    log_debug("[Kling Omni] Error images_list[{}]: {}".format(idx, str(e)))

        elif reference_images is not None:
            log_debug("[Kling Omni] reference_images: {} frames".format(reference_images.shape[0]))
            for i in range(reference_images.shape[0]):
                try:
                    img_bytes = tensor_to_bytes(reference_images[i].unsqueeze(0))
                    img_url = upload_image_to_replicate(img_bytes, api_key)
                    ref_urls.append(img_url)
                except Exception as e:
                    log_debug("[Kling Omni] Error reference_images[{}]: {}".format(i, str(e)))

        # ── Payload ──────────────────────────────────────────────────────────
        payload_input = {
            "mode":          mode,
            "prompt":        prompt,
            "duration":      max(3, min(15, int(round(duration)))),
            "aspect_ratio":  aspect_ratio,
            "generate_audio": generate_audio,
            "keep_original_sound": keep_original_sound,
            "seed":          seed,
        }

        if ref_urls:
            payload_input["reference_images"]     = ref_urls
            payload_input["video_reference_type"] = video_reference_type

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type":  "application/json",
        }

        url = "{}/models/{}/predictions".format(REPLICATE_BASE, REPLICATE_MODEL)
        log_debug("[Kling Omni] Enviando prediccion seed={}".format(seed))
        log_debug("[Kling Omni] Payload: {}".format(json.dumps(payload_input, default=str)[:600]))

        max_retries = 3
        data = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                log_debug("[Kling Omni] Intento {}/{}".format(attempt, max_retries))
                resp = requests.post(url, json={"input": payload_input}, headers=headers, timeout=30)
                log_debug("[Kling Omni] Response: {} {}".format(resp.status_code, resp.text[:300]))
                resp.raise_for_status()
                prediction = resp.json()
                pred_id = prediction.get("id")
                if not pred_id:
                    raise ValueError("[Kling Omni] Sin ID: {}".format(prediction))
                data = poll_prediction(pred_id, api_key)
                break
            except Exception as e:
                last_error = e
                log_debug("[Kling Omni] Intento {} fallo: {}".format(attempt, str(e)[:200]))
                if attempt < max_retries:
                    log_debug("[Kling Omni] Reintentando en 5s...")
                    time.sleep(5)

        if data is None:
            raise RuntimeError("[Kling Omni] Fallaron los {} intentos. Ultimo error: {}".format(
                max_retries, str(last_error)[:300]))

        output = data.get("output")
        if isinstance(output, list):
            video_url = output[0]
        else:
            video_url = output

        filepath, video_url = download_video(video_url, api_key)
        log_debug("[Kling Omni] OK — seed: {} | archivo: {}".format(seed, filepath))

        return (_make_video_output(filepath), video_url)


NODE_CLASS_MAPPINGS = {
    "KlingOmniByGeorge": KlingOmniByGeorge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingOmniByGeorge": "Kling 3.0 Omni by George",
}
