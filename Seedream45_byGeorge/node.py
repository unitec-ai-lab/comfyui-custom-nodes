import requests
import time
import random
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO

REPLICATE_BASE  = "https://api.replicate.com/v1"
REPLICATE_MODEL = "bytedance/seedream-4.5"
DEBUG_FILE      = "C:/Users/Georg/Documents/seedream45_debug.txt"

SIZES         = ["4K", "2K", "1K", "512"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"]
SEQ_GEN       = ["disabled", "enabled"]
AFTER_GEN     = ["randomize", "fixed"]


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
        log_debug("[Seedream45] Imagen reducida a {}x{}".format(img.width, img.height))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


def upload_image_to_replicate(image_bytes, api_key, filename="input.jpg"):
    log_debug("[Seedream45] Subiendo imagen ({} bytes)...".format(len(image_bytes)))
    resp = requests.post(
        "{}/files".format(REPLICATE_BASE),
        headers={"Authorization": "Bearer {}".format(api_key)},
        files={"content": (filename, BytesIO(image_bytes), "image/jpeg")},
        timeout=60
    )
    log_debug("[Seedream45] Upload: {} {}".format(resp.status_code, resp.text[:300]))
    resp.raise_for_status()
    data = resp.json()
    url = data.get("urls", {}).get("get") or data.get("url")
    if not url:
        raise ValueError("[Seedream45] No URL en respuesta: {}".format(data))
    log_debug("[Seedream45] Imagen subida OK: {}".format(url))
    return url


def poll_prediction(prediction_id, api_key, timeout=300):
    url = "{}/predictions/{}".format(REPLICATE_BASE, prediction_id)
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    elapsed = 0
    interval = 3
    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        log_debug("[Seedream45] status: {} ({:.0f}s)".format(status, elapsed))
        if status == "succeeded":
            return data
        elif status in ("failed", "canceled"):
            logs  = data.get("logs")  or ""
            error = data.get("error") or ""
            log_debug("[Seedream45] FALLO — error: {} logs: {}".format(error, logs[-500:]))
            raise RuntimeError("[Seedream45] {} — error: {} | logs: {}".format(
                status, error, logs[-300:]))
    raise TimeoutError("[Seedream45] Timeout ({:.0f}s)".format(timeout))


def url_to_tensor(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class Seedream45ByGeorge:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":      ("STRING", {"multiline": False, "default": "r8_..."}),
                "prompt":       ("STRING", {"multiline": True,  "default": ""}),
                "size":         (SIZES,         {"default": "2K"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9"}),
                "max_images":   ("INT",    {"default": 1, "min": 1, "max": 4}),
                "sequential_image_generation": (SEQ_GEN, {"default": "disabled"}),
                "seed":         ("INT",    {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "control_after_generate": (AFTER_GEN, {"default": "randomize"}),
            },
            "optional": {
                "image_input":  ("IMAGE",),
                "images_list":  ("LIST",),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("IMAGEN", "image_url")
    FUNCTION      = "generate"
    CATEGORY      = "Seedream by George"
    OUTPUT_NODE   = False

    def generate(self, api_key, prompt, size, aspect_ratio, max_images,
                 sequential_image_generation, seed, control_after_generate,
                 image_input=None, images_list=None):

        with open(DEBUG_FILE, "w") as f:
            f.write("=== Seedream 4.5 by George Debug ===\n")

        if control_after_generate == "randomize":
            seed = random.randint(0, 0x7FFFFFFF)

        # ── Subir imágenes de referencia ─────────────────────────────────────
        image_urls = []

        if images_list and isinstance(images_list, (list, tuple)) and len(images_list) > 0:
            log_debug("[Seedream45] images_list: {} imágenes".format(len(images_list)))
            for idx, img_t in enumerate(images_list):
                if img_t is None:
                    continue
                try:
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    img_bytes = tensor_to_bytes(img_t)
                    img_url = upload_image_to_replicate(img_bytes, api_key, "image_{}.jpg".format(idx))
                    image_urls.append(img_url)
                except Exception as e:
                    log_debug("[Seedream45] Error images_list[{}]: {}".format(idx, str(e)))

        elif image_input is not None:
            log_debug("[Seedream45] image_input: {} frames".format(image_input.shape[0]))
            for i in range(image_input.shape[0]):
                try:
                    img_bytes = tensor_to_bytes(image_input[i].unsqueeze(0))
                    img_url = upload_image_to_replicate(img_bytes, api_key, "image_{}.jpg".format(i))
                    image_urls.append(img_url)
                except Exception as e:
                    log_debug("[Seedream45] Error image_input[{}]: {}".format(i, str(e)))

        # ── Payload ──────────────────────────────────────────────────────────
        payload_input = {
            "prompt":       prompt,
            "size":         size,
            "aspect_ratio": aspect_ratio,
            "max_images":   max_images,
            "image_input":  image_urls,
            "sequential_image_generation": sequential_image_generation,
        }

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type":  "application/json",
        }

        url = "{}/models/{}/predictions".format(REPLICATE_BASE, REPLICATE_MODEL)
        log_debug("[Seedream45] Enviando prediccion seed={}".format(seed))
        log_debug("[Seedream45] Payload: {}".format(json.dumps(payload_input, default=str)[:600]))

        max_retries = 3
        data = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                log_debug("[Seedream45] Intento {}/{}".format(attempt, max_retries))
                resp = requests.post(url, json={"input": payload_input}, headers=headers, timeout=30)
                log_debug("[Seedream45] Response: {} {}".format(resp.status_code, resp.text[:300]))
                resp.raise_for_status()
                prediction = resp.json()
                pred_id = prediction.get("id")
                if not pred_id:
                    raise ValueError("[Seedream45] Sin ID: {}".format(prediction))
                data = poll_prediction(pred_id, api_key)
                break
            except Exception as e:
                last_error = e
                log_debug("[Seedream45] Intento {} fallo: {}".format(attempt, str(e)[:200]))
                if attempt < max_retries:
                    log_debug("[Seedream45] Reintentando en 5s...")
                    time.sleep(5)

        if data is None:
            raise RuntimeError("[Seedream45] Fallaron los {} intentos. Ultimo error: {}".format(
                max_retries, str(last_error)[:300]))

        output = data.get("output")

        # Output puede ser lista de URLs o una sola URL
        if isinstance(output, list):
            img_url = output[0]
        else:
            img_url = output

        imagen_tensor = url_to_tensor(img_url)
        log_debug("[Seedream45] OK — seed: {} | url: {}".format(seed, img_url))

        return (imagen_tensor, img_url)


NODE_CLASS_MAPPINGS = {
    "Seedream45ByGeorge": Seedream45ByGeorge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Seedream45ByGeorge": "Seedream 4.5 by George",
}
