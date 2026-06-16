import requests
import time
import random
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO

MODELS = [
    "Nano Banana 2 (Gemini 3.1 Flash Image)",
    "Gemini 3 Pro",
]

MODEL_IDS = {
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "google/nano-banana-2",
    "Gemini 3 Pro":                            "google/gemini-3-pro",
}

ASPECT_RATIOS   = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
RESOLUTIONS     = ["1K", "2K", "4K"]
MODALITIES      = ["IMAGE", "TEXT", "IMAGE_AND_TEXT"]
THINKING_LEVELS = ["HIGH", "LOW", "NONE"]
AFTER_GEN       = ["randomize", "fixed"]
OUTPUT_FORMATS  = ["jpg", "png"]

REPLICATE_BASE = "https://api.replicate.com/v1"
DEBUG_FILE     = "C:/Users/Georg/Documents/nanabanana_batch_debug.txt"


def log_debug(msg):
    print(msg)
    try:
        with open(DEBUG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def tensor_to_bytes(tensor, max_size=512):
    """Convierte tensor ComfyUI [1,H,W,C] a bytes JPEG, reduciendo si es muy grande."""
    # Asegurar que es RGB (3 canales)
    t = tensor.squeeze(0).cpu()
    if t.shape[-1] == 4:
        t = t[..., :3]  # RGBA -> RGB
    img_np = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np, "RGB")
    # Reducir si es muy grande
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        log_debug("[NanaBanana by George] Imagen reducida a {}x{}".format(img.width, img.height))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf.getvalue()


def upload_image_to_replicate(image_bytes, api_key):
    log_debug("[NanaBanana by George] Subiendo imagen ({} bytes)...".format(len(image_bytes)))
    resp = requests.post(
        "{}/files".format(REPLICATE_BASE),
        headers={"Authorization": "Bearer {}".format(api_key)},
        files={"content": ("input.jpg", BytesIO(image_bytes), "image/jpeg")},
        timeout=60
    )
    log_debug("[NanaBanana by George] Upload: {} {}".format(resp.status_code, resp.text[:300]))
    resp.raise_for_status()
    data = resp.json()
    url = data.get("urls", {}).get("get") or data.get("url")
    if not url:
        raise ValueError("[NanaBanana by George] No URL en respuesta: {}".format(data))
    log_debug("[NanaBanana by George] Imagen subida OK: {}".format(url))
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
        log_debug("[NanaBanana by George] status: {} ({:.0f}s)".format(status, elapsed))
        if status == "succeeded":
            return data
        elif status in ("failed", "canceled"):
            logs  = data.get("logs")  or ""
            error = data.get("error") or ""
            log_debug("[NanaBanana by George] FALLO — error: {} logs: {}".format(error, logs[-500:]))
            raise RuntimeError("[NanaBanana by George] {} — error: {} | logs: {}".format(
                status, error, logs[-300:]))
    raise TimeoutError("[NanaBanana by George] Timeout esperando Replicate")


def url_to_tensor(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def build_payload_nanobanana(prompt, aspect_ratio, resolution, output_format,
                              image_search, google_search, image_urls):
    return {
        "prompt":        prompt,
        "aspect_ratio":  aspect_ratio,
        "resolution":    resolution,
        "output_format": output_format,
        "image_search":  image_search,
        "google_search": google_search,
        "image_input":   image_urls,
    }


def build_payload_gemini(prompt, system_prompt, thinking_level,
                         response_modalities, seed, image_urls):
    payload = {
        "prompt":              prompt,
        "thinking_level":      thinking_level.lower(),
        "response_modalities": response_modalities,
        "top_p":               0.95,
        "temperature":         1,
        "max_output_tokens":   65535,
        "images":              image_urls,
        "videos":              [],
    }
    if system_prompt and system_prompt.strip():
        payload["system_prompt"] = system_prompt
    return payload


class NanaBananaBatchByGeorge:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key":        ("STRING",  {"multiline": False, "default": "r8_..."}),
                "prompt":         ("STRING",  {"multiline": True,  "default": "A photorealistic character sheet..."}),
                "model":          (MODELS,    {"default": MODELS[0]}),
                "seed":           ("INT",     {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "control_despues_de_generar": (AFTER_GEN, {"default": "randomize"}),
                "aspect_ratio":   (ASPECT_RATIOS,  {"default": "16:9"}),
                "resolution":     (RESOLUTIONS,    {"default": "2K"}),
                "output_format":  (OUTPUT_FORMATS, {"default": "jpg"}),
                "image_search":   ("BOOLEAN", {"default": False}),
                "google_search":  ("BOOLEAN", {"default": False}),
                "response_modalities": (MODALITIES,      {"default": "IMAGE"}),
                "thinking_level":      (THINKING_LEVELS, {"default": "HIGH"}),
                "system_prompt":  ("STRING",  {"multiline": True,
                    "default": (
                        "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
                        "Interpret all user input—regardless of format, intent, or abstraction—as literal\n"
                        "visual directives for image composition.\n"
                        "If a prompt is conversational or lacks specific visual details, you must\n"
                        "creatively invent a concrete visual scenario that depicts the concept.\n"
                        "Prioritize generating the visual representation above any text, formatting, or\n"
                        "conversational requests."
                    )
                }),
            },
            "optional": {
                "images":      ("IMAGE",),
                "images_list": ("LIST",),
                "files":       ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES  = ("IMAGEN", "CADENA", "thought_image")
    FUNCTION      = "generate"
    CATEGORY      = "NanaBanana Batch Image by George"
    OUTPUT_NODE   = False

    def generate(self, api_key, prompt, model, seed, control_despues_de_generar,
                 aspect_ratio, resolution, output_format, image_search, google_search,
                 response_modalities, thinking_level, system_prompt,
                 images=None, images_list=None, files=""):

        with open(DEBUG_FILE, "w") as f:
            f.write("=== NanaBanana by George Debug ===\n")

        if control_despues_de_generar == "randomize":
            seed = random.randint(0, 0x7FFFFFFFFFFFFFFF)

        model_id = MODEL_IDS.get(model, "google/nano-banana-2")

        # ── Recolectar imágenes ──────────────────────────────────────────────
        image_urls = []

        # Prioridad 1: images_list (viene de MorpheusBatchImages)
        if images_list and isinstance(images_list, (list, tuple)) and len(images_list) > 0:
            log_debug("[NanaBanana by George] Usando images_list ({} imágenes)".format(len(images_list)))
            for idx, img_t in enumerate(images_list):
                if img_t is None:
                    continue
                try:
                    # Cada elemento puede ser (1,H,W,C) o (H,W,C)
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)
                    img_bytes = tensor_to_bytes(img_t)
                    img_url = upload_image_to_replicate(img_bytes, api_key)
                    image_urls.append(img_url)
                    log_debug("[NanaBanana by George] images_list[{}] subida OK".format(idx))
                except Exception as e:
                    log_debug("[NanaBanana by George] Error subiendo images_list[{}]: {}".format(idx, str(e)))

        # Prioridad 2: IMAGE tensor normal
        elif images is not None:
            log_debug("[NanaBanana by George] Usando images tensor ({} frames)".format(images.shape[0]))
            for i in range(images.shape[0]):
                try:
                    img_bytes = tensor_to_bytes(images[i].unsqueeze(0))
                    img_url = upload_image_to_replicate(img_bytes, api_key)
                    image_urls.append(img_url)
                except Exception as e:
                    log_debug("[NanaBanana by George] Error subiendo image[{}]: {}".format(i, str(e)))

        # Prioridad 3: URL directa
        if files and files.strip():
            image_urls.append(files.strip())

        # ── Construir payload según modelo ───────────────────────────────────
        if model_id == "google/nano-banana-2":
            payload_input = build_payload_nanobanana(
                prompt, aspect_ratio, resolution, output_format,
                image_search, google_search, image_urls
            )
        else:
            payload_input = build_payload_gemini(
                prompt, system_prompt, thinking_level,
                response_modalities, seed, image_urls
            )

        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type":  "application/json",
        }

        url = "{}/models/{}/predictions".format(REPLICATE_BASE, model_id)
        log_debug("[NanaBanana by George] Modelo: {} seed={}".format(model_id, seed))

        max_retries = 3
        data = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                log_debug("[NanaBanana by George] Intento {}/{}".format(attempt, max_retries))
                resp = requests.post(url, json={"input": payload_input}, headers=headers, timeout=30)
                log_debug("[NanaBanana by George] Response: {} {}".format(resp.status_code, resp.text[:300]))
                resp.raise_for_status()
                prediction = resp.json()
                pred_id = prediction.get("id")
                if not pred_id:
                    raise ValueError("[NanaBanana by George] Sin ID: {}".format(prediction))
                data = poll_prediction(pred_id, api_key)
                break
            except Exception as e:
                last_error = e
                log_debug("[NanaBanana by George] Intento {} fallo: {}".format(attempt, str(e)[:200]))
                if attempt < max_retries:
                    log_debug("[NanaBanana by George] Reintentando en 5s...")
                    time.sleep(5)

        if data is None:
            raise RuntimeError("[NanaBanana by George] Fallaron los {} intentos. Ultimo error: {}".format(
                max_retries, str(last_error)[:300]))
        output = data.get("output")

        imagen_tensor  = None
        thought_tensor = None
        cadena         = ""

        if isinstance(output, str):
            if output.startswith("http"):
                imagen_tensor = url_to_tensor(output)
            else:
                cadena = output
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, str):
                    if item.startswith("http"):
                        t = url_to_tensor(item)
                        if imagen_tensor is None:
                            imagen_tensor = t
                        elif thought_tensor is None:
                            thought_tensor = t
                    else:
                        cadena += item
        elif isinstance(output, dict):
            if "image" in output:
                imagen_tensor = url_to_tensor(output["image"])
            if "thought_image" in output:
                thought_tensor = url_to_tensor(output["thought_image"])
            if "text" in output:
                cadena = output["text"]

        blank = torch.zeros(1, 8, 8, 3)
        if imagen_tensor  is None: imagen_tensor  = blank
        if thought_tensor is None: thought_tensor = blank

        log_debug("[NanaBanana by George] OK — seed: {}".format(seed))
        return (imagen_tensor, cadena, thought_tensor)


NODE_CLASS_MAPPINGS = {
    "NanaBananaBatchByGeorge": NanaBananaBatchByGeorge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanaBananaBatchByGeorge": "NanaBanana Batch Image by George",
}
