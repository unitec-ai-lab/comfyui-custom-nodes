import json
import base64
import os
import numpy as np
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "Pillow"])
    from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from .aion_api import (verify_api_key, analyze_face_images, compose_aion_prompt,
                       generate_face_image, compose_flux_prompt_local, compose_flux_prompt_gemini,
                       generate_fusion_portrait)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_config(filename: str) -> dict:
    filepath = os.path.join(CONFIG_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_options(config: dict, key: str) -> list:
    return config.get(key, {}).get("options", ["auto"])


_photo_types_config = _load_config("photo_types.json")
_photo_types_list = _photo_types_config.get("photo_types", [])
PHOTO_TYPE_OPTIONS = [pt["label"] for pt in _photo_types_list]
_photo_types_by_label = {pt["label"]: pt for pt in _photo_types_list}

ASPECT_RATIO_OPTIONS = ["16:9", "9:16", "1:1", "4:3", "3:4", "2:3", "3:2", "4:5", "5:4", "21:9"]
RESOLUTION_OPTIONS = ["512px", "1K", "2K", "4K"]

IMAGE_MODEL_OPTIONS = [
    "Nano Banana Pro (gemini-3-pro-image-preview)",
    "Nano Banana 2 (gemini-3.1-flash-image-preview)",
]
_IMAGE_MODEL_MAP = {
    "Nano Banana Pro (gemini-3-pro-image-preview)": "gemini-3-pro-image-preview",
    "Nano Banana 2 (gemini-3.1-flash-image-preview)": "gemini-3.1-flash-image-preview",
}
_PRO_UNSUPPORTED_RESOLUTIONS = {"512px"}

_demographics_config = _load_config("demographics.json")
_eyes_config = _load_config("eyes.json")
_eyebrows_config = _load_config("eyebrows.json")
_nose_config = _load_config("nose.json")
_lips_config = _load_config("lips.json")
_structure_config = _load_config("structure.json")
_volumes_config = _load_config("volumes.json")
_hair_config = _load_config("hair.json")
_skin_config = _load_config("skin.json")
_defects_config = _load_config("defects.json")
_expressions_config = _load_config("expressions.json")


def _image_to_base64(image_tensor) -> tuple:
    if torch is not None and isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.squeeze(0).cpu().numpy()
    else:
        img_np = np.array(image_tensor)
        if img_np.ndim == 4:
            img_np = img_np[0]

    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str, "image/png"


def _base64_to_image_tensor(b64_str: str, mime_type: str = "image/png"):
    img_data = base64.b64decode(b64_str)
    pil_img = Image.open(BytesIO(img_data)).convert("RGB")
    img_np = np.array(pil_img).astype(np.float32) / 255.0

    if torch is not None:
        return torch.from_numpy(img_np).unsqueeze(0)
    else:
        return np.expand_dims(img_np, axis=0)


class AionThetaNode:
    """
    AION - Gemini Portrait Master Generator

    Generates new faces from reference images or manual attribute selection
    using Google Gemini API (3.1 Pro / 3 Flash) and Nano Banana Pro.

    Flow with images (3 steps):
      1. Analyze images with Gemini -> facial feature JSON
      2. Compose AION prompt with system prompt -> nano_banana_prompt_sentence
      3. Generate image with Nano Banana Pro

    Flow without images (2 steps):
      1. Combine selected attributes with system prompt -> nano_banana_prompt_sentence
      2. Generate image with Nano Banana Pro

    The "Verify API Key" button is added by the JS extension and calls
    the server-side /aion/verify_key endpoint (single call, no loop).
    The api_key_status widget is display-only, updated by the button.
    """

    CATEGORY = "AION"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "log", "aion_json")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter your Gemini API key"
                }),
                "api_key_status": ("STRING", {
                    "default": "Not Verified",
                    "multiline": False,
                }),
                "model": (["gemini-3.1-pro-preview", "gemini-3-flash-preview"], {
                    "default": "gemini-3.1-pro-preview"
                }),
                "image_model": (IMAGE_MODEL_OPTIONS, {
                    "default": "Nano Banana Pro (gemini-3-pro-image-preview)"
                }),
                "photo_type": (PHOTO_TYPE_OPTIONS, {
                    "default": "-- Not selected / System inferred --"
                }),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {
                    "default": "16:9"
                }),
                "resolution": (RESOLUTION_OPTIONS, {
                    "default": "1K"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "brief_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional brief text to guide generation"
                }),

                "sex": (_get_options(_demographics_config, "sex"), {"default": "auto"}),
                "ethnicity": (_get_options(_demographics_config, "ethnicity"), {"default": "auto"}),

                "eye_shape": (_get_options(_eyes_config, "eye_shape"), {"default": "auto"}),
                "eye_size": (_get_options(_eyes_config, "eye_size"), {"default": "auto"}),
                "eye_tilt": (_get_options(_eyes_config, "eye_tilt"), {"default": "auto"}),
                "eye_color": (_get_options(_eyes_config, "eye_color"), {"default": "auto"}),

                "eyebrow_thickness": (_get_options(_eyebrows_config, "eyebrow_thickness"), {"default": "auto"}),
                "eyebrow_shape": (_get_options(_eyebrows_config, "eyebrow_shape"), {"default": "auto"}),
                "eyebrow_color": (_get_options(_eyebrows_config, "eyebrow_color"), {"default": "auto"}),

                "nose_profile": (_get_options(_nose_config, "nose_profile"), {"default": "auto"}),
                "nose_base": (_get_options(_nose_config, "nose_base"), {"default": "auto"}),
                "nose_tip": (_get_options(_nose_config, "nose_tip"), {"default": "auto"}),

                "lips_volume": (_get_options(_lips_config, "lips_volume"), {"default": "auto"}),
                "cupid_bow": (_get_options(_lips_config, "cupid_bow"), {"default": "auto"}),
                "lips_proportion": (_get_options(_lips_config, "lips_proportion"), {"default": "auto"}),
                "lips_color": (_get_options(_lips_config, "lips_color"), {"default": "auto"}),

                "forehead": (_get_options(_structure_config, "forehead"), {"default": "auto"}),
                "cheekbones": (_get_options(_structure_config, "cheekbones"), {"default": "auto"}),
                "jawline": (_get_options(_structure_config, "jawline"), {"default": "auto"}),
                "chin": (_get_options(_structure_config, "chin"), {"default": "auto"}),

                "cheeks": (_get_options(_volumes_config, "cheeks"), {"default": "auto"}),
                "submental": (_get_options(_volumes_config, "submental"), {"default": "auto"}),
                "face_neck_transition": (_get_options(_volumes_config, "face_neck_transition"), {"default": "auto"}),

                "hair_structure": (_get_options(_hair_config, "hair_structure"), {"default": "auto"}),
                "hair_length": (_get_options(_hair_config, "hair_length"), {"default": "auto"}),
                "hair_volume": (_get_options(_hair_config, "hair_volume"), {"default": "auto"}),
                "hair_color": (_get_options(_hair_config, "hair_color"), {"default": "auto"}),

                "skin_tone": (_get_options(_skin_config, "skin_tone"), {"default": "auto"}),
                "skin_undertone": (_get_options(_skin_config, "skin_undertone"), {"default": "auto"}),
                "skin_texture": (_get_options(_skin_config, "skin_texture"), {"default": "auto"}),
                "skin_micro_texture": (_get_options(_skin_config, "skin_micro_texture"), {"default": "auto"}),
                "skin_imperfections": (_get_options(_skin_config, "skin_imperfections"), {"default": "auto"}),
                "skin_reflection": (_get_options(_skin_config, "skin_reflection"), {"default": "auto"}),

                "wrinkles": (_get_options(_defects_config, "wrinkles"), {"default": "auto"}),
                "scars": (_get_options(_defects_config, "scars"), {"default": "auto"}),
                "deformations": (_get_options(_defects_config, "deformations"), {"default": "auto"}),
                "tone_loss": (_get_options(_defects_config, "tone_loss"), {"default": "auto"}),
                "skin_marks": (_get_options(_defects_config, "skin_marks"), {"default": "auto"}),
                "vitiligo": (_get_options(_defects_config, "vitiligo"), {"default": "auto"}),
                "under_eye": (_get_options(_defects_config, "under_eye"), {"default": "auto"}),

                "expression": (_get_options(_expressions_config, "expression"), {"default": "auto"}),
                "expression_variant": (_get_options(_expressions_config, "expression_variant"), {"default": "auto"}),
            },
            "optional": {
                "image_eyes": ("IMAGE",),
                "image_eyebrows": ("IMAGE",),
                "image_nose": ("IMAGE",),
                "image_lips": ("IMAGE",),
                "image_forehead": ("IMAGE",),
                "image_cheekbones": ("IMAGE",),
                "image_jawline": ("IMAGE",),
                "image_hair": ("IMAGE",),
                "image_skin": ("IMAGE",),
                "image_full_face": ("IMAGE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, gemini_api_key, api_key_status, model, image_model,
                 photo_type, aspect_ratio, resolution, top_p,
                 brief_text,
                 sex, ethnicity,
                 eye_shape, eye_size, eye_tilt, eye_color,
                 eyebrow_thickness, eyebrow_shape, eyebrow_color,
                 nose_profile, nose_base, nose_tip,
                 lips_volume, cupid_bow, lips_proportion, lips_color,
                 forehead, cheekbones, jawline, chin,
                 cheeks, submental, face_neck_transition,
                 hair_structure, hair_length, hair_volume, hair_color,
                 skin_tone, skin_undertone, skin_texture,
                 skin_micro_texture, skin_imperfections, skin_reflection,
                 wrinkles, scars, deformations, tone_loss, skin_marks, vitiligo, under_eye,
                 expression, expression_variant,
                 image_eyes=None, image_eyebrows=None, image_nose=None,
                 image_lips=None, image_forehead=None, image_cheekbones=None,
                 image_jawline=None, image_hair=None, image_skin=None,
                 image_full_face=None):

        all_logs = []
        all_logs.append("=" * 60)
        all_logs.append("AION - Gemini Portrait Master Generator")
        all_logs.append("=" * 60)

        resolved_image_model = _IMAGE_MODEL_MAP.get(image_model, "gemini-3-pro-image-preview")
        if resolution in _PRO_UNSUPPORTED_RESOLUTIONS and resolved_image_model == "gemini-3-pro-image-preview":
            all_logs.append(f"[WARN] Resolution '{resolution}' not supported by Nano Banana Pro, falling back to 1K")
            resolution = "1K"
        all_logs.append(f"[INFO] Image generation model: {resolved_image_model}")

        if not gemini_api_key:
            all_logs.append("[ERROR] No Gemini API key provided")
            return self._error_output(all_logs)

        image_inputs = {
            "eyes": image_eyes,
            "eyebrows": image_eyebrows,
            "nose": image_nose,
            "lips": image_lips,
            "forehead": image_forehead,
            "cheekbones": image_cheekbones,
            "jawline": image_jawline,
            "hair": image_hair,
            "skin": image_skin,
            "full_face": image_full_face,
        }

        connected_images = []
        for slot, img in image_inputs.items():
            if img is not None:
                b64, mime = _image_to_base64(img)
                connected_images.append({"slot": slot, "data": b64, "mime_type": mime})

        has_images = len(connected_images) > 0

        photo_type_config = _photo_types_by_label.get(photo_type, _photo_types_list[0] if _photo_types_list else None)
        is_default_photo_type = (photo_type == "-- Not selected / System inferred --")

        all_logs.append(f"[INFO] Connected image inputs: {len(connected_images)}")
        all_logs.append(f"[INFO] Model: {model}")
        all_logs.append(f"[INFO] Photo type: {photo_type}")
        if photo_type_config:
            all_logs.append(f"[INFO] Photo type constraints: {photo_type_config.get('fixed_constraints_fragment', 'N/A')}")
        all_logs.append(f"[INFO] Aspect ratio: {aspect_ratio}, Resolution: {resolution}, top_p: {top_p}")

        manual_attrs = {
            "demographics.sex": sex,
            "demographics.ethnicity": ethnicity,
            "eye_system.eyes.shape_descriptor": eye_shape,
            "eye_system.eyes.size_descriptor": eye_size,
            "eye_system.eyes.tilt_descriptor": eye_tilt,
            "eye_system.eyes.color_descriptor": eye_color,
            "eye_system.eyebrows.thickness_descriptor": eyebrow_thickness,
            "eye_system.eyebrows.shape_descriptor": eyebrow_shape,
            "eye_system.eyebrows.color_descriptor": eyebrow_color,
            "central_system.nose.profile_descriptor": nose_profile,
            "central_system.nose.base_descriptor": nose_base,
            "central_system.nose.tip_descriptor": nose_tip,
            "central_system.lips.volume_descriptor": lips_volume,
            "central_system.lips.cupid_bow_descriptor": cupid_bow,
            "central_system.lips.proportion_descriptor": lips_proportion,
            "central_system.lips.natural_color_descriptor": lips_color,
            "structure.forehead_descriptor": forehead,
            "structure.cheekbones_descriptor": cheekbones,
            "structure.jawline_descriptor": jawline,
            "structure.chin_descriptor": chin,
            "volumes.cheeks_descriptor": cheeks,
            "volumes.submental_descriptor": submental,
            "volumes.face_neck_transition_descriptor": face_neck_transition,
            "hair.structure_descriptor": hair_structure,
            "hair.length_descriptor": hair_length,
            "hair.volume_descriptor": hair_volume,
            "hair.color_descriptor": hair_color,
            "skin.tone_descriptor": skin_tone,
            "skin.undertone_descriptor": skin_undertone,
            "skin.texture_descriptor": skin_texture,
            "skin.micro_texture_descriptor": skin_micro_texture,
            "skin.imperfections_descriptor": skin_imperfections,
            "skin.surface_reflection_descriptor": skin_reflection,
            "defects.wrinkles_descriptor": wrinkles,
            "defects.scars_descriptor": scars,
            "defects.deformations_descriptor": deformations,
            "defects.tone_loss_descriptor": tone_loss,
            "defects.skin_marks_descriptor": skin_marks,
            "defects.vitiligo_descriptor": vitiligo,
            "defects.under_eye_descriptor": under_eye,
            "expression.expression_descriptor": expression,
            "expression.variant_descriptor": expression_variant,
        }

        if brief_text and brief_text.strip():
            manual_attrs["_brief_text"] = brief_text.strip()

        non_auto_count = sum(1 for v in manual_attrs.values() if v and v != "auto")
        has_manual = non_auto_count > 0
        all_logs.append(f"[INFO] Manual attributes set: {non_auto_count}")
        if has_manual:
            non_auto = {k: v for k, v in manual_attrs.items() if v and v != "auto"}
            all_logs.append("[INFO] Selected manual attributes:")
            for attr_key, attr_val in non_auto.items():
                all_logs.append(f"  - {attr_key}: {attr_val}")

        image_analysis = None

        if has_images:
            all_logs.append("\n[FLOW] 3-step flow: Image Analysis -> Prompt Composition -> Image Generation")
            all_logs.append("-" * 40)

            result = analyze_face_images(gemini_api_key, model, connected_images)
            all_logs.extend(result["logs"])

            if not result["success"]:
                all_logs.append(f"[ERROR] Image analysis failed: {result.get('error', 'Unknown')}")
                return self._error_output(all_logs)

            image_analysis = result["analysis"]
            all_logs.append("[STEP 1] Image analysis result:")
            all_logs.append(json.dumps(image_analysis, indent=2))
            mode = "auto_detect"

        elif has_manual:
            all_logs.append("\n[FLOW] 2-step flow: Prompt Composition -> Image Generation")
            mode = "manual_select"

        else:
            all_logs.append("\n[FLOW] 2-step flow: Generate New -> Image Generation")
            mode = "generate_new"

        all_logs.append("-" * 40)

        photo_constraints = None
        if photo_type_config and not is_default_photo_type:
            photo_constraints = {
                "fixed_constraints_fragment": photo_type_config["fixed_constraints_fragment"],
                "system_constraints": photo_type_config["system_constraints"],
            }
            all_logs.append(f"[INFO] Passing photo type constraints to compose step: {photo_type_config['label']}")
        else:
            all_logs.append("[INFO] Using default system prompt constraints (no photo type override)")

        compose_result = compose_aion_prompt(
            api_key=gemini_api_key,
            model=model,
            mode=mode,
            manual_attributes=manual_attrs if has_manual else None,
            image_analysis=image_analysis,
            photo_constraints=photo_constraints
        )
        all_logs.extend(compose_result["logs"])

        if not compose_result["success"]:
            all_logs.append(f"[ERROR] Prompt composition failed: {compose_result.get('error', 'Unknown')}")
            return self._error_output(all_logs, compose_result.get("result", {}))

        nano_prompt = compose_result["prompt"]
        aion_result = compose_result["result"]

        all_logs.append("[STEP 2] Full AION_THETA_RESULT JSON:")
        all_logs.append(json.dumps(aion_result, indent=2))

        if not nano_prompt:
            all_logs.append("[ERROR] No nano_banana_prompt_sentence generated")
            return self._error_output(all_logs, aion_result)

        all_logs.append("-" * 40)

        gen_result = generate_face_image(
            api_key=gemini_api_key,
            prompt=nano_prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            top_p=top_p,
            reference_images=connected_images if has_images else None,
            image_model=resolved_image_model
        )
        all_logs.extend(gen_result["logs"])

        if not gen_result["success"]:
            all_logs.append(f"[ERROR] Image generation failed: {gen_result.get('error', 'Unknown')}")
            return self._error_output(all_logs, aion_result)

        image_tensor = _base64_to_image_tensor(gen_result["image_base64"], gen_result["mime_type"])

        all_logs.append("=" * 60)
        all_logs.append("[SUCCESS] Face generation completed successfully")
        all_logs.append("=" * 60)

        log_text = "\n".join(all_logs)
        json_text = json.dumps(aion_result, indent=2)

        return (image_tensor, log_text, json_text)

    def _error_output(self, logs: list, aion_result: dict = None):
        logs.append("=" * 60)
        logs.append("[FAILED] Generation did not complete")
        logs.append("=" * 60)

        if torch is not None:
            blank = torch.zeros(1, 64, 64, 3)
        else:
            blank = np.zeros((1, 64, 64, 3), dtype=np.float32)

        log_text = "\n".join(logs)
        json_text = json.dumps(aion_result if aion_result else {}, indent=2)

        return (blank, log_text, json_text)


class AionFluxPrompterNode:
    """
    AION - Portrait Master Prompter

    Generates optimized text prompts for Flux2 Klein and zimage from
    facial attribute selections. Two composition modes:

    - Local (no Gemini): Builds prompts by combining selected attributes
      with optimized templates. Fast, free, deterministic.
    - Gemini-assisted: Sends attributes to Gemini 3.0 Flash for richer,
      more natural prompt composition.

    Two outputs optimized for different models:
    - zimage_prompt: keyword-rich with technical photography specs
    - flux_prompt: natural language descriptive paragraph

    No image inputs. Connect outputs to Flux2 Klein or zimage nodes.
    """

    CATEGORY = "AION"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("zimage_prompt", "flux_prompt", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional - leave empty for local mode"
                }),
                "api_key_status": ("STRING", {
                    "default": "Not Verified",
                    "multiline": False,
                }),
                "photo_type": (PHOTO_TYPE_OPTIONS, {
                    "default": "-- Not selected / System inferred --"
                }),
                "brief_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional brief text to guide prompt generation"
                }),

                "sex": (_get_options(_demographics_config, "sex"), {"default": "auto"}),
                "ethnicity": (_get_options(_demographics_config, "ethnicity"), {"default": "auto"}),

                "eye_shape": (_get_options(_eyes_config, "eye_shape"), {"default": "auto"}),
                "eye_size": (_get_options(_eyes_config, "eye_size"), {"default": "auto"}),
                "eye_tilt": (_get_options(_eyes_config, "eye_tilt"), {"default": "auto"}),
                "eye_color": (_get_options(_eyes_config, "eye_color"), {"default": "auto"}),

                "eyebrow_thickness": (_get_options(_eyebrows_config, "eyebrow_thickness"), {"default": "auto"}),
                "eyebrow_shape": (_get_options(_eyebrows_config, "eyebrow_shape"), {"default": "auto"}),
                "eyebrow_color": (_get_options(_eyebrows_config, "eyebrow_color"), {"default": "auto"}),

                "nose_profile": (_get_options(_nose_config, "nose_profile"), {"default": "auto"}),
                "nose_base": (_get_options(_nose_config, "nose_base"), {"default": "auto"}),
                "nose_tip": (_get_options(_nose_config, "nose_tip"), {"default": "auto"}),

                "lips_volume": (_get_options(_lips_config, "lips_volume"), {"default": "auto"}),
                "cupid_bow": (_get_options(_lips_config, "cupid_bow"), {"default": "auto"}),
                "lips_proportion": (_get_options(_lips_config, "lips_proportion"), {"default": "auto"}),
                "lips_color": (_get_options(_lips_config, "lips_color"), {"default": "auto"}),

                "forehead": (_get_options(_structure_config, "forehead"), {"default": "auto"}),
                "cheekbones": (_get_options(_structure_config, "cheekbones"), {"default": "auto"}),
                "jawline": (_get_options(_structure_config, "jawline"), {"default": "auto"}),
                "chin": (_get_options(_structure_config, "chin"), {"default": "auto"}),

                "cheeks": (_get_options(_volumes_config, "cheeks"), {"default": "auto"}),
                "submental": (_get_options(_volumes_config, "submental"), {"default": "auto"}),
                "face_neck_transition": (_get_options(_volumes_config, "face_neck_transition"), {"default": "auto"}),

                "hair_structure": (_get_options(_hair_config, "hair_structure"), {"default": "auto"}),
                "hair_length": (_get_options(_hair_config, "hair_length"), {"default": "auto"}),
                "hair_volume": (_get_options(_hair_config, "hair_volume"), {"default": "auto"}),
                "hair_color": (_get_options(_hair_config, "hair_color"), {"default": "auto"}),

                "skin_tone": (_get_options(_skin_config, "skin_tone"), {"default": "auto"}),
                "skin_undertone": (_get_options(_skin_config, "skin_undertone"), {"default": "auto"}),
                "skin_texture": (_get_options(_skin_config, "skin_texture"), {"default": "auto"}),
                "skin_micro_texture": (_get_options(_skin_config, "skin_micro_texture"), {"default": "auto"}),
                "skin_imperfections": (_get_options(_skin_config, "skin_imperfections"), {"default": "auto"}),
                "skin_reflection": (_get_options(_skin_config, "skin_reflection"), {"default": "auto"}),

                "wrinkles": (_get_options(_defects_config, "wrinkles"), {"default": "auto"}),
                "scars": (_get_options(_defects_config, "scars"), {"default": "auto"}),
                "deformations": (_get_options(_defects_config, "deformations"), {"default": "auto"}),
                "tone_loss": (_get_options(_defects_config, "tone_loss"), {"default": "auto"}),
                "skin_marks": (_get_options(_defects_config, "skin_marks"), {"default": "auto"}),
                "vitiligo": (_get_options(_defects_config, "vitiligo"), {"default": "auto"}),
                "under_eye": (_get_options(_defects_config, "under_eye"), {"default": "auto"}),

                "expression": (_get_options(_expressions_config, "expression"), {"default": "auto"}),
                "expression_variant": (_get_options(_expressions_config, "expression_variant"), {"default": "auto"}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def compose(self, gemini_api_key, api_key_status,
                photo_type, brief_text,
                sex, ethnicity,
                eye_shape, eye_size, eye_tilt, eye_color,
                eyebrow_thickness, eyebrow_shape, eyebrow_color,
                nose_profile, nose_base, nose_tip,
                lips_volume, cupid_bow, lips_proportion, lips_color,
                forehead, cheekbones, jawline, chin,
                cheeks, submental, face_neck_transition,
                hair_structure, hair_length, hair_volume, hair_color,
                skin_tone, skin_undertone, skin_texture,
                skin_micro_texture, skin_imperfections, skin_reflection,
                wrinkles, scars, deformations, tone_loss, skin_marks, vitiligo, under_eye,
                expression, expression_variant):

        all_logs = []
        all_logs.append("=" * 60)
        all_logs.append("AION - Portrait Master Prompter")
        all_logs.append("=" * 60)

        use_gemini = gemini_api_key and gemini_api_key.strip()
        all_logs.append(f"[INFO] Mode: {'Gemini-assisted' if use_gemini else 'Local (no API)'}")

        photo_type_config = _photo_types_by_label.get(photo_type, _photo_types_list[0] if _photo_types_list else None)
        is_default_photo_type = (photo_type == "-- Not selected / System inferred --")
        all_logs.append(f"[INFO] Photo type: {photo_type}")

        manual_attrs = {
            "demographics.sex": sex,
            "demographics.ethnicity": ethnicity,
            "eye_system.eyes.shape_descriptor": eye_shape,
            "eye_system.eyes.size_descriptor": eye_size,
            "eye_system.eyes.tilt_descriptor": eye_tilt,
            "eye_system.eyes.color_descriptor": eye_color,
            "eye_system.eyebrows.thickness_descriptor": eyebrow_thickness,
            "eye_system.eyebrows.shape_descriptor": eyebrow_shape,
            "eye_system.eyebrows.color_descriptor": eyebrow_color,
            "central_system.nose.profile_descriptor": nose_profile,
            "central_system.nose.base_descriptor": nose_base,
            "central_system.nose.tip_descriptor": nose_tip,
            "central_system.lips.volume_descriptor": lips_volume,
            "central_system.lips.cupid_bow_descriptor": cupid_bow,
            "central_system.lips.proportion_descriptor": lips_proportion,
            "central_system.lips.natural_color_descriptor": lips_color,
            "structure.forehead_descriptor": forehead,
            "structure.cheekbones_descriptor": cheekbones,
            "structure.jawline_descriptor": jawline,
            "structure.chin_descriptor": chin,
            "volumes.cheeks_descriptor": cheeks,
            "volumes.submental_descriptor": submental,
            "volumes.face_neck_transition_descriptor": face_neck_transition,
            "hair.structure_descriptor": hair_structure,
            "hair.length_descriptor": hair_length,
            "hair.volume_descriptor": hair_volume,
            "hair.color_descriptor": hair_color,
            "skin.tone_descriptor": skin_tone,
            "skin.undertone_descriptor": skin_undertone,
            "skin.texture_descriptor": skin_texture,
            "skin.micro_texture_descriptor": skin_micro_texture,
            "skin.imperfections_descriptor": skin_imperfections,
            "skin.surface_reflection_descriptor": skin_reflection,
            "defects.wrinkles_descriptor": wrinkles,
            "defects.scars_descriptor": scars,
            "defects.deformations_descriptor": deformations,
            "defects.tone_loss_descriptor": tone_loss,
            "defects.skin_marks_descriptor": skin_marks,
            "defects.vitiligo_descriptor": vitiligo,
            "defects.under_eye_descriptor": under_eye,
            "expression.expression_descriptor": expression,
            "expression.variant_descriptor": expression_variant,
        }

        if brief_text and brief_text.strip():
            manual_attrs["_brief_text"] = brief_text.strip()

        non_auto_count = sum(1 for v in manual_attrs.values() if v and v != "auto")
        all_logs.append(f"[INFO] Attributes set: {non_auto_count}")
        if non_auto_count > 0:
            non_auto = {k: v for k, v in manual_attrs.items() if v and v != "auto"}
            all_logs.append("[INFO] Selected attributes:")
            for attr_key, attr_val in non_auto.items():
                all_logs.append(f"  - {attr_key}: {attr_val}")

        photo_constraints = None
        if photo_type_config and not is_default_photo_type:
            photo_constraints = {
                "fixed_constraints_fragment": photo_type_config["fixed_constraints_fragment"],
                "system_constraints": photo_type_config["system_constraints"],
            }
            all_logs.append(f"[INFO] Photo type constraints: {photo_type_config['label']}")
        else:
            all_logs.append("[INFO] Using default constraints")

        all_logs.append("-" * 40)

        brief = brief_text.strip() if brief_text else ""

        if use_gemini:
            result = compose_flux_prompt_gemini(
                api_key=gemini_api_key.strip(),
                manual_attributes=manual_attrs,
                photo_constraints=photo_constraints,
                brief_text=brief,
            )
        else:
            result = compose_flux_prompt_local(
                manual_attributes=manual_attrs,
                photo_constraints=photo_constraints,
                brief_text=brief,
            )

        all_logs.extend(result["logs"])

        if not result["success"]:
            all_logs.append(f"[ERROR] Prompt composition failed: {result.get('error', 'Unknown')}")
            return self._error_output(all_logs)

        flux_prompt = result["flux_prompt"]
        zimage_prompt = result["zimage_prompt"]

        all_logs.append("-" * 40)
        all_logs.append("=" * 60)
        all_logs.append("[SUCCESS] Prompt generation completed")
        all_logs.append(f"[OUTPUT] Flux prompt: {flux_prompt}")
        all_logs.append(f"[OUTPUT] zimage prompt: {zimage_prompt}")
        all_logs.append("=" * 60)

        log_text = "\n".join(all_logs)
        return (zimage_prompt, flux_prompt, log_text)

    def _error_output(self, logs: list):
        logs.append("=" * 60)
        logs.append("[FAILED] Prompt generation did not complete")
        logs.append("=" * 60)
        log_text = "\n".join(logs)
        return ("", "", log_text)


class AionFusionNode:
    """
    AION - Portrait Master Fusion

    Merges individual facial-part reference images into a single coherent
    realistic portrait using Nano Banana Pro (Gemini image generation).

    Flow (1 step - direct):
      Send reference images + generic fusion prompt directly to Nano Banana Pro.
      The model merges all facial parts (hair, eyes, nose, lips, etc.) into
      one consistent person with natural blending.

    Optional combo attributes act as subtle modifiers to guide the result.
    No intermediate Gemini analysis step — images go straight to generation.
    """

    CATEGORY = "AION"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Enter your Gemini API key"
                }),
                "api_key_status": ("STRING", {
                    "default": "Not Verified",
                    "multiline": False,
                }),
                "image_model": (IMAGE_MODEL_OPTIONS, {
                    "default": "Nano Banana Pro (gemini-3-pro-image-preview)"
                }),
                "photo_type": (PHOTO_TYPE_OPTIONS, {
                    "default": "-- Not selected / System inferred --"
                }),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {
                    "default": "1:1"
                }),
                "resolution": (RESOLUTION_OPTIONS, {
                    "default": "1K"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "brief_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional brief text to guide fusion"
                }),

                "sex": (_get_options(_demographics_config, "sex"), {"default": "auto"}),
                "ethnicity": (_get_options(_demographics_config, "ethnicity"), {"default": "auto"}),

                "eye_shape": (_get_options(_eyes_config, "eye_shape"), {"default": "auto"}),
                "eye_size": (_get_options(_eyes_config, "eye_size"), {"default": "auto"}),
                "eye_tilt": (_get_options(_eyes_config, "eye_tilt"), {"default": "auto"}),
                "eye_color": (_get_options(_eyes_config, "eye_color"), {"default": "auto"}),

                "eyebrow_thickness": (_get_options(_eyebrows_config, "eyebrow_thickness"), {"default": "auto"}),
                "eyebrow_shape": (_get_options(_eyebrows_config, "eyebrow_shape"), {"default": "auto"}),
                "eyebrow_color": (_get_options(_eyebrows_config, "eyebrow_color"), {"default": "auto"}),

                "nose_profile": (_get_options(_nose_config, "nose_profile"), {"default": "auto"}),
                "nose_base": (_get_options(_nose_config, "nose_base"), {"default": "auto"}),
                "nose_tip": (_get_options(_nose_config, "nose_tip"), {"default": "auto"}),

                "lips_volume": (_get_options(_lips_config, "lips_volume"), {"default": "auto"}),
                "cupid_bow": (_get_options(_lips_config, "cupid_bow"), {"default": "auto"}),
                "lips_proportion": (_get_options(_lips_config, "lips_proportion"), {"default": "auto"}),
                "lips_color": (_get_options(_lips_config, "lips_color"), {"default": "auto"}),

                "forehead": (_get_options(_structure_config, "forehead"), {"default": "auto"}),
                "cheekbones": (_get_options(_structure_config, "cheekbones"), {"default": "auto"}),
                "jawline": (_get_options(_structure_config, "jawline"), {"default": "auto"}),
                "chin": (_get_options(_structure_config, "chin"), {"default": "auto"}),

                "cheeks": (_get_options(_volumes_config, "cheeks"), {"default": "auto"}),
                "submental": (_get_options(_volumes_config, "submental"), {"default": "auto"}),
                "face_neck_transition": (_get_options(_volumes_config, "face_neck_transition"), {"default": "auto"}),

                "hair_structure": (_get_options(_hair_config, "hair_structure"), {"default": "auto"}),
                "hair_length": (_get_options(_hair_config, "hair_length"), {"default": "auto"}),
                "hair_volume": (_get_options(_hair_config, "hair_volume"), {"default": "auto"}),
                "hair_color": (_get_options(_hair_config, "hair_color"), {"default": "auto"}),

                "skin_tone": (_get_options(_skin_config, "skin_tone"), {"default": "auto"}),
                "skin_undertone": (_get_options(_skin_config, "skin_undertone"), {"default": "auto"}),
                "skin_texture": (_get_options(_skin_config, "skin_texture"), {"default": "auto"}),
                "skin_micro_texture": (_get_options(_skin_config, "skin_micro_texture"), {"default": "auto"}),
                "skin_imperfections": (_get_options(_skin_config, "skin_imperfections"), {"default": "auto"}),
                "skin_reflection": (_get_options(_skin_config, "skin_reflection"), {"default": "auto"}),

                "wrinkles": (_get_options(_defects_config, "wrinkles"), {"default": "auto"}),
                "scars": (_get_options(_defects_config, "scars"), {"default": "auto"}),
                "deformations": (_get_options(_defects_config, "deformations"), {"default": "auto"}),
                "tone_loss": (_get_options(_defects_config, "tone_loss"), {"default": "auto"}),
                "skin_marks": (_get_options(_defects_config, "skin_marks"), {"default": "auto"}),
                "vitiligo": (_get_options(_defects_config, "vitiligo"), {"default": "auto"}),
                "under_eye": (_get_options(_defects_config, "under_eye"), {"default": "auto"}),

                "expression": (_get_options(_expressions_config, "expression"), {"default": "auto"}),
                "expression_variant": (_get_options(_expressions_config, "expression_variant"), {"default": "auto"}),
            },
            "optional": {
                "image_eyes": ("IMAGE",),
                "image_eyebrows": ("IMAGE",),
                "image_nose": ("IMAGE",),
                "image_lips": ("IMAGE",),
                "image_forehead": ("IMAGE",),
                "image_cheekbones": ("IMAGE",),
                "image_ears": ("IMAGE",),
                "image_hair": ("IMAGE",),
                "image_skin": ("IMAGE",),
                "image_chin": ("IMAGE",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, gemini_api_key, api_key_status, image_model,
                 photo_type, aspect_ratio, resolution, top_p,
                 brief_text,
                 sex, ethnicity,
                 eye_shape, eye_size, eye_tilt, eye_color,
                 eyebrow_thickness, eyebrow_shape, eyebrow_color,
                 nose_profile, nose_base, nose_tip,
                 lips_volume, cupid_bow, lips_proportion, lips_color,
                 forehead, cheekbones, jawline, chin,
                 cheeks, submental, face_neck_transition,
                 hair_structure, hair_length, hair_volume, hair_color,
                 skin_tone, skin_undertone, skin_texture,
                 skin_micro_texture, skin_imperfections, skin_reflection,
                 wrinkles, scars, deformations, tone_loss, skin_marks, vitiligo, under_eye,
                 expression, expression_variant,
                 image_eyes=None, image_eyebrows=None, image_nose=None,
                 image_lips=None, image_forehead=None, image_cheekbones=None,
                 image_ears=None, image_hair=None, image_skin=None,
                 image_chin=None):

        all_logs = []
        all_logs.append("=" * 60)
        all_logs.append("AION - Portrait Master Fusion")
        all_logs.append("=" * 60)

        resolved_image_model = _IMAGE_MODEL_MAP.get(image_model, "gemini-3-pro-image-preview")
        if resolution in _PRO_UNSUPPORTED_RESOLUTIONS and resolved_image_model == "gemini-3-pro-image-preview":
            all_logs.append(f"[WARN] Resolution '{resolution}' not supported by Nano Banana Pro, falling back to 1K")
            resolution = "1K"
        all_logs.append(f"[INFO] Image generation model: {resolved_image_model}")

        if not gemini_api_key:
            all_logs.append("[ERROR] No Gemini API key provided")
            return self._error_output(all_logs)

        image_inputs = {
            "eyes": image_eyes,
            "eyebrows": image_eyebrows,
            "nose": image_nose,
            "lips": image_lips,
            "forehead": image_forehead,
            "cheekbones": image_cheekbones,
            "ears": image_ears,
            "hair": image_hair,
            "skin": image_skin,
            "chin": image_chin,
        }

        connected_images = []
        for slot, img in image_inputs.items():
            if img is not None:
                b64, mime = _image_to_base64(img)
                connected_images.append({"slot": slot, "data": b64, "mime_type": mime})

        if len(connected_images) == 0:
            all_logs.append("[ERROR] No reference images connected. At least one image input is required for fusion.")
            return self._error_output(all_logs)

        all_logs.append(f"[INFO] Reference images connected: {len(connected_images)}")
        for ci in connected_images:
            all_logs.append(f"  - {ci['slot']}")

        photo_type_config = _photo_types_by_label.get(photo_type, _photo_types_list[0] if _photo_types_list else None)
        is_default_photo_type = (photo_type == "-- Not selected / System inferred --")
        all_logs.append(f"[INFO] Photo type: {photo_type}")
        all_logs.append(f"[INFO] Aspect ratio: {aspect_ratio}, Resolution: {resolution}, top_p: {top_p}")

        manual_attrs = {
            "demographics.sex": sex,
            "demographics.ethnicity": ethnicity,
            "eye_system.eyes.shape_descriptor": eye_shape,
            "eye_system.eyes.size_descriptor": eye_size,
            "eye_system.eyes.tilt_descriptor": eye_tilt,
            "eye_system.eyes.color_descriptor": eye_color,
            "eye_system.eyebrows.thickness_descriptor": eyebrow_thickness,
            "eye_system.eyebrows.shape_descriptor": eyebrow_shape,
            "eye_system.eyebrows.color_descriptor": eyebrow_color,
            "central_system.nose.profile_descriptor": nose_profile,
            "central_system.nose.base_descriptor": nose_base,
            "central_system.nose.tip_descriptor": nose_tip,
            "central_system.lips.volume_descriptor": lips_volume,
            "central_system.lips.cupid_bow_descriptor": cupid_bow,
            "central_system.lips.proportion_descriptor": lips_proportion,
            "central_system.lips.natural_color_descriptor": lips_color,
            "structure.forehead_descriptor": forehead,
            "structure.cheekbones_descriptor": cheekbones,
            "structure.jawline_descriptor": jawline,
            "structure.chin_descriptor": chin,
            "volumes.cheeks_descriptor": cheeks,
            "volumes.submental_descriptor": submental,
            "volumes.face_neck_transition_descriptor": face_neck_transition,
            "hair.structure_descriptor": hair_structure,
            "hair.length_descriptor": hair_length,
            "hair.volume_descriptor": hair_volume,
            "hair.color_descriptor": hair_color,
            "skin.tone_descriptor": skin_tone,
            "skin.undertone_descriptor": skin_undertone,
            "skin.texture_descriptor": skin_texture,
            "skin.micro_texture_descriptor": skin_micro_texture,
            "skin.imperfections_descriptor": skin_imperfections,
            "skin.surface_reflection_descriptor": skin_reflection,
            "defects.wrinkles_descriptor": wrinkles,
            "defects.scars_descriptor": scars,
            "defects.deformations_descriptor": deformations,
            "defects.tone_loss_descriptor": tone_loss,
            "defects.skin_marks_descriptor": skin_marks,
            "defects.vitiligo_descriptor": vitiligo,
            "defects.under_eye_descriptor": under_eye,
            "expression.expression_descriptor": expression,
            "expression.variant_descriptor": expression_variant,
        }

        non_auto_count = sum(1 for v in manual_attrs.values() if v and v != "auto")
        if non_auto_count > 0:
            non_auto = {k: v for k, v in manual_attrs.items() if v and v != "auto"}
            all_logs.append(f"[INFO] Attribute modifiers set: {non_auto_count}")
            for attr_key, attr_val in non_auto.items():
                all_logs.append(f"  - {attr_key}: {attr_val}")
        else:
            all_logs.append("[INFO] No attribute modifiers (using reference images only)")

        photo_constraints = None
        if photo_type_config and not is_default_photo_type:
            photo_constraints = {
                "fixed_constraints_fragment": photo_type_config["fixed_constraints_fragment"],
                "system_constraints": photo_type_config["system_constraints"],
            }
            all_logs.append(f"[INFO] Photo type constraints applied: {photo_type_config['label']}")

        all_logs.append("-" * 40)
        all_logs.append("[FLOW] Direct fusion: Reference images + prompt -> Nano Banana Pro")
        all_logs.append("-" * 40)

        brief = brief_text.strip() if brief_text else ""

        result = generate_fusion_portrait(
            api_key=gemini_api_key,
            images=connected_images,
            manual_attributes=manual_attrs if non_auto_count > 0 else None,
            photo_constraints=photo_constraints,
            brief_text=brief,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            top_p=top_p,
            image_model=resolved_image_model,
        )
        all_logs.extend(result["logs"])

        if not result["success"]:
            all_logs.append(f"[ERROR] Fusion generation failed: {result.get('error', 'Unknown')}")
            return self._error_output(all_logs)

        image_tensor = _base64_to_image_tensor(result["image_base64"], result["mime_type"])

        all_logs.append("=" * 60)
        all_logs.append("[SUCCESS] Fusion portrait generated successfully")
        all_logs.append("=" * 60)

        log_text = "\n".join(all_logs)
        return (image_tensor, log_text)

    def _error_output(self, logs: list):
        logs.append("=" * 60)
        logs.append("[FAILED] Fusion generation did not complete")
        logs.append("=" * 60)

        if torch is not None:
            blank = torch.zeros(1, 64, 64, 3)
        else:
            blank = np.zeros((1, 64, 64, 3), dtype=np.float32)

        log_text = "\n".join(logs)
        return (blank, log_text)


NODE_CLASS_MAPPINGS = {
    "AionThetaNode": AionThetaNode,
    "AionFluxPrompterNode": AionFluxPrompterNode,
    "AionFusionNode": AionFusionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AionThetaNode": "AION - Gemini Portrait Master Generator",
    "AionFluxPrompterNode": "AION - Portrait Master Prompter",
    "AionFusionNode": "AION - Portrait Master Fusion",
}
