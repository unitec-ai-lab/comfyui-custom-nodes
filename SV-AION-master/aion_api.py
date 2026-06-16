import json
import base64
import urllib.request
import urllib.error
import os

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
NANO_BANANA_PRO = "gemini-3-pro-image-preview"
NANO_BANANA_2 = "gemini-3.1-flash-image-preview"


def _try_repair_json(text: str):
    """
    Attempt to repair truncated JSON by finding the last complete key-value pair
    and closing all open braces/brackets.
    Returns parsed dict or None if repair fails.
    """
    if not text or not text.strip().startswith("{"):
        return None

    clean = text.rstrip()

    strategies = []

    pos = len(clean)
    while pos > 0:
        pos = clean.rfind('"', 0, pos)
        if pos < 0:
            break

        after = clean[pos + 1:].strip()

        if not after or after[0] in ',}]':
            candidate = clean[:pos + 1]
            if candidate.rstrip().endswith('"'):
                strategies.append(candidate)
            pos -= 1
            continue

        if after.startswith(':'):
            key_start = clean.rfind('"', 0, pos)
            if key_start >= 0:
                candidate = clean[:key_start]
                candidate = candidate.rstrip().rstrip(',')
                strategies.append(candidate)
            pos -= 1
            continue

        pos -= 1

    for candidate in strategies:
        open_braces = candidate.count('{') - candidate.count('}')
        open_brackets = candidate.count('[') - candidate.count(']')

        if open_braces < 0 or open_brackets < 0:
            continue

        attempt = candidate.rstrip().rstrip(',')
        attempt += '}' * open_braces

        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue

    return None


def verify_api_key(api_key: str) -> dict:
    """
    Verify the Gemini API key by making a single GET request to the models endpoint.
    Returns {"valid": True/False, "error": "..." if invalid}.
    Only ONE call, no retries, no loops.
    """
    url = f"{GEMINI_API_BASE}/models?key={api_key}"
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            model_names = [m.get("name", "") for m in data.get("models", [])]
            return {"valid": True, "models": model_names}
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")
        except Exception:
            pass
        return {"valid": False, "error": f"HTTP {e.code}: {error_body}"}
    except urllib.error.URLError as e:
        return {"valid": False, "error": f"Connection error: {str(e.reason)}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def load_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _gemini_generate_content(api_key: str, model: str, contents: list, generation_config: dict = None) -> dict:
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent?key={api_key}"

    payload = {"contents": contents}
    if generation_config:
        payload["generationConfig"] = generation_config

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def analyze_face_images(api_key: str, model: str, images: list) -> dict:
    """
    Step 1 (with images): Send images to Gemini 3.1 Pro or 3 Flash
    to analyze facial features and produce a structured JSON description.

    images: list of dicts with {"slot": str, "data": base64_str, "mime_type": str}
    Returns: {"success": bool, "analysis": dict, "logs": list, "error": str}
    """
    logs = []
    logs.append(f"[STEP 1] Analyzing {len(images)} image(s) with model: {model}")

    parts = []

    slot_descriptions = {
        "eyes": "eyes region",
        "eyebrows": "eyebrows region",
        "nose": "nose region",
        "lips": "lips and mouth region",
        "forehead": "forehead region",
        "cheekbones": "cheekbones region",
        "jawline": "jawline region",
        "hair": "hair",
        "skin": "skin texture and tone",
        "full_face": "complete face"
    }

    analysis_prompt = (
        "You are a facial morphology analyst. Analyze the provided image(s) and describe "
        "each facial feature using short descriptors (max 5 words each). "
        "Do NOT infer ethnicity or sex. Set those to 'unspecified'.\n\n"
        "Each image is labeled with the facial region it represents:\n\n"
    )

    for i, img in enumerate(images):
        slot = img.get("slot", f"image_{i}")
        desc = slot_descriptions.get(slot, slot)
        analysis_prompt += f"Image {i+1}: {desc}\n"
        logs.append(f"  - Slot '{slot}' ({desc}): image loaded")

    analysis_prompt += (
        "\nReturn ONLY a valid JSON object (no markdown fences, no text before or after) "
        "with this exact structure. Use short descriptors (max 5 words per field):\n"
        '{"eye_system":{"eyes":{"shape_descriptor":"","size_descriptor":"","tilt_descriptor":"","color_descriptor":""},'
        '"eyebrows":{"thickness_descriptor":"","shape_descriptor":"","color_descriptor":""}},'
        '"central_system":{"nose":{"profile_descriptor":"","base_descriptor":"","tip_descriptor":""},'
        '"lips":{"volume_descriptor":"","cupid_bow_descriptor":"","proportion_descriptor":"","natural_color_descriptor":""}},'
        '"structure":{"forehead_descriptor":"","cheekbones_descriptor":"","jawline_descriptor":"","chin_descriptor":""},'
        '"volumes":{"cheeks_descriptor":"","submental_descriptor":"","face_neck_transition_descriptor":""},'
        '"hair":{"structure_descriptor":"","length_descriptor":"","volume_descriptor":"","color_descriptor":""},'
        '"skin":{"tone_descriptor":"","undertone_descriptor":"","texture_descriptor":"","micro_texture_descriptor":"",'
        '"imperfections_descriptor":"","surface_reflection_descriptor":""}}\n\n'
        "Fill only fields for features visible in the images. Leave others as empty strings."
    )

    parts.append({"text": analysis_prompt})

    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime_type"],
                "data": img["data"]
            }
        })

    try:
        response = _gemini_generate_content(
            api_key, model,
            [{"parts": parts}],
            {"temperature": 0.3, "maxOutputTokens": 8192}
        )

        text_response = ""
        finish_reason = ""
        candidates = response.get("candidates", [])
        if candidates:
            finish_reason = candidates[0].get("finishReason", "")
            content_parts = candidates[0].get("content", {}).get("parts", [])
            for part in content_parts:
                if "text" in part:
                    text_response += part["text"]

        logs.append(f"[STEP 1] Raw response length: {len(text_response)} chars")
        if finish_reason:
            logs.append(f"[STEP 1] Finish reason: {finish_reason}")

        clean_text = text_response.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines)

        try:
            analysis = json.loads(clean_text)
        except json.JSONDecodeError:
            analysis = _try_repair_json(clean_text)
            if analysis is not None:
                logs.append("[STEP 1] JSON was truncated, repaired successfully")
            else:
                raise

        logs.append("[STEP 1] Successfully parsed facial analysis JSON")

        return {"success": True, "analysis": analysis, "logs": logs}

    except json.JSONDecodeError as e:
        logs.append(f"[STEP 1] ERROR: Failed to parse JSON response: {e}")
        logs.append(f"[STEP 1] Raw text: {text_response[:500]}")
        return {"success": False, "analysis": {}, "logs": logs, "error": f"JSON parse error: {e}"}
    except Exception as e:
        logs.append(f"[STEP 1] ERROR: {str(e)}")
        return {"success": False, "analysis": {}, "logs": logs, "error": str(e)}


def compose_aion_prompt(api_key: str, model: str, mode: str,
                        manual_attributes: dict = None,
                        image_analysis: dict = None,
                        photo_constraints: dict = None) -> dict:
    """
    Step 2: Combine attributes (from selects or image analysis) with the AION Theta
    system prompt and send to Gemini to produce the full AION_THETA_RESULT JSON
    including the nano_banana_prompt_sentence.

    mode: "manual_select" | "auto_detect" | "generate_new"
    manual_attributes: dict of user-selected values (when mode is manual_select)
    image_analysis: dict from analyze_face_images (when mode is auto_detect)
    photo_constraints: dict with "fixed_constraints_fragment" and "system_constraints" overrides

    Returns: {"success": bool, "result": dict, "prompt": str, "logs": list}
    """
    logs = []
    logs.append(f"[STEP 2] Composing AION Theta prompt with mode: {mode}, model: {model}")

    system_prompt = load_system_prompt()

    if photo_constraints:
        override_constraints = photo_constraints.get("system_constraints", "")
        override_fragment = photo_constraints.get("fixed_constraints_fragment", "")
        logs.append(f"[STEP 2] Overriding system prompt constraints with photo type")
        logs.append(f"[STEP 2] New system_constraints: {override_constraints}")
        logs.append(f"[STEP 2] New fixed_constraints_fragment: {override_fragment}")

        old_constraints = (
            "FIXED STUDIO CONSTRAINTS (MUST ALWAYS APPLY)\n"
            "- Pure white background (#FFFFFF), high-key studio, clean and textureless background.\n"
            "- Front-facing full-face close-up portrait, centered and symmetrical.\n"
            "- Only shoulders visible, wearing a plain white tank top (white camisole).\n"
            "- Minimal natural editorial makeup, realistic anatomy, no stylization."
        )
        new_constraints = (
            f"FIXED STUDIO CONSTRAINTS (MUST ALWAYS APPLY)\n"
            f"{override_constraints}"
        )
        system_prompt = system_prompt.replace(old_constraints, new_constraints)

        old_fragment = '"fixed_constraints_fragment": "Pure white #FFFFFF background, high-key studio, centered full-face close-up, shoulders only, plain white tank top, minimal natural editorial makeup."'
        new_fragment = f'"fixed_constraints_fragment": "{override_fragment}"'
        system_prompt = system_prompt.replace(old_fragment, new_fragment)

        old_final_line = "The final output must always be a centered, symmetrical, full-face close-up on a pure white background, with only shoulders visible and a plain white tank top."
        system_prompt = system_prompt.replace(old_final_line, "The final output must always be a centered, symmetrical, full-face close-up portrait following the FIXED STUDIO CONSTRAINTS below.")

        logs.append("[STEP 2] System prompt patched with photo type constraints")
    else:
        logs.append("[STEP 2] Using default system prompt constraints (white background studio)")

    user_message = ""

    if mode == "auto_detect" and image_analysis:
        user_message = (
            f"Mode: auto_detect\n\n"
            f"The following facial analysis was obtained from reference images:\n"
            f"```json\n{json.dumps(image_analysis, indent=2)}\n```\n\n"
            f"Using this analysis, generate the complete AION_THETA_RESULT JSON. "
            f"Remember: do NOT infer ethnicity or sex — set them to 'unspecified'. "
            f"Generate the nano_banana_prompt_sentence as one English sentence."
        )
        logs.append("[STEP 2] Using image analysis data for auto_detect mode")

    elif mode == "manual_select" and manual_attributes:
        non_auto = {k: v for k, v in manual_attributes.items() if v and v != "auto"}
        user_message = (
            f"Mode: manual_select\n\n"
            f"The user has selected the following facial attributes:\n"
            f"```json\n{json.dumps(non_auto, indent=2)}\n```\n\n"
            f"Copy these values exactly into the corresponding fields and set confidence high. "
            f"For any fields not provided, generate coherent values that complement the selected ones. "
            f"Generate the complete AION_THETA_RESULT JSON with the nano_banana_prompt_sentence."
        )
        logs.append(f"[STEP 2] Using {len(non_auto)} manually selected attributes")

    else:
        user_message = (
            "Mode: generate_new\n\n"
            "No images or manual selections provided. "
            "Autonomously generate a new, distinct, diverse face profile with coherent traits. "
            "Generate the complete AION_THETA_RESULT JSON with the nano_banana_prompt_sentence."
        )
        logs.append("[STEP 2] Using generate_new mode (no inputs)")

    try:
        response = _gemini_generate_content(
            api_key, model,
            [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will output only the AION_THETA_RESULT JSON following all rules."}]},
                {"role": "user", "parts": [{"text": user_message}]}
            ],
            {"temperature": 0.7, "maxOutputTokens": 8192}
        )

        text_response = ""
        finish_reason = ""
        candidates = response.get("candidates", [])
        if candidates:
            finish_reason = candidates[0].get("finishReason", "")
            content_parts = candidates[0].get("content", {}).get("parts", [])
            for part in content_parts:
                if "text" in part:
                    text_response += part["text"]

        logs.append(f"[STEP 2] Raw response length: {len(text_response)} chars")
        if finish_reason:
            logs.append(f"[STEP 2] Finish reason: {finish_reason}")

        clean_text = text_response.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines)

        try:
            result = json.loads(clean_text)
        except json.JSONDecodeError:
            result = _try_repair_json(clean_text)
            if result is not None:
                logs.append("[STEP 2] JSON was truncated, repaired successfully")
            else:
                raise

        logs.append("[STEP 2] Successfully parsed AION_THETA_RESULT JSON")

        prompt = ""
        aion_data = result.get("AION_THETA_RESULT", result)

        prompt = aion_data.get("nano_banana_prompt_sentence", "")

        result_mode = aion_data.get("mode", "unknown")
        logs.append(f"[STEP 2] Result mode: {result_mode}")

        brief_fragments = aion_data.get("brief_fragments", {})
        if brief_fragments:
            logs.append("[STEP 2] Brief fragments from result:")
            for frag_key, frag_val in brief_fragments.items():
                logs.append(f"  - {frag_key}: {frag_val}")

        confidence = aion_data.get("confidence", {})
        if confidence:
            logs.append("[STEP 2] Confidence scores:")
            for conf_key, conf_val in confidence.items():
                logs.append(f"  - {conf_key}: {conf_val}")

        if prompt:
            logs.append(f"[STEP 2] Nano Banana prompt sentence: {prompt}")
        else:
            logs.append("[STEP 2] WARNING: No nano_banana_prompt_sentence found in result")

        return {"success": True, "result": result, "prompt": prompt, "logs": logs}

    except json.JSONDecodeError as e:
        logs.append(f"[STEP 2] ERROR: Failed to parse JSON: {e}")
        logs.append(f"[STEP 2] Raw text: {text_response[:500]}")
        return {"success": False, "result": {}, "prompt": "", "logs": logs, "error": f"JSON parse error: {e}"}
    except Exception as e:
        logs.append(f"[STEP 2] ERROR: {str(e)}")
        return {"success": False, "result": {}, "prompt": "", "logs": logs, "error": str(e)}


def generate_face_image(api_key: str, prompt: str,
                        aspect_ratio: str = "1:1",
                        resolution: str = "1K",
                        top_p: float = 0.95,
                        reference_images: list = None,
                        image_model: str = None) -> dict:
    """
    Step 3 (final): Send the nano_banana_prompt_sentence to the selected
    image generation model (Nano Banana Pro or Nano Banana 2).

    The prompt already includes the correct photo type constraints from compose step.

    aspect_ratio: e.g. "16:9", "9:16", "1:1", "4:3", "3:4"
    resolution: e.g. "1K", "2K", "4K"
    top_p: float 0.0-1.0, default 0.95
    reference_images: optional list of {"slot": str, "data": base64_str, "mime_type": str}
        When provided, images are sent alongside the prompt so Nano Banana Pro
        can use them as visual references for the corresponding facial features.

    Returns: {"success": bool, "image_base64": str, "mime_type": str, "logs": list}
    """
    selected_model = image_model or NANO_BANANA_PRO
    logs = []
    logs.append(f"[STEP 3] Generating image with model: {selected_model}")
    logs.append(f"[STEP 3] Final prompt sent to image model: {prompt}")
    logs.append(f"[STEP 3] Aspect ratio: {aspect_ratio}, Resolution: {resolution}, top_p: {top_p}")

    generation_config = {
        "temperature": 0.8,
        "topP": top_p,
        "maxOutputTokens": 8192,
        "responseModalities": ["IMAGE", "TEXT"],
        "imageConfig": {
            "aspectRatio": aspect_ratio,
            "imageSize": resolution,
        },
    }

    parts = []

    if reference_images:
        logs.append(f"[STEP 3] Including {len(reference_images)} reference image(s) for Nano Banana Pro")

        ref_slot_descs = {
            "eyes": "eyes",
            "eyebrows": "eyebrows",
            "nose": "nose",
            "lips": "lips and mouth",
            "forehead": "forehead",
            "cheekbones": "cheekbones",
            "jawline": "jawline",
            "ears": "ears",
            "hair": "hair",
            "skin": "skin texture and tone",
            "full_face": "complete face",
            "chin": "chin",
        }

        ref_intro = (
            "REFERENCE IMAGES:\n"
            "The following reference images show specific facial features. "
            "Use each image as the visual reference for the indicated feature, "
            "reproducing its appearance faithfully in the generated portrait.\n\n"
        )
        for i, img in enumerate(reference_images):
            slot = img.get("slot", f"image_{i}")
            desc = ref_slot_descs.get(slot, slot)
            ref_intro += f"Reference Image {i + 1}: {desc}\n"
            logs.append(f"[STEP 3]   - Reference {i + 1}: {desc} ({slot})")

        ref_intro += f"\nPORTRAIT GENERATION PROMPT:\n{prompt}"

        parts.append({"text": ref_intro})

        for img in reference_images:
            parts.append({
                "inline_data": {
                    "mime_type": img["mime_type"],
                    "data": img["data"]
                }
            })
    else:
        logs.append("[STEP 3] No reference images — text-only prompt")
        parts.append({"text": prompt})

    try:
        response = _gemini_generate_content(
            api_key, selected_model,
            [{"parts": parts}],
            generation_config
        )

        candidates = response.get("candidates", [])
        if not candidates:
            logs.append("[STEP 3] ERROR: No candidates in response")
            logs.append(f"[STEP 3] Full API response keys: {list(response.keys())}")
            return {"success": False, "image_base64": "", "mime_type": "", "logs": logs,
                    "error": "No candidates returned from Nano Banana Pro"}

        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason:
            logs.append(f"[STEP 3] Finish reason: {finish_reason}")

        content_parts = candidates[0].get("content", {}).get("parts", [])

        for part in content_parts:
            inline_data = part.get("inlineData", part.get("inline_data", None))
            if inline_data:
                image_b64 = inline_data.get("data", "")
                mime_type = inline_data.get("mimeType", inline_data.get("mime_type", "image/png"))
                logs.append(f"[STEP 3] Image generated successfully, mime: {mime_type}, size: {len(image_b64)} chars base64")
                return {"success": True, "image_base64": image_b64, "mime_type": mime_type, "logs": logs}

        text_parts = [p.get("text", "") for p in content_parts if "text" in p]
        if text_parts:
            logs.append(f"[STEP 3] Model returned text instead of image: {' '.join(text_parts)[:300]}")

        logs.append("[STEP 3] ERROR: No image data found in response")
        return {"success": False, "image_base64": "", "mime_type": "", "logs": logs,
                "error": "No image data in Nano Banana Pro response"}

    except Exception as e:
        logs.append(f"[STEP 3] ERROR: {str(e)}")
        return {"success": False, "image_base64": "", "mime_type": "", "logs": logs, "error": str(e)}


def load_flux_system_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "flux_system_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


_ATTR_LABEL_MAP = {
    "demographics.sex": "sex",
    "demographics.ethnicity": "ethnicity",
    "eye_system.eyes.shape_descriptor": "eye shape",
    "eye_system.eyes.size_descriptor": "eye size",
    "eye_system.eyes.tilt_descriptor": "eye tilt",
    "eye_system.eyes.color_descriptor": "eye color",
    "eye_system.eyebrows.thickness_descriptor": "eyebrow thickness",
    "eye_system.eyebrows.shape_descriptor": "eyebrow shape",
    "eye_system.eyebrows.color_descriptor": "eyebrow color",
    "central_system.nose.profile_descriptor": "nose profile",
    "central_system.nose.base_descriptor": "nose base",
    "central_system.nose.tip_descriptor": "nose tip",
    "central_system.lips.volume_descriptor": "lip volume",
    "central_system.lips.cupid_bow_descriptor": "cupid's bow",
    "central_system.lips.proportion_descriptor": "lip proportion",
    "central_system.lips.natural_color_descriptor": "lip color",
    "structure.forehead_descriptor": "forehead",
    "structure.cheekbones_descriptor": "cheekbones",
    "structure.jawline_descriptor": "jawline",
    "structure.chin_descriptor": "chin",
    "volumes.cheeks_descriptor": "cheeks",
    "volumes.submental_descriptor": "submental area",
    "volumes.face_neck_transition_descriptor": "face-neck transition",
    "hair.structure_descriptor": "hair structure",
    "hair.length_descriptor": "hair length",
    "hair.volume_descriptor": "hair volume",
    "hair.color_descriptor": "hair color",
    "skin.tone_descriptor": "skin tone",
    "skin.undertone_descriptor": "skin undertone",
    "skin.texture_descriptor": "skin texture",
    "skin.micro_texture_descriptor": "skin micro-texture",
    "skin.imperfections_descriptor": "skin imperfections",
    "skin.surface_reflection_descriptor": "skin reflection",
    "defects.wrinkles_descriptor": "wrinkles",
    "defects.scars_descriptor": "scars",
    "defects.deformations_descriptor": "deformations",
    "defects.tone_loss_descriptor": "tone and volume loss",
    "defects.skin_marks_descriptor": "skin marks",
    "defects.vitiligo_descriptor": "vitiligo",
    "defects.under_eye_descriptor": "under-eye condition",
    "expression.expression_descriptor": "expression",
    "expression.variant_descriptor": "expression variant",
}


def _build_attr_groups(attrs: dict) -> dict:
    groups = {
        "demographics": [],
        "eyes": [],
        "eyebrows": [],
        "nose": [],
        "lips": [],
        "structure": [],
        "volumes": [],
        "hair": [],
        "skin": [],
        "defects": [],
        "expression": [],
    }
    for key, val in attrs.items():
        if not val or val == "auto" or val == "none" or key == "_brief_text":
            continue
        label = _ATTR_LABEL_MAP.get(key, key)
        if key.startswith("demographics"):
            groups["demographics"].append(f"{val}")
        elif "eyes." in key:
            groups["eyes"].append(f"{val} {label.replace('eye ', '')}")
        elif "eyebrows" in key:
            groups["eyebrows"].append(f"{val} {label.replace('eyebrow ', '')}")
        elif "nose" in key:
            groups["nose"].append(f"{val} {label.replace('nose ', '')}")
        elif "lips" in key:
            groups["lips"].append(f"{val} {label.replace('lip ', '')}")
        elif "structure" in key:
            groups["structure"].append(f"{val} {label}")
        elif "volumes" in key:
            groups["volumes"].append(f"{val} {label}")
        elif "hair" in key:
            groups["hair"].append(f"{val} {label.replace('hair ', '')}")
        elif key.startswith("defects"):
            groups["defects"].append(f"{val}")
        elif key.startswith("expression"):
            groups["expression"].append(f"{val}")
        elif "skin" in key:
            groups["skin"].append(f"{val} {label.replace('skin ', '')}")
    return {k: v for k, v in groups.items() if v}


def load_flux_local_template() -> dict:
    template_path = os.path.join(os.path.dirname(__file__), "flux_local_template.txt")
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    result = {"flux": "", "zimage": ""}
    current_section = None
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped == "[FLUX_PROMPT]":
            current_section = "flux"
            continue
        elif stripped == "[ZIMAGE_PROMPT]":
            current_section = "zimage"
            continue
        if current_section and stripped:
            result[current_section] = stripped if not result[current_section] else result[current_section] + "\n" + stripped

    return result


def compose_flux_prompt_local(manual_attributes: dict = None,
                               photo_constraints: dict = None,
                               brief_text: str = "") -> dict:
    logs = []
    logs.append("[FLUX LOCAL] Composing prompts locally (no Gemini API)")

    templates = load_flux_local_template()
    flux_template = templates.get("flux", "")
    zimage_template = templates.get("zimage", "")
    logs.append(f"[FLUX LOCAL] Loaded templates - flux: {len(flux_template)} chars, zimage: {len(zimage_template)} chars")

    attrs = manual_attributes or {}
    non_auto = {k: v for k, v in attrs.items() if v and v != "auto" and k != "_brief_text"}
    logs.append(f"[FLUX LOCAL] Attributes provided: {len(non_auto)}")

    groups = _build_attr_groups(attrs)

    constraint_fragment = "pure white #FFFFFF background, high-key studio lighting, even diffused light"
    if photo_constraints:
        constraint_fragment = photo_constraints.get("fixed_constraints_fragment", constraint_fragment)
        logs.append(f"[FLUX LOCAL] Using photo type constraints: {constraint_fragment[:80]}...")

    subject_parts = []
    if "demographics" in groups:
        subject_parts.append(" ".join(groups["demographics"]))
    subject = " ".join(subject_parts) if subject_parts else "a person"

    feature_parts = []
    if "eyes" in groups:
        feature_parts.append(", ".join(groups["eyes"]) + " eyes")
    if "eyebrows" in groups:
        feature_parts.append(", ".join(groups["eyebrows"]) + " eyebrows")
    if "nose" in groups:
        feature_parts.append("a " + ", ".join(groups["nose"]) + " nose")
    if "lips" in groups:
        feature_parts.append(", ".join(groups["lips"]) + " lips")
    features = "with " + ", ".join(feature_parts) if feature_parts else ""

    structure_parts = []
    if "structure" in groups:
        structure_parts.extend(groups["structure"])
    if "volumes" in groups:
        structure_parts.extend(groups["volumes"])
    structure = "featuring " + ", ".join(structure_parts) if structure_parts else ""

    hair_parts = []
    if "hair" in groups:
        hair_parts.append(", ".join(groups["hair"]) + " hair")
    hair = "with " + ", ".join(hair_parts) if hair_parts else ""

    skin_parts = []
    if "skin" in groups:
        skin_parts.extend(groups["skin"])
    skin_parts.append("natural skin grain with visible fine pores")
    skin_parts.append("subtle tonal variation")
    skin = "showing " + ", ".join(skin_parts)

    defects_parts = []
    if "defects" in groups:
        defects_parts.extend(groups["defects"])
    defects = "with " + ", ".join(defects_parts) if defects_parts else ""

    expression_parts = []
    if "expression" in groups:
        expression_parts.extend(groups["expression"])
    expression_str = ", ".join(expression_parts) + " expression" if expression_parts else ""

    brief = ", " + brief_text if brief_text else ""

    replacements = {
        "{subject}": subject,
        "{features}": features,
        "{structure}": structure,
        "{hair}": hair,
        "{skin}": skin,
        "{defects}": defects,
        "{expression}": expression_str,
        "{clothing}": "a plain white tank top",
        "{constraints}": constraint_fragment,
        "{style}": "editorial portrait photography style with an intimate, authentic mood",
        "{camera_specs}": "85mm lens, f/2.8, shallow depth of field",
        "{quality}": "photorealistic, ultra-realistic, professional portrait photography, editorial style, 8K, RAW photo, sharp focus, high detail",
        "{brief}": brief,
    }

    flux_prompt = flux_template
    zimage_prompt = zimage_template
    for token, value in replacements.items():
        flux_prompt = flux_prompt.replace(token, value)
        zimage_prompt = zimage_prompt.replace(token, value)

    flux_prompt = ", ".join(seg.strip() for seg in flux_prompt.split(",") if seg.strip())
    zimage_prompt = ", ".join(seg.strip() for seg in zimage_prompt.split(",") if seg.strip())

    logs.append(f"[FLUX LOCAL] Flux prompt ({len(flux_prompt)} chars): {flux_prompt}")
    logs.append(f"[FLUX LOCAL] zimage prompt ({len(zimage_prompt)} chars): {zimage_prompt}")

    return {
        "success": True,
        "flux_prompt": flux_prompt,
        "zimage_prompt": zimage_prompt,
        "logs": logs,
    }


def compose_flux_prompt_gemini(api_key: str,
                                manual_attributes: dict = None,
                                photo_constraints: dict = None,
                                brief_text: str = "") -> dict:
    logs = []
    logs.append("[FLUX GEMINI] Composing prompts with Gemini 3.0 Flash")

    system_prompt = load_flux_system_prompt()

    if photo_constraints:
        override_constraints = photo_constraints.get("system_constraints", "")
        override_fragment = photo_constraints.get("fixed_constraints_fragment", "")
        logs.append(f"[FLUX GEMINI] Overriding constraints with photo type")

        old_constraints = (
            "FIXED STUDIO CONSTRAINTS (MUST ALWAYS APPLY)\n"
            "- Pure white background (#FFFFFF), high-key studio, clean and textureless background.\n"
            "- Front-facing full-face close-up portrait, centered and symmetrical.\n"
            "- Only shoulders visible, wearing a plain white tank top (white camisole).\n"
            "- Minimal natural editorial makeup, realistic anatomy, no stylization."
        )
        new_constraints = (
            f"FIXED STUDIO CONSTRAINTS (MUST ALWAYS APPLY)\n"
            f"{override_constraints}"
        )
        system_prompt = system_prompt.replace(old_constraints, new_constraints)

        logs.append("[FLUX GEMINI] System prompt patched with photo type constraints")
    else:
        logs.append("[FLUX GEMINI] Using default system prompt constraints")

    attrs = manual_attributes or {}
    non_auto = {k: v for k, v in attrs.items() if v and v != "auto" and k != "_brief_text"}

    user_message_parts = []
    if non_auto:
        readable = {}
        for k, v in non_auto.items():
            label = _ATTR_LABEL_MAP.get(k, k)
            readable[label] = v
        user_message_parts.append(
            f"The user has selected these facial attributes:\n"
            f"{json.dumps(readable, indent=2)}\n\n"
            f"Use only these provided attributes. Do not invent values for attributes not listed."
        )
    else:
        user_message_parts.append(
            "No specific facial attributes were selected. "
            "Generate prompts for a diverse, realistic portrait with coherent traits."
        )

    if brief_text:
        user_message_parts.append(f"\nAdditional user instructions: {brief_text}")

    user_message_parts.append(
        "\nGenerate the JSON with flux_prompt and zimage_prompt fields. "
        "Output only valid JSON, no markdown fences, no extra text."
    )

    user_message = "\n".join(user_message_parts)
    logs.append(f"[FLUX GEMINI] User message length: {len(user_message)} chars")
    logs.append(f"[FLUX GEMINI] Attributes sent: {len(non_auto)}")

    model = "gemini-3-flash-preview"

    try:
        response = _gemini_generate_content(
            api_key, model,
            [
                {"role": "user", "parts": [{"text": system_prompt}]},
                {"role": "model", "parts": [{"text": "Understood. I will output only the JSON with flux_prompt and zimage_prompt fields."}]},
                {"role": "user", "parts": [{"text": user_message}]}
            ],
            {"temperature": 0.7, "maxOutputTokens": 4096}
        )

        text_response = ""
        finish_reason = ""
        candidates = response.get("candidates", [])
        if candidates:
            finish_reason = candidates[0].get("finishReason", "")
            content_parts = candidates[0].get("content", {}).get("parts", [])
            for part in content_parts:
                if "text" in part:
                    text_response += part["text"]

        logs.append(f"[FLUX GEMINI] Raw response length: {len(text_response)} chars")
        if finish_reason:
            logs.append(f"[FLUX GEMINI] Finish reason: {finish_reason}")

        clean_text = text_response.strip()
        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_text = "\n".join(lines)

        try:
            result = json.loads(clean_text)
        except json.JSONDecodeError:
            result = _try_repair_json(clean_text)
            if result is not None:
                logs.append("[FLUX GEMINI] JSON was truncated, repaired successfully")
            else:
                raise

        flux_prompt = result.get("flux_prompt", "")
        zimage_prompt = result.get("zimage_prompt", "")

        if flux_prompt:
            logs.append(f"[FLUX GEMINI] Flux prompt ({len(flux_prompt)} chars): {flux_prompt}")
        else:
            logs.append("[FLUX GEMINI] WARNING: No flux_prompt in response")

        if zimage_prompt:
            logs.append(f"[FLUX GEMINI] zimage prompt ({len(zimage_prompt)} chars): {zimage_prompt}")
        else:
            logs.append("[FLUX GEMINI] WARNING: No zimage_prompt in response")

        return {
            "success": True,
            "flux_prompt": flux_prompt,
            "zimage_prompt": zimage_prompt,
            "logs": logs,
        }

    except json.JSONDecodeError as e:
        logs.append(f"[FLUX GEMINI] ERROR: Failed to parse JSON: {e}")
        logs.append(f"[FLUX GEMINI] Raw text: {text_response[:500]}")
        return {"success": False, "flux_prompt": "", "zimage_prompt": "", "logs": logs, "error": f"JSON parse error: {e}"}
    except Exception as e:
        logs.append(f"[FLUX GEMINI] ERROR: {str(e)}")
        return {"success": False, "flux_prompt": "", "zimage_prompt": "", "logs": logs, "error": str(e)}


def load_fusion_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "fusion_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


_SLOT_DESCRIPTIONS = {
    "eyes": "eyes reference",
    "eyebrows": "eyebrows reference",
    "nose": "nose reference",
    "lips": "lips and mouth reference",
    "forehead": "forehead reference",
    "cheekbones": "cheekbones reference",
    "jawline": "jawline reference",
    "ears": "ears reference",
    "hair": "hair reference",
    "skin": "skin texture and tone reference",
    "full_face": "complete face reference",
    "chin": "chin reference",
}

DEFAULT_STUDIO_CONSTRAINTS = (
    "- Subject centered on a clean seamless pure white background.\n"
    "- Wearing a white tank top, minimal styling, no heavy makeup.\n"
    "- Front-facing head-and-shoulders framing, centered and symmetrical."
)


def generate_fusion_portrait(api_key: str, images: list,
                              manual_attributes: dict = None,
                              photo_constraints: dict = None,
                              brief_text: str = "",
                              aspect_ratio: str = "1:1",
                              resolution: str = "1K",
                              top_p: float = 0.95,
                              image_model: str = None) -> dict:
    selected_model = image_model or NANO_BANANA_PRO
    logs = []
    logs.append(f"[FUSION] Generating fusion portrait with model: {selected_model}")
    logs.append(f"[FUSION] Reference images: {len(images)}")

    fusion_template = load_fusion_prompt()

    image_labels_lines = []
    for i, img in enumerate(images):
        slot = img.get("slot", f"image_{i}")
        desc = _SLOT_DESCRIPTIONS.get(slot, slot)
        label = f"Image {i + 1}: {desc}"
        image_labels_lines.append(label)
        logs.append(f"[FUSION]   - {label}")

    image_labels_str = "\n".join(image_labels_lines)

    studio_constraints = DEFAULT_STUDIO_CONSTRAINTS
    if photo_constraints:
        override = photo_constraints.get("system_constraints", "")
        if override:
            studio_constraints = override
            logs.append(f"[FUSION] Using photo type constraints override")
    logs.append(f"[FUSION] Studio constraints: {studio_constraints[:80]}...")

    attr_modifier_text = ""
    if manual_attributes:
        non_auto = {k: v for k, v in manual_attributes.items()
                    if v and v != "auto" and k != "_brief_text"}
        if non_auto:
            modifier_lines = ["ADDITIONAL ATTRIBUTE GUIDANCE (apply subtly to the merged result):"]
            for k, v in non_auto.items():
                label = _ATTR_LABEL_MAP.get(k, k)
                modifier_lines.append(f"- {label}: {v}")
            attr_modifier_text = "\n".join(modifier_lines)
            logs.append(f"[FUSION] Attribute modifiers: {len(non_auto)} applied")

    brief_section = ""
    if brief_text:
        brief_section = f"ADDITIONAL USER INSTRUCTIONS:\n{brief_text}"
        logs.append(f"[FUSION] Brief text: {brief_text[:80]}...")

    prompt_text = fusion_template.format(
        image_labels=image_labels_str,
        studio_constraints=studio_constraints,
        attribute_modifiers=attr_modifier_text,
        brief_section=brief_section,
    )

    prompt_text = "\n".join(line for line in prompt_text.split("\n") if line.strip())

    logs.append(f"[FUSION] Final prompt length: {len(prompt_text)} chars")
    logs.append(f"[FUSION] Final prompt:\n{prompt_text}")
    logs.append(f"[FUSION] Aspect ratio: {aspect_ratio}, Resolution: {resolution}, top_p: {top_p}")

    parts = [{"text": prompt_text}]
    for img in images:
        parts.append({
            "inline_data": {
                "mime_type": img["mime_type"],
                "data": img["data"]
            }
        })

    generation_config = {
        "temperature": 0.8,
        "topP": top_p,
        "maxOutputTokens": 8192,
        "responseModalities": ["IMAGE", "TEXT"],
        "imageConfig": {
            "aspectRatio": aspect_ratio,
            "imageSize": resolution,
        },
    }

    try:
        response = _gemini_generate_content(
            api_key, selected_model,
            [{"parts": parts}],
            generation_config
        )

        candidates = response.get("candidates", [])
        if not candidates:
            logs.append("[FUSION] ERROR: No candidates in response")
            logs.append(f"[FUSION] Full API response keys: {list(response.keys())}")
            return {"success": False, "image_base64": "", "mime_type": "", "logs": logs,
                    "error": "No candidates returned from Nano Banana Pro"}

        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason:
            logs.append(f"[FUSION] Finish reason: {finish_reason}")

        content_parts = candidates[0].get("content", {}).get("parts", [])

        for part in content_parts:
            inline_data = part.get("inlineData", part.get("inline_data", None))
            if inline_data:
                image_b64 = inline_data.get("data", "")
                mime_type = inline_data.get("mimeType", inline_data.get("mime_type", "image/png"))
                logs.append(f"[FUSION] Image generated successfully, mime: {mime_type}, size: {len(image_b64)} chars base64")
                return {"success": True, "image_base64": image_b64, "mime_type": mime_type, "logs": logs}

        text_parts = [p.get("text", "") for p in content_parts if "text" in p]
        if text_parts:
            logs.append(f"[FUSION] Model returned text instead of image: {' '.join(text_parts)[:300]}")

        logs.append("[FUSION] ERROR: No image data found in response")
        return {"success": False, "image_base64": "", "mime_type": "", "logs": logs,
                "error": "No image data in Nano Banana Pro response"}

    except Exception as e:
        logs.append(f"[FUSION] ERROR: {str(e)}")
        return {"success": False, "image_base64": "", "mime_type": "", "logs": logs, "error": str(e)}
