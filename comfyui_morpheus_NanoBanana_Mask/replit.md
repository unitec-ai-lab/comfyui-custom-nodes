# ComfyUI Morpheus NanoBanana Mask

## Overview

This project is a ComfyUI custom node package designed for advanced multi-image composition using the Google Gemini 2.5 Flash Image API. It functions as a plugin for existing ComfyUI installations, enabling sophisticated image generation workflows. The core functionality revolves around a three-node system:

1. **Batch Analyzer** - Processes input images using Gemini for detailed analysis
2. **Image Editing Prompt** - Generates editing-style prompts with customizable templates following Google's recommended patterns
3. **Composer** - Uses analysis or editing prompts to construct optimized API calls for image generation with seed-based variation control

The package leverages Google's `google-genai` SDK for robust API integration and advanced compositional techniques. The business vision is to empower ComfyUI users with powerful, Gemini-driven multi-image compositional capabilities, expanding creative possibilities in image generation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Design Principles

The system operates on a **Three-Node Compositional Pattern**, offering flexible workflows for image generation:

1. **Batch Analyzer Path**: Analyzer node processes images and provides detailed Gemini analysis → feeds into Composer
2. **Image Editing Path**: Image Editing Prompt node generates customizable editing-style prompts with brief analysis → feeds into Composer
3. **Composer**: Receives images and prompt (from either path), adds seed-based variation, and generates final image

Key principles include: **No Image Manipulation Philosophy** (images passed "as-is" without internal resizing), **Subject-First Ordering** (primary subject image always first, followed by reference images), **Seed-Based Variation** (deterministic prompt diversity using seed control), and **ComfyUI Custom Node Integration** (standard node interface patterns throughout).

### Node Components

1.  **MorpheusBatchImagesCropV25Fix (Analyzer Node)**:
    *   **Purpose**: Collects 1-6 images and performs simple Gemini analysis, returning original images and text analysis.
    *   **Inputs**: `image_0_subject` (required), `image_1` through `image_5` (optional reference images), `context`, `api_key`.
    *   **Outputs**: `crop_image` (passthrough), `images_list` (passthrough), `analysis_text` (from Gemini).
    *   **Logic**: Converts ComfyUI tensors to PIL Images, constructs a prompt for `gemini-2.5-flash` describing images for composition, and returns original tensors plus Gemini analysis.
    *   **Error Handling**: Returns fallback text on analysis failure, allowing workflow continuity.

2.  **MorpheusImageEditingPrompt (Image Editing Prompt Node)**:
    *   **Purpose**: Generates an editing-style prompt using Google's recommended pattern, incorporating a brief Gemini analysis.
    *   **Inputs**: `image_0_subject` (required), `image_1` through `image_5` (optional), `prompt_template` (editable with `{action}` placeholder), `action` text, `context`, `api_key`.
    *   **Outputs**: `crop_image` (passthrough), `images_list` (passthrough), `analysis_text` (brief Gemini analysis), `editing_prompt` (formatted template with {action} replaced).
    *   **Logic**: Performs a brief Gemini 2.5 Flash analysis (2-3 sentences), replaces the `{action}` placeholder in the template, adds context, and returns both analysis_text and editing_prompt separately.

3.  **MorpheusNanoBananaMaskGeminiV25Fix (Composer Node)**:
    *   **Purpose**: Generates images using `gemini-2.5-flash-image` by combining analysis results, user prompts, and seed control.
    *   **Inputs**: `crop_image`, `images_list`, `analysis_text`, `user_prompt`, `api_key`, `seed` (integer with control modes), `control_after_generate` (fixed/randomize/increment/decrement), `aspect_ratio` (auto, original, 1:1, etc.).
    *   **Outputs**: `image` (generated tensor or error image), `info` (detailed generation metadata).
    *   **Logic**: Converts tensors to PIL Images, generates a seed-based variation suffix for prompt diversity, combines analysis, user prompt, and suffix, adds an aspect ratio hint, calls `gemini-2.5-flash-image`, extracts the generated image, and returns it with detailed info.
    *   **Seed Variation**: Uses the seed to select a variation phrase from a predefined list, ensuring diverse outputs.
    *   **Error Handling**: Creates a red error image with text overlay, saves to an `errors/` folder, and includes error details in the info output.

### Data Flow Pattern

The system supports two workflow paths:

**Path 1 - Analyzer Workflow:**
1. Images flow from user → Batch Analyzer node
2. Analyzer sends images to Gemini 2.5 Flash for detailed analysis
3. Analyzer outputs: processed images + reference images list + analysis text
4. These outputs + user-defined task → Composer node
5. Composer adds seed variation, constructs request to `gemini-2.5-flash-image`
6. Generated image and detailed info are returned

**Path 2 - Image Editing Workflow:**
1. Images flow from user → Image Editing Prompt node
2. Node sends images to Gemini 2.5 Flash for brief analysis
3. Node formats editing-style prompt using customizable template + analysis
4. Node outputs: processed images + reference images list + editing prompt
5. These outputs → Composer node (uses editing_prompt as analysis_text input)
6. Composer adds seed variation, constructs request to `gemini-2.5-flash-image`
7. Generated image and detailed info are returned

Both paths converge at the Composer, which handles aspect ratio control, seed-based variation, and final image generation.

### Tensor Handling

ComfyUI images (PyTorch tensors `[batch, height, width, channels]`, float32 `[0,1]`) are converted to PIL Images (`tensor_to_pil`) for Gemini API calls and back to tensors (`pil_to_tensor`) for ComfyUI.

### Error Resilience

Both Analyzer and Composer nodes are designed for error resilience. Analyzer returns safe analysis on failure, allowing workflows to continue. Composer generates informative error images and logs detailed diagnostics if generation fails.

## External Dependencies

### Required Python Packages
-   **pillow**: For image processing and PIL Image objects.
-   **numpy**: For numerical operations, especially during tensor conversions.
-   **google-genai**: The official Google Generative AI SDK, crucial for interacting with Gemini APIs and supporting Structured Output.
-   **torch**: The PyTorch framework, provided by the ComfyUI environment itself, used for tensor manipulation.

### Third-Party APIs
-   **Google Gemini 2.5 Flash Image**:
    *   **Models Used**: `gemini-2.5-flash-image` (for the Composer node's image generation) and `gemini-2.5-flash` (for the Analyzer node's image analysis).
    *   **Endpoint**: Access is via the official Google GenAI REST API, facilitated by the `google-genai` SDK.
    *   **Authentication**: Requires an API key provided as a parameter.
    *   **Key Features**: Supports multi-image composition and aspect ratio control (either via the dimensions of the last image in the input list or explicit prompt hints). PIL Images are passed directly in the API call's contents list.
    *   **Documentation**: Detailed usage can be found at https://ai.google.dev/gemini-api/docs/image-generation.

### ComfyUI Framework
-   **Host Application**: This custom node package operates within the ComfyUI environment, leveraging its node execution system, `IMAGE` tensor type, and workflow graph capabilities.
-   **Integration**: Nodes are registered for discovery via `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` within the `__init__.py` file.