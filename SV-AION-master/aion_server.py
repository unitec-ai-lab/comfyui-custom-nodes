"""
Server-side routes for AION Theta.
Registers the /aion/verify_key endpoint in ComfyUI's server
so API key verification happens on the backend, never client-side.
"""

from aiohttp import web
from .aion_api import verify_api_key


def setup_routes(server_instance):
    """Register AION routes with the ComfyUI PromptServer instance."""

    @server_instance.routes.post("/aion/verify_key")
    async def verify_key_handler(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"status": "Not Verified - Invalid request"},
                status=400
            )

        api_key = data.get("api_key", "")
        if not api_key:
            return web.json_response(
                {"status": "Not Verified - No key provided"}
            )

        result = verify_api_key(api_key)

        if result["valid"]:
            model_count = len(result.get("models", []))
            status = f"Verified ({model_count} models available)"
        else:
            error = result.get("error", "Unknown error")
            if len(error) > 100:
                error = error[:100] + "..."
            status = f"Not Verified - {error}"

        return web.json_response({"status": status})
