import os
import json
import base64
import urllib.request
import urllib.error
from server import PromptServer
from aiohttp import web

# ── Configuración desde variables de entorno ──────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "")   # formato: usuario/repo
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

API_BASE = "https://api.github.com"

def gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "ComfyUI-GitHub-Workflows"
    }

def gh_request(method, path, data=None):
    """Hace una petición a la API de GitHub."""
    url = f"{API_BASE}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=gh_headers(), method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise Exception(f"GitHub API error {e.code}: {error_body}")

# ── Rutas del servidor ────────────────────────────────────────────────────────

routes = PromptServer.instance.routes

@routes.get("/github_workflows/list")
async def list_workflows(request):
    """Lista todos los .json del repo."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return web.json_response({"error": "GITHUB_TOKEN o GITHUB_REPO no configurados"}, status=400)
    try:
        contents = gh_request("GET", f"/repos/{GITHUB_REPO}/contents/?ref={GITHUB_BRANCH}")
        workflows = [
            {"name": f["name"], "path": f["path"], "sha": f["sha"]}
            for f in contents
            if f["name"].endswith(".json")
        ]
        return web.json_response({"workflows": workflows})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/github_workflows/load")
async def load_workflow(request):
    """Carga el contenido de un workflow por su path."""
    path = request.query.get("path")
    if not path:
        return web.json_response({"error": "Falta el parámetro path"}, status=400)
    try:
        file_info = gh_request("GET", f"/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}")
        content = base64.b64decode(file_info["content"]).decode("utf-8")
        workflow = json.loads(content)
        return web.json_response({"workflow": workflow, "sha": file_info["sha"]})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@routes.post("/github_workflows/save")
async def save_workflow(request):
    """Guarda o sobreescribe un workflow en GitHub."""
    try:
        body = await request.json()
        filename = body.get("filename")
        workflow  = body.get("workflow")
        sha       = body.get("sha")  # None si es archivo nuevo

        if not filename or not workflow:
            return web.json_response({"error": "Faltan datos"}, status=400)

        content_b64 = base64.b64encode(
            json.dumps(workflow, indent=2).encode("utf-8")
        ).decode("utf-8")

        payload = {
            "message": f"autoguardado: {filename}",
            "content": content_b64,
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha  # necesario para sobreescribir

        result = gh_request("PUT", f"/repos/{GITHUB_REPO}/contents/{filename}", payload)
        new_sha = result["content"]["sha"]
        return web.json_response({"ok": True, "sha": new_sha})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


@routes.get("/github_workflows/status")
async def status(request):
    """Verifica si el token y repo están configurados."""
    configured = bool(GITHUB_TOKEN and GITHUB_REPO)
    return web.json_response({
        "configured": configured,
        "repo": GITHUB_REPO if configured else None
    })


# ComfyUI requiere esto aunque no usemos nodos visuales
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

print("✅ ComfyUI GitHub Workflows cargado")
