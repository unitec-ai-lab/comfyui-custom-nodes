import { app } from "/scripts/app.js";

// ── Estado global ─────────────────────────────────────────────────────────────
let currentFile = null;   // { name, path, sha }
let autoSaveInterval = null;
let lastSavedWorkflow = null;
let statusEl = null;

const AUTOSAVE_MS = 5 * 60 * 1000; // 5 minutos

// ── Utilidades ────────────────────────────────────────────────────────────────
async function apiFetch(url, options = {}) {
    const resp = await fetch(url, options);
    return resp.json();
}

function getCurrentWorkflow() {
    return JSON.stringify(app.graph.serialize());
}

function showStatus(msg, color = "#aaa") {
    if (statusEl) {
        statusEl.textContent = msg;
        statusEl.style.color = color;
    }
}

// ── Guardar en GitHub ─────────────────────────────────────────────────────────
async function saveToGitHub(silent = false) {
    if (!currentFile) {
        if (!silent) showStatus("⚠️ Selecciona o crea un workflow primero", "#f90");
        return;
    }
    try {
        const workflow = JSON.parse(getCurrentWorkflow());
        showStatus("💾 Guardando...", "#4af");
        const result = await apiFetch("/github_workflows/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: currentFile.path,
                workflow,
                sha: currentFile.sha || null
            })
        });
        if (result.ok) {
            currentFile.sha = result.sha;
            const now = new Date().toLocaleTimeString();
            showStatus(`✅ Guardado a las ${now}`, "#4f4");
            lastSavedWorkflow = getCurrentWorkflow();
        } else {
            showStatus(`❌ Error: ${result.error}`, "#f44");
        }
    } catch (e) {
        showStatus(`❌ Error: ${e.message}`, "#f44");
    }
}

// ── Autoguardado ──────────────────────────────────────────────────────────────
function startAutoSave() {
    if (autoSaveInterval) clearInterval(autoSaveInterval);
    autoSaveInterval = setInterval(async () => {
        const current = getCurrentWorkflow();
        if (current !== lastSavedWorkflow) {
            await saveToGitHub(true);
        }
    }, AUTOSAVE_MS);
}

// ── Cargar workflow desde GitHub ──────────────────────────────────────────────
async function loadWorkflow(file, listEl) {
    showStatus("⏳ Cargando...", "#4af");
    try {
        const result = await apiFetch(`/github_workflows/load?path=${encodeURIComponent(file.path)}`);
        if (result.workflow) {
            app.loadGraphData(result.workflow);
            currentFile = { ...file, sha: result.sha };
            lastSavedWorkflow = getCurrentWorkflow();
            showStatus(`📂 ${file.name}`, "#4f4");
            // Resaltar el seleccionado
            listEl.querySelectorAll(".wf-item").forEach(el => el.classList.remove("wf-active"));
            const active = listEl.querySelector(`[data-path="${file.path}"]`);
            if (active) active.classList.add("wf-active");
        } else {
            showStatus(`❌ ${result.error}`, "#f44");
        }
    } catch (e) {
        showStatus(`❌ ${e.message}`, "#f44");
    }
}

// ── Crear nuevo workflow ──────────────────────────────────────────────────────
async function createNewWorkflow(listEl) {
    const name = prompt("Nombre del nuevo workflow (sin extensión):");
    if (!name || !name.trim()) return;
    const filename = name.trim().replace(/\s+/g, "_") + ".json";
    currentFile = { name: filename, path: filename, sha: null };
    lastSavedWorkflow = null;
    app.graph.clear();
    showStatus(`🆕 ${filename} — guarda cuando quieras`, "#4af");
    await saveToGitHub();
    await refreshList(listEl);
}

// ── Refrescar lista de workflows ──────────────────────────────────────────────
async function refreshList(listEl) {
    listEl.innerHTML = `<div style="color:#aaa;padding:4px">Cargando...</div>`;
    try {
        const result = await apiFetch("/github_workflows/list");
        listEl.innerHTML = "";
        if (result.error) {
            listEl.innerHTML = `<div style="color:#f44;font-size:11px">${result.error}</div>`;
            return;
        }
        if (!result.workflows.length) {
            listEl.innerHTML = `<div style="color:#aaa;font-size:11px">No hay workflows aún</div>`;
            return;
        }
        result.workflows.forEach(file => {
            const item = document.createElement("div");
            item.className = "wf-item";
            item.dataset.path = file.path;
            item.textContent = file.name.replace(".json", "");
            if (currentFile && currentFile.path === file.path) item.classList.add("wf-active");
            item.onclick = () => loadWorkflow(file, listEl);
            listEl.appendChild(item);
        });
    } catch (e) {
        listEl.innerHTML = `<div style="color:#f44;font-size:11px">Error: ${e.message}</div>`;
    }
}

// ── Crear el panel flotante ───────────────────────────────────────────────────
function createPanel() {
    const panel = document.createElement("div");
    panel.id = "gh-workflows-panel";
    panel.innerHTML = `
        <div id="gh-panel-header">
            <span>📂 Mis Workflows</span>
            <button id="gh-toggle-btn" title="Minimizar">−</button>
        </div>
        <div id="gh-panel-body">
            <div id="gh-workflow-list"></div>
            <div id="gh-panel-actions">
                <button id="gh-new-btn">＋ Nuevo</button>
                <button id="gh-refresh-btn">🔄</button>
                <button id="gh-save-btn">💾 Guardar</button>
            </div>
            <div id="gh-status">Iniciando...</div>
        </div>
    `;

    // Estilos
    const style = document.createElement("style");
    style.textContent = `
        #gh-workflows-panel {
            position: fixed;
            top: 60px;
            right: 16px;
            width: 220px;
            background: #1a1a2e;
            border: 1px solid #444;
            border-radius: 8px;
            z-index: 9999;
            font-family: sans-serif;
            font-size: 13px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        #gh-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 10px;
            background: #16213e;
            border-radius: 8px 8px 0 0;
            cursor: move;
            color: #eee;
            font-weight: bold;
        }
        #gh-toggle-btn {
            background: none;
            border: none;
            color: #aaa;
            cursor: pointer;
            font-size: 16px;
            padding: 0 4px;
        }
        #gh-panel-body { padding: 8px; }
        #gh-workflow-list {
            max-height: 180px;
            overflow-y: auto;
            margin-bottom: 8px;
        }
        .wf-item {
            padding: 5px 8px;
            border-radius: 4px;
            cursor: pointer;
            color: #ccc;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .wf-item:hover { background: #0f3460; color: #fff; }
        .wf-active { background: #0f3460; color: #4af !important; font-weight: bold; }
        #gh-panel-actions {
            display: flex;
            gap: 4px;
            margin-bottom: 6px;
        }
        #gh-panel-actions button {
            flex: 1;
            padding: 4px;
            border: none;
            border-radius: 4px;
            background: #0f3460;
            color: #eee;
            cursor: pointer;
            font-size: 11px;
        }
        #gh-panel-actions button:hover { background: #1a5276; }
        #gh-save-btn { background: #1a472a !important; }
        #gh-save-btn:hover { background: #27ae60 !important; }
        #gh-status {
            font-size: 11px;
            color: #aaa;
            text-align: center;
            min-height: 16px;
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(panel);

    statusEl = panel.querySelector("#gh-status");
    const listEl = panel.querySelector("#gh-workflow-list");
    const body = panel.querySelector("#gh-panel-body");

    // Botones
    panel.querySelector("#gh-save-btn").onclick = () => saveToGitHub(false);
    panel.querySelector("#gh-refresh-btn").onclick = () => refreshList(listEl);
    panel.querySelector("#gh-new-btn").onclick = () => createNewWorkflow(listEl);

    // Minimizar
    let minimized = false;
    panel.querySelector("#gh-toggle-btn").onclick = () => {
        minimized = !minimized;
        body.style.display = minimized ? "none" : "block";
        panel.querySelector("#gh-toggle-btn").textContent = minimized ? "+" : "−";
    };

    // Arrastrar panel
    let dragging = false, ox = 0, oy = 0;
    panel.querySelector("#gh-panel-header").addEventListener("mousedown", e => {
        dragging = true;
        ox = e.clientX - panel.offsetLeft;
        oy = e.clientY - panel.offsetTop;
    });
    document.addEventListener("mousemove", e => {
        if (!dragging) return;
        panel.style.left = (e.clientX - ox) + "px";
        panel.style.top  = (e.clientY - oy) + "px";
        panel.style.right = "auto";
    });
    document.addEventListener("mouseup", () => dragging = false);

    return listEl;
}

// ── Init ──────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "ComfyUI.GitHubWorkflows",
    async setup() {
        console.log("🟢 GitHub Workflows: extensión iniciando...");

        // Crear el panel SIEMPRE (aunque falle la config, para diagnóstico)
        const listEl = createPanel();
        showStatus("Verificando conexión...", "#4af");

        try {
            const status = await apiFetch("/github_workflows/status");
            if (!status.configured) {
                showStatus("⚠️ Falta GITHUB_TOKEN o GITHUB_REPO", "#f90");
                console.warn("⚠️ GitHub Workflows: configura los secrets en tu Space");
                return;
            }
            showStatus(`Conectado a ${status.repo}`, "#4f4");
            await refreshList(listEl);
            startAutoSave();
        } catch (e) {
            showStatus(`❌ ${e.message}`, "#f44");
            console.error("GitHub Workflows error:", e);
        }
    }
});
