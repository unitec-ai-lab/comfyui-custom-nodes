"""
Wrapper around the notebooklm CLI for subprocess-based integration.
All functions call the CLI with --json and return parsed dicts.
"""

import json
import logging
import subprocess
import shutil
from typing import Any, Optional

logger = logging.getLogger("comfyui-notebooklm")

import os

# Search common locations since ComfyUI's venv may not have notebooklm in PATH
_SEARCH_PATHS = [
    shutil.which("notebooklm"),
    os.path.expanduser("~/anaconda3/bin/notebooklm"),
    os.path.expanduser("~/miniconda3/bin/notebooklm"),
    "/usr/local/bin/notebooklm",
    os.path.expanduser("~/.local/bin/notebooklm"),
]

NOTEBOOKLM_BIN = next((p for p in _SEARCH_PATHS if p and os.path.isfile(p)), None)

if NOTEBOOKLM_BIN is None:
    logger.warning(
        "notebooklm CLI not found. "
        "Install with: pip install 'notebooklm-py[browser]'"
    )
else:
    logger.info(f"notebooklm CLI found at: {NOTEBOOKLM_BIN}")


class NotebookLMCLIError(Exception):
    """Raised when a notebooklm CLI command fails."""

    def __init__(self, command: str, stderr: str, returncode: int):
        self.command = command
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(f"notebooklm {command} failed (exit {returncode}): {stderr}")


def _run_cmd(
    args: list[str],
    timeout: int = 120,
    parse_json: bool = True,
    input_text: Optional[str] = None,
) -> Any:
    """Run a notebooklm CLI command and return parsed output."""
    if NOTEBOOKLM_BIN is None:
        raise NotebookLMCLIError(
            " ".join(args),
            "notebooklm CLI not found. Install with: pip install 'notebooklm-py[browser]'",
            127,
        )

    cmd = [NOTEBOOKLM_BIN] + args
    cmd_str = " ".join(args)

    logger.info(f"Running: notebooklm {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
        )
    except subprocess.TimeoutExpired:
        raise NotebookLMCLIError(cmd_str, f"Command timed out after {timeout}s", -1)

    if result.returncode != 0:
        raise NotebookLMCLIError(cmd_str, result.stderr.strip(), result.returncode)

    if not parse_json:
        return result.stdout.strip()

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        # Some commands return plain text even with --json
        return {"raw_output": result.stdout.strip()}


# ── Notebook Operations ──────────────────────────────────────────────


def create_notebook(title: str) -> dict:
    """Create a new notebook. Returns {notebook: {id, title}}."""
    return _run_cmd(["create", title, "--json"])


def list_notebooks() -> dict:
    """List all notebooks. Returns {notebooks: [...]}."""
    return _run_cmd(["list", "--json"])


def set_context(notebook_id: str) -> None:
    """Set the active notebook context."""
    _run_cmd(["use", notebook_id], parse_json=False)


def get_summary(notebook_id: str) -> str:
    """Get AI-generated summary for a notebook."""
    set_context(notebook_id)
    result = _run_cmd(["summary", "--json"], timeout=60)
    if isinstance(result, dict):
        return result.get("summary", result.get("raw_output", ""))
    return str(result)


# ── Source Operations ────────────────────────────────────────────────


def add_source(notebook_id: str, url: str) -> dict:
    """Add a URL source to a notebook. Returns {source: {id, title, type}}."""
    set_context(notebook_id)
    return _run_cmd(["source", "add", url, "--json"], timeout=60)


def add_research(
    notebook_id: str,
    query: str,
    mode: str = "fast",
    timeout: int = 600,
) -> dict:
    """Run web research and import sources. Blocks until complete."""
    set_context(notebook_id)
    return _run_cmd(
        ["source", "add-research", query, "--mode", mode, "--import-all"],
        timeout=timeout,
        parse_json=False,
    )


def list_sources(notebook_id: str) -> dict:
    """List all sources in a notebook. Returns {sources: [...]}."""
    set_context(notebook_id)
    return _run_cmd(["source", "list", "--json"])


# ── Chat Operations ─────────────────────────────────────────────────


def ask_question(
    notebook_id: str,
    question: str,
    conversation_id: Optional[str] = None,
) -> dict:
    """Ask a question to the notebook.

    Returns {answer, conversation_id, turn_number, references}.
    """
    set_context(notebook_id)
    args = ["ask", question, "--json"]
    if conversation_id:
        args.extend(["-c", conversation_id])
    return _run_cmd(args, timeout=120)


# ── High-Level Workflows ────────────────────────────────────────────


def research_product(
    product_name: str,
    brand_url: str,
    extra_urls: Optional[list[str]] = None,
    search_queries: Optional[list[str]] = None,
    research_mode: str = "fast",
) -> dict:
    """Full product research workflow.

    1. Creates notebook
    2. Adds brand URL
    3. Adds extra URLs
    4. Runs web research
    5. Returns notebook data with summary

    Returns:
        {
            notebook_id: str,
            title: str,
            sources: list,
            source_count: int,
            summary: str,
            errors: list[str],
        }
    """
    errors = []

    # 1. Create notebook
    logger.info(f"Creating notebook: {product_name}")
    create_result = create_notebook(product_name)
    notebook_id = create_result["notebook"]["id"]
    title = create_result["notebook"]["title"]

    logger.info(f"Notebook created: {notebook_id}")

    # 2. Add brand URL
    try:
        logger.info(f"Adding brand URL: {brand_url}")
        add_source(notebook_id, brand_url)
    except NotebookLMCLIError as e:
        msg = f"Failed to add brand URL {brand_url}: {e.stderr}"
        logger.warning(msg)
        errors.append(msg)

    # 3. Add extra URLs
    if extra_urls:
        for url in extra_urls:
            url = url.strip()
            if not url:
                continue
            try:
                logger.info(f"Adding extra URL: {url}")
                add_source(notebook_id, url)
            except NotebookLMCLIError as e:
                msg = f"Failed to add URL {url}: {e.stderr}"
                logger.warning(msg)
                errors.append(msg)

    # 4. Run web research
    search_terms = product_name
    if search_queries:
        extra = " ".join(q.strip() for q in search_queries if q.strip())
        if extra:
            search_terms = f"{product_name} {extra}"

    try:
        logger.info(f"Running web research: '{search_terms}' (mode={research_mode})")
        add_research(
            notebook_id,
            search_terms,
            mode=research_mode,
            timeout=600 if research_mode == "deep" else 180,
        )
    except NotebookLMCLIError as e:
        msg = f"Web research failed or timed out: {e.stderr}"
        logger.warning(msg)
        errors.append(msg)

    # 5. List final sources
    try:
        sources_result = list_sources(notebook_id)
        sources = sources_result.get("sources", [])
    except NotebookLMCLIError:
        sources = []

    # 6. Get summary
    try:
        summary = get_summary(notebook_id)
    except NotebookLMCLIError:
        summary = ""

    return {
        "notebook_id": notebook_id,
        "title": title,
        "sources": sources,
        "source_count": len(sources),
        "summary": summary,
        "errors": errors,
    }
