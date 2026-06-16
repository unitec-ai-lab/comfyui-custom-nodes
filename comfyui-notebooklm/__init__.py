"""
ComfyUI NotebookLM Integration
───────────────────────────────
Send messages to NotebookLM notebooks and get responses.

Requirements:
  - notebooklm-py CLI installed and authenticated (notebooklm login)
"""

from .nodes.script_generator import NotebookLM_ScriptGenerator

NODE_CLASS_MAPPINGS = {
    "NotebookLM_ScriptGenerator": NotebookLM_ScriptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NotebookLM_ScriptGenerator": "NotebookLM Chat",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
