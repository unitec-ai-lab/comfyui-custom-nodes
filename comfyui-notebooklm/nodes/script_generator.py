"""
NotebookLM Chat Node
────────────────────
Sends a message to an existing NotebookLM notebook and returns the response.
Always starts a new conversation.
"""

import logging

logger = logging.getLogger("comfyui-notebooklm")

DEFAULT_MESSAGE = (
    "Necesito que crees una secuencia de prompts para Nano Banana para una "
    "publicidad en estilo [Claymotion] que represente todo lo que [PRODUCTO] "
    "mostraria en un anuncio de TV de 60 segundos. Genera los prompts de "
    "imagen para Nano Banana en orden para contar la mejor historia posible. "
    "Debe tener un inicio, desarrollo y cierre claros, y estar disenado para "
    "un anuncio de 60 segundos. Optimizalo para lograr un alto CTR en Facebook. "
    "Sos un marketer de elite con una creatividad sobresaliente."
)


class NotebookLM_ScriptGenerator:
    """Send a message to a NotebookLM notebook and get a response."""

    CATEGORY = "NotebookLM"
    FUNCTION = "send_message"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "notebook_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Notebook ID (from URL or notebooklm list)",
                    },
                ),
                "message": (
                    "STRING",
                    {
                        "default": DEFAULT_MESSAGE,
                        "multiline": True,
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def send_message(self, notebook_id: str, message: str):
        from ..utils.notebooklm_cli import ask_question

        notebook_id = notebook_id.strip()
        if not notebook_id:
            raise ValueError("notebook_id is required. Get it from the notebook URL.")
        if not message.strip():
            raise ValueError("message is required")

        logger.info(f"Sending message to notebook {notebook_id[:8]}...")

        result = ask_question(notebook_id, message.strip())
        response = result.get("answer", "")

        logger.info(f"Response: {len(response)} chars")

        return (response,)
