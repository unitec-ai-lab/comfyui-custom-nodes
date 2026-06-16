"""
NotebookLM Product Researcher Node
───────────────────────────────────
Takes a brand/product URL + product name, researches the product via
notebooklm CLI, and creates a fully loaded NotebookLM notebook with
relevant sources.

Outputs NOTEBOOK_DATA for chaining into the Script Generator node.
"""

import logging

logger = logging.getLogger("comfyui-notebooklm")


class NotebookLM_ProductResearcher:
    """Research a product and create a NotebookLM notebook with sources."""

    CATEGORY = "NotebookLM"
    FUNCTION = "research"
    RETURN_TYPES = ("NOTEBOOK_DATA", "STRING", "STRING")
    RETURN_NAMES = ("notebook_data", "summary", "notebook_id")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "brand_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "https://brand.com/product-page",
                    },
                ),
                "product_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Product Name - Brand",
                    },
                ),
            },
            "optional": {
                "extra_urls": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "One URL per line\nhttps://...\nhttps://...",
                    },
                ),
                "search_queries": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Additional search terms\none per line",
                    },
                ),
                "research_mode": (["fast", "deep"],),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run — research creates new notebooks each time
        return float("nan")

    def research(
        self,
        brand_url: str,
        product_name: str,
        extra_urls: str = "",
        search_queries: str = "",
        research_mode: str = "fast",
    ):
        from ..utils.notebooklm_cli import research_product

        if not brand_url.strip():
            raise ValueError("brand_url is required")
        if not product_name.strip():
            raise ValueError("product_name is required")

        # Parse multiline inputs into lists
        extra_url_list = [u for u in extra_urls.strip().split("\n") if u.strip()]
        query_list = [q for q in search_queries.strip().split("\n") if q.strip()]

        logger.info(
            f"Starting product research: {product_name} | "
            f"URL: {brand_url} | "
            f"Extra URLs: {len(extra_url_list)} | "
            f"Queries: {len(query_list)} | "
            f"Mode: {research_mode}"
        )

        # Run the full research workflow
        notebook_data = research_product(
            product_name=product_name.strip(),
            brand_url=brand_url.strip(),
            extra_urls=extra_url_list if extra_url_list else None,
            search_queries=query_list if query_list else None,
            research_mode=research_mode,
        )

        summary = notebook_data.get("summary", "")
        notebook_id = notebook_data.get("notebook_id", "")

        # Log results
        source_count = notebook_data.get("source_count", 0)
        errors = notebook_data.get("errors", [])
        logger.info(
            f"Research complete: {source_count} sources | "
            f"{len(errors)} errors | "
            f"Notebook: {notebook_id}"
        )
        if errors:
            for err in errors:
                logger.warning(f"  - {err}")

        return (notebook_data, summary, notebook_id)
