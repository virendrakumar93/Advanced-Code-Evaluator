"""Multi-model router with ordered fallback.

Tries models in config order. First successful model wins.
If all models fail, returns None so the caller can use deterministic fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from llm.provider_adapter import invoke_with_retry

logger = logging.getLogger(__name__)


def _get_client():
    """Create a HuggingFace InferenceClient, or None if unavailable."""
    api_key = os.environ.get("HF_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_key:
        logger.warning("No HuggingFace API key found (HF_API_KEY / HF_TOKEN). LLM layer disabled.")
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(token=api_key)
    except ImportError:
        logger.warning("huggingface_hub not installed. LLM layer disabled.")
        return None


def _extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from model output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block in markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def query_models(
    models: List[str],
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    retries: int = 3,
) -> Tuple[Optional[dict], str]:
    """Try each model in order. Return (parsed_json, model_used) or (None, "")."""
    client = _get_client()
    if client is None:
        return None, ""

    for model_id in models:
        logger.info("Trying model: %s", model_id)
        raw = invoke_with_retry(
            client, model_id, system_prompt, user_prompt,
            max_tokens=max_tokens, temperature=temperature, retries=retries,
        )
        if raw is None:
            logger.warning("Model %s returned no response, trying next.", model_id)
            continue

        parsed = _extract_json(raw)
        if parsed is not None and "scores" in parsed:
            logger.info("Model %s returned valid response.", model_id)
            return parsed, model_id

        logger.warning("Model %s returned unparseable response, trying next.", model_id)

    logger.warning("All models failed. Falling back to deterministic.")
    return None, ""
