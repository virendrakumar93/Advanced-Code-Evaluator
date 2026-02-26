"""Capability-aware HuggingFace Inference API adapter.

Queries model metadata to detect supported tasks and routes to the correct API
(chat completions vs text generation). Never hardcodes the task.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache: model_id -> set of supported tasks
_task_cache: Dict[str, set] = {}


def _query_model_tasks(model_id: str) -> set:
    """Query HuggingFace model metadata to find supported inference tasks."""
    if model_id in _task_cache:
        return _task_cache[model_id]

    import requests

    tasks = set()
    url = f"https://huggingface.co/api/models/{model_id}?expand=inferenceProviderMapping"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Top-level pipeline_tag
            tag = data.get("pipeline_tag", "")
            if tag:
                tasks.add(tag)
            # Provider mapping has tasks per provider
            ipm = data.get("inferenceProviderMapping", {})
            for provider, info in ipm.items():
                task = info.get("task", "") if isinstance(info, dict) else ""
                if task:
                    tasks.add(task)
        logger.debug("Model %s supports tasks: %s", model_id, tasks)
    except Exception as exc:
        logger.warning("Failed to query model metadata for %s: %s", model_id, exc)

    _task_cache[model_id] = tasks
    return tasks


def invoke_model(
    client,  # huggingface_hub.InferenceClient
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> Optional[str]:
    """Invoke a HuggingFace model using the appropriate API based on capabilities.

    Returns the model's text response, or None on failure.
    """
    tasks = _query_model_tasks(model_id)

    # Strategy 1: chat completions (preferred for instruction-tuned models)
    if "conversational" in tasks or "text-generation" in tasks:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            if text:
                return text.strip()
        except Exception as exc:
            logger.warning("Chat completions failed for %s: %s", model_id, exc)

    # Strategy 2: text generation fallback
    if "text-generation" in tasks:
        try:
            prompt = f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
            response = client.text_generation(
                prompt, model=model_id,
                max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
            )
            if response:
                return response.strip()
        except Exception as exc:
            logger.warning("Text generation failed for %s: %s", model_id, exc)

    return None


def invoke_with_retry(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    retries: int = 3,
) -> Optional[str]:
    """Invoke model with exponential-backoff retry."""
    for attempt in range(retries):
        result = invoke_model(client, model_id, system_prompt, user_prompt, max_tokens, temperature)
        if result is not None:
            return result
        if attempt < retries - 1:
            wait = 2 ** attempt
            logger.info("Retry %d/%d for %s in %ds", attempt + 1, retries, model_id, wait)
            time.sleep(wait)
    return None
