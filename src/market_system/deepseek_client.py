from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .config import load_env_values
from .http import FetchError, post_json


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def deepseek_configured() -> bool:
    return bool(load_env_values().get("DEEPSEEK_API_KEY"))


def call_deepseek_json(
    system_prompt: str,
    user_payload: Dict[str, Any],
    temperature: float = 0.2,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    env = load_env_values()
    api_key = env.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    base_url = env.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
    model = env.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = post_json(url, payload, headers=headers, timeout=timeout)
        content = response["choices"][0]["message"]["content"]
        return json.loads(_extract_json_object(content))
    except (FetchError, KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def _extract_json_object(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned
