from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from .cache import read_cache, write_cache


class FetchError(Exception):
    pass


def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    return json.dumps({"url": url, "params": params or {}}, sort_keys=True)


def get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15, ttl_seconds: int = 0) -> Any:
    key = _cache_key(url, params)
    if ttl_seconds > 0:
        cached = read_cache(key, ttl_seconds)
        if cached is not None:
            return json.loads(cached)
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        body = response.text
        if ttl_seconds > 0:
            write_cache(key, body)
        return json.loads(body)
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"GET {url} failed: {exc}") from exc


def get_text(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15, ttl_seconds: int = 0) -> str:
    key = _cache_key(url, params)
    if ttl_seconds > 0:
        cached = read_cache(key, ttl_seconds)
        if cached is not None:
            return cached
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        body = response.text
        if ttl_seconds > 0:
            write_cache(key, body)
        return body
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"GET {url} failed: {exc}") from exc


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Any:
    try:
        response = requests.post(url, json=payload, headers=headers or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"POST {url} failed: {exc}") from exc
