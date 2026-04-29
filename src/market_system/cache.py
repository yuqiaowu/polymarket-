from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT


CACHE_DIR = PROJECT_ROOT / ".cache" / "http"


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def read_cache(key: str, ttl_seconds: int) -> Optional[str]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(payload["created_at"]) > ttl_seconds:
            return None
        return str(payload["body"])
    except Exception:  # noqa: BLE001
        return None


def write_cache(key: str, body: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"created_at": time.time(), "body": body}
    _cache_path(key).write_text(json.dumps(payload), encoding="utf-8")
