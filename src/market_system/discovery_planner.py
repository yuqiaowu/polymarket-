from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from .config import DISCOVERY_DIRECTIONS
from .deepseek_client import call_deepseek_json, deepseek_configured


PLANNER_SCHEMA_VERSION = "planner_v0.1"
MAX_KEYWORDS_PER_DIRECTION = 36
MAX_SERIES_PER_DIRECTION = 12


PLANNER_SYSTEM_PROMPT = """
You are a prediction-market discovery planner for a US equity index trading research system.

Task:
- Expand search keywords for two directions: US_EQUITY_MARKET and AI_SEMICONDUCTOR.
- Use private chain-of-thought and tree-of-thought style deliberation internally:
  1. Generate several candidate keyword branches for each direction.
  2. Compare durable macro/industry concepts vs. transient names.
  3. Reject branches likely to retrieve sports, entertainment, stale, or weakly relevant markets.
  4. Select the smallest keyword set that maximizes durable market discovery.
- Do not output hidden reasoning, chain-of-thought, prose, markdown, or external facts.
- Do not invent probabilities, market prices, API results, or trading orders.
- Keywords should help code search Polymarket/Kalshi markets.
- Prefer durable concepts over one-off names when possible.
- For each direction, add at most 12 keywords, at most 6 exclude keywords, and at most 5 Kalshi series.
- Keep every keyword short, usually 1-3 words.

Allowed output JSON shape:
{
  "schema_version": "planner_v0.1",
  "directions": {
    "US_EQUITY_MARKET": {
      "add_keywords": ["string"],
      "exclude_keywords": ["string"],
      "kalshi_series": ["string"],
      "rationale_summary": "brief non-factual reason for keyword set"
    },
    "AI_SEMICONDUCTOR": {
      "add_keywords": ["string"],
      "exclude_keywords": ["string"],
      "kalshi_series": ["string"],
      "rationale_summary": "brief non-factual reason for keyword set"
    }
  }
}
""".strip()


def build_discovery_plan(context: Optional[Dict[str, Any]] = None, use_llm: bool = True) -> dict:
    base = _default_plan()
    llm_raw = None
    llm_used = False
    llm_error = None

    if use_llm and deepseek_configured():
        user_payload = {
            "schema_version": PLANNER_SCHEMA_VERSION,
            "base_directions": base["directions"],
            "context": context or {},
        }
        llm_raw = call_deepseek_json(PLANNER_SYSTEM_PROMPT, user_payload)
        if llm_raw:
            merged = _merge_llm_plan(base, llm_raw)
            merged["planner"]["llm_used"] = True
            merged["planner"]["llm_raw_valid"] = True
            return merged
        llm_error = "DeepSeek returned no valid JSON or request failed."

    base["planner"]["llm_used"] = llm_used
    base["planner"]["llm_raw_valid"] = False
    base["planner"]["llm_error"] = llm_error
    return base


def _default_plan() -> dict:
    directions = {}
    for direction, payload in DISCOVERY_DIRECTIONS.items():
        directions[direction] = {
            "description": payload.get("description"),
            "keywords": list(payload.get("polymarket_keywords", [])),
            "exclude_keywords": _default_excludes(direction),
            "kalshi_series": list(payload.get("kalshi_series", [])),
            "source": "static_default",
            "rationale_summary": "Static durable search terms.",
        }
    return {
        "schema_version": PLANNER_SCHEMA_VERSION,
        "planner": {
            "engine": "deepseek_planner_optional",
            "llm_configured": deepseek_configured(),
            "llm_used": False,
            "llm_raw_valid": False,
            "llm_error": None,
        },
        "directions": directions,
    }


def _merge_llm_plan(base: dict, llm_raw: Dict[str, Any]) -> dict:
    merged = deepcopy(base)
    if llm_raw.get("schema_version") != PLANNER_SCHEMA_VERSION:
        return merged

    raw_directions = llm_raw.get("directions")
    if not isinstance(raw_directions, dict):
        return merged

    for direction, current in merged["directions"].items():
        raw = raw_directions.get(direction)
        if not isinstance(raw, dict):
            continue
        current["keywords"] = _limited_unique(
            current["keywords"] + _string_list(raw.get("add_keywords")),
            MAX_KEYWORDS_PER_DIRECTION,
        )
        current["exclude_keywords"] = _limited_unique(
            current["exclude_keywords"] + _string_list(raw.get("exclude_keywords")),
            MAX_KEYWORDS_PER_DIRECTION,
        )
        current["kalshi_series"] = _limited_unique(
            current["kalshi_series"] + _string_list(raw.get("kalshi_series")),
            MAX_SERIES_PER_DIRECTION,
        )
        summary = raw.get("rationale_summary")
        if isinstance(summary, str) and summary.strip():
            current["rationale_summary"] = summary.strip()[:300]
        current["source"] = "static_plus_deepseek"
    return merged


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.strip().lower().split())
        if 2 <= len(cleaned) <= 48:
            output.append(cleaned)
    return output


def _limited_unique(items: List[str], limit: int) -> List[str]:
    output = []
    seen = set()
    for item in items:
        cleaned = " ".join(str(item).strip().lower().split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _default_excludes(direction: str) -> List[str]:
    common = ["nba", "nfl", "mlb", "nhl", "soccer", "world cup", "grammy", "oscars"]
    if direction == "US_EQUITY_MARKET":
        return common + ["presidential nominee"]
    if direction == "AI_SEMICONDUCTOR":
        return common + ["anime", "openai trial"]
    return common
