from __future__ import annotations

import math
from typing import Optional


SHARED_SCHEMA_VERSION = "shared_schema_v0.1"

QUALITY_GRADE_RULES = {
    "A": {
        "min_score": 0.75,
        "meaning": "High-quality prediction-market evidence. May enter theme synthesis and insight context.",
        "policy": "accepted",
    },
    "B": {
        "min_score": 0.55,
        "meaning": "Usable prediction-market evidence. May enter theme synthesis with lower weight than A.",
        "policy": "accepted",
    },
    "C": {
        "min_score": 0.35,
        "meaning": "Watch-only evidence. Useful for context, but not allowed to drive direction.",
        "policy": "watch",
    },
    "D": {
        "min_score": 0.0,
        "meaning": "Discard or display only as a data-quality gap.",
        "policy": "discard",
    },
}

DISCOVERY_BUCKETS = {
    "accepted": {
        "grades": ["A", "B"],
        "meaning": "Allowed into research context and theme synthesis. Never sufficient by itself to trigger a trade.",
    },
    "watch": {
        "grades": ["C"],
        "meaning": "Displayed as context only. Must not influence direction directly.",
    },
    "discard": {
        "grades": ["D"],
        "meaning": "Excluded from theme synthesis and insight conclusions.",
    },
}

SIGNAL_STATUS = {
    "AVAILABLE": "Enough accepted evidence exists for a theme-level value.",
    "LOW_CONFIDENCE": "Only watch-quality evidence exists, or confidence is too low.",
    "UNAVAILABLE": "No usable evidence exists.",
}

CORE_THEME_MIN_CONFIDENCE = 0.5
CORE_THEME_MIN_EVIDENCE = 1

ALLOWED_ACTIONS = {"NO_TRADE", "CONSIDER_TQQQ", "CONSIDER_SQQQ", "CONSIDER_SOXL", "CONSIDER_SOXS"}


def quality_grade(score: float) -> str:
    if score >= QUALITY_GRADE_RULES["A"]["min_score"]:
        return "A"
    if score >= QUALITY_GRADE_RULES["B"]["min_score"]:
        return "B"
    if score >= QUALITY_GRADE_RULES["C"]["min_score"]:
        return "C"
    return "D"


def is_accepted_grade(grade: str) -> bool:
    return grade in DISCOVERY_BUCKETS["accepted"]["grades"]


def is_watch_grade(grade: str) -> bool:
    return grade in DISCOVERY_BUCKETS["watch"]["grades"]


def is_core_theme(status: str, confidence: float, evidence_count: int) -> bool:
    return status == "AVAILABLE" and confidence >= CORE_THEME_MIN_CONFIDENCE and evidence_count >= CORE_THEME_MIN_EVIDENCE


def score_prediction_market(
    volume: Optional[float],
    liquidity: Optional[float],
    spread: Optional[float],
    relevance: float = 1.0,
) -> float:
    volume_quality = min(math.log10(max(volume or 0, 1)) / 6.0, 1.0)
    liquidity_quality = min(math.log10(max(liquidity or 0, 1)) / 5.0, 1.0)
    if spread is None:
        spread_quality = 0.5
    else:
        spread_quality = max(0.0, min(1.0, 1.0 - spread / 0.25))
    recency_quality = 0.8
    depth_quality = liquidity_quality
    return round(
        0.25 * volume_quality
        + 0.20 * liquidity_quality
        + 0.20 * spread_quality
        + 0.15 * recency_quality
        + 0.10 * depth_quality
        + 0.10 * max(0.0, min(relevance, 1.0)),
        4,
    )


def schema_definitions() -> dict:
    return {
        "schema_version": SHARED_SCHEMA_VERSION,
        "quality_grade_rules": QUALITY_GRADE_RULES,
        "discovery_buckets": DISCOVERY_BUCKETS,
        "signal_status": SIGNAL_STATUS,
        "core_theme_rules": {
            "min_confidence": CORE_THEME_MIN_CONFIDENCE,
            "min_evidence": CORE_THEME_MIN_EVIDENCE,
            "meaning": "Only core themes are shown in the main Theme Scores table and used as primary insight context.",
        },
        "allowed_actions": sorted(ALLOWED_ACTIONS),
    }
