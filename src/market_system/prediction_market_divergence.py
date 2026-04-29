from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from .shared_schema import is_accepted_grade, is_watch_grade


THEME_ALIASES = {
    "AI_BUBBLE_RISK": "AI_BUBBLE_RISK",
    "AI_SEMICONDUCTOR_SENTIMENT": "AI_SEMICONDUCTOR_SENTIMENT",
    "SEMICONDUCTOR_IPO_WINDOW": "SEMICONDUCTOR_CAPITAL_MARKETS",
    "CHINA_TAIWAN_RISK": "CHINA_TAIWAN_RISK",
    "CONSUMER_GPU_TIGHTNESS": "GPU_COMPUTE_PRICE_PRESSURE",
    "CURRENT_GEN_GPU_TIGHTNESS": "GPU_COMPUTE_PRICE_PRESSURE",
    "LEGACY_GPU_PRICE_PRESSURE": "GPU_COMPUTE_PRICE_PRESSURE",
}


def build_prediction_market_divergence(report: dict) -> dict:
    observations = _collect_observations(report)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in observations:
        grouped[item["normalized_theme"]].append(item)

    themes = []
    for theme, items in sorted(grouped.items()):
        themes.append(_summarize_theme(theme, items))

    return {
        "schema_version": "prediction_market_divergence_v0.1",
        "policy": (
            "Compares prediction-market evidence by normalized theme and platform. "
            "A/B markets are primary, C markets are watch-only, D markets are ignored."
        ),
        "theme_count": len(themes),
        "themes": themes,
        "cross_platform_divergences": [item for item in themes if item["divergence_type"] == "CROSS_PLATFORM_PROBABILITY_GAP"],
        "one_sided_consensus": [item for item in themes if item["divergence_type"] == "ONE_SIDED_PLATFORM_CONSENSUS"],
        "data_gaps": _data_gaps(themes),
    }


def _collect_observations(report: dict) -> List[dict]:
    observations = []
    for item in report.get("prediction_market_evidence", []):
        theme = THEME_ALIASES.get(item.get("theme"), item.get("theme") or "UNKNOWN")
        if _ignore_grade(item.get("quality_grade")):
            continue
        observations.append(_observation_from_item(item, theme, "fixed_theme_evidence"))

    for direction, payload in report.get("market_discovery", {}).get("directions", {}).items():
        for market in payload.get("top_markets", []):
            if _ignore_grade(market.get("quality_grade")):
                continue
            theme = _classify_discovered_theme(market.get("title", ""), direction)
            observations.append(_observation_from_item(market, theme, "dynamic_discovery"))
    return observations


def _observation_from_item(item: dict, normalized_theme: str, source: str) -> dict:
    return {
        "normalized_theme": normalized_theme,
        "source": source,
        "platform": item.get("platform"),
        "market_id": item.get("market_id"),
        "title": item.get("question") or item.get("title"),
        "probability": _safe_float(item.get("probability")),
        "volume": _safe_float(item.get("volume")),
        "liquidity": _safe_float(item.get("liquidity")),
        "spread": _safe_float(item.get("spread")),
        "quality_grade": item.get("quality_grade"),
        "quality_score": _safe_float(item.get("quality_score")),
        "relevance_score": _safe_float(item.get("relevance_score")),
        "is_primary": is_accepted_grade(item.get("quality_grade")),
        "is_watch": is_watch_grade(item.get("quality_grade")),
    }


def _summarize_theme(theme: str, items: List[dict]) -> dict:
    primary = [item for item in items if item["is_primary"]]
    usable = primary or [item for item in items if item["is_watch"]]
    platform_summaries = [_platform_summary(platform, platform_items) for platform, platform_items in _group_by_platform(usable).items()]
    platforms_with_primary = sorted({item["platform"] for item in primary if item.get("platform")})
    probabilities = [item["weighted_probability"] for item in platform_summaries if item["weighted_probability"] is not None]
    gap = round(max(probabilities) - min(probabilities), 4) if len(probabilities) >= 2 else None

    if len(platforms_with_primary) >= 2 and gap is not None and gap >= 0.12:
        divergence_type = "CROSS_PLATFORM_PROBABILITY_GAP"
        interpretation = "Prediction markets disagree on this theme; treat it as a divergence requiring price confirmation."
        quality = "MEDIUM"
    elif len(platforms_with_primary) >= 2:
        divergence_type = "CROSS_PLATFORM_CONSENSUS"
        interpretation = "Multiple platforms broadly agree on this theme."
        quality = "HIGH"
    elif primary:
        divergence_type = "ONE_SIDED_PLATFORM_CONSENSUS"
        interpretation = "Only one platform has accepted evidence; useful context, but not cross-platform confirmation."
        quality = "MEDIUM"
    else:
        divergence_type = "WATCH_ONLY"
        interpretation = "Evidence is watch-only and should not drive the final conclusion."
        quality = "LOW"

    return {
        "theme": theme,
        "platforms": platform_summaries,
        "accepted_count": len(primary),
        "watch_count": len([item for item in items if item["is_watch"]]),
        "platform_count": len({item["platform"] for item in usable if item.get("platform")}),
        "probability_gap": gap,
        "divergence_type": divergence_type,
        "quality": quality,
        "interpretation": interpretation,
        "top_markets": sorted(usable, key=lambda item: (item.get("quality_score") or 0.0, item.get("volume") or 0.0), reverse=True)[:5],
    }


def _platform_summary(platform: str, items: List[dict]) -> dict:
    weighted_probability = _weighted_probability(items)
    return {
        "platform": platform,
        "market_count": len(items),
        "accepted_count": len([item for item in items if item["is_primary"]]),
        "watch_count": len([item for item in items if item["is_watch"]]),
        "weighted_probability": weighted_probability,
        "avg_quality_score": round(mean([item["quality_score"] for item in items if item["quality_score"] is not None]), 4)
        if any(item["quality_score"] is not None for item in items)
        else None,
        "total_volume": round(sum(item["volume"] or 0.0 for item in items), 2),
        "best_market": max(items, key=lambda item: (item.get("quality_score") or 0.0, item.get("volume") or 0.0)),
    }


def _weighted_probability(items: Iterable[dict]) -> Optional[float]:
    weighted_sum = 0.0
    weight_total = 0.0
    for item in items:
        probability = item.get("probability")
        if probability is None:
            continue
        quality = item.get("quality_score") or 0.0
        volume = min((item.get("volume") or 0.0) / 1_000_000.0, 10.0)
        weight = max(0.01, quality) * (1.0 + volume)
        weighted_sum += probability * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return round(weighted_sum / weight_total, 4)


def _group_by_platform(items: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        grouped[item.get("platform") or "unknown"].append(item)
    return grouped


def _classify_discovered_theme(title: str, direction: str) -> str:
    lower = title.lower()
    if "fed" in lower and ("rate" in lower or "interest" in lower):
        return "FED_RATE_DECISION"
    if "cpi" in lower or "pce" in lower or "inflation" in lower:
        return "INFLATION_DATA"
    if "unemployment" in lower or "payroll" in lower or "jobs" in lower:
        return "EMPLOYMENT_GROWTH"
    if "nasdaq" in lower or "s&p" in lower or "stock market" in lower:
        return "US_EQUITY_DIRECTION"
    if "vix" in lower or "volatility" in lower:
        return "MARKET_VOLATILITY"
    if "tariff" in lower or "trade" in lower:
        return "TARIFF_TRADE_POLICY"
    if "ceasefire" in lower or "war" in lower or "ukraine" in lower or "israel" in lower:
        return "WAR_GEOPOLITICS"
    if "taiwan" in lower or "china invade" in lower:
        return "CHINA_TAIWAN_RISK"
    if "ai bubble" in lower:
        return "AI_BUBBLE_RISK"
    if "nvidia" in lower or "semiconductor" in lower or "gpu" in lower or "hbm" in lower or "memory" in lower:
        return "AI_SEMICONDUCTOR_SENTIMENT"
    return direction


def _data_gaps(themes: List[dict]) -> List[str]:
    gaps = []
    for item in themes:
        if item["divergence_type"] == "ONE_SIDED_PLATFORM_CONSENSUS":
            gaps.append(f"{item['theme']}: only one platform has accepted evidence.")
        elif item["divergence_type"] == "WATCH_ONLY":
            gaps.append(f"{item['theme']}: only watch-grade evidence is available.")
    return gaps[:12]


def _ignore_grade(grade: Optional[str]) -> bool:
    return grade not in {"A", "B", "C"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
