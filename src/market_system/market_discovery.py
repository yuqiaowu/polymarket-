from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .discovery_planner import build_discovery_plan
from .http import FetchError, get_json
from .prediction_markets import (
    KALSHI_BASE,
    POLYMARKET_BASE,
    _parse_outcome_prices,
    _safe_float,
)
from .shared_schema import is_accepted_grade, is_watch_grade, quality_grade, score_prediction_market


@dataclass
class DiscoveryMarket:
    direction: str
    platform: str
    market_id: str
    title: str
    probability: Optional[float]
    volume: Optional[float]
    liquidity: Optional[float]
    spread: Optional[float]
    relevance_score: float
    quality_score: float
    quality_grade: str
    status: str
    raw: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def _keyword_relevance(text: str, keywords: List[str], exclude_keywords: Optional[List[str]] = None) -> float:
    lower = text.lower()
    if any(_keyword_matches(lower, keyword.lower()) for keyword in (exclude_keywords or [])):
        return 0.0
    hits = 0
    weighted_hits = 0.0
    for keyword in keywords:
        key = keyword.lower()
        if _keyword_matches(lower, key):
            hits += 1
            weighted_hits += 1.5 if " " in key else 1.0
    if hits == 0:
        return 0.0
    return min(1.0, weighted_hits / 3.0)


def _keyword_matches(text: str, keyword: str) -> bool:
    if len(keyword) <= 4 and keyword.isalpha():
        return bool(re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", text))
    return keyword in text


def _is_past_end_date(raw: Optional[str]) -> bool:
    if not raw:
        return False
    try:
        normalized = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone(timezone.utc) < datetime.now(timezone.utc)
    except ValueError:
        return False


def _polymarket_candidate_to_discovery(
    market: Dict[str, Any],
    direction: str,
    keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
) -> Optional[DiscoveryMarket]:
    title = market.get("question") or market.get("title") or market.get("slug") or ""
    relevance = _keyword_relevance(title, keywords, exclude_keywords)
    if relevance <= 0:
        return None
    active = bool(market.get("active", True))
    closed = bool(market.get("closed", False))
    if not active or closed:
        return None
    end_date = market.get("endDate") or market.get("endDateIso")
    if _is_past_end_date(end_date):
        return None
    prices = _parse_outcome_prices(market.get("outcomePrices"))
    probability = prices[0] if prices else _safe_float(market.get("lastTradePrice"))
    volume = _safe_float(market.get("volumeNum") or market.get("volume"))
    liquidity = _safe_float(market.get("liquidityNum") or market.get("liquidity"))
    spread = _safe_float(market.get("spread"))
    score = score_prediction_market(volume, liquidity, spread, relevance=relevance)
    if relevance < 0.5:
        score = min(score, 0.54)
    return DiscoveryMarket(
        direction=direction,
        platform="polymarket",
        market_id=market.get("slug") or market.get("conditionId") or market.get("id") or title,
        title=title,
        probability=probability,
        volume=volume,
        liquidity=liquidity,
        spread=spread,
        relevance_score=round(relevance, 4),
        quality_score=score,
        quality_grade=quality_grade(score),
        status="ACTIVE",
        raw={
            "endDate": market.get("endDate") or market.get("endDateIso"),
            "oneDayPriceChange": market.get("oneDayPriceChange"),
            "oneWeekPriceChange": market.get("oneWeekPriceChange"),
        },
    )


def fetch_polymarket_discovery(
    direction: str,
    keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
    limit: int = 300,
    timeout: int = 15,
) -> List[DiscoveryMarket]:
    try:
        markets = get_json(
            f"{POLYMARKET_BASE}/markets",
            params={"active": "true", "closed": "false", "limit": limit, "order": "volumeNum", "ascending": "false"},
            timeout=timeout,
            ttl_seconds=120,
        )
    except FetchError:
        return []
    output: List[DiscoveryMarket] = []
    seen = set()
    for market in markets if isinstance(markets, list) else []:
        item = _polymarket_candidate_to_discovery(market, direction, keywords, exclude_keywords)
        if item is None or item.market_id in seen:
            continue
        seen.add(item.market_id)
        output.append(item)
    return sorted(output, key=lambda item: (item.quality_score, item.volume or 0), reverse=True)


def _kalshi_market_probability_and_spread(market: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    yes_bid = _safe_float(market.get("yes_bid_dollars"))
    yes_ask = _safe_float(market.get("yes_ask_dollars"))
    last = _safe_float(market.get("last_price_dollars"))
    probability = yes_bid if yes_bid is not None and yes_bid > 0 else last
    spread = None
    if yes_bid is not None and yes_ask is not None and yes_ask > 0:
        spread = max(0.0, yes_ask - yes_bid)
    return probability, spread


def fetch_kalshi_series_discovery(
    series_ticker: str,
    direction: str,
    keywords: List[str],
    exclude_keywords: Optional[List[str]] = None,
    timeout: int = 15,
) -> List[DiscoveryMarket]:
    try:
        data = get_json(
            f"{KALSHI_BASE}/markets",
            params={"series_ticker": series_ticker, "limit": 50, "status": "open"},
            timeout=timeout,
            ttl_seconds=120,
        )
    except FetchError:
        return []
    output: List[DiscoveryMarket] = []
    for market in data.get("markets", []):
        title = market.get("title") or market.get("ticker") or ""
        relevance = max(0.35, _keyword_relevance(title, keywords, exclude_keywords))
        probability, spread = _kalshi_market_probability_and_spread(market)
        volume = _safe_float(market.get("volume") or market.get("volume_fp"))
        liquidity = _safe_float(market.get("liquidity_dollars"))
        score = score_prediction_market(volume, liquidity, spread, relevance=relevance)
        if relevance < 0.5:
            score = min(score, 0.54)
        output.append(
            DiscoveryMarket(
                direction=direction,
                platform="kalshi",
                market_id=market.get("ticker") or title,
                title=title,
                probability=probability,
                volume=volume,
                liquidity=liquidity,
                spread=spread,
                relevance_score=round(relevance, 4),
                quality_score=score,
                quality_grade=quality_grade(score),
                status=market.get("status") or "UNKNOWN",
                raw={
                    "series_ticker": series_ticker,
                    "event_ticker": market.get("event_ticker"),
                    "close_time": market.get("close_time"),
                    "expiration_time": market.get("expiration_time"),
                },
            )
        )
    return sorted(output, key=lambda item: (item.quality_score, item.volume or 0), reverse=True)


def build_market_discovery(directions: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> dict:
    plan = build_discovery_plan(context=context, use_llm=True)
    selected = directions or list(plan["directions"].keys())
    by_direction: Dict[str, dict] = {}
    all_markets: List[DiscoveryMarket] = []

    jobs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for direction in selected:
            direction_plan = plan["directions"].get(direction)
            if not direction_plan:
                continue
            keywords = direction_plan["keywords"]
            excludes = direction_plan.get("exclude_keywords", [])
            jobs.append(executor.submit(fetch_polymarket_discovery, direction, keywords, excludes))
            for series in direction_plan.get("kalshi_series", []):
                jobs.append(executor.submit(fetch_kalshi_series_discovery, series, direction, keywords, excludes))

        for future in as_completed(jobs):
            all_markets.extend(future.result())

    for direction in selected:
        direction_plan = plan["directions"].get(direction, {})
        items = [item for item in all_markets if item.direction == direction]
        accepted = [item for item in items if is_accepted_grade(item.quality_grade)]
        watch = [item for item in items if is_watch_grade(item.quality_grade)]
        by_direction[direction] = {
            "description": direction_plan.get("description"),
            "keywords": direction_plan.get("keywords", []),
            "exclude_keywords": direction_plan.get("exclude_keywords", []),
            "kalshi_series": direction_plan.get("kalshi_series", []),
            "planner_source": direction_plan.get("source"),
            "planner_rationale_summary": direction_plan.get("rationale_summary"),
            "market_count": len(items),
            "accepted_count": len(accepted),
            "watch_count": len(watch),
            "top_markets": [item.to_dict() for item in sorted(items, key=lambda x: x.quality_score, reverse=True)[:12]],
        }

    return {
        "planner": plan["planner"],
        "planner_schema_version": plan["schema_version"],
        "directions": by_direction,
        "policy": "Discovery markets identify what prediction markets are pricing. A/B can enter research context; C is watch-only; D is discarded.",
    }
