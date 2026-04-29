from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

from .config import KALSHI_GPU_MARKETS, POLYMARKET_THEME_SLUGS, THEME_KEYWORDS
from .http import FetchError, get_json
from .shared_schema import quality_grade, score_prediction_market


POLYMARKET_BASE = "https://gamma-api.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_outcome_prices(raw: Any) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [_safe_float(item) for item in raw if _safe_float(item) is not None]
    if isinstance(raw, str):
        cleaned = raw.strip().replace("[", "").replace("]", "").replace('"', "")
        values = []
        for part in cleaned.split(","):
            parsed = _safe_float(part.strip())
            if parsed is not None:
                values.append(parsed)
        return values or None
    return None


@dataclass
class MarketEvidence:
    platform: str
    market_id: str
    theme: str
    question: str
    probability: Optional[float]
    volume: Optional[float]
    liquidity: Optional[float]
    spread: Optional[float]
    quality_score: float
    quality_grade: str
    status: str
    raw: Dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def score_market(volume: Optional[float], liquidity: Optional[float], spread: Optional[float], relevance: float = 1.0) -> float:
    return score_prediction_market(volume, liquidity, spread, relevance)


def classify_theme(text: str) -> str:
    lower = text.lower()
    best_theme = "OTHER_NOISE"
    best_hits = 0
    for theme, keywords in THEME_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lower)
        if hits > best_hits:
            best_theme = theme
            best_hits = hits
    return best_theme


def fetch_polymarket_event(slug: str, theme: str, timeout: int = 15) -> List[MarketEvidence]:
    try:
        event = get_json(f"{POLYMARKET_BASE}/events/slug/{slug}", timeout=timeout, ttl_seconds=60)
    except FetchError:
        return []

    markets = event.get("markets") or []
    evidence: List[MarketEvidence] = []
    for market in markets:
        active = bool(market.get("active", event.get("active")))
        closed = bool(market.get("closed", event.get("closed")))
        if not active or closed:
            continue
        question = market.get("question") or event.get("title") or slug
        question_lower = question.lower()
        if theme == "AI_SEMICONDUCTOR_SENTIMENT" and "nvidia" not in question_lower:
            continue
        if theme == "SEMICONDUCTOR_IPO_WINDOW" and "cerebras" not in question_lower:
            continue

        prices = _parse_outcome_prices(market.get("outcomePrices"))
        probability = prices[0] if prices else None
        volume = _safe_float(market.get("volumeNum") or event.get("volume"))
        liquidity = _safe_float(market.get("liquidityNum") or event.get("liquidity"))
        score = score_market(volume, liquidity, None)
        evidence.append(
            MarketEvidence(
                platform="polymarket",
                market_id=slug,
                theme=theme,
                question=question,
                probability=probability,
                volume=volume,
                liquidity=liquidity,
                spread=None,
                quality_score=score,
                quality_grade=quality_grade(score),
                status="ACTIVE",
                raw={
                    "event_title": event.get("title"),
                    "event_slug": event.get("slug"),
                    "endDate": event.get("endDate"),
                },
            )
        )
    return evidence


def fetch_polymarket_theme_evidence(timeout: int = 15) -> List[MarketEvidence]:
    evidence: List[MarketEvidence] = []
    jobs = [(theme, slug) for theme, slugs in POLYMARKET_THEME_SLUGS.items() for slug in slugs]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as executor:
        futures = {executor.submit(fetch_polymarket_event, slug, theme, timeout): (theme, slug) for theme, slug in jobs}
        for future in as_completed(futures):
            evidence.extend(future.result())
    return evidence


def fetch_polymarket_top_events(limit: int = 50, timeout: int = 15) -> List[dict]:
    try:
        events = get_json(
            f"{POLYMARKET_BASE}/events",
            params={"active": "true", "closed": "false", "limit": limit, "order": "volume", "ascending": "false"},
            timeout=timeout,
            ttl_seconds=60,
        )
    except FetchError:
        return []
    output = []
    for event in events:
        title = event.get("title") or event.get("slug") or ""
        theme = classify_theme(title)
        if theme == "OTHER_NOISE":
            continue
        output.append(
            {
                "platform": "polymarket",
                "theme": theme,
                "title": title,
                "market_id": event.get("slug"),
                "volume": _safe_float(event.get("volume")),
                "liquidity": _safe_float(event.get("liquidity")),
            }
        )
    return output


def _best_yes_no(orderbook: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    book = orderbook.get("orderbook_fp") or orderbook.get("orderbook") or {}
    yes = book.get("yes_dollars") or []
    no = book.get("no_dollars") or []
    best_yes = _safe_float(yes[-1][0]) if yes else None
    best_no = _safe_float(no[-1][0]) if no else None
    yes_depth = sum((_safe_float(row[1]) or 0.0) for row in yes)
    no_depth = sum((_safe_float(row[1]) or 0.0) for row in no)
    depth = yes_depth + no_depth if yes or no else None
    spread = None
    if best_yes is not None and best_no is not None:
        implied_ask_yes = 1.0 - best_no
        spread = max(0.0, implied_ask_yes - best_yes)
    return best_yes, best_no, spread, depth


def fetch_kalshi_market(ticker: str, theme: str, timeout: int = 15) -> Optional[MarketEvidence]:
    try:
        market = get_json(f"{KALSHI_BASE}/markets/{ticker}", timeout=timeout, ttl_seconds=60).get("market", {})
        orderbook = get_json(f"{KALSHI_BASE}/markets/{ticker}/orderbook", timeout=timeout, ttl_seconds=30)
    except FetchError:
        return None

    best_yes, best_no, spread, depth = _best_yes_no(orderbook)
    probability = best_yes
    volume = _safe_float(market.get("volume"))
    liquidity = _safe_float(market.get("liquidity"))
    if liquidity is None:
        liquidity = depth
    score = score_market(volume, liquidity, spread)
    return MarketEvidence(
        platform="kalshi",
        market_id=ticker,
        theme=theme,
        question=market.get("title") or ticker,
        probability=probability,
        volume=volume,
        liquidity=liquidity,
        spread=spread,
        quality_score=score,
        quality_grade=quality_grade(score),
        status=market.get("status") or "UNKNOWN",
        raw={
            "yes_best": best_yes,
            "no_best": best_no,
            "close_time": market.get("close_time"),
        },
    )


def fetch_kalshi_gpu_evidence(timeout: int = 15) -> List[MarketEvidence]:
    evidence: List[MarketEvidence] = []
    jobs = [(theme, ticker) for theme, tickers in KALSHI_GPU_MARKETS.items() for ticker in tickers]
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as executor:
        futures = {executor.submit(fetch_kalshi_market, ticker, theme, timeout): (theme, ticker) for theme, ticker in jobs}
        for future in as_completed(futures):
            item = future.result()
            if item is not None:
                evidence.append(item)
    return evidence


def group_by_theme(evidence: Iterable[MarketEvidence]) -> Dict[str, List[MarketEvidence]]:
    grouped: Dict[str, List[MarketEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.theme, []).append(item)
    return grouped
