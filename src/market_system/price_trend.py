from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from .market_data import PriceBar


def build_price_trends(daily_bars: Dict[str, List[PriceBar]]) -> dict:
    symbols = sorted(daily_bars.keys())
    trends = {symbol: _trend_for_symbol(symbol, daily_bars.get(symbol, [])) for symbol in symbols}
    return {
        "schema_version": "price_trend_v0.1",
        "policy": (
            "Short-term momentum is confirmed with daily bars using MA5/MA10/MA20, 5-day return, "
            "20-day return, and volume vs 20-day average. Single-day returns alone are not sufficient."
        ),
        "symbols": trends,
        "groups": {
            "broad_market": _group_summary(trends, ["QQQ", "SPY", "TQQQ", "SQQQ"], inverse_symbols={"SQQQ"}),
            "semiconductor": _group_summary(trends, ["SOXX", "SMH", "SOXL", "SOXS"], inverse_symbols={"SOXS"}),
        },
    }


def _trend_for_symbol(symbol: str, bars: List[PriceBar]) -> dict:
    clean = [bar for bar in bars if bar.close is not None]
    if len(clean) < 21:
        return {
            "symbol": symbol,
            "status": "UNAVAILABLE",
            "trend": "UNKNOWN",
            "score": 0,
            "reason": "Fewer than 21 daily bars.",
        }
    closes = [bar.close for bar in clean]
    volumes = [bar.volume for bar in clean if bar.volume is not None]
    close = closes[-1]
    ma5 = mean(closes[-5:])
    ma10 = mean(closes[-10:])
    ma20 = mean(closes[-20:])
    ret_1d = _ret(closes, 1)
    ret_5d = _ret(closes, 5)
    ret_20d = _ret(closes, 20)
    volume_ratio = None
    if len(volumes) >= 20 and clean[-1].volume is not None:
        avg_volume = mean(volumes[-20:])
        if avg_volume:
            volume_ratio = clean[-1].volume / avg_volume

    score = 0
    reasons = []
    if close > ma5 > ma10:
        score += 1
        reasons.append("close above rising short MAs")
    if close < ma5 < ma10:
        score -= 1
        reasons.append("close below falling short MAs")
    if close > ma20:
        score += 1
        reasons.append("close above MA20")
    elif close < ma20:
        score -= 1
        reasons.append("close below MA20")
    if ret_5d is not None and ret_5d > 2:
        score += 1
        reasons.append("5-day return positive")
    elif ret_5d is not None and ret_5d < -2:
        score -= 1
        reasons.append("5-day return negative")
    if ret_20d is not None and ret_20d > 4:
        score += 1
        reasons.append("20-day return positive")
    elif ret_20d is not None and ret_20d < -4:
        score -= 1
        reasons.append("20-day return negative")
    if volume_ratio is not None and volume_ratio >= 1.5 and ret_1d is not None:
        if ret_1d < -1:
            score -= 1
            reasons.append("down day on elevated volume")
        elif ret_1d > 1:
            score += 1
            reasons.append("up day on elevated volume")

    if score >= 2:
        trend = "STRONG"
    elif score <= -2:
        trend = "WEAK"
    else:
        trend = "NEUTRAL"

    return {
        "symbol": symbol,
        "status": "OK",
        "trend": trend,
        "score": score,
        "close": round(close, 4),
        "ma5": round(ma5, 4),
        "ma10": round(ma10, 4),
        "ma20": round(ma20, 4),
        "return_1d_pct": _round(ret_1d),
        "return_5d_pct": _round(ret_5d),
        "return_20d_pct": _round(ret_20d),
        "volume_ratio_20d": _round(volume_ratio),
        "reason": "; ".join(reasons) or "No clear trend confirmation.",
    }


def _group_summary(trends: Dict[str, dict], symbols: List[str], inverse_symbols: set[str]) -> dict:
    scores = []
    evidence = []
    for symbol in symbols:
        item = trends.get(symbol, {})
        if item.get("status") != "OK":
            continue
        score = int(item.get("score") or 0)
        if symbol in inverse_symbols:
            score = -score
        scores.append(score)
        evidence.append({"symbol": symbol, "trend": item.get("trend"), "adjusted_score": score, "reason": item.get("reason")})
    if not scores:
        trend = "UNKNOWN"
        score = 0
    else:
        score = round(mean(scores), 2)
        if score >= 1:
            trend = "STRONG"
        elif score <= -1:
            trend = "WEAK"
        else:
            trend = "NEUTRAL"
    return {
        "trend": trend,
        "score": score,
        "evidence": evidence,
    }


def _ret(closes: List[float], lookback: int) -> Optional[float]:
    if len(closes) <= lookback:
        return None
    base = closes[-lookback - 1]
    if not base:
        return None
    return (closes[-1] / base - 1.0) * 100.0


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)
