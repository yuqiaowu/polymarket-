from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .market_data import Quote
from .prediction_markets import MarketEvidence, group_by_theme
from .shared_schema import is_accepted_grade, is_core_theme, is_watch_grade


@dataclass
class RiskFilterResult:
    can_trade: bool
    risk_appetite_score: int
    regime: str
    reasons: List[str]
    warnings: List[str]
    data_gaps: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThemeScore:
    theme: str
    value: Optional[float]
    confidence: float
    status: str
    interpretation: str
    evidence_count: int
    top_evidence: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _quote_return(quotes: Dict[str, Quote], symbol: str) -> Optional[float]:
    quote = quotes.get(symbol)
    if not quote:
        return None
    return quote.intraday_return_pct


def macro_risk_filter(quotes: Dict[str, Quote]) -> RiskFilterResult:
    score = 0
    reasons: List[str] = []
    warnings: List[str] = []
    data_gaps: List[str] = []

    us2y = _quote_return(quotes, "US2Y")
    us10y = _quote_return(quotes, "US10Y")
    dxy = _quote_return(quotes, "DXY")
    usdjpy = _quote_return(quotes, "USDJPY")
    vix_quote = quotes.get("VIX")
    vix9d_quote = quotes.get("VIX9D")
    vix3m_quote = quotes.get("VIX3M")

    for symbol in ["VIX", "VIX9D", "VIX3M", "US2Y", "US10Y", "DXY", "USDJPY"]:
        quote = quotes.get(symbol)
        if not quote or quote.status != "OK":
            data_gaps.append(symbol)

    if us2y is not None:
        if us2y > 0.35:
            score -= 1
            reasons.append(f"US2Y rising ({us2y:.2f}%), hawkish policy repricing.")
        elif us2y < -0.35:
            score += 1
            reasons.append(f"US2Y falling ({us2y:.2f}%), dovish policy repricing.")

    if us10y is not None:
        if us10y > 0.35:
            score -= 1
            reasons.append(f"US10Y rising ({us10y:.2f}%), valuation headwind.")
        elif us10y < -0.35:
            score += 1
            reasons.append(f"US10Y falling ({us10y:.2f}%), valuation support.")

    if dxy is not None:
        if dxy > 0.35:
            score -= 1
            reasons.append(f"DXY rising ({dxy:.2f}%), risk appetite headwind.")
        elif dxy < -0.35:
            score += 1
            reasons.append(f"DXY falling ({dxy:.2f}%), risk appetite support.")

    if usdjpy is not None and usdjpy < -0.6:
        score -= 1
        reasons.append(f"USDJPY falling ({usdjpy:.2f}%), possible carry unwind.")

    if vix_quote and vix_quote.close is not None:
        if vix_quote.time == "CBOE_DAILY":
            warnings.append("VIX uses CBOE daily history, not real-time intraday data.")
        elif vix_quote.source_symbol == "^VIX":
            warnings.append("VIX uses yfinance intraday data; validate with IBKR before live execution.")
        if vix_quote.close >= 24:
            score -= 2
            reasons.append(f"VIX elevated ({vix_quote.close:.2f}), high-risk regime.")
        elif vix_quote.close >= 18:
            score -= 1
            reasons.append(f"VIX above calm range ({vix_quote.close:.2f}).")
        elif vix_quote.close <= 14:
            score += 1
            reasons.append(f"VIX calm ({vix_quote.close:.2f}), supports risk-taking.")

    if vix9d_quote and vix3m_quote and vix9d_quote.close is not None and vix3m_quote.close is not None:
        ratio = vix9d_quote.close / vix3m_quote.close if vix3m_quote.close else None
        if ratio and ratio > 1.0:
            score -= 1
            reasons.append(f"VIX9D/VIX3M inverted ({ratio:.2f}), short-term stress.")
        elif ratio and ratio < 0.8:
            score += 1
            reasons.append(f"VIX9D/VIX3M calm ({ratio:.2f}).")

    missing_vol = [symbol for symbol in ["VIX", "VIX9D", "VIX3M"] if symbol in data_gaps]
    if missing_vol:
        warnings.append(f"Volatility data unavailable: {', '.join(missing_vol)}; volatility regime downgraded.")

    capped = max(-2, min(2, score))
    if capped >= 1:
        regime = "RISK_ON"
    elif capped <= -1:
        regime = "RISK_OFF"
    else:
        regime = "NEUTRAL"

    can_trade = len([gap for gap in data_gaps if gap in {"VIX", "US2Y", "US10Y", "DXY"}]) == 0
    if not can_trade:
        warnings.append("Core macro data missing; no trade.")

    return RiskFilterResult(
        can_trade=can_trade,
        risk_appetite_score=capped,
        regime=regime,
        reasons=reasons or ["No strong macro/risk filter signal."],
        warnings=warnings,
        data_gaps=data_gaps,
    )


def _weighted_theme_value(items: List[MarketEvidence]) -> tuple[Optional[float], float, str]:
    usable = [item for item in items if is_accepted_grade(item.quality_grade) and item.probability is not None]
    if not usable:
        watch = [item for item in items if is_watch_grade(item.quality_grade)]
        if watch:
            return None, 0.25, "LOW_CONFIDENCE"
        return None, 0.0, "UNAVAILABLE"
    total_weight = 0.0
    weighted = 0.0
    for item in usable:
        weight = item.quality_score * (1.0 if item.quality_grade == "A" else 0.5)
        total_weight += weight
        weighted += (item.probability or 0.0) * weight
    value = weighted / total_weight if total_weight else None
    confidence = min(1.0, total_weight / max(1, len(usable)))
    return value, confidence, "AVAILABLE"


def score_themes(evidence: List[MarketEvidence]) -> List[ThemeScore]:
    output: List[ThemeScore] = []
    for theme, items in sorted(group_by_theme(evidence).items()):
        value, confidence, status = _weighted_theme_value(items)
        if value is None:
            interpretation = "NO_ACTIONABLE_SIGNAL"
        elif theme in {"AI_BUBBLE_RISK", "LEGACY_GPU_PRICE_PRESSURE"}:
            interpretation = "BEARISH_FOR_SOXL" if value >= 0.5 else "NOT_BEARISH"
        elif "GPU_TIGHTNESS" in theme or theme in {"AI_SEMICONDUCTOR_SENTIMENT", "SEMICONDUCTOR_IPO_WINDOW"}:
            interpretation = "BULLISH_FOR_SOXL" if value >= 0.5 else "NOT_BULLISH"
        else:
            interpretation = "CONTEXT"
        top = sorted(items, key=lambda item: item.quality_score, reverse=True)[:3]
        output.append(
            ThemeScore(
                theme=theme,
                value=None if value is None else round(value, 4),
                confidence=round(confidence, 4),
                status=status,
                interpretation=interpretation,
                evidence_count=len(items),
                top_evidence=[
                    {
                        "platform": item.platform,
                        "market_id": item.market_id,
                        "question": item.question,
                        "probability": item.probability,
                        "quality_grade": item.quality_grade,
                        "quality_score": item.quality_score,
                    }
                    for item in top
                ],
            )
        )
    return output


def semiconductor_direction(theme_scores: List[ThemeScore], quotes: Dict[str, Quote]) -> dict:
    by_theme = {
        item.theme: item
        for item in theme_scores
        if is_core_theme(item.status, item.confidence, item.evidence_count)
    }
    semi_score = 0
    reasons: List[str] = []

    ai_sentiment = by_theme.get("AI_SEMICONDUCTOR_SENTIMENT")
    current_gpu = by_theme.get("CURRENT_GEN_GPU_TIGHTNESS")
    legacy_pressure = by_theme.get("LEGACY_GPU_PRICE_PRESSURE")
    ai_bubble = by_theme.get("AI_BUBBLE_RISK")

    if ai_sentiment and ai_sentiment.value is not None and ai_sentiment.value >= 0.6:
        semi_score += 1
        reasons.append("AI semiconductor prediction-market sentiment is supportive.")
    if current_gpu and current_gpu.value is not None and current_gpu.value >= 0.5:
        semi_score += 1
        reasons.append("Current-gen GPU tightness is supportive.")
    if legacy_pressure and legacy_pressure.value is not None and legacy_pressure.value < 0.3:
        reasons.append("Legacy GPU price pressure is weak enough not to block SOXL.")
    elif legacy_pressure and legacy_pressure.value is not None and legacy_pressure.value >= 0.5:
        semi_score -= 1
        reasons.append("Legacy GPU price pressure is elevated; avoid chasing SOXL.")
    if ai_bubble and ai_bubble.value is not None and ai_bubble.value >= 0.25:
        semi_score -= 1
        reasons.append("AI bubble risk is elevated.")

    qqq = _quote_return(quotes, "QQQ")
    soxx = _quote_return(quotes, "SOXX")
    if qqq is not None and soxx is not None:
        rel = soxx - qqq
        if rel > 0.5:
            semi_score += 1
            reasons.append(f"SOXX confirms strength vs QQQ ({rel:.2f}pp).")
        elif rel < -0.5:
            semi_score -= 1
            reasons.append(f"SOXX confirms weakness vs QQQ ({rel:.2f}pp).")

    capped = max(-2, min(2, semi_score))
    if capped >= 1:
        bias = "SOXL_BIAS"
    elif capped <= -1:
        bias = "SOXS_BIAS"
    else:
        bias = "NO_CLEAR_EDGE"
    return {"semiconductor_score": capped, "bias": bias, "reasons": reasons or ["No clear semiconductor edge."]}
