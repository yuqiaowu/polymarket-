from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .market_data import Quote


@dataclass
class RiskTemperature:
    score: int
    label: str
    fed_policy_bias: str
    risk_appetite: str
    components: List[dict]
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _ret(quotes: Dict[str, Quote], symbol: str) -> Optional[float]:
    quote = quotes.get(symbol)
    if not quote or quote.status != "OK":
        return None
    return quote.intraday_return_pct


def _close(quotes: Dict[str, Quote], symbol: str) -> Optional[float]:
    quote = quotes.get(symbol)
    if not quote or quote.status != "OK":
        return None
    return quote.close


def _add_component(components: List[dict], name: str, value, points: int, reason: str) -> None:
    components.append(
        {
            "name": name,
            "value": value,
            "points": points,
            "reason": reason,
        }
    )


def build_risk_temperature(quotes: Dict[str, Quote]) -> RiskTemperature:
    score = 0
    fed_score = 0
    components: List[dict] = []
    warnings: List[str] = []

    us2y = _ret(quotes, "US2Y")
    us10y = _ret(quotes, "US10Y")
    dxy = _ret(quotes, "DXY")
    usdjpy = _ret(quotes, "USDJPY")
    vix = _close(quotes, "VIX")
    vix9d = _close(quotes, "VIX9D")
    vix3m = _close(quotes, "VIX3M")

    missing = [symbol for symbol in ["US2Y", "US10Y", "DXY", "USDJPY", "VIX", "VIX9D", "VIX3M"] if _close(quotes, symbol) is None]
    if missing:
        warnings.append(f"Missing risk-temperature inputs: {', '.join(missing)}.")

    if us2y is not None:
        if us2y >= 0.35:
            score -= 1
            fed_score += 1
            _add_component(components, "US2Y intraday", round(us2y, 4), -1, "2Y yield is rising; market is pricing a more hawkish Fed path.")
        elif us2y <= -0.35:
            score += 1
            fed_score -= 1
            _add_component(components, "US2Y intraday", round(us2y, 4), 1, "2Y yield is falling; market is pricing a more dovish Fed path.")

    if us10y is not None:
        if us10y >= 0.35:
            score -= 1
            fed_score += 1
            _add_component(components, "US10Y intraday", round(us10y, 4), -1, "10Y yield is rising; valuation pressure for long-duration equities.")
        elif us10y <= -0.35:
            score += 1
            fed_score -= 1
            _add_component(components, "US10Y intraday", round(us10y, 4), 1, "10Y yield is falling; valuation pressure eases.")

    if us2y is not None and us10y is not None:
        curve_shift = us2y - us10y
        if curve_shift >= 0.25:
            fed_score += 1
            _add_component(components, "2Y-10Y intraday curve shift", round(curve_shift, 4), -1, "Short-end rates are rising faster than long-end rates; hawkish policy shock.")
        elif curve_shift <= -0.25:
            fed_score -= 1
            _add_component(components, "2Y-10Y intraday curve shift", round(curve_shift, 4), 1, "Short-end rates are easing faster than long-end rates; dovish policy shock.")

    if dxy is not None:
        if dxy >= 0.35:
            score -= 1
            fed_score += 1
            _add_component(components, "DXY intraday", round(dxy, 4), -1, "Dollar strength tightens financial conditions.")
        elif dxy <= -0.35:
            score += 1
            fed_score -= 1
            _add_component(components, "DXY intraday", round(dxy, 4), 1, "Dollar weakness supports risk appetite.")

    if usdjpy is not None:
        if usdjpy <= -0.6:
            score -= 1
            _add_component(components, "USDJPY intraday", round(usdjpy, 4), -1, "JPY strength can signal carry unwind/risk-off.")
        elif usdjpy >= 0.6:
            score += 1
            _add_component(components, "USDJPY intraday", round(usdjpy, 4), 1, "JPY weakness is consistent with carry/risk-on.")

    if vix is not None:
        if vix >= 24:
            score -= 2
            _add_component(components, "VIX level", round(vix, 4), -2, "VIX is elevated; avoid aggressive long leverage.")
        elif vix >= 18:
            score -= 1
            _add_component(components, "VIX level", round(vix, 4), -1, "VIX is above calm range.")
        elif vix <= 14:
            score += 1
            _add_component(components, "VIX level", round(vix, 4), 1, "VIX is calm.")

    if vix9d is not None and vix3m is not None and vix3m:
        ratio = vix9d / vix3m
        if ratio >= 1.0:
            score -= 1
            _add_component(components, "VIX9D/VIX3M", round(ratio, 4), -1, "Short-term volatility is inverted versus 3M volatility.")
        elif ratio <= 0.8:
            score += 1
            _add_component(components, "VIX9D/VIX3M", round(ratio, 4), 1, "Volatility curve is calm.")

    capped = max(-5, min(5, score))
    if capped >= 3:
        label = "GREED_RISK_ON"
        appetite = "RISK_ON"
    elif capped >= 1:
        label = "MILD_RISK_ON"
        appetite = "RISK_ON"
    elif capped <= -3:
        label = "FEAR_RISK_OFF"
        appetite = "RISK_OFF"
    elif capped <= -1:
        label = "MILD_RISK_OFF"
        appetite = "RISK_OFF"
    else:
        label = "NEUTRAL"
        appetite = "NEUTRAL"

    if fed_score >= 2:
        fed_bias = "HAWKISH"
    elif fed_score <= -2:
        fed_bias = "DOVISH"
    elif fed_score != 0:
        fed_bias = "MIXED_LEAN_HAWKISH" if fed_score > 0 else "MIXED_LEAN_DOVISH"
    else:
        fed_bias = "NEUTRAL"

    return RiskTemperature(
        score=capped,
        label=label,
        fed_policy_bias=fed_bias,
        risk_appetite=appetite,
        components=components,
        warnings=warnings,
    )
