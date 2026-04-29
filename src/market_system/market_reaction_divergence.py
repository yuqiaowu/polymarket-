from __future__ import annotations

from typing import Dict, Optional

from .market_data import Quote


def build_market_reaction_divergence(
    macro_filter: dict,
    risk_temperature: dict,
    semiconductor_direction: dict,
    prediction_divergence: dict,
    reaction_quotes: Dict[str, Quote],
) -> dict:
    broad = _broad_risk_reaction(macro_filter, risk_temperature, reaction_quotes)
    semiconductor = _semiconductor_reaction(semiconductor_direction, prediction_divergence, reaction_quotes)
    return {
        "schema_version": "market_reaction_divergence_v0.1",
        "policy": (
            "Compares prediction-market/macro expectations with equity ETF price reaction. "
            "This is confirmation context only and does not change the macro filter."
        ),
        "broad_risk": broad,
        "semiconductor": semiconductor,
        "overall": _overall(broad, semiconductor),
        "data_quality": _data_quality(reaction_quotes),
    }


def _broad_risk_reaction(macro_filter: dict, risk_temperature: dict, quotes: Dict[str, Quote]) -> dict:
    qqq = _ret(quotes, "QQQ")
    spy = _ret(quotes, "SPY")
    sqqq = _ret(quotes, "SQQQ")
    tqqq = _ret(quotes, "TQQQ")
    expected = _expected_broad_bias(macro_filter, risk_temperature)
    evidence = {
        "QQQ_day_pct": qqq,
        "SPY_day_pct": spy,
        "TQQQ_day_pct": tqqq,
        "SQQQ_day_pct": sqqq,
    }
    if expected == "RISK_OFF":
        confirmations = [value for value in [qqq, spy, tqqq] if value is not None and value < -0.15]
        inverse_confirms = sqqq is not None and sqqq > 0.15
        status = "CONFIRMED" if len(confirmations) >= 1 or inverse_confirms else "NOT_CONFIRMED"
        interpretation = "Risk-off macro is confirmed by weak broad equity reaction." if status == "CONFIRMED" else "Risk-off macro is not yet confirmed by broad equity price reaction."
    elif expected == "RISK_ON":
        confirmations = [value for value in [qqq, spy, tqqq] if value is not None and value > 0.15]
        inverse_confirms = sqqq is not None and sqqq < -0.15
        status = "CONFIRMED" if len(confirmations) >= 1 or inverse_confirms else "NOT_CONFIRMED"
        interpretation = "Risk-on macro is confirmed by broad equity strength." if status == "CONFIRMED" else "Risk-on macro is not yet confirmed by broad equity price reaction."
    else:
        status = "MIXED_OR_UNAVAILABLE"
        interpretation = "Macro expectation is mixed or unavailable."
    return {
        "expected_bias": expected,
        "reaction_status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def _semiconductor_reaction(semi: dict, prediction_divergence: dict, quotes: Dict[str, Quote]) -> dict:
    qqq = _ret(quotes, "QQQ")
    soxx = _ret(quotes, "SOXX")
    smh = _ret(quotes, "SMH")
    soxl = _ret(quotes, "SOXL")
    soxs = _ret(quotes, "SOXS")
    relative_soxx = _diff(soxx, qqq)
    relative_smh = _diff(smh, qqq)
    expected = semi.get("bias") or "UNKNOWN"
    prediction_quality = _theme_quality(prediction_divergence, {"AI_SEMICONDUCTOR_SENTIMENT", "SEMICONDUCTOR_CAPITAL_MARKETS", "GPU_COMPUTE_PRICE_PRESSURE"})
    evidence = {
        "QQQ_day_pct": qqq,
        "SOXX_day_pct": soxx,
        "SMH_day_pct": smh,
        "SOXL_day_pct": soxl,
        "SOXS_day_pct": soxs,
        "SOXX_minus_QQQ_pct": relative_soxx,
        "SMH_minus_QQQ_pct": relative_smh,
        "prediction_market_quality": prediction_quality,
    }
    if expected == "SOXL_BIAS":
        leadership = [value for value in [relative_soxx, relative_smh] if value is not None and value > 0.25]
        leveraged_confirms = soxl is not None and soxl > 0.4
        status = "CONFIRMED" if leadership or leveraged_confirms else "NOT_CONFIRMED"
        interpretation = "Semiconductor bullish theme has price confirmation." if status == "CONFIRMED" else "Semiconductor bullish theme lacks price confirmation."
    elif expected == "SOXS_BIAS":
        weakness = [value for value in [relative_soxx, relative_smh] if value is not None and value < -0.25]
        inverse_confirms = soxs is not None and soxs > 0.4
        status = "CONFIRMED" if weakness or inverse_confirms else "NOT_CONFIRMED"
        interpretation = "Semiconductor bearish theme has price confirmation." if status == "CONFIRMED" else "Semiconductor bearish theme lacks price confirmation."
    else:
        status = "MIXED_OR_UNAVAILABLE"
        interpretation = "Semiconductor theme has no clear directional expectation."
    return {
        "expected_bias": expected,
        "reaction_status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def _overall(broad: dict, semiconductor: dict) -> dict:
    divergences = []
    confirmations = []
    if broad.get("reaction_status") == "CONFIRMED":
        confirmations.append("BROAD_RISK")
    elif broad.get("reaction_status") == "NOT_CONFIRMED":
        divergences.append("BROAD_RISK")
    if semiconductor.get("reaction_status") == "CONFIRMED":
        confirmations.append("SEMICONDUCTOR")
    elif semiconductor.get("reaction_status") == "NOT_CONFIRMED":
        divergences.append("SEMICONDUCTOR")
    if confirmations and not divergences:
        status = "PRICE_CONFIRMS_EXPECTATIONS"
    elif divergences and not confirmations:
        status = "PRICE_DIVERGES_FROM_EXPECTATIONS"
    elif confirmations and divergences:
        status = "MIXED_CONFIRMATION"
    else:
        status = "INSUFFICIENT_PRICE_CONFIRMATION"
    return {
        "status": status,
        "confirmations": confirmations,
        "divergences": divergences,
    }


def _expected_broad_bias(macro_filter: dict, risk_temperature: dict) -> str:
    if macro_filter.get("regime") == "RISK_OFF" or risk_temperature.get("risk_appetite") == "RISK_OFF":
        return "RISK_OFF"
    if macro_filter.get("regime") == "RISK_ON" or risk_temperature.get("risk_appetite") == "RISK_ON":
        return "RISK_ON"
    return "MIXED"


def _theme_quality(prediction_divergence: dict, themes: set[str]) -> str:
    for item in prediction_divergence.get("themes", []):
        if item.get("theme") in themes and item.get("quality") in {"HIGH", "MEDIUM"}:
            return item["quality"]
    return "LOW_OR_UNAVAILABLE"


def _data_quality(quotes: Dict[str, Quote]) -> dict:
    expected = ["QQQ", "SPY", "TQQQ", "SQQQ", "SOXX", "SMH", "SOXL", "SOXS"]
    ok = [symbol for symbol in expected if quotes.get(symbol) and quotes[symbol].status == "OK"]
    return {
        "ok_symbols": ok,
        "missing_or_bad_symbols": [symbol for symbol in expected if symbol not in ok],
        "note": "Reaction uses latest price vs previous close when available; IBKR should validate live execution signals.",
    }


def _ret(quotes: Dict[str, Quote], symbol: str) -> Optional[float]:
    quote = quotes.get(symbol)
    if quote is None or quote.status != "OK":
        return None
    value = quote.day_return_pct
    return round(value, 4) if value is not None else None


def _diff(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return round(left - right, 4)
