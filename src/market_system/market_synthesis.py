from __future__ import annotations

from typing import Any, Dict, List, Optional

from .shared_schema import is_core_theme


def build_synthesis(report: dict) -> dict:
    macro = report.get("macro_filter", {})
    risk = report.get("risk_temperature", {})
    event_gate = report.get("economic_calendar", {}).get("event_gate", {})
    discovery = report.get("market_discovery", {}).get("directions", {})
    prediction_divergence = report.get("prediction_market_divergence", {})
    reaction_divergence = report.get("market_reaction_divergence", {})
    themes = report.get("theme_scores", [])
    semi = report.get("semiconductor_direction", {})

    fed_consensus = _fed_decision_consensus(discovery.get("US_EQUITY_MARKET", {}))
    theme_synthesis = _theme_synthesis(themes, semi)
    macro_synthesis = _macro_synthesis(macro, risk, event_gate, fed_consensus)
    conflicts = _conflicts(macro_synthesis, theme_synthesis, fed_consensus, prediction_divergence, reaction_divergence)

    return {
        "schema_version": "synthesis_v0.1",
        "macro": macro_synthesis,
        "fed_decision_consensus": fed_consensus,
        "themes": theme_synthesis,
        "prediction_market_divergence_summary": _prediction_divergence_summary(prediction_divergence),
        "market_reaction_summary": _reaction_summary(reaction_divergence),
        "conflicts": conflicts,
        "integrated_conclusion": _integrated_conclusion(macro_synthesis, theme_synthesis, fed_consensus, conflicts),
    }


def _fed_decision_consensus(payload: dict) -> dict:
    buckets = {
        "NO_CHANGE": None,
        "CUT_25": None,
        "CUT_50_PLUS": None,
        "HIKE_25_PLUS": None,
    }
    evidence = []
    for market in payload.get("top_markets", []):
        title = market.get("title", "")
        if market.get("quality_grade") not in {"A", "B"}:
            continue
        bucket = _classify_fed_decision(title)
        if not bucket:
            continue
        probability = market.get("probability")
        buckets[bucket] = probability
        evidence.append(
            {
                "bucket": bucket,
                "title": title,
                "probability": probability,
                "quality_grade": market.get("quality_grade"),
                "market_id": market.get("market_id"),
            }
        )

    no_change = _safe_float(buckets["NO_CHANGE"])
    if no_change is not None and no_change >= 0.9:
        consensus = "NO_CHANGE_PRICED_IN"
        surprise_risk = "LOW_FOR_RATE_DECISION"
        guidance_risk = "MEDIUM"
    elif no_change is not None:
        consensus = "NO_STRONG_CONSENSUS"
        surprise_risk = "MEDIUM"
        guidance_risk = "MEDIUM"
    else:
        consensus = "UNAVAILABLE"
        surprise_risk = "UNKNOWN"
        guidance_risk = "UNKNOWN"

    return {
        "consensus": consensus,
        "no_change_probability": no_change,
        "cut_25_probability": _safe_float(buckets["CUT_25"]),
        "cut_50_plus_probability": _safe_float(buckets["CUT_50_PLUS"]),
        "hike_25_plus_probability": _safe_float(buckets["HIKE_25_PLUS"]),
        "decision_surprise_risk": surprise_risk,
        "guidance_risk": guidance_risk,
        "evidence_count": len(evidence),
        "evidence": evidence,
        "interpretation": _fed_interpretation(consensus),
    }


def _macro_synthesis(macro: dict, risk: dict, event_gate: dict, fed: dict) -> dict:
    regime = macro.get("regime")
    risk_appetite = risk.get("risk_appetite")
    fed_bias = risk.get("fed_policy_bias")
    event_status = event_gate.get("status")
    if regime == "RISK_OFF" and risk_appetite == "RISK_OFF":
        stance = "MACRO_RISK_OFF"
    elif regime == "RISK_ON" and risk_appetite == "RISK_ON":
        stance = "MACRO_RISK_ON"
    else:
        stance = "MACRO_MIXED"

    return {
        "stance": stance,
        "macro_regime": regime,
        "risk_temperature": risk.get("label"),
        "fed_policy_bias": fed_bias,
        "event_gate": event_status,
        "decision_consensus": fed.get("consensus"),
        "dominant_drivers": _dominant_macro_drivers(risk),
        "interpretation": _macro_interpretation(stance, fed_bias, fed.get("consensus")),
    }


def _theme_synthesis(themes: List[dict], semi: dict) -> dict:
    core = [
        theme
        for theme in themes
        if is_core_theme(theme.get("status"), float(theme.get("confidence") or 0.0), int(theme.get("evidence_count") or 0))
    ]
    watch = [theme for theme in themes if theme not in core]
    bullish = [theme for theme in core if "BULLISH" in str(theme.get("interpretation"))]
    bearish = [theme for theme in core if "BEARISH" in str(theme.get("interpretation"))]
    return {
        "semiconductor_bias": semi.get("bias"),
        "core_theme_count": len(core),
        "watch_theme_count": len(watch),
        "bullish_core_themes": [theme.get("theme") for theme in bullish],
        "bearish_core_themes": [theme.get("theme") for theme in bearish],
        "watch_themes": [theme.get("theme") for theme in watch],
        "interpretation": _theme_interpretation(semi.get("bias"), bullish, bearish, watch),
    }


def _conflicts(macro: dict, themes: dict, fed: dict, prediction_divergence: dict, reaction_divergence: dict) -> List[dict]:
    output = []
    if macro.get("stance") == "MACRO_RISK_OFF" and themes.get("semiconductor_bias") == "SOXL_BIAS":
        output.append(
            {
                "kind": "MACRO_VS_SEMICONDUCTOR",
                "summary": "Macro/rates/volatility are risk-off while semiconductor narrative remains constructive.",
                "effect": "Do not chase leveraged semiconductor long exposure until macro confirms.",
            }
        )
    if fed.get("consensus") == "NO_CHANGE_PRICED_IN" and macro.get("fed_policy_bias") == "HAWKISH":
        output.append(
            {
                "kind": "FED_DECISION_VS_RATES_REACTION",
                "summary": "Prediction markets price no rate change, but rates proxy is hawkish.",
                "effect": "Focus on guidance/rate-market reaction rather than the binary FOMC decision.",
            }
        )
    for item in prediction_divergence.get("cross_platform_divergences", [])[:3]:
        output.append(
            {
                "kind": "PREDICTION_MARKET_CROSS_PLATFORM_GAP",
                "summary": f"{item.get('theme')} has a cross-platform probability gap.",
                "effect": "Treat the theme as unresolved until higher-quality price or market confirmation appears.",
            }
        )
    reaction_status = reaction_divergence.get("overall", {}).get("status")
    if reaction_status in {"PRICE_DIVERGES_FROM_EXPECTATIONS", "MIXED_CONFIRMATION"}:
        output.append(
            {
                "kind": "PRICE_REACTION_DIVERGENCE",
                "summary": f"ETF price reaction status is {reaction_status}.",
                "effect": "Downgrade conviction or require confirmation before escalating from shadow mode.",
            }
        )
    return output


def _prediction_divergence_summary(payload: dict) -> dict:
    return {
        "theme_count": payload.get("theme_count", 0),
        "cross_platform_divergence_count": len(payload.get("cross_platform_divergences", [])),
        "one_sided_consensus_count": len(payload.get("one_sided_consensus", [])),
        "data_gaps": payload.get("data_gaps", [])[:5],
    }


def _reaction_summary(payload: dict) -> dict:
    return {
        "overall_status": payload.get("overall", {}).get("status"),
        "broad_risk_status": payload.get("broad_risk", {}).get("reaction_status"),
        "semiconductor_status": payload.get("semiconductor", {}).get("reaction_status"),
        "confirmations": payload.get("overall", {}).get("confirmations", []),
        "divergences": payload.get("overall", {}).get("divergences", []),
    }


def _integrated_conclusion(macro: dict, themes: dict, fed: dict, conflicts: List[dict]) -> dict:
    if macro.get("stance") == "MACRO_RISK_OFF":
        posture = "DEFENSIVE"
        action_bias = "SQQQ_OVER_LONG_LEVERAGE"
    elif macro.get("stance") == "MACRO_RISK_ON" and themes.get("semiconductor_bias") == "SOXL_BIAS":
        posture = "RISK_ON_SEMICONDUCTOR"
        action_bias = "SOXL_CANDIDATE"
    elif macro.get("stance") == "MACRO_RISK_ON":
        posture = "RISK_ON_BROAD"
        action_bias = "TQQQ_CANDIDATE"
    else:
        posture = "WAIT_FOR_CONFIRMATION"
        action_bias = "NO_TRADE_OR_SMALL_SHADOW_ONLY"

    return {
        "posture": posture,
        "action_bias": action_bias,
        "summary": _summary_sentence(macro, themes, fed, conflicts),
        "what_would_change_view": [
            "Macro stance changes to MACRO_RISK_ON.",
            "VIX falls back into calm range while rates stop tightening.",
            "Prediction-market semiconductor discovery finds A/B evidence that confirms fixed themes.",
            "Post-FOMC guidance/rates reaction confirms dovish or risk-on repricing.",
        ],
    }


def _classify_fed_decision(title: str) -> Optional[str]:
    lower = title.lower()
    if "fed" not in lower and "interest rates" not in lower:
        return None
    if "no change" in lower:
        return "NO_CHANGE"
    if "decrease" in lower and "50" in lower:
        return "CUT_50_PLUS"
    if "decrease" in lower and "25" in lower:
        return "CUT_25"
    if "increase" in lower and "25" in lower:
        return "HIKE_25_PLUS"
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _dominant_macro_drivers(risk: dict) -> List[str]:
    components = sorted(risk.get("components", []), key=lambda item: abs(item.get("points", 0)), reverse=True)
    return [item.get("name") for item in components[:4] if item.get("name")]


def _fed_interpretation(consensus: str) -> str:
    if consensus == "NO_CHANGE_PRICED_IN":
        return "The rate decision itself is not the main surprise channel; guidance and rate-market reaction matter more."
    if consensus == "NO_STRONG_CONSENSUS":
        return "Prediction markets do not show a clean policy consensus."
    return "No reliable Fed decision consensus was found from accepted prediction markets."


def _macro_interpretation(stance: str, fed_bias: str, decision_consensus: str) -> str:
    if stance == "MACRO_RISK_OFF" and fed_bias == "HAWKISH":
        return "Macro inputs point to defensive positioning, driven by hawkish rates pressure and volatility."
    if stance == "MACRO_RISK_ON":
        return "Macro inputs support risk exposure, subject to event and data-quality gates."
    if decision_consensus == "NO_CHANGE_PRICED_IN":
        return "Policy decision is priced, so market reaction should be read through rates, volatility, and dollar moves."
    return "Macro inputs are mixed."


def _theme_interpretation(semi_bias: str, bullish: List[dict], bearish: List[dict], watch: List[dict]) -> str:
    if semi_bias == "SOXL_BIAS" and bullish:
        return "Core semiconductor themes are constructive, but watchlist themes should not be treated as confirmation."
    if bearish:
        return "Core semiconductor themes include bearish pressure."
    if watch:
        return "Theme evidence exists but much of it is watch-only."
    return "No strong theme evidence."


def _summary_sentence(macro: dict, themes: dict, fed: dict, conflicts: List[dict]) -> str:
    if macro.get("stance") == "MACRO_RISK_OFF":
        return (
            "Effective evidence favors a defensive macro posture: rates/volatility are risk-off, "
            "Fed no-change is already priced, and constructive semiconductor themes are not enough to override macro pressure."
        )
    if macro.get("stance") == "MACRO_RISK_ON" and themes.get("semiconductor_bias") == "SOXL_BIAS":
        return "Effective evidence favors risk-on with semiconductor leadership, assuming event gates remain open."
    if conflicts:
        return "Effective evidence is conflicted; wait for macro and theme confirmation to align."
    return "Effective evidence does not produce a strong directional conclusion."
