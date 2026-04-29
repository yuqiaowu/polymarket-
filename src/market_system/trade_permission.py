from __future__ import annotations

from typing import Any, Dict, Optional


OPEN_PERMISSIONS = {"BLOCKED", "ALLOWED_SHADOW_ONLY", "ALLOWED"}
DIRECTION_PERMISSIONS = {"NONE", "LONG_ONLY", "SHORT_ONLY", "BOTH"}
POSITION_SIZES = {"NONE", "LIGHT", "NORMAL", "HEAVY"}
MOMENTUM_STATES = {"STRONG", "NEUTRAL", "WEAK", "UNKNOWN"}


def build_rules_trade_permission(report: dict) -> dict:
    macro = report.get("macro_filter", {})
    risk = report.get("risk_temperature", {})
    event_gate = report.get("economic_calendar", {}).get("event_gate", {})
    semi = report.get("semiconductor_direction", {})
    reaction = report.get("market_reaction_divergence", {})
    price_trends = report.get("price_trends", {})
    prediction = report.get("prediction_market_divergence", {})
    draft_action = report.get("draft_action", {}).get("action")
    candidate_layer = report.get("candidate_layer", {})
    candidates = _directional_candidates(candidate_layer)

    hard_blocks = []
    if not macro.get("can_trade", False):
        hard_blocks.append("MACRO_FILTER_BLOCK")
    if event_gate.get("status") == "EVENT_RISK":
        hard_blocks.append("HIGH_IMPACT_EVENT_WINDOW")

    broad = reaction.get("broad_risk", {})
    semi_reaction = reaction.get("semiconductor", {})
    overall = reaction.get("overall", {})
    broad_short = _broad_short_term(macro, risk, broad, price_trends)
    semi_short = _semi_short_term(semi, semi_reaction, price_trends)
    broad_long = _broad_long_term(prediction, risk)
    semi_long = _semi_long_term(semi, prediction)

    if hard_blocks or not candidates:
        return _permission(
            open_permission="BLOCKED",
            direction_permission="NONE",
            position_size="NONE",
            preferred_symbols=[],
            broad_long_term=broad_long,
            broad_short_term=broad_short,
            semiconductor_long_term=semi_long,
            semiconductor_short_term=semi_short,
            hard_blocks=hard_blocks or ["NO_ACTIONABLE_CANDIDATE"],
            warnings=_warnings(macro, risk, reaction, prediction),
            rationale="Hard gate blocks opening a new position.",
        )

    short_setup = broad_short == "WEAK" and broad.get("reaction_status") == "CONFIRMED"
    semi_short_setup = semi_short == "WEAK" and semi_reaction.get("reaction_status") == "NOT_CONFIRMED"
    long_setup = broad_short == "STRONG" and broad.get("reaction_status") == "CONFIRMED"
    semi_long_setup = semi_short == "STRONG" and semi_reaction.get("reaction_status") == "CONFIRMED"

    if short_setup and semi_short_setup:
        direction = "SHORT_ONLY"
        preferred = ["SOXS", "SQQQ"]
        rationale = "Broad risk-off is price-confirmed and semiconductor short-term momentum is weak."
    elif short_setup:
        direction = "SHORT_ONLY"
        preferred = ["SQQQ"]
        rationale = "Broad risk-off is price-confirmed."
    elif long_setup and semi_long_setup:
        direction = "LONG_ONLY"
        preferred = ["SOXL", "TQQQ"]
        rationale = "Broad risk-on and semiconductor strength are price-confirmed."
    elif long_setup:
        direction = "LONG_ONLY"
        preferred = ["TQQQ"]
        rationale = "Broad risk-on is price-confirmed."
    elif overall.get("status") == "MIXED_CONFIRMATION":
        direction = "BOTH"
        preferred = _preferred_from_candidates(candidates)
        rationale = "Signals are mixed; both directions require stricter confirmation."
    else:
        direction = "NONE"
        preferred = []
        rationale = "No clean direction after macro, prediction-market, and price-reaction checks."

    preferred = _filter_preferred_by_candidates(preferred, candidates)
    if direction != "NONE" and not preferred:
        direction = "NONE"
        rationale = "Rule direction had no matching candidate from the candidate layer."

    if direction == "NONE":
        open_permission = "BLOCKED"
        size = "NONE"
    else:
        open_permission = "ALLOWED_SHADOW_ONLY"
        size = _position_size(direction, overall, prediction)

    return _permission(
        open_permission=open_permission,
        direction_permission=direction,
        position_size=size,
        preferred_symbols=preferred,
        broad_long_term=broad_long,
        broad_short_term=broad_short,
        semiconductor_long_term=semi_long,
        semiconductor_short_term=semi_short,
        hard_blocks=[],
        warnings=_warnings(macro, risk, reaction, prediction),
        rationale=rationale,
    )


def validate_trade_permission(value: Any, fallback: dict) -> dict:
    if not isinstance(value, dict):
        return fallback
    return _permission(
        open_permission=_enum(value.get("open_permission"), OPEN_PERMISSIONS, fallback.get("open_permission")),
        direction_permission=_enum(
            value.get("direction_permission"), DIRECTION_PERMISSIONS, fallback.get("direction_permission")
        ),
        position_size=_enum(value.get("position_size"), POSITION_SIZES, fallback.get("position_size")),
        preferred_symbols=_symbols(value.get("preferred_symbols")) or fallback.get("preferred_symbols", []),
        broad_long_term=_enum(value.get("broad_market_long_term"), MOMENTUM_STATES, fallback.get("broad_market_long_term")),
        broad_short_term=_enum(value.get("broad_market_short_term"), MOMENTUM_STATES, fallback.get("broad_market_short_term")),
        semiconductor_long_term=_enum(
            value.get("semiconductor_long_term"), MOMENTUM_STATES, fallback.get("semiconductor_long_term")
        ),
        semiconductor_short_term=_enum(
            value.get("semiconductor_short_term"), MOMENTUM_STATES, fallback.get("semiconductor_short_term")
        ),
        hard_blocks=_string_list(value.get("hard_blocks"), 8) or fallback.get("hard_blocks", []),
        warnings=_string_list(value.get("warnings"), 8) or fallback.get("warnings", []),
        rationale=_short(value.get("rationale") or fallback.get("rationale"), 500),
    )


def _permission(
    open_permission: str,
    direction_permission: str,
    position_size: str,
    preferred_symbols: list[str],
    broad_long_term: str,
    broad_short_term: str,
    semiconductor_long_term: str,
    semiconductor_short_term: str,
    hard_blocks: list[str],
    warnings: list[str],
    rationale: str,
) -> dict:
    return {
        "schema_version": "trade_permission_v0.1",
        "open_permission": open_permission,
        "direction_permission": direction_permission,
        "position_size": position_size,
        "preferred_symbols": preferred_symbols,
        "momentum": {
            "broad_market": {
                "long_term": broad_long_term,
                "short_term": broad_short_term,
            },
            "semiconductor": {
                "long_term": semiconductor_long_term,
                "short_term": semiconductor_short_term,
            },
        },
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "rationale": rationale,
        "allowed_values": {
            "open_permission": sorted(OPEN_PERMISSIONS),
            "direction_permission": sorted(DIRECTION_PERMISSIONS),
            "position_size": sorted(POSITION_SIZES),
            "momentum": sorted(MOMENTUM_STATES),
        },
    }


def _broad_short_term(macro: dict, risk: dict, broad_reaction: dict, price_trends: dict) -> str:
    trend = price_trends.get("groups", {}).get("broad_market", {}).get("trend")
    if macro.get("regime") == "RISK_OFF" and broad_reaction.get("reaction_status") == "CONFIRMED" and trend == "WEAK":
        return "WEAK"
    if macro.get("regime") == "RISK_ON" and broad_reaction.get("reaction_status") == "CONFIRMED" and trend == "STRONG":
        return "STRONG"
    if trend in {"STRONG", "WEAK"} and broad_reaction.get("reaction_status") == "CONFIRMED":
        return "NEUTRAL"
    if trend in {"STRONG", "WEAK"} and macro.get("regime") not in {"RISK_OFF", "RISK_ON"}:
        return trend
    if risk.get("risk_appetite") == "RISK_OFF":
        return "WEAK"
    if risk.get("risk_appetite") == "RISK_ON":
        return "STRONG"
    return "NEUTRAL"


def _semi_short_term(semi: dict, semi_reaction: dict, price_trends: dict) -> str:
    trend = price_trends.get("groups", {}).get("semiconductor", {}).get("trend")
    expected = semi.get("bias")
    status = semi_reaction.get("reaction_status")
    if expected == "SOXL_BIAS" and status == "CONFIRMED" and trend == "STRONG":
        return "STRONG"
    if expected == "SOXL_BIAS" and status == "NOT_CONFIRMED" and trend == "WEAK":
        evidence = semi_reaction.get("evidence", {})
        if (evidence.get("SOXL_day_pct") or 0) < -2 or (evidence.get("SOXX_minus_QQQ_pct") or 0) < -0.5:
            return "WEAK"
        return "NEUTRAL"
    if expected == "SOXL_BIAS" and status == "NOT_CONFIRMED" and trend == "STRONG":
        return "NEUTRAL"
    if expected == "SOXS_BIAS" and status == "CONFIRMED" and trend == "WEAK":
        return "WEAK"
    if trend in {"STRONG", "WEAK"} and expected not in {"SOXL_BIAS", "SOXS_BIAS"}:
        return trend
    return "NEUTRAL"


def _broad_long_term(prediction: dict, risk: dict) -> str:
    if risk.get("fed_policy_bias") == "HAWKISH" and prediction.get("one_sided_consensus"):
        return "NEUTRAL"
    return "UNKNOWN"


def _semi_long_term(semi: dict, prediction: dict) -> str:
    if semi.get("bias") == "SOXL_BIAS":
        for item in prediction.get("themes", []):
            if item.get("theme") in {"AI_SEMICONDUCTOR_SENTIMENT", "SEMICONDUCTOR_CAPITAL_MARKETS"}:
                if item.get("quality") == "MEDIUM":
                    return "STRONG"
        return "NEUTRAL"
    if semi.get("bias") == "SOXS_BIAS":
        return "WEAK"
    return "UNKNOWN"


def _position_size(direction: str, overall: dict, prediction: dict) -> str:
    if direction == "NONE":
        return "NONE"
    if overall.get("status") == "PRICE_CONFIRMS_EXPECTATIONS" and not prediction.get("data_gaps"):
        return "NORMAL"
    return "LIGHT"


def _warnings(macro: dict, risk: dict, reaction: dict, prediction: dict) -> list[str]:
    output = []
    output.extend(macro.get("warnings") or [])
    output.extend(risk.get("warnings") or [])
    if reaction.get("overall", {}).get("status") == "MIXED_CONFIRMATION":
        output.append("PRICE_REACTION_MIXED_CONFIRMATION")
    if prediction.get("one_sided_consensus"):
        output.append("PREDICTION_MARKET_ONE_SIDED_CONFIRMATION")
    return output[:8]


def _preferred_from_draft(action: Optional[str]) -> list[str]:
    mapping = {
        "CONSIDER_TQQQ": ["TQQQ"],
        "CONSIDER_SQQQ": ["SQQQ"],
        "CONSIDER_SOXL": ["SOXL"],
        "CONSIDER_SOXS": ["SOXS"],
    }
    return mapping.get(action or "", [])


def _directional_candidates(candidate_layer: dict) -> list[dict]:
    if not isinstance(candidate_layer, dict):
        return []
    candidates = candidate_layer.get("candidates")
    if not isinstance(candidates, list):
        return []
    output = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("action") == "NO_TRADE":
            continue
        if item.get("target_symbol") not in {"TQQQ", "SQQQ", "SOXL", "SOXS"}:
            continue
        output.append(item)
    return output


def _preferred_from_candidates(candidates: list[dict]) -> list[str]:
    return _unique_symbols([item.get("target_symbol") for item in candidates])


def _filter_preferred_by_candidates(preferred: list[str], candidates: list[dict]) -> list[str]:
    candidate_symbols = set(_preferred_from_candidates(candidates))
    if not preferred:
        return []
    return [symbol for symbol in preferred if symbol in candidate_symbols]


def _unique_symbols(symbols: list[Optional[str]]) -> list[str]:
    output = []
    seen = set()
    for symbol in symbols:
        if symbol not in {"TQQQ", "SQQQ", "SOXL", "SOXS"} or symbol in seen:
            continue
        seen.add(symbol)
        output.append(symbol)
    return output


def _enum(value: Any, allowed: set[str], fallback: Optional[str]) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    if fallback in allowed:
        return fallback
    return sorted(allowed)[0]


def _symbols(value: Any) -> list[str]:
    allowed = {"TQQQ", "SQQQ", "SOXL", "SOXS"}
    if not isinstance(value, list):
        return []
    return [item for item in value[:4] if isinstance(item, str) and item in allowed]


def _string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_short(item, 120) for item in value[:limit] if isinstance(item, str) and item.strip()]


def _short(value: Any, max_len: int = 120) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]
