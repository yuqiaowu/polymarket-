from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, List, Optional


CANDIDATE_SCHEMA_VERSION = "candidate_layer_v0.1"
ALLOWED_CANDIDATE_ACTIONS = {"NO_TRADE", "CONSIDER_TQQQ", "CONSIDER_SQQQ", "CONSIDER_SOXL", "CONSIDER_SOXS"}


@dataclass
class TradeCandidate:
    candidate_id: str
    strategy: str
    action: str
    target_symbol: Optional[str]
    direction: str
    thesis: str
    confidence: float
    evidence: List[str]
    invalidations: List[str]
    tags: List[str]
    rule_status: str = "PENDING_RULE_REVIEW"
    rule_blocks: Optional[List[str]] = None
    rule_warnings: Optional[List[str]] = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["rule_blocks"] = self.rule_blocks or []
        value["rule_warnings"] = self.rule_warnings or []
        return value


def build_candidate_layer(report: dict) -> dict:
    candidates = []
    candidates.extend(_macro_risk_candidates(report))
    candidates.extend(_semiconductor_candidates(report))
    candidates.extend(_synthesis_candidates(report))
    candidates.extend(_price_confirmation_candidates(report))

    deduped = _dedupe_candidates(candidates)
    if not deduped:
        deduped = [
            _candidate(
                strategy="no_trade_default",
                action="NO_TRADE",
                thesis="No strategy produced a directional candidate.",
                confidence=0.1,
                evidence=["macro_filter", "risk_temperature", "theme_scores", "price_trends"],
                invalidations=["A strategy produces a higher-confidence directional candidate."],
                tags=["fallback"],
            )
        ]
    ranked = sorted(deduped, key=lambda item: (item.action != "NO_TRADE", item.confidence), reverse=True)
    primary = ranked[0]
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "policy": (
            "Strategies generate trade candidates only. Rule engine and risk review decide whether a candidate "
            "is allowed, blocked, or shadow-only."
        ),
        "strategies": sorted({item.strategy for item in ranked}),
        "primary_candidate_id": primary.candidate_id,
        "primary_candidate": primary.to_dict(),
        "candidates": [item.to_dict() for item in ranked],
        "summary": {
            "candidate_count": len(ranked),
            "directional_candidate_count": len([item for item in ranked if item.action != "NO_TRADE"]),
            "top_action": primary.action,
            "top_symbol": primary.target_symbol,
        },
    }


def candidate_to_draft_action(candidate_layer: dict) -> dict:
    primary = candidate_layer.get("primary_candidate") or {}
    action = primary.get("action")
    if action not in ALLOWED_CANDIDATE_ACTIONS:
        action = "NO_TRADE"
    rule_blocks = primary.get("rule_blocks") or []
    rule_status = primary.get("rule_status")
    reason = primary.get("thesis") or "No candidate thesis available."
    if rule_status in {"BLOCKED", "FILTERED"}:
        suffix = f" Rule status: {rule_status}."
        if rule_blocks:
            suffix += f" Blocks: {', '.join(rule_blocks[:4])}."
        reason = reason + suffix
    return {
        "action": action,
        "reason": reason,
        "candidate_id": primary.get("candidate_id"),
        "strategy": primary.get("strategy"),
        "rule_status": rule_status,
    }


def apply_rule_review(candidate_layer: dict, trade_permission: dict) -> dict:
    if not isinstance(candidate_layer, dict):
        return candidate_layer
    direction_permission = trade_permission.get("direction_permission")
    open_permission = trade_permission.get("open_permission")
    preferred = set(trade_permission.get("preferred_symbols") or [])
    hard_blocks = list(trade_permission.get("hard_blocks") or [])
    warnings = list(trade_permission.get("warnings") or [])

    reviewed = []
    for raw in candidate_layer.get("candidates", []):
        item = dict(raw)
        blocks = list(hard_blocks)
        symbol = item.get("target_symbol")
        action = item.get("action")
        if action == "NO_TRADE":
            status = "BLOCKED"
            blocks = _unique(blocks + ["NO_DIRECTIONAL_CANDIDATE"])
        elif open_permission == "BLOCKED":
            status = "BLOCKED"
        elif direction_permission not in {"LONG_ONLY", "BOTH"}:
            status = "BLOCKED"
            blocks = _unique(blocks + ["DIRECTION_NOT_PERMITTED"])
        elif preferred and symbol not in preferred:
            status = "FILTERED"
            blocks = _unique(blocks + ["SYMBOL_NOT_PREFERRED_BY_RULES"])
        else:
            status = "ALLOWED_SHADOW_ONLY" if open_permission == "ALLOWED_SHADOW_ONLY" else "ALLOWED"
        item["rule_status"] = status
        item["rule_blocks"] = blocks
        item["rule_warnings"] = warnings
        reviewed.append(item)

    primary = _select_reviewed_primary(reviewed)
    output = dict(candidate_layer)
    output["candidates"] = reviewed
    output["primary_candidate"] = primary
    output["primary_candidate_id"] = primary.get("candidate_id")
    output["rule_review"] = {
        "open_permission": open_permission,
        "direction_permission": direction_permission,
        "preferred_symbols": sorted(preferred),
        "allowed_candidate_count": len([item for item in reviewed if item.get("rule_status") in {"ALLOWED", "ALLOWED_SHADOW_ONLY"}]),
        "blocked_candidate_count": len([item for item in reviewed if item.get("rule_status") == "BLOCKED"]),
        "filtered_candidate_count": len([item for item in reviewed if item.get("rule_status") == "FILTERED"]),
    }
    return output


def _macro_risk_candidates(report: dict) -> List[TradeCandidate]:
    macro = report.get("macro_filter", {})
    risk = report.get("risk_temperature", {})
    if macro.get("regime") == "RISK_ON" and risk.get("risk_appetite") == "RISK_ON":
        return [
            _candidate(
                strategy="macro_risk_regime",
                action="CONSIDER_TQQQ",
                thesis="Macro filter and risk temperature both support broad risk-on exposure.",
                confidence=_confidence_from_score(0.48, risk.get("score"), positive=True),
                evidence=["macro_filter.regime", "risk_temperature.risk_appetite", "risk_temperature.components"],
                invalidations=["macro_filter.regime turns RISK_OFF", "risk_temperature.risk_appetite turns RISK_OFF"],
                tags=["broad_market", "macro"],
            )
        ]
    if macro.get("regime") == "RISK_OFF" and risk.get("risk_appetite") == "RISK_OFF":
        return [
            _candidate(
                strategy="macro_risk_regime",
                action="CONSIDER_SQQQ",
                thesis="Macro filter and risk temperature both support broad risk-off exposure.",
                confidence=_confidence_from_score(0.48, risk.get("score"), positive=False),
                evidence=["macro_filter.regime", "risk_temperature.risk_appetite", "risk_temperature.components"],
                invalidations=["macro_filter.regime turns RISK_ON", "risk_temperature.risk_appetite turns RISK_ON"],
                tags=["broad_market", "macro"],
            )
        ]
    return []


def _semiconductor_candidates(report: dict) -> List[TradeCandidate]:
    semi = report.get("semiconductor_direction", {})
    risk = report.get("risk_temperature", {})
    score = int(semi.get("semiconductor_score") or 0)
    risk_penalty = 0.1 if risk.get("risk_appetite") == "RISK_OFF" else 0.0
    if semi.get("bias") == "SOXL_BIAS":
        return [
            _candidate(
                strategy="semiconductor_theme",
                action="CONSIDER_SOXL",
                thesis="Semiconductor theme score supports leveraged semiconductor long exposure candidate.",
                confidence=max(0.2, min(0.8, 0.45 + 0.1 * score - risk_penalty)),
                evidence=["semiconductor_direction", "theme_scores", "market_reaction_divergence.semiconductor"],
                invalidations=["semiconductor_direction.bias changes away from SOXL_BIAS", "semiconductor price reaction fails confirmation"],
                tags=["semiconductor", "theme"],
            )
        ]
    if semi.get("bias") == "SOXS_BIAS":
        return [
            _candidate(
                strategy="semiconductor_theme",
                action="CONSIDER_SOXS",
                thesis="Semiconductor theme score supports leveraged semiconductor short exposure candidate.",
                confidence=max(0.2, min(0.8, 0.45 + 0.1 * abs(score))),
                evidence=["semiconductor_direction", "theme_scores", "market_reaction_divergence.semiconductor"],
                invalidations=["semiconductor_direction.bias changes away from SOXS_BIAS", "semiconductor price reaction confirms strength"],
                tags=["semiconductor", "theme"],
            )
        ]
    return []


def _synthesis_candidates(report: dict) -> List[TradeCandidate]:
    conclusion = report.get("synthesis", {}).get("integrated_conclusion", {})
    action_bias = conclusion.get("action_bias")
    mapping = {
        "SOXL_CANDIDATE": "CONSIDER_SOXL",
        "TQQQ_CANDIDATE": "CONSIDER_TQQQ",
        "SQQQ_OVER_LONG_LEVERAGE": "CONSIDER_SQQQ",
    }
    action = mapping.get(action_bias)
    if not action:
        return []
    return [
        _candidate(
            strategy="integrated_synthesis",
            action=action,
            thesis=conclusion.get("summary") or f"Synthesis action bias is {action_bias}.",
            confidence=0.5,
            evidence=["synthesis.integrated_conclusion", "synthesis.conflicts"],
            invalidations=conclusion.get("what_would_change_view", [])[:4] or ["Synthesis posture changes."],
            tags=["synthesis"],
        )
    ]


def _price_confirmation_candidates(report: dict) -> List[TradeCandidate]:
    trends = report.get("price_trends", {}).get("groups", {})
    candidates = []
    broad = trends.get("broad_market", {})
    semi = trends.get("semiconductor", {})
    if broad.get("trend") == "STRONG":
        candidates.append(
            _candidate(
                strategy="price_trend_confirmation",
                action="CONSIDER_TQQQ",
                thesis="Broad-market daily trend is strong.",
                confidence=0.42,
                evidence=["price_trends.groups.broad_market"],
                invalidations=["price_trends.groups.broad_market.trend weakens to NEUTRAL or WEAK"],
                tags=["price_confirmation", "broad_market"],
            )
        )
    elif broad.get("trend") == "WEAK":
        candidates.append(
            _candidate(
                strategy="price_trend_confirmation",
                action="CONSIDER_SQQQ",
                thesis="Broad-market daily trend is weak.",
                confidence=0.42,
                evidence=["price_trends.groups.broad_market"],
                invalidations=["price_trends.groups.broad_market.trend improves to NEUTRAL or STRONG"],
                tags=["price_confirmation", "broad_market"],
            )
        )
    if semi.get("trend") == "STRONG":
        candidates.append(
            _candidate(
                strategy="price_trend_confirmation",
                action="CONSIDER_SOXL",
                thesis="Semiconductor daily trend is strong.",
                confidence=0.42,
                evidence=["price_trends.groups.semiconductor"],
                invalidations=["price_trends.groups.semiconductor.trend weakens to NEUTRAL or WEAK"],
                tags=["price_confirmation", "semiconductor"],
            )
        )
    elif semi.get("trend") == "WEAK":
        candidates.append(
            _candidate(
                strategy="price_trend_confirmation",
                action="CONSIDER_SOXS",
                thesis="Semiconductor daily trend is weak.",
                confidence=0.42,
                evidence=["price_trends.groups.semiconductor"],
                invalidations=["price_trends.groups.semiconductor.trend improves to NEUTRAL or STRONG"],
                tags=["price_confirmation", "semiconductor"],
            )
        )
    return candidates


def _candidate(
    strategy: str,
    action: str,
    thesis: str,
    confidence: float,
    evidence: List[str],
    invalidations: List[str],
    tags: List[str],
) -> TradeCandidate:
    symbol = _target_symbol(action)
    candidate_id = _stable_id("cand", [strategy, action, thesis])
    return TradeCandidate(
        candidate_id=candidate_id,
        strategy=strategy,
        action=action,
        target_symbol=symbol,
        direction="LONG" if symbol else "NONE",
        thesis=thesis,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        evidence=evidence[:8],
        invalidations=invalidations[:8],
        tags=tags[:8],
    )


def _dedupe_candidates(candidates: List[TradeCandidate]) -> List[TradeCandidate]:
    by_action: dict[str, TradeCandidate] = {}
    for item in candidates:
        existing = by_action.get(item.action)
        if existing is None or item.confidence > existing.confidence:
            by_action[item.action] = item
    return list(by_action.values())


def _select_reviewed_primary(candidates: List[dict]) -> dict:
    allowed = [item for item in candidates if item.get("rule_status") in {"ALLOWED", "ALLOWED_SHADOW_ONLY"}]
    pool = allowed or candidates
    if not pool:
        return {}
    return sorted(pool, key=lambda item: (item.get("rule_status") in {"ALLOWED", "ALLOWED_SHADOW_ONLY"}, float(item.get("confidence") or 0)), reverse=True)[0]


def _confidence_from_score(base: float, raw_score: Any, positive: bool) -> float:
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = 0
    directional_score = score if positive else -score
    return max(0.2, min(0.85, base + 0.08 * max(0, directional_score)))


def _target_symbol(action: str) -> Optional[str]:
    mapping = {
        "CONSIDER_TQQQ": "TQQQ",
        "CONSIDER_SQQQ": "SQQQ",
        "CONSIDER_SOXL": "SOXL",
        "CONSIDER_SOXS": "SOXS",
    }
    return mapping.get(action)


def _stable_id(prefix: str, parts: List[str]) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _unique(items: List[str]) -> List[str]:
    output = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
