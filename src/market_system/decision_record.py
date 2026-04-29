from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _stable_id(prefix: str, parts: List[str]) -> str:
    raw = "|".join(parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def build_trade_decision_record(report: dict) -> dict:
    generated_at = report["generated_at_utc"]
    cycle_id = _stable_id("cycle", [generated_at[:16], report["draft_action"]["action"]])
    candidate_layer = report.get("candidate_layer", {})
    primary_candidate = candidate_layer.get("primary_candidate") or {}
    action = primary_candidate.get("action") or report["draft_action"]["action"]
    target_symbol = primary_candidate.get("target_symbol") or _target_symbol(action)
    decision_id = _stable_id("decision", [cycle_id, action, target_symbol or "NONE"])
    macro = report.get("macro_filter", {})
    risk_temperature = report.get("risk_temperature", {})
    calendar_gate = report.get("economic_calendar", {}).get("event_gate", {})
    insight = report.get("insight", {})
    permission = report.get("trade_permission", {})

    hard_blocks = []
    if not macro.get("can_trade", False):
        hard_blocks.append("MACRO_FILTER_BLOCK")
    if calendar_gate.get("status") == "EVENT_RISK":
        hard_blocks.append("HIGH_IMPACT_EVENT_WINDOW")
    if action == "NO_TRADE":
        hard_blocks.append("NO_ACTIONABLE_SIGNAL")
    if primary_candidate.get("rule_status") == "BLOCKED":
        hard_blocks.extend(primary_candidate.get("rule_blocks") or [])

    return {
        "cycleId": cycle_id,
        "decisionId": decision_id,
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "mode": "shadow",
        "candidate": {
            "candidateId": primary_candidate.get("candidate_id"),
            "strategy": primary_candidate.get("strategy"),
            "action": action,
            "targetSymbol": target_symbol,
            "confidence": primary_candidate.get("confidence"),
            "ruleStatus": primary_candidate.get("rule_status"),
            "reason": primary_candidate.get("thesis") or report["draft_action"]["reason"],
            "evidence": primary_candidate.get("evidence") or [],
            "invalidations": primary_candidate.get("invalidations") or [],
        },
        "inputs": {
            "macroRegime": macro.get("regime"),
            "macroRiskScore": macro.get("risk_appetite_score"),
            "riskTemperatureScore": risk_temperature.get("score"),
            "riskTemperatureLabel": risk_temperature.get("label"),
            "fedPolicyBias": risk_temperature.get("fed_policy_bias"),
            "semiconductorBias": report.get("semiconductor_direction", {}).get("bias"),
            "eventGate": calendar_gate.get("status"),
            "insightRecommendedAction": insight.get("recommended_action"),
            "insightConfidence": insight.get("confidence"),
            "openPermission": permission.get("open_permission"),
            "directionPermission": permission.get("direction_permission"),
            "positionSize": permission.get("position_size"),
            "candidateCount": candidate_layer.get("summary", {}).get("candidate_count"),
            "allowedCandidateCount": candidate_layer.get("rule_review", {}).get("allowed_candidate_count"),
        },
        "insight": {
            "schemaVersion": insight.get("schema_version"),
            "engine": insight.get("engine"),
            "llmUsed": insight.get("llm_used"),
            "recommendedAction": insight.get("recommended_action"),
            "confidence": insight.get("confidence"),
            "convergenceCount": len(insight.get("convergences", [])),
            "divergenceCount": len(insight.get("divergences", [])),
        },
        "riskReview": {
            "allowed": not hard_blocks and permission.get("open_permission") in {"ALLOWED", "ALLOWED_SHADOW_ONLY"},
            "hardBlocks": _unique(hard_blocks + (permission.get("hard_blocks") or [])),
            "warnings": _unique(
                (macro.get("warnings") or [])
                + (risk_temperature.get("warnings") or [])
                + (permission.get("warnings") or [])
            ),
        },
        "execution": {
            "status": "SHADOW_ONLY",
            "submitted": False,
            "broker": None,
            "orderId": None,
            "filledQuantity": 0,
            "averageFillPrice": None,
        },
    }


def _target_symbol(action: str) -> Optional[str]:
    mapping: Dict[str, str] = {
        "CONSIDER_TQQQ": "TQQQ",
        "CONSIDER_SQQQ": "SQQQ",
        "CONSIDER_SOXL": "SOXL",
        "CONSIDER_SOXS": "SOXS",
    }
    return mapping.get(action)


def _unique(items: List[str]) -> List[str]:
    output = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
