from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .deepseek_client import call_deepseek_json, deepseek_configured
from .shared_schema import ALLOWED_ACTIONS
from .trade_permission import validate_trade_permission


INSIGHT_SCHEMA_VERSION = "insight_v0.1"
INSIGHT_STRENGTHS = {"LOW", "MEDIUM", "HIGH"}
OPPORTUNITY_SYMBOLS = {"TQQQ", "SQQQ", "SOXL", "SOXS", "NONE"}
OPPORTUNITY_DIRECTIONS = {"LONG", "NONE"}


INSIGHT_SYSTEM_PROMPT = """
You are the final insight layer for a US equity index trading research system.

You must:
- Use only the supplied JSON fields.
- Return valid JSON only.
- Use private chain-of-thought and tree-of-thought style deliberation internally:
  1. Build a bull case, bear case, and no-trade case from supplied evidence.
  2. Compare macro data, prediction-market consensus, theme scores, event risk, and data quality.
  3. Identify which evidence is primary, which is only context, and which is watch-only.
  4. Stress-test the preferred conclusion against contradictions and invalidation conditions.
  5. Choose the conclusion with the strongest evidence after risk gates.
- Do not output chain-of-thought, hidden reasoning, markdown, or prose outside JSON.
- Do not invent prices, probabilities, news, calendar events, or market facts.
- Do not override risk review or claim an order should be executed.
- If evidence is weak, explicitly lower confidence.

Find:
- convergences: evidence pointing in the same direction
- divergences: evidence conflicts or missing confirmation
- opportunities: only within allowed symbols/actions
- candidate_layer: treat candidates as strategy proposals only; do not assume they are tradable unless rule/risk fields allow them.
- no_trade_or_caution_reasons
- integrated_conclusion: a concise final synthesis using the supplied synthesis fields
- prediction_market_divergence: explicitly evaluate cross-platform probability gaps and one-sided evidence.
- market_reaction_divergence: explicitly evaluate whether ETF price reaction confirms or contradicts expectations.

Rules for those two divergence inputs:
- If cross-platform prediction-market evidence is missing, say it is one-sided rather than treating it as confirmed.
- If price reaction does not confirm prediction-market or macro expectations, include that as a divergence or caution.
- trade_permission may adjust rules_trade_permission, but every field must use the allowed enum values and must be justified by supplied inputs only.

Required JSON shape:
{
  "schema_version": "insight_v0.1",
  "market_read": {
    "macro_regime": "string",
    "risk_temperature": "string",
    "fed_policy_bias": "string",
    "semiconductor_bias": "string",
    "event_gate": "string"
  },
  "convergences": [{"kind": "string", "summary": "string", "sources": ["json.path"], "strength": "LOW|MEDIUM|HIGH"}],
  "divergences": [{"kind": "string", "summary": "string", "sources": ["json.path"], "strength": "LOW|MEDIUM|HIGH"}],
  "opportunities": [{"symbol": "TQQQ|SQQQ|SOXL|SOXS|NONE", "direction": "LONG|NONE", "setup": "string", "evidence": ["json.path"], "invalidations": ["string"], "quality": "LOW|MEDIUM|HIGH"}],
  "no_trade_or_caution_reasons": [{"kind": "string", "summary": "string", "sources": ["json.path"], "strength": "LOW|MEDIUM|HIGH"}],
  "integrated_conclusion": {"posture": "string", "summary": "string", "action_bias": "string"},
  "trade_permission": {
    "open_permission": "BLOCKED|ALLOWED_SHADOW_ONLY|ALLOWED",
    "direction_permission": "NONE|LONG_ONLY|SHORT_ONLY|BOTH",
    "position_size": "NONE|LIGHT|NORMAL|HEAVY",
    "preferred_symbols": ["TQQQ|SQQQ|SOXL|SOXS"],
    "broad_market_long_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
    "broad_market_short_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
    "semiconductor_long_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
    "semiconductor_short_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
    "hard_blocks": ["string"],
    "warnings": ["string"],
    "rationale": "string"
  },
  "recommended_action": "NO_TRADE|CONSIDER_TQQQ|CONSIDER_SQQQ|CONSIDER_SOXL|CONSIDER_SOXS",
  "confidence": 0.0
}
""".strip()


@dataclass
class InsightItem:
    kind: str
    summary: str
    sources: List[str]
    strength: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_insight(report: dict) -> dict:
    rules = _build_rules_insight(report)
    if not deepseek_configured():
        rules["llm_error"] = "DEEPSEEK_API_KEY is not configured."
        return rules

    llm_payload = build_llm_insight_payload(report)
    llm_payload["rules_fallback"] = {
        "recommended_action": rules.get("recommended_action"),
        "confidence": rules.get("confidence"),
        "convergences": rules.get("convergences"),
        "divergences": rules.get("divergences"),
        "opportunities": rules.get("opportunities"),
        "no_trade_or_caution_reasons": rules.get("no_trade_or_caution_reasons"),
        "trade_permission": rules.get("trade_permission"),
    }
    llm = call_deepseek_json(INSIGHT_SYSTEM_PROMPT, llm_payload, temperature=0.15, timeout=30)
    validated = _validate_llm_insight(llm, report)
    if validated is None:
        rules["llm_error"] = "DeepSeek insight returned no valid schema; using rules fallback."
        return rules
    validated["engine"] = "deepseek_insight_v0"
    validated["llm_used"] = True
    validated["rules_fallback"] = {
        "engine": rules.get("engine"),
        "recommended_action": rules.get("recommended_action"),
        "confidence": rules.get("confidence"),
        "trade_permission": rules.get("trade_permission"),
    }
    validated["constraints"] = rules["constraints"]
    return validated


def _build_rules_insight(report: dict) -> dict:
    macro = report.get("macro_filter", {})
    risk_temp = report.get("risk_temperature", {})
    semi = report.get("semiconductor_direction", {})
    event_gate = report.get("economic_calendar", {}).get("event_gate", {})
    discovery = report.get("market_discovery", {}).get("directions", {})
    prediction_divergence = report.get("prediction_market_divergence", {})
    reaction_divergence = report.get("market_reaction_divergence", {})
    draft = report.get("draft_action", {})

    convergences: List[InsightItem] = []
    divergences: List[InsightItem] = []
    no_trade_reasons: List[InsightItem] = []

    macro_regime = macro.get("regime")
    temp_appetite = risk_temp.get("risk_appetite")
    temp_label = risk_temp.get("label")
    fed_bias = risk_temp.get("fed_policy_bias")
    semi_bias = semi.get("bias")

    if macro_regime == "RISK_OFF" and temp_appetite == "RISK_OFF":
        convergences.append(
            InsightItem(
                kind="MACRO_RISK_CONVERGENCE",
                summary="Macro filter and risk temperature both point to risk-off.",
                sources=["macro_filter.regime", "risk_temperature.risk_appetite"],
                strength="MEDIUM" if temp_label == "MILD_RISK_OFF" else "HIGH",
            )
        )

    if fed_bias in {"HAWKISH", "MIXED_LEAN_HAWKISH"} and temp_appetite == "RISK_OFF":
        convergences.append(
            InsightItem(
                kind="RATES_RISK_CONVERGENCE",
                summary="Rates/Fed proxy is hawkish while risk appetite is weak.",
                sources=["risk_temperature.fed_policy_bias", "risk_temperature.components"],
                strength="HIGH" if fed_bias == "HAWKISH" else "MEDIUM",
            )
        )

    fed_markets = _top_discovery_titles(discovery.get("US_EQUITY_MARKET", {}), include=["Fed", "interest rates"])
    if fed_markets:
        convergences.append(
            InsightItem(
                kind="PREDICTION_MARKET_POLICY_CONTEXT",
                summary="Prediction markets are actively pricing the near-term Fed decision.",
                sources=["market_discovery.US_EQUITY_MARKET.top_markets"],
                strength="MEDIUM",
            )
        )

    if macro_regime == "RISK_OFF" and semi_bias == "SOXL_BIAS":
        divergences.append(
            InsightItem(
                kind="MACRO_VS_SEMICONDUCTOR_DIVERGENCE",
                summary="Macro tape is risk-off, but semiconductor theme evidence is still constructive.",
                sources=["macro_filter.regime", "semiconductor_direction.bias", "theme_scores"],
                strength="MEDIUM",
            )
        )

    if _accepted_count(discovery.get("AI_SEMICONDUCTOR", {})) == 0 and semi_bias == "SOXL_BIAS":
        divergences.append(
            InsightItem(
                kind="FIXED_THEME_VS_DISCOVERY_GAP",
                summary="Fixed semiconductor themes are bullish, but automatic discovery found no A/B semiconductor markets.",
                sources=["theme_scores", "market_discovery.AI_SEMICONDUCTOR.accepted_count"],
                strength="MEDIUM",
            )
        )

    if prediction_divergence.get("one_sided_consensus"):
        divergences.append(
            InsightItem(
                kind="PREDICTION_MARKET_ONE_SIDED_CONFIRMATION",
                summary="Several themes have accepted evidence from only one prediction-market platform, so cross-platform confirmation is missing.",
                sources=["prediction_market_divergence.one_sided_consensus", "prediction_market_divergence.data_gaps"],
                strength="MEDIUM",
            )
        )

    if prediction_divergence.get("cross_platform_divergences"):
        divergences.append(
            InsightItem(
                kind="PREDICTION_MARKET_CROSS_PLATFORM_DIVERGENCE",
                summary="At least one theme shows a cross-platform prediction-market probability gap.",
                sources=["prediction_market_divergence.cross_platform_divergences"],
                strength="HIGH",
            )
        )

    reaction_status = reaction_divergence.get("overall", {}).get("status")
    if reaction_status == "PRICE_CONFIRMS_EXPECTATIONS":
        convergences.append(
            InsightItem(
                kind="PRICE_REACTION_CONFIRMATION",
                summary="ETF price reaction confirms the current macro/theme expectations.",
                sources=["market_reaction_divergence.overall"],
                strength="MEDIUM",
            )
        )
    elif reaction_status in {"PRICE_DIVERGES_FROM_EXPECTATIONS", "MIXED_CONFIRMATION"}:
        divergences.append(
            InsightItem(
                kind="PRICE_REACTION_DIVERGENCE",
                summary="ETF price reaction is mixed or diverges from macro/theme expectations.",
                sources=[
                    "market_reaction_divergence.overall",
                    "market_reaction_divergence.broad_risk",
                    "market_reaction_divergence.semiconductor",
                ],
                strength="MEDIUM",
            )
        )

    if event_gate.get("status") == "EVENT_RISK":
        no_trade_reasons.append(
            InsightItem(
                kind="EVENT_RISK_BLOCK",
                summary="High-impact event window is active.",
                sources=["economic_calendar.event_gate"],
                strength="HIGH",
            )
        )
    elif event_gate.get("next_high_impact_event"):
        no_trade_reasons.append(
            InsightItem(
                kind="UPCOMING_EVENT_CAUTION",
                summary="A high-impact macro event is upcoming; avoid treating signals as clean trend confirmation.",
                sources=["economic_calendar.event_gate.next_high_impact_event"],
                strength="MEDIUM",
            )
        )

    if macro.get("warnings"):
        no_trade_reasons.append(
            InsightItem(
                kind="DATA_SOURCE_WARNING",
                summary="Some macro inputs require validation before live execution.",
                sources=["macro_filter.warnings"],
                strength="MEDIUM",
            )
        )

    opportunities = _build_opportunities(report, convergences, divergences, no_trade_reasons)
    recommended_action = _recommend_action(draft.get("action"), opportunities, no_trade_reasons)
    confidence = _confidence(convergences, divergences, no_trade_reasons, opportunities)

    return {
        "schema_version": INSIGHT_SCHEMA_VERSION,
        "engine": "rules_v0_llm_ready",
        "llm_used": False,
        "llm_error": None,
        "constraints": {
            "allowed_actions": sorted(ALLOWED_ACTIONS),
            "must_use_only_report_fields": True,
            "must_not_create_external_facts": True,
            "must_not_override_risk_review": True,
        },
        "market_read": {
            "macro_regime": macro_regime,
            "risk_temperature": temp_label,
            "fed_policy_bias": fed_bias,
            "semiconductor_bias": semi_bias,
            "event_gate": event_gate.get("status"),
        },
        "convergences": [item.to_dict() for item in convergences],
        "divergences": [item.to_dict() for item in divergences],
        "opportunities": opportunities,
        "no_trade_or_caution_reasons": [item.to_dict() for item in no_trade_reasons],
        "integrated_conclusion": report.get("synthesis", {}).get("integrated_conclusion", {}),
        "trade_permission": report.get("rules_trade_permission", {}),
        "recommended_action": recommended_action,
        "confidence": confidence,
    }


def build_llm_insight_payload(report: dict) -> dict:
    """Payload that a future DeepSeek insight call may read. It excludes raw secrets and large raw API blobs."""
    return {
        "schema_version": INSIGHT_SCHEMA_VERSION,
        "instruction": (
            "Use only supplied JSON fields. Return valid JSON only. Do not invent market data, news, prices, "
            "or probabilities. If evidence is insufficient, say so and lower confidence."
        ),
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "inputs": {
            "macro_filter": report.get("macro_filter"),
            "risk_temperature": report.get("risk_temperature"),
            "economic_calendar_event_gate": report.get("economic_calendar", {}).get("event_gate"),
            "theme_scores": report.get("theme_scores"),
            "semiconductor_direction": report.get("semiconductor_direction"),
            "market_discovery": report.get("market_discovery"),
            "prediction_market_divergence": report.get("prediction_market_divergence"),
            "market_reaction_divergence": report.get("market_reaction_divergence"),
            "price_trends": report.get("price_trends"),
            "candidate_layer": report.get("candidate_layer"),
            "rules_trade_permission": report.get("rules_trade_permission"),
            "synthesis": report.get("synthesis"),
            "draft_action": report.get("draft_action"),
            "trade_decision_record": report.get("trade_decision_record"),
        },
        "required_output_shape": {
            "convergences": [{"kind": "string", "summary": "string", "sources": ["json.path"], "strength": "LOW|MEDIUM|HIGH"}],
            "divergences": [{"kind": "string", "summary": "string", "sources": ["json.path"], "strength": "LOW|MEDIUM|HIGH"}],
            "opportunities": [
                {
                    "symbol": "TQQQ|SQQQ|SOXL|SOXS|NONE",
                    "direction": "LONG|NONE",
                    "setup": "string",
                    "evidence": ["json.path"],
                    "invalidations": ["string"],
                    "quality": "LOW|MEDIUM|HIGH",
                }
            ],
            "recommended_action": "NO_TRADE|CONSIDER_TQQQ|CONSIDER_SQQQ|CONSIDER_SOXL|CONSIDER_SOXS",
            "integrated_conclusion": {"posture": "string", "summary": "string", "action_bias": "string"},
            "trade_permission": {
                "open_permission": "BLOCKED|ALLOWED_SHADOW_ONLY|ALLOWED",
                "direction_permission": "NONE|LONG_ONLY|SHORT_ONLY|BOTH",
                "position_size": "NONE|LIGHT|NORMAL|HEAVY",
                "preferred_symbols": ["TQQQ|SQQQ|SOXL|SOXS"],
                "broad_market_long_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
                "broad_market_short_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
                "semiconductor_long_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
                "semiconductor_short_term": "STRONG|NEUTRAL|WEAK|UNKNOWN",
                "hard_blocks": ["string"],
                "warnings": ["string"],
                "rationale": "string",
            },
            "confidence": 0.0,
        },
    }


def _validate_llm_insight(value: Optional[Dict[str, Any]], report: dict) -> Optional[dict]:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != INSIGHT_SCHEMA_VERSION:
        return None
    recommended = value.get("recommended_action")
    if recommended not in ALLOWED_ACTIONS:
        return None

    event_gate = report.get("economic_calendar", {}).get("event_gate", {})
    if event_gate.get("status") == "EVENT_RISK":
        recommended = "NO_TRADE"

    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError):
        return None
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    market_read = value.get("market_read") if isinstance(value.get("market_read"), dict) else {}
    trade_permission = validate_trade_permission(value.get("trade_permission"), report.get("rules_trade_permission", {}))
    if event_gate.get("status") == "EVENT_RISK":
        trade_permission["open_permission"] = "BLOCKED"
        trade_permission["direction_permission"] = "NONE"
        trade_permission["position_size"] = "NONE"
        if "HIGH_IMPACT_EVENT_WINDOW" not in trade_permission["hard_blocks"]:
            trade_permission["hard_blocks"].append("HIGH_IMPACT_EVENT_WINDOW")
    return {
        "schema_version": INSIGHT_SCHEMA_VERSION,
        "market_read": {
            "macro_regime": _short_string(market_read.get("macro_regime")),
            "risk_temperature": _short_string(market_read.get("risk_temperature")),
            "fed_policy_bias": _short_string(market_read.get("fed_policy_bias")),
            "semiconductor_bias": _short_string(market_read.get("semiconductor_bias")),
            "event_gate": _short_string(market_read.get("event_gate")),
        },
        "convergences": _validate_items(value.get("convergences")),
        "divergences": _validate_items(value.get("divergences")),
        "opportunities": _validate_opportunities(value.get("opportunities")),
        "no_trade_or_caution_reasons": _validate_items(value.get("no_trade_or_caution_reasons")),
        "integrated_conclusion": _validate_integrated_conclusion(value.get("integrated_conclusion"), report),
        "trade_permission": trade_permission,
        "recommended_action": recommended,
        "confidence": confidence,
    }


def _validate_integrated_conclusion(value: Any, report: dict) -> dict:
    fallback = report.get("synthesis", {}).get("integrated_conclusion", {})
    if not isinstance(value, dict):
        return fallback
    return {
        "posture": _short_string(value.get("posture") or fallback.get("posture"), 80),
        "action_bias": _short_string(value.get("action_bias") or fallback.get("action_bias"), 80),
        "summary": _short_string(value.get("summary") or fallback.get("summary"), 500),
    }


def _validate_items(value: Any) -> List[dict]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        strength = item.get("strength")
        if strength not in INSIGHT_STRENGTHS:
            strength = "LOW"
        output.append(
            {
                "kind": _short_string(item.get("kind"), 80),
                "summary": _short_string(item.get("summary"), 300),
                "sources": _string_list(item.get("sources"), 8),
                "strength": strength,
            }
        )
    return output


def _validate_opportunities(value: Any) -> List[dict]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value[:5]:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol")
        direction = item.get("direction")
        quality = item.get("quality")
        if symbol not in OPPORTUNITY_SYMBOLS:
            symbol = "NONE"
        if direction not in OPPORTUNITY_DIRECTIONS:
            direction = "NONE"
        if quality not in INSIGHT_STRENGTHS:
            quality = "LOW"
        output.append(
            {
                "symbol": symbol,
                "direction": direction,
                "setup": _short_string(item.get("setup"), 400),
                "evidence": _string_list(item.get("evidence"), 8),
                "invalidations": _string_list(item.get("invalidations"), 8, max_len=180),
                "quality": quality,
            }
        )
    return output


def _string_list(value: Any, limit: int, max_len: int = 120) -> List[str]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value[:limit]:
        if isinstance(item, str) and item.strip():
            output.append(_short_string(item, max_len))
    return output


def _short_string(value: Any, max_len: int = 120) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())[:max_len]


def _build_opportunities(
    report: dict,
    convergences: List[InsightItem],
    divergences: List[InsightItem],
    no_trade_reasons: List[InsightItem],
) -> List[dict]:
    draft_action = report.get("draft_action", {}).get("action")
    risk_temp = report.get("risk_temperature", {})
    semi = report.get("semiconductor_direction", {})
    opportunities: List[dict] = []

    has_event_block = any(item.kind == "EVENT_RISK_BLOCK" for item in no_trade_reasons)
    if has_event_block:
        return [
            {
                "symbol": "NONE",
                "direction": "NONE",
                "setup": "Event risk blocks new trade setup.",
                "evidence": ["economic_calendar.event_gate"],
                "invalidations": ["Event gate returns to NORMAL after release waiting window."],
                "quality": "LOW",
            }
        ]

    if draft_action == "CONSIDER_SQQQ":
        opportunities.append(
            {
                "symbol": "SQQQ",
                "direction": "LONG",
                "setup": "Risk-off macro setup favors inverse Nasdaq exposure in shadow mode.",
                "evidence": ["macro_filter", "risk_temperature", "draft_action"],
                "invalidations": [
                    "risk_temperature.risk_appetite turns RISK_ON",
                    "macro_filter.regime turns RISK_ON",
                    "VIX falls back into calm range",
                ],
                "quality": "MEDIUM" if risk_temp.get("score", 0) <= -2 else "LOW",
            }
        )

    if semi.get("bias") == "SOXL_BIAS" and risk_temp.get("risk_appetite") == "RISK_OFF":
        opportunities.append(
            {
                "symbol": "SOXL",
                "direction": "LONG",
                "setup": "Semiconductor theme is constructive, but macro risk-off blocks chasing leveraged long exposure.",
                "evidence": ["semiconductor_direction", "theme_scores", "risk_temperature"],
                "invalidations": [
                    "macro_filter.regime remains RISK_OFF",
                    "risk_temperature.fed_policy_bias remains HAWKISH",
                ],
                "quality": "LOW",
            }
        )

    if not opportunities:
        opportunities.append(
            {
                "symbol": "NONE",
                "direction": "NONE",
                "setup": "No clean opportunity after combining macro, prediction-market themes, and risk gates.",
                "evidence": ["macro_filter", "risk_temperature", "theme_scores"],
                "invalidations": ["A stronger convergence appears in future cycle."],
                "quality": "LOW",
            }
        )
    return opportunities


def _recommend_action(draft_action: Optional[str], opportunities: List[dict], cautions: List[InsightItem]) -> str:
    if any(item.kind == "EVENT_RISK_BLOCK" for item in cautions):
        return "NO_TRADE"
    if draft_action in ALLOWED_ACTIONS:
        return draft_action
    first = opportunities[0] if opportunities else {}
    symbol = first.get("symbol")
    mapping = {"TQQQ": "CONSIDER_TQQQ", "SQQQ": "CONSIDER_SQQQ", "SOXL": "CONSIDER_SOXL", "SOXS": "CONSIDER_SOXS"}
    return mapping.get(symbol, "NO_TRADE")


def _confidence(
    convergences: List[InsightItem],
    divergences: List[InsightItem],
    cautions: List[InsightItem],
    opportunities: List[dict],
) -> float:
    score = 0.45
    score += 0.08 * len([item for item in convergences if item.strength in {"MEDIUM", "HIGH"}])
    score -= 0.06 * len([item for item in divergences if item.strength in {"MEDIUM", "HIGH"}])
    score -= 0.05 * len(cautions)
    if opportunities and opportunities[0].get("quality") == "MEDIUM":
        score += 0.08
    return round(max(0.05, min(0.9, score)), 4)


def _top_discovery_titles(payload: dict, include: List[str]) -> List[str]:
    titles = []
    for market in payload.get("top_markets", []):
        title = market.get("title") or ""
        if any(token.lower() in title.lower() for token in include):
            titles.append(title)
    return titles


def _accepted_count(payload: dict) -> int:
    try:
        return int(payload.get("accepted_count", 0))
    except (TypeError, ValueError):
        return 0
