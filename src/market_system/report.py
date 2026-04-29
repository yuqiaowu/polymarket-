from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import (
    MACRO_SYMBOLS,
    REFERENCE_SYMBOLS,
    REPORTS_DIR,
    TRADE_SYMBOLS,
    configured,
    load_env_names,
)
from .candidate_layer import apply_rule_review, build_candidate_layer, candidate_to_draft_action
from .decision_record import build_trade_decision_record
from .economic_calendar import build_economic_calendar
from .insight_engine import build_insight, build_llm_insight_payload
from .market_discovery import build_market_discovery
from .market_reaction_divergence import build_market_reaction_divergence
from .market_data import Quote, data_status, fetch_daily_bars, fetch_quotes
from .prediction_markets import (
    MarketEvidence,
    fetch_kalshi_gpu_evidence,
    fetch_polymarket_theme_evidence,
    fetch_polymarket_top_events,
)
from .prediction_market_divergence import build_prediction_market_divergence
from .price_trend import build_price_trends
from .market_synthesis import build_synthesis
from .risk_temperature import build_risk_temperature
from .scoring import macro_risk_filter, score_themes, semiconductor_direction
from .shared_schema import is_core_theme, schema_definitions
from .trade_permission import build_rules_trade_permission


def build_report() -> dict:
    env_names = load_env_names()
    quotes = fetch_quotes(MACRO_SYMBOLS)
    reaction_symbols = TRADE_SYMBOLS + REFERENCE_SYMBOLS
    reaction_quotes = fetch_quotes(reaction_symbols)
    reaction_daily_bars = fetch_daily_bars(reaction_symbols, period="3mo")
    price_trends = build_price_trends(reaction_daily_bars)
    polymarket_evidence = fetch_polymarket_theme_evidence()
    kalshi_evidence = fetch_kalshi_gpu_evidence()
    evidence: List[MarketEvidence] = polymarket_evidence + kalshi_evidence
    theme_scores = score_themes(evidence)
    risk_filter = macro_risk_filter(quotes)
    risk_temperature = build_risk_temperature(quotes)
    semi = semiconductor_direction(theme_scores, quotes)
    top_attention = fetch_polymarket_top_events(limit=80)
    economic_calendar = build_economic_calendar(window_days=21)
    market_discovery = build_market_discovery(
        ["US_EQUITY_MARKET", "AI_SEMICONDUCTOR"],
        context={
            "macro_filter": risk_filter.to_dict(),
            "risk_temperature": risk_temperature.to_dict(),
            "semiconductor_direction": semi,
            "economic_calendar_event_gate": economic_calendar.get("event_gate"),
        },
    )
    partial_report = {
        "prediction_market_evidence": [item.to_dict() for item in evidence],
        "macro_filter": risk_filter.to_dict(),
        "risk_temperature": risk_temperature.to_dict(),
        "theme_scores": [item.to_dict() for item in theme_scores],
        "semiconductor_direction": semi,
        "market_discovery": market_discovery,
        "economic_calendar": economic_calendar,
    }
    prediction_divergence = build_prediction_market_divergence(partial_report)
    reaction_divergence = build_market_reaction_divergence(
        risk_filter.to_dict(),
        risk_temperature.to_dict(),
        semi,
        prediction_divergence,
        reaction_quotes,
    )
    partial_report["prediction_market_divergence"] = prediction_divergence
    partial_report["market_reaction_divergence"] = reaction_divergence
    partial_report["price_trends"] = price_trends
    synthesis = build_synthesis(partial_report)
    candidate_layer = build_candidate_layer({**partial_report, "synthesis": synthesis})
    draft = candidate_to_draft_action(candidate_layer)
    rules_trade_permission = build_rules_trade_permission(
        {**partial_report, "synthesis": synthesis, "candidate_layer": candidate_layer, "draft_action": draft}
    )
    candidate_layer = apply_rule_review(candidate_layer, rules_trade_permission)
    draft = candidate_to_draft_action(candidate_layer)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "env_status": {
            "configured": configured(
                [
                    "DEEPSEEK_API_KEY",
                    "GEMINI_API_KEY",
                    "TELEGRAM_BOT_TOKEN",
                    "TELEGRAM_CHAT_ID",
                    "DISCORD_WEBHOOK_URL",
                    "MONGODB_URI",
                    "IBKR_HOST",
                    "IBKR_PORT",
                    "IBKR_CLIENT_ID",
                    "IBKR_ACCOUNT",
                ],
                env_names,
            ),
            "all_env_names": sorted(env_names.keys()),
            "note": "Values are intentionally not exposed.",
        },
        "shared_schema": schema_definitions(),
        "macro_data": {symbol: quote.to_dict() for symbol, quote in quotes.items()},
        "quotes": {symbol: quote.to_dict() for symbol, quote in quotes.items()},
        "reaction_quotes": {symbol: quote.to_dict() for symbol, quote in reaction_quotes.items()},
        "price_trends": price_trends,
        "data_status": data_status(quotes),
        "macro_filter": risk_filter.to_dict(),
        "risk_temperature": risk_temperature.to_dict(),
        "prediction_market_evidence": [item.to_dict() for item in evidence],
        "theme_scores": [item.to_dict() for item in theme_scores],
        "semiconductor_direction": semi,
        "market_attention": top_attention[:20],
        "market_discovery": market_discovery,
        "prediction_market_divergence": prediction_divergence,
        "market_reaction_divergence": reaction_divergence,
        "economic_calendar": economic_calendar,
        "synthesis": synthesis,
        "candidate_layer": candidate_layer,
        "rules_trade_permission": rules_trade_permission,
        "draft_action": draft,
    }
    report["insight"] = build_insight(report)
    report["trade_permission"] = report["insight"].get("trade_permission") or rules_trade_permission
    report["candidate_layer"] = apply_rule_review(report["candidate_layer"], report["trade_permission"])
    report["draft_action"] = candidate_to_draft_action(report["candidate_layer"])
    report["llm_insight_payload"] = build_llm_insight_payload(report)
    report["trade_decision_record"] = build_trade_decision_record(report)
    return report


def draft_action(macro: dict, semi: dict) -> dict:
    if not macro["can_trade"]:
        return {"action": "NO_TRADE", "reason": "Macro/data filter blocks trading."}
    risk = macro["risk_appetite_score"]
    semi_score = semi["semiconductor_score"]
    if risk >= 1 and semi_score >= 1:
        return {"action": "CONSIDER_SOXL", "reason": "Risk appetite and semiconductor edge are both positive."}
    if risk <= -1 and semi_score <= -1:
        return {"action": "CONSIDER_SOXS", "reason": "Risk appetite and semiconductor edge are both negative."}
    if risk >= 1:
        return {"action": "CONSIDER_TQQQ", "reason": "Risk appetite is positive but semiconductor edge is not strong."}
    if risk <= -1:
        return {"action": "CONSIDER_SQQQ", "reason": "Risk appetite is negative but semiconductor-specific edge is not strong."}
    return {"action": "NO_TRADE", "reason": "No strong macro/semiconductor confluence."}


def write_reports(report: dict, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"market_report_{stamp}.json"
    md_path = reports_dir / f"market_report_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = [
        "# Market Regime Prototype Report",
        "",
        f"Generated UTC: `{report['generated_at_utc']}`",
        "",
        "## Trade Permission",
        "",
    ]
    permission = report.get("trade_permission", {})
    momentum = permission.get("momentum", {})
    broad_momentum = momentum.get("broad_market", {})
    semi_momentum = momentum.get("semiconductor", {})
    lines.extend(
        [
            f"- Open permission: `{permission.get('open_permission', 'N/A')}`",
            f"- Direction permission: `{permission.get('direction_permission', 'N/A')}`",
            f"- Position size: `{permission.get('position_size', 'N/A')}`",
            f"- Preferred symbols: `{', '.join(permission.get('preferred_symbols', [])) or 'none'}`",
            f"- Broad market momentum: long=`{broad_momentum.get('long_term', 'N/A')}`, short=`{broad_momentum.get('short_term', 'N/A')}`",
            f"- Semiconductor momentum: long=`{semi_momentum.get('long_term', 'N/A')}`, short=`{semi_momentum.get('short_term', 'N/A')}`",
            f"- Rationale: {permission.get('rationale', 'N/A')}",
        ]
    )
    if permission.get("hard_blocks"):
        lines.append(f"- Hard blocks: `{', '.join(permission.get('hard_blocks', []))}`")
    if permission.get("warnings"):
        lines.append(f"- Warnings: `{', '.join(permission.get('warnings', []))}`")
    lines.extend(
        [
            "",
            "## Draft Action",
            "",
            f"- Action: `{report['draft_action']['action']}`",
            f"- Reason: {report['draft_action']['reason']}",
            "",
            "## Candidate Layer",
            "",
        ]
    )
    candidate_layer = report.get("candidate_layer", {})
    candidate_summary = candidate_layer.get("summary", {})
    candidate_rule_review = candidate_layer.get("rule_review", {})
    lines.extend(
        [
            f"- Candidates: `{candidate_summary.get('candidate_count', 0)}`",
            f"- Directional candidates: `{candidate_summary.get('directional_candidate_count', 0)}`",
            f"- Allowed after rules: `{candidate_rule_review.get('allowed_candidate_count', 0)}`",
            f"- Blocked after rules: `{candidate_rule_review.get('blocked_candidate_count', 0)}`",
            "",
            "| Candidate | Strategy | Action | Confidence | Rule status | Thesis |",
            "|---|---|---|---:|---|---|",
        ]
    )
    candidates = candidate_layer.get("candidates", [])
    if not candidates:
        lines.append("| none | N/A | N/A | N/A | N/A | No candidate generated |")
    for item in candidates[:8]:
        lines.append(
            f"| `{item.get('candidate_id', 'N/A')}` | `{item.get('strategy', 'N/A')}` | "
            f"`{item.get('action', 'N/A')}` `{item.get('target_symbol') or 'NONE'}` | "
            f"{_fmt(item.get('confidence'))} | `{item.get('rule_status', 'N/A')}` | {item.get('thesis', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "## Macro Filter",
            "",
            f"- Can trade: `{report['macro_filter']['can_trade']}`",
            f"- Regime: `{report['macro_filter']['regime']}`",
            f"- Risk appetite score: `{report['macro_filter']['risk_appetite_score']}`",
            f"- Data gaps: `{', '.join(report['macro_filter']['data_gaps']) or 'none'}`",
            "",
            "Reasons:",
        ]
    )
    lines.extend([f"- {reason}" for reason in report["macro_filter"]["reasons"]])
    if report["macro_filter"]["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend([f"- {warning}" for warning in report["macro_filter"]["warnings"]])

    risk_temp = report.get("risk_temperature", {})
    lines.extend(["", "## Risk Temperature", ""])
    lines.append(f"- Label: `{risk_temp.get('label', 'UNKNOWN')}`")
    lines.append(f"- Score: `{risk_temp.get('score', 'N/A')}`")
    lines.append(f"- Fed policy bias: `{risk_temp.get('fed_policy_bias', 'UNKNOWN')}`")
    lines.append(f"- Risk appetite: `{risk_temp.get('risk_appetite', 'UNKNOWN')}`")
    lines.append("")
    lines.append("Top components:")
    for component in sorted(risk_temp.get("components", []), key=lambda item: abs(item.get("points", 0)), reverse=True)[:8]:
        lines.append(
            f"- `{component['name']}` value={_fmt(component['value'])}, points={component['points']}: {component['reason']}"
        )
    if risk_temp.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        lines.extend([f"- {warning}" for warning in risk_temp["warnings"]])

    insight = report.get("insight", {})
    lines.extend(["", "## Insight Layer", ""])
    lines.append(f"- Engine: `{insight.get('engine', 'N/A')}`")
    lines.append(f"- LLM used: `{insight.get('llm_used', False)}`")
    lines.append(f"- Recommended action: `{insight.get('recommended_action', 'N/A')}`")
    lines.append(f"- Confidence: `{_fmt(insight.get('confidence'))}`")
    lines.append("")
    lines.append("Convergences:")
    convergences = insight.get("convergences", [])
    if convergences:
        for item in convergences:
            lines.append(f"- `{item['strength']}` {item['summary']} sources={item['sources']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Divergences:")
    divergences = insight.get("divergences", [])
    if divergences:
        for item in divergences:
            lines.append(f"- `{item['strength']}` {item['summary']} sources={item['sources']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Opportunities:")
    for item in insight.get("opportunities", []):
        lines.append(
            f"- `{item['quality']}` `{item['symbol']}` {item['setup']} evidence={item['evidence']}"
        )

    synthesis = report.get("synthesis", {})
    conclusion = synthesis.get("integrated_conclusion", {})
    lines.extend(["", "## Integrated Synthesis", ""])
    lines.append(f"- Posture: `{conclusion.get('posture', 'N/A')}`")
    lines.append(f"- Action bias: `{conclusion.get('action_bias', 'N/A')}`")
    lines.append(f"- Summary: {conclusion.get('summary', 'N/A')}")
    fed = synthesis.get("fed_decision_consensus", {})
    lines.append("")
    lines.append("Fed decision consensus:")
    lines.append(f"- Consensus: `{fed.get('consensus', 'N/A')}`")
    lines.append(f"- No-change probability: `{_fmt(fed.get('no_change_probability'))}`")
    lines.append(f"- Decision surprise risk: `{fed.get('decision_surprise_risk', 'N/A')}`")
    lines.append(f"- Guidance risk: `{fed.get('guidance_risk', 'N/A')}`")
    if synthesis.get("conflicts"):
        lines.append("")
        lines.append("Conflicts:")
        for conflict in synthesis["conflicts"]:
            lines.append(f"- `{conflict['kind']}` {conflict['summary']} Effect: {conflict['effect']}")

    pm_divergence = report.get("prediction_market_divergence", {})
    lines.extend(["", "## Prediction Market Divergence", ""])
    lines.append(pm_divergence.get("policy", "N/A"))
    lines.append(f"- Themes analyzed: `{pm_divergence.get('theme_count', 0)}`")
    lines.append(f"- Cross-platform divergences: `{len(pm_divergence.get('cross_platform_divergences', []))}`")
    lines.append(f"- One-sided consensus themes: `{len(pm_divergence.get('one_sided_consensus', []))}`")
    lines.append("")
    lines.append("| Theme | Type | Quality | Gap | Interpretation |")
    lines.append("|---|---|---|---:|---|")
    for item in pm_divergence.get("themes", [])[:10]:
        lines.append(
            f"| {item.get('theme')} | {item.get('divergence_type')} | {item.get('quality')} | "
            f"{_fmt(item.get('probability_gap'))} | {item.get('interpretation')} |"
        )
    if not pm_divergence.get("themes"):
        lines.append("| none | N/A | N/A | N/A | No prediction-market themes available |")

    reaction = report.get("market_reaction_divergence", {})
    lines.extend(["", "## Market Reaction Divergence", ""])
    lines.append(reaction.get("policy", "N/A"))
    overall = reaction.get("overall", {})
    lines.append(f"- Overall: `{overall.get('status', 'N/A')}`")
    lines.append(f"- Confirmations: `{', '.join(overall.get('confirmations', [])) or 'none'}`")
    lines.append(f"- Divergences: `{', '.join(overall.get('divergences', [])) or 'none'}`")
    for key, label in [("broad_risk", "Broad risk"), ("semiconductor", "Semiconductor")]:
        section = reaction.get(key, {})
        lines.append(
            f"- {label}: expected=`{section.get('expected_bias', 'N/A')}`, "
            f"reaction=`{section.get('reaction_status', 'N/A')}` - {section.get('interpretation', 'N/A')}"
        )

    trends = report.get("price_trends", {})
    lines.extend(["", "## Price Trend Confirmation", ""])
    lines.append(trends.get("policy", "N/A"))
    for key, label in [("broad_market", "Broad market"), ("semiconductor", "Semiconductor")]:
        group = trends.get("groups", {}).get(key, {})
        lines.append(f"- {label}: trend=`{group.get('trend', 'N/A')}`, score=`{group.get('score', 'N/A')}`")
    lines.append("")
    lines.append("| Symbol | Trend | Score | Close | MA5 | MA10 | MA20 | 5D % | 20D % | Vol/20D |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for symbol in ["QQQ", "SPY", "SOXX", "SMH", "TQQQ", "SQQQ", "SOXL", "SOXS"]:
        item = trends.get("symbols", {}).get(symbol, {})
        lines.append(
            f"| {symbol} | {item.get('trend', 'N/A')} | {_fmt(item.get('score'))} | {_fmt(item.get('close'))} | "
            f"{_fmt(item.get('ma5'))} | {_fmt(item.get('ma10'))} | {_fmt(item.get('ma20'))} | "
            f"{_fmt(item.get('return_5d_pct'))} | {_fmt(item.get('return_20d_pct'))} | {_fmt(item.get('volume_ratio_20d'))} |"
        )

    lines.extend(
        [
            "",
            "## Macro Data Snapshot",
            "",
            "Macro note: these rows are macro/rates/volatility snapshots for regime filtering, not 4H trade signals.",
            "",
            "| Symbol | Date | Time | Source | Value | Status |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for symbol, quote in report["macro_data"].items():
        lines.append(
            f"| {symbol} | {quote.get('date') or 'N/A'} | {quote.get('time') or 'N/A'} | "
            f"{quote.get('source_symbol') or 'N/A'} | {_fmt(quote['close'])} | {quote['status']} |"
        )

    core_themes, watch_themes = _split_theme_scores(report["theme_scores"])
    lines.extend(["", "## Core Theme Scores", "", "| Theme | Value | Confidence | Status | Interpretation | Evidence |", "|---|---:|---:|---|---|---:|"])
    if not core_themes:
        lines.append("| none | N/A | N/A | N/A | No core theme passed confidence filter | 0 |")
    for theme in core_themes:
        lines.append(
            f"| {theme['theme']} | {_fmt(theme['value'])} | {_fmt(theme['confidence'])} | "
            f"{theme['status']} | {theme['interpretation']} | {theme['evidence_count']} |"
        )
    lines.extend(["", "## Theme Watchlist / Data Gaps", "", "| Theme | Confidence | Status | Reason | Evidence |", "|---|---:|---|---|---:|"])
    if not watch_themes:
        lines.append("| none | N/A | N/A | No downgraded themes | 0 |")
    for theme in watch_themes:
        reason = _theme_watch_reason(theme)
        lines.append(
            f"| {theme['theme']} | {_fmt(theme['confidence'])} | {theme['status']} | "
            f"{reason} | {theme['evidence_count']} |"
        )

    lines.extend(["", "## Semiconductor Direction", ""])
    lines.append(f"- Bias: `{report['semiconductor_direction']['bias']}`")
    lines.append(f"- Score: `{report['semiconductor_direction']['semiconductor_score']}`")
    lines.append("")
    lines.append("Reasons:")
    lines.extend([f"- {reason}" for reason in report["semiconductor_direction"]["reasons"]])

    lines.extend(["", "## Market Attention Sample", ""])
    if not report["market_attention"]:
        lines.append("- No non-noise top attention markets found from current fetch.")
    else:
        for item in report["market_attention"][:10]:
            lines.append(f"- `{item['theme']}` {item['title']} ({item['platform']}, volume={_fmt(item.get('volume'))})")

    discovery = report.get("market_discovery", {})
    lines.extend(["", "## Prediction Market Discovery", ""])
    lines.append(discovery.get("policy", "N/A"))
    shared = report.get("shared_schema", {})
    buckets = shared.get("discovery_buckets", {})
    if buckets:
        lines.append(
            f"Definitions: accepted={buckets['accepted']['grades']} means {buckets['accepted']['meaning']} "
            f"watch={buckets['watch']['grades']} means {buckets['watch']['meaning']}"
        )
    planner = discovery.get("planner", {})
    lines.append(
        f"Planner: engine=`{planner.get('engine', 'N/A')}`, "
        f"llm_configured=`{planner.get('llm_configured', False)}`, "
        f"llm_used=`{planner.get('llm_used', False)}`"
    )
    for direction, payload in discovery.get("directions", {}).items():
        lines.append("")
        lines.append(f"### {direction}")
        lines.append(f"- Planner source: `{payload.get('planner_source', 'N/A')}`")
        lines.append(f"- Markets found: `{payload.get('market_count', 0)}`")
        lines.append(f"- Accepted A/B: `{payload.get('accepted_count', 0)}`")
        lines.append(f"- Watch C: `{payload.get('watch_count', 0)}`")
        top = payload.get("top_markets", [])
        if not top:
            lines.append("- No relevant markets found.")
            continue
        for market in top[:6]:
            lines.append(
                f"- `{market['quality_grade']}` `{market['platform']}` "
                f"p={_fmt(market.get('probability'))}, q={_fmt(market.get('quality_score'))}, "
                f"vol={_fmt(market.get('volume'))}: {market['title']}"
            )

    calendar = report.get("economic_calendar", {})
    event_gate = calendar.get("event_gate", {})
    lines.extend(["", "## Economic Calendar", ""])
    lines.append(f"- Event gate: `{event_gate.get('status', 'UNKNOWN')}`")
    lines.append(f"- Rule: {event_gate.get('rule', 'N/A')}")
    next_event = event_gate.get("next_high_impact_event")
    if next_event:
        lines.append(
            f"- Next high-impact event: `{next_event['date']} {next_event.get('time_et') or ''} ET` "
            f"{next_event['title']} ({next_event['source']})"
        )
    lines.append("")
    lines.append("Source status:")
    for source in calendar.get("source_status", []):
        suffix = f" - {source['error']}" if source.get("error") else ""
        lines.append(f"- `{source['source']}`: `{source['status']}`, items={source['item_count']}{suffix}")
    lines.append("")
    lines.append("Upcoming scheduled events:")
    scheduled = calendar.get("scheduled_events", [])
    if not scheduled:
        lines.append("- No scheduled official events found in the current window.")
    else:
        for event in scheduled[:12]:
            lines.append(
                f"- `{event['impact']}` `{event['date']} {event.get('time_et') or ''} ET` "
                f"{event['title']} ({event['source']}, {event['event_type']})"
            )

    decision = report.get("trade_decision_record", {})
    candidate = decision.get("candidate", {})
    risk_review = decision.get("riskReview", {})
    lines.extend(["", "## Trade Decision Record", ""])
    lines.append(f"- Cycle ID: `{decision.get('cycleId', 'N/A')}`")
    lines.append(f"- Decision ID: `{decision.get('decisionId', 'N/A')}`")
    lines.append(f"- Mode: `{decision.get('mode', 'N/A')}`")
    lines.append(f"- Candidate: `{candidate.get('action', 'N/A')}` `{candidate.get('targetSymbol') or 'NONE'}`")
    lines.append(f"- Risk allowed: `{risk_review.get('allowed', False)}`")
    lines.append(f"- Hard blocks: `{', '.join(risk_review.get('hardBlocks', [])) or 'none'}`")
    lines.append(f"- Execution: `{decision.get('execution', {}).get('status', 'N/A')}`")

    lines.extend(["", "## Env Status", ""])
    for key, is_set in report["env_status"]["configured"].items():
        lines.append(f"- `{key}`: `{is_set}`")

    return "\n".join(lines) + "\n"


def _split_theme_scores(themes: List[dict]) -> tuple[List[dict], List[dict]]:
    core = []
    watch = []
    for theme in themes:
        if is_core_theme(theme.get("status"), float(theme.get("confidence") or 0.0), int(theme.get("evidence_count") or 0)):
            core.append(theme)
        else:
            watch.append(theme)
    return core, watch


def _theme_watch_reason(theme: dict) -> str:
    status = theme.get("status")
    confidence = float(theme.get("confidence") or 0.0)
    if status != "AVAILABLE":
        return "Not enough accepted evidence."
    if confidence < 0.5:
        return "Confidence below core threshold."
    return "Downgraded by core theme filter."
