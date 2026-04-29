from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from market_system.config import REPORTS_DIR, TRADE_SYMBOLS
from market_system.market_data import PriceBar, fetch_daily_bars
from market_system.timesfm_strategy import (
    TimesFMForecast,
    TimesFMForecaster,
    TimesFMUnavailable,
    build_timesfm_forecast_matrix,
)


DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
DEFAULT_SYMBOLS = ["TQQQ", "SQQQ", "SOXL", "SOXS"]


@dataclass
class BacktestTrade:
    symbol: str
    decision_date: str
    exit_date: str
    action: str
    confidence: float
    forecast: dict
    entry_close: float
    exit_close: float
    gross_return_pct: float
    realized_return_pct: float
    rule_status: str


def run_backtest(args: argparse.Namespace) -> dict:
    symbols = _validate_symbols(args.symbols)
    bars_by_symbol = fetch_daily_bars(symbols, period=args.period, timeout=args.timeout)
    forecaster = TimesFMForecaster(args.model, args.context, args.horizon)

    forecast_rows: list[BacktestTrade] = []
    data_status = {}
    clean_bars_by_symbol = {}
    for symbol in symbols:
        bars = _clean_bars(bars_by_symbol.get(symbol, []))
        clean_bars_by_symbol[symbol] = bars
        data_status[symbol] = {"bar_count": len(bars), "first_date": bars[0].date if bars else None, "last_date": bars[-1].date if bars else None}
        forecast_rows.extend(_backtest_symbol(symbol, bars, forecaster, args))

    trades = forecast_rows if args.include_no_trade else [item for item in forecast_rows if item.action != "NO_TRADE"]
    summary = _summary(forecast_rows)
    return {
        "schema_version": "timesfm_backtest_v0.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": args.model,
            "context": args.context,
            "horizon_days": args.horizon,
            "period": args.period,
            "step": args.step,
            "max_windows_per_symbol": args.max_windows,
        },
        "policy": {
            "tradeable_symbols": symbols,
            "entry": "decision day close",
            "exit": f"close after {args.horizon} trading day(s)",
            "costs": f"{args.cost_bps} bps round-trip deducted from directional trades",
            "note": "Walk-forward forecast uses only bars available up to each decision date.",
        },
        "thresholds": {
            "positive_threshold_pct": args.positive_threshold,
            "negative_threshold_pct": args.negative_threshold,
            "probability_threshold": args.probability_threshold,
            "max_adverse_threshold_pct": args.max_adverse_threshold,
        },
        "data_status": data_status,
        "summary": summary,
        "calibration": _calibration(forecast_rows),
        "ranking_evaluation": _ranking_evaluation(forecast_rows),
        "by_year": _by_year(forecast_rows),
        "baselines": _build_baselines(clean_bars_by_symbol, args),
        "grid_search": _grid_search(forecast_rows, args) if args.grid else [],
        "latest_forecast_matrix": _latest_forecast_matrix(forecast_rows, args),
        "trades": [asdict(item) for item in trades],
    }


def _backtest_symbol(symbol: str, bars: list[PriceBar], forecaster: TimesFMForecaster, args: argparse.Namespace) -> list[BacktestTrade]:
    if len(bars) < args.context + args.horizon + 1:
        return []

    output: list[BacktestTrade] = []
    end_indexes = list(range(args.context, len(bars) - args.horizon + 1, args.step))
    if args.max_windows:
        end_indexes = end_indexes[-args.max_windows :]

    for end_idx in end_indexes:
        context_bars = bars[end_idx - args.context : end_idx]
        entry_bar = bars[end_idx - 1]
        exit_bar = bars[end_idx + args.horizon - 1]
        closes = [float(bar.close) for bar in context_bars if bar.close is not None]
        if len(closes) < args.context:
            continue

        forecast = forecaster.forecast(closes, args.horizon)
        action, confidence, rule_status = _candidate_from_forecast(symbol, forecast, args)
        if entry_bar.close is None or exit_bar.close is None:
            continue

        gross = (float(exit_bar.close) / float(entry_bar.close) - 1.0) * 100.0
        realized = _apply_trade_cost(gross, args.cost_bps) if action != "NO_TRADE" else gross
        output.append(
            BacktestTrade(
                symbol=symbol,
                decision_date=entry_bar.date,
                exit_date=exit_bar.date,
                action=action,
                confidence=confidence,
                forecast=asdict(forecast),
                entry_close=round(float(entry_bar.close), 4),
                exit_close=round(float(exit_bar.close), 4),
                gross_return_pct=round(gross, 4),
                realized_return_pct=round(realized, 4),
                rule_status=rule_status,
            )
        )
    return output


def _candidate_from_forecast(symbol: str, forecast: TimesFMForecast, args: argparse.Namespace) -> tuple[str, float, str]:
    q50 = forecast.q50_return_pct if forecast.q50_return_pct is not None else forecast.point_return_pct
    q10 = forecast.q10_return_pct
    q90 = forecast.q90_return_pct
    prob_positive = forecast.probability_positive
    prob_negative = None if prob_positive is None else 1.0 - prob_positive

    long_ok = (
        q50 >= args.positive_threshold
        and (prob_positive or 0.0) >= args.probability_threshold
        and (q10 is None or q10 >= args.max_adverse_threshold)
    )
    short_etf_ok = (
        q50 >= args.positive_threshold
        and (prob_positive or 0.0) >= args.probability_threshold
        and (q10 is None or q10 >= args.max_adverse_threshold)
    )
    # SQQQ and SOXS are already inverse ETF symbols. A positive forecast on them is a bearish underlying candidate.
    if symbol in {"TQQQ", "SOXL"} and long_ok:
        return f"CONSIDER_{symbol}", _confidence(q50, prob_positive), "PENDING_RULE_REVIEW"
    if symbol in {"SQQQ", "SOXS"} and short_etf_ok:
        return f"CONSIDER_{symbol}", _confidence(q50, prob_positive), "PENDING_RULE_REVIEW"

    downside_ok = (
        q50 <= args.negative_threshold
        and (prob_negative or 0.0) >= args.probability_threshold
        and (q90 is None or q90 <= abs(args.max_adverse_threshold))
    )
    if downside_ok:
        return "NO_TRADE", _confidence(abs(q50), prob_negative), "FORECAST_OPPOSES_SYMBOL"
    return "NO_TRADE", 0.0, "NO_FORECAST_CANDIDATE"


def _summary(trades: list[BacktestTrade]) -> dict:
    directional = [item for item in trades if item.action != "NO_TRADE"]
    returns = [item.realized_return_pct for item in directional]
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    by_symbol = {}
    for symbol in DEFAULT_SYMBOLS:
        symbol_trades = [item for item in directional if item.symbol == symbol]
        by_symbol[symbol] = _return_stats([item.realized_return_pct for item in symbol_trades])
    return {
        "all_rows": len(trades),
        "directional_trades": len(directional),
        "win_rate": _round_optional(len(wins) / len(returns)) if returns else None,
        "avg_return_pct": _round_optional(sum(returns) / len(returns)) if returns else None,
        "median_return_pct": _median(returns),
        "total_compounded_return_pct": _compound_return(returns),
        "avg_win_pct": _round_optional(sum(wins) / len(wins)) if wins else None,
        "avg_loss_pct": _round_optional(sum(losses) / len(losses)) if losses else None,
        "max_drawdown_pct": _max_drawdown(returns),
        "by_symbol": by_symbol,
    }


def _return_stats(returns: list[float]) -> dict:
    return {
        "trades": len(returns),
        "win_rate": _round_optional(len([value for value in returns if value > 0]) / len(returns)) if returns else None,
        "avg_return_pct": _round_optional(sum(returns) / len(returns)) if returns else None,
        "total_compounded_return_pct": _compound_return(returns),
    }


def _build_baselines(bars_by_symbol: dict[str, list[PriceBar]], args: argparse.Namespace) -> dict:
    always_returns = []
    ma20_returns = []
    by_symbol = {}
    for symbol, bars in bars_by_symbol.items():
        windows = _baseline_windows(bars, args)
        symbol_always = [_apply_trade_cost(item["gross_return_pct"], args.cost_bps) for item in windows]
        symbol_ma20 = [
            _apply_trade_cost(item["gross_return_pct"], args.cost_bps)
            for item in windows
            if item["entry_close"] > item["ma20"]
        ]
        always_returns.extend(symbol_always)
        ma20_returns.extend(symbol_ma20)
        by_symbol[symbol] = {
            "always_hold": _return_stats(symbol_always),
            "ma20_trend": _return_stats(symbol_ma20),
        }
    return {
        "always_hold": _summary_from_returns(always_returns),
        "ma20_trend": _summary_from_returns(ma20_returns),
        "by_symbol": by_symbol,
    }


def _baseline_windows(bars: list[PriceBar], args: argparse.Namespace) -> list[dict]:
    if len(bars) < args.context + args.horizon + 1:
        return []
    output = []
    end_indexes = list(range(args.context, len(bars) - args.horizon + 1, args.step))
    if args.max_windows:
        end_indexes = end_indexes[-args.max_windows :]
    for end_idx in end_indexes:
        context_bars = bars[end_idx - args.context : end_idx]
        entry_bar = bars[end_idx - 1]
        exit_bar = bars[end_idx + args.horizon - 1]
        ma_closes = [float(bar.close) for bar in context_bars[-20:] if bar.close is not None]
        if len(ma_closes) < 20 or entry_bar.close is None or exit_bar.close is None:
            continue
        gross = (float(exit_bar.close) / float(entry_bar.close) - 1.0) * 100.0
        output.append(
            {
                "entry_close": float(entry_bar.close),
                "ma20": sum(ma_closes) / len(ma_closes),
                "gross_return_pct": gross,
            }
        )
    return output


def _grid_search(rows: list[BacktestTrade], args: argparse.Namespace) -> list[dict]:
    output = []
    for positive in _float_list(args.grid_positive_thresholds):
        for probability in _float_list(args.grid_probability_thresholds):
            for adverse in _float_list(args.grid_max_adverse_thresholds):
                grid_args = argparse.Namespace(
                    positive_threshold=positive,
                    negative_threshold=-abs(positive),
                    probability_threshold=probability,
                    max_adverse_threshold=adverse,
                )
                returns = []
                trade_count_by_symbol: dict[str, int] = {}
                for row in rows:
                    forecast = TimesFMForecast(**row.forecast)
                    action, _, _ = _candidate_from_forecast(row.symbol, forecast, grid_args)
                    if action == "NO_TRADE":
                        continue
                    returns.append(_apply_trade_cost(row.gross_return_pct, args.cost_bps))
                    trade_count_by_symbol[row.symbol] = trade_count_by_symbol.get(row.symbol, 0) + 1
                stats = _summary_from_returns(returns)
                output.append(
                    {
                        "positive_threshold_pct": positive,
                        "probability_threshold": probability,
                        "max_adverse_threshold_pct": adverse,
                        "trade_count_by_symbol": trade_count_by_symbol,
                        **stats,
                    }
                )
    return sorted(
        output,
        key=lambda item: (
            item.get("total_compounded_return_pct") is not None,
            item.get("total_compounded_return_pct") or -10**9,
            item.get("directional_trades") or 0,
        ),
        reverse=True,
    )


def _summary_from_returns(returns: list[float]) -> dict:
    wins = [value for value in returns if value > 0]
    losses = [value for value in returns if value <= 0]
    return {
        "directional_trades": len(returns),
        "win_rate": _round_optional(len(wins) / len(returns)) if returns else None,
        "avg_return_pct": _round_optional(sum(returns) / len(returns)) if returns else None,
        "median_return_pct": _median(returns),
        "total_compounded_return_pct": _compound_return(returns),
        "avg_win_pct": _round_optional(sum(wins) / len(wins)) if wins else None,
        "avg_loss_pct": _round_optional(sum(losses) / len(losses)) if losses else None,
        "max_drawdown_pct": _max_drawdown(returns),
    }


def _calibration(rows: list[BacktestTrade]) -> dict:
    clean = [
        row
        for row in rows
        if row.forecast.get("probability_positive") is not None
        and row.forecast.get("q50_return_pct") is not None
    ]
    buckets = []
    for low, high in [(0.0, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 1.01)]:
        bucket = [
            row
            for row in clean
            if low <= float(row.forecast.get("probability_positive")) < high
        ]
        returns = [row.gross_return_pct for row in bucket]
        buckets.append(
            {
                "probability_bucket": f"{low:.2f}-{min(high, 1.0):.2f}",
                "count": len(bucket),
                "avg_probability_positive": _round_optional(
                    sum(float(row.forecast.get("probability_positive")) for row in bucket) / len(bucket)
                )
                if bucket
                else None,
                "actual_positive_rate": _round_optional(len([value for value in returns if value > 0]) / len(returns))
                if returns
                else None,
                "avg_actual_return_pct": _round_optional(sum(returns) / len(returns)) if returns else None,
            }
        )
    q50_positive = [row for row in clean if float(row.forecast.get("q50_return_pct")) > 0]
    q50_negative = [row for row in clean if float(row.forecast.get("q50_return_pct")) < 0]
    return {
        "row_count": len(clean),
        "probability_buckets": buckets,
        "q50_positive": _directional_quality(q50_positive),
        "q50_negative": _directional_quality(q50_negative, expect_negative=True),
    }


def _directional_quality(rows: list[BacktestTrade], expect_negative: bool = False) -> dict:
    returns = [row.gross_return_pct for row in rows]
    if not returns:
        return {"count": 0, "hit_rate": None, "avg_actual_return_pct": None}
    hits = [value for value in returns if value < 0] if expect_negative else [value for value in returns if value > 0]
    return {
        "count": len(returns),
        "hit_rate": _round_optional(len(hits) / len(returns)),
        "avg_actual_return_pct": _round_optional(sum(returns) / len(returns)),
    }


def _ranking_evaluation(rows: list[BacktestTrade]) -> dict:
    by_date: dict[str, list[BacktestTrade]] = {}
    for row in rows:
        by_date.setdefault(row.decision_date, []).append(row)

    evaluated = []
    for decision_date, group in sorted(by_date.items()):
        symbols = {row.symbol for row in group}
        if not set(DEFAULT_SYMBOLS).issubset(symbols):
            continue
        ranked_by_forecast = sorted(
            group,
            key=lambda row: float(row.forecast.get("probability_positive") or 0.0),
            reverse=True,
        )
        ranked_by_actual = sorted(group, key=lambda row: row.gross_return_pct, reverse=True)
        top_forecast = ranked_by_forecast[0]
        top_actual = ranked_by_actual[0]
        evaluated.append(
            {
                "decision_date": decision_date,
                "top_forecast_symbol": top_forecast.symbol,
                "top_actual_symbol": top_actual.symbol,
                "top_forecast_actual_return_pct": top_forecast.gross_return_pct,
                "top_actual_return_pct": top_actual.gross_return_pct,
                "hit": top_forecast.symbol == top_actual.symbol,
            }
        )
    returns = [item["top_forecast_actual_return_pct"] for item in evaluated]
    hits = [item for item in evaluated if item["hit"]]
    return {
        "evaluated_dates": len(evaluated),
        "top1_hit_rate": _round_optional(len(hits) / len(evaluated)) if evaluated else None,
        "top_forecast_avg_return_pct": _round_optional(sum(returns) / len(returns)) if returns else None,
        "top_forecast_total_compounded_return_pct": _compound_return(returns),
        "recent": evaluated[-10:],
    }


def _by_year(rows: list[BacktestTrade]) -> dict:
    by_year: dict[str, list[BacktestTrade]] = {}
    for row in rows:
        by_year.setdefault(row.decision_date[:4], []).append(row)
    return {year: _summary(items) for year, items in sorted(by_year.items())}


def _latest_forecast_matrix(rows: list[BacktestTrade], args: argparse.Namespace) -> dict:
    latest_by_symbol: dict[str, BacktestTrade] = {}
    for row in rows:
        current = latest_by_symbol.get(row.symbol)
        if current is None or row.decision_date > current.decision_date:
            latest_by_symbol[row.symbol] = row
    forecasts = {
        symbol: TimesFMForecast(**row.forecast)
        for symbol, row in latest_by_symbol.items()
    }
    return build_timesfm_forecast_matrix(
        forecasts,
        horizon_days=args.horizon,
        q10_floor=args.max_adverse_threshold,
    )


def write_outputs(result: dict, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"timesfm_backtest_{stamp}.json"
    md_path = reports_dir / f"timesfm_backtest_{stamp}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def render_markdown(result: dict) -> str:
    summary = result.get("summary", {})
    model = result.get("model", {})
    lines = [
        "# TimesFM Walk-Forward Backtest",
        "",
        f"Generated UTC: `{result.get('generated_at_utc')}`",
        "",
        "## Model",
        "",
        f"- Model: `{model.get('model_id')}`",
        f"- Context: `{model.get('context')}` daily bars",
        f"- Horizon: `{model.get('horizon_days')}` trading day(s)",
        f"- Period: `{model.get('period')}`",
        f"- Cost: `{result.get('policy', {}).get('costs')}`",
        "",
        "## Summary",
        "",
        f"- Directional trades: `{summary.get('directional_trades')}`",
        f"- Win rate: `{_fmt_pct_ratio(summary.get('win_rate'))}`",
        f"- Average return: `{_fmt_pct(summary.get('avg_return_pct'))}`",
        f"- Median return: `{_fmt_pct(summary.get('median_return_pct'))}`",
        f"- Total compounded return: `{_fmt_pct(summary.get('total_compounded_return_pct'))}`",
        f"- Max drawdown: `{_fmt_pct(summary.get('max_drawdown_pct'))}`",
        "",
        "## By Symbol",
        "",
        "| Symbol | Trades | Win rate | Avg return | Compounded return |",
        "|---|---:|---:|---:|---:|",
    ]
    for symbol, payload in summary.get("by_symbol", {}).items():
        lines.append(
            f"| {symbol} | {payload.get('trades', 0)} | {_fmt_pct_ratio(payload.get('win_rate'))} | "
            f"{_fmt_pct(payload.get('avg_return_pct'))} | {_fmt_pct(payload.get('total_compounded_return_pct'))} |"
        )
    calibration = result.get("calibration", {})
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- Rows evaluated: `{calibration.get('row_count', 0)}`",
            f"- q50 > 0 hit rate: `{_fmt_pct_ratio(calibration.get('q50_positive', {}).get('hit_rate'))}` "
            f"avg actual `{_fmt_pct(calibration.get('q50_positive', {}).get('avg_actual_return_pct'))}`",
            f"- q50 < 0 hit rate: `{_fmt_pct_ratio(calibration.get('q50_negative', {}).get('hit_rate'))}` "
            f"avg actual `{_fmt_pct(calibration.get('q50_negative', {}).get('avg_actual_return_pct'))}`",
            "",
            "| Prob+ bucket | Count | Avg prob+ | Actual positive rate | Avg actual return |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket in calibration.get("probability_buckets", []):
        lines.append(
            f"| `{bucket.get('probability_bucket')}` | {bucket.get('count', 0)} | "
            f"{_fmt_pct_ratio(bucket.get('avg_probability_positive'))} | "
            f"{_fmt_pct_ratio(bucket.get('actual_positive_rate'))} | "
            f"{_fmt_pct(bucket.get('avg_actual_return_pct'))} |"
        )
    ranking_eval = result.get("ranking_evaluation", {})
    lines.extend(
        [
            "",
            "## Ranking Evaluation",
            "",
            f"- Evaluated dates: `{ranking_eval.get('evaluated_dates', 0)}`",
            f"- Top-1 hit rate: `{_fmt_pct_ratio(ranking_eval.get('top1_hit_rate'))}`",
            f"- Top forecast avg return: `{_fmt_pct(ranking_eval.get('top_forecast_avg_return_pct'))}`",
            f"- Top forecast compounded return: `{_fmt_pct(ranking_eval.get('top_forecast_total_compounded_return_pct'))}`",
        ]
    )
    by_year = result.get("by_year", {})
    if by_year:
        lines.extend(
            [
                "",
                "## By Year",
                "",
                "| Year | Forecast rows | Directional trades | Win rate | Avg return | Compounded |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for year, payload in by_year.items():
            lines.append(
                f"| {year} | {payload.get('all_rows', 0)} | {payload.get('directional_trades', 0)} | "
                f"{_fmt_pct_ratio(payload.get('win_rate'))} | {_fmt_pct(payload.get('avg_return_pct'))} | "
                f"{_fmt_pct(payload.get('total_compounded_return_pct'))} |"
            )
    matrix = result.get("latest_forecast_matrix", {})
    matrix_symbols = matrix.get("symbols", {})
    if matrix_symbols:
        lines.extend(
            [
                "",
                "## Latest TimesFM Matrix",
                "",
                f"- Ranking: `{', '.join(matrix.get('ranking', [])) or 'none'}`",
                f"- Warnings: `{', '.join(matrix.get('warnings', [])) or 'none'}`",
                "",
                "| Symbol | Hint | Score | q10 | q50 | q90 | Prob+ |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for symbol in ["TQQQ", "SQQQ", "SOXL", "SOXS"]:
            payload = matrix_symbols.get(symbol, {})
            if not payload:
                continue
            lines.append(
                f"| {symbol} | `{payload.get('candidate_hint')}` | {payload.get('forecast_score')} | "
                f"{_fmt_pct(payload.get('q10_return_pct'))} | {_fmt_pct(payload.get('q50_return_pct'))} | "
                f"{_fmt_pct(payload.get('q90_return_pct'))} | {_fmt_pct_ratio(payload.get('probability_positive'))} |"
            )
    baselines = result.get("baselines", {})
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | Trades | Win rate | Avg return | Compounded return | Max drawdown |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ["always_hold", "ma20_trend"]:
        payload = baselines.get(name, {})
        lines.append(
            f"| `{name}` | {payload.get('directional_trades', 0)} | {_fmt_pct_ratio(payload.get('win_rate'))} | "
            f"{_fmt_pct(payload.get('avg_return_pct'))} | {_fmt_pct(payload.get('total_compounded_return_pct'))} | "
            f"{_fmt_pct(payload.get('max_drawdown_pct'))} |"
        )
    grid = result.get("grid_search", [])
    if grid:
        lines.extend(
            [
                "",
                "## Grid Search Top 10",
                "",
                "| Rank | Pos threshold | Prob threshold | q10 floor | Trades | Win rate | Avg return | Compounded | Max DD |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for idx, item in enumerate(grid[:10], start=1):
            lines.append(
                f"| {idx} | {_fmt_pct(item.get('positive_threshold_pct'))} | {_fmt_pct_ratio(item.get('probability_threshold'))} | "
                f"{_fmt_pct(item.get('max_adverse_threshold_pct'))} | {item.get('directional_trades', 0)} | "
                f"{_fmt_pct_ratio(item.get('win_rate'))} | {_fmt_pct(item.get('avg_return_pct'))} | "
                f"{_fmt_pct(item.get('total_compounded_return_pct'))} | {_fmt_pct(item.get('max_drawdown_pct'))} |"
            )
    lines.extend(["", "## Recent Trades", "", "| Date | Exit | Symbol | Action | Forecast q50 | Prob+ | Realized |", "|---|---|---|---|---:|---:|---:|"])
    directional = [item for item in result.get("trades", []) if item.get("action") != "NO_TRADE"]
    for item in directional[-20:]:
        forecast = item.get("forecast", {})
        lines.append(
            f"| {item.get('decision_date')} | {item.get('exit_date')} | {item.get('symbol')} | `{item.get('action')}` | "
            f"{_fmt_pct(forecast.get('q50_return_pct'))} | {_fmt_pct_ratio(forecast.get('probability_positive'))} | "
            f"{_fmt_pct(item.get('realized_return_pct'))} |"
        )
    if not directional:
        lines.append("| none | none | none | `NO_TRADE` | N/A | N/A | N/A |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TimesFM daily walk-forward backtest for leveraged ETF candidates.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Tradeable symbols. Only TQQQ/SQQQ/SOXL/SOXS are allowed.")
    parser.add_argument("--period", default="5y", help="yfinance history period, for example 2y, 5y, 10y.")
    parser.add_argument("--context", type=int, default=256, help="Daily bars supplied to TimesFM for each decision.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast and holding horizon in trading days.")
    parser.add_argument("--step", type=int, default=5, help="Walk-forward stride in trading days.")
    parser.add_argument("--max-windows", type=int, default=80, help="Limit windows per symbol from the most recent history. Use 0 for all.")
    parser.add_argument("--timeout", type=int, default=20, help="Data request timeout in seconds.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face TimesFM model id.")
    parser.add_argument("--positive-threshold", type=float, default=1.5, help="Minimum q50/point forecast return for a candidate.")
    parser.add_argument("--negative-threshold", type=float, default=-1.5, help="Forecast threshold for rejecting a symbol.")
    parser.add_argument("--probability-threshold", type=float, default=0.60, help="Minimum positive probability for a candidate.")
    parser.add_argument("--max-adverse-threshold", type=float, default=-2.0, help="Minimum acceptable q10 return for a candidate.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Round-trip cost in basis points deducted from directional trades.")
    parser.add_argument("--grid", action="store_true", help="Run threshold grid search over cached TimesFM forecast rows.")
    parser.add_argument("--grid-positive-thresholds", default="0,0.5,1.0,1.5,2.0", help="Comma-separated q50/point return thresholds.")
    parser.add_argument("--grid-probability-thresholds", default="0.5,0.55,0.6,0.65", help="Comma-separated positive probability thresholds.")
    parser.add_argument("--grid-max-adverse-thresholds", default="-20,-10,-5,-2", help="Comma-separated q10 floor thresholds.")
    parser.add_argument("--include-no-trade", action="store_true", help="Include NO_TRADE rows in output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_windows == 0:
        args.max_windows = None
    try:
        result = run_backtest(args)
    except TimesFMUnavailable as exc:
        print(f"TimesFM unavailable: {exc}")
        raise SystemExit(2) from exc
    json_path, md_path = write_outputs(result)
    summary = result["summary"]
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(
        "TimesFM backtest: "
        f"trades={summary.get('directional_trades')}, "
        f"win_rate={_fmt_pct_ratio(summary.get('win_rate'))}, "
        f"avg_return={_fmt_pct(summary.get('avg_return_pct'))}, "
        f"compounded={_fmt_pct(summary.get('total_compounded_return_pct'))}"
    )


def _clean_bars(bars: list[PriceBar]) -> list[PriceBar]:
    return [bar for bar in bars if bar.close is not None and bar.close > 0]


def _validate_symbols(symbols: Iterable[str]) -> list[str]:
    allowed = set(TRADE_SYMBOLS)
    output = []
    for symbol in symbols:
        normalized = symbol.upper().strip()
        if normalized not in allowed:
            raise ValueError(f"{normalized} is not allowed. TimesFM backtest only supports {sorted(allowed)}.")
        if normalized not in output:
            output.append(normalized)
    return output


def _confidence(return_pct: float, probability: Optional[float]) -> float:
    prob = probability if probability is not None else 0.5
    raw = 0.35 + min(abs(return_pct), 8.0) / 20.0 + max(0.0, prob - 0.5)
    return round(max(0.0, min(0.9, raw)), 4)


def _apply_trade_cost(gross_return_pct: float, cost_bps: float) -> float:
    return gross_return_pct - cost_bps / 100.0


def _float_list(raw: str) -> list[float]:
    output = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        output.append(float(item))
    return output


def _compound_return(returns: list[float]) -> Optional[float]:
    if not returns:
        return None
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value / 100.0
    return round((equity - 1.0) * 100.0, 4)


def _max_drawdown(returns: list[float]) -> Optional[float]:
    if not returns:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in returns:
        equity *= 1.0 + value / 100.0
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        max_dd = min(max_dd, drawdown)
    return round(max_dd * 100.0, 4)


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return round(sorted_values[mid], 4)
    return round((sorted_values[mid - 1] + sorted_values[mid]) / 2.0, 4)


def _round_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def _fmt_pct_ratio(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    main()
