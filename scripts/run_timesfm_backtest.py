from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from market_system.config import REPORTS_DIR, TRADE_SYMBOLS
from market_system.market_data import PriceBar, fetch_daily_bars


DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
DEFAULT_SYMBOLS = ["TQQQ", "SQQQ", "SOXL", "SOXS"]


@dataclass
class ForecastResult:
    point_return_pct: float
    q10_return_pct: Optional[float]
    q50_return_pct: Optional[float]
    q90_return_pct: Optional[float]
    probability_positive: Optional[float]


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
    realized_return_pct: float
    rule_status: str


class TimesFMUnavailable(RuntimeError):
    pass


class TimesFMForecaster:
    def __init__(self, model_id: str, max_context: int, max_horizon: int) -> None:
        self.model_id = model_id
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.model = self._load_model(model_id, max_context, max_horizon)

    def forecast(self, closes: list[float], horizon: int) -> ForecastResult:
        import numpy as np

        if len(closes) < 2:
            raise ValueError("at least two closes are required")
        if horizon > self.max_horizon:
            raise ValueError(f"horizon {horizon} exceeds max_horizon {self.max_horizon}")

        safe_closes = np.asarray([max(float(value), 1e-9) for value in closes], dtype=np.float32)
        log_closes = np.log(safe_closes)
        last_log = float(log_closes[-1])
        point_forecast, quantile_forecast = self.model.forecast(horizon=horizon, inputs=[log_closes])

        point_value = float(point_forecast[0][horizon - 1])
        point_return = _log_forecast_to_return_pct(point_value, last_log)

        q10 = q50 = q90 = probability_positive = None
        if quantile_forecast is not None:
            q_values = _quantile_values(quantile_forecast, horizon)
            if q_values:
                q10 = _log_forecast_to_return_pct(q_values.get("q10"), last_log)
                q50 = _log_forecast_to_return_pct(q_values.get("q50"), last_log)
                q90 = _log_forecast_to_return_pct(q_values.get("q90"), last_log)
                probability_positive = _probability_positive(q_values, last_log)

        if probability_positive is None:
            probability_positive = 1.0 if point_return > 0 else 0.0
        return ForecastResult(
            point_return_pct=round(point_return, 4),
            q10_return_pct=_round_optional(q10),
            q50_return_pct=_round_optional(q50),
            q90_return_pct=_round_optional(q90),
            probability_positive=_round_optional(probability_positive),
        )

    def _load_model(self, model_id: str, max_context: int, max_horizon: int) -> Any:
        try:
            import torch
            import timesfm
        except Exception as exc:  # noqa: BLE001
            raise TimesFMUnavailable(
                "TimesFM dependencies are not installed. Run: "
                "python -m pip install --index-url https://pypi.org/simple -r requirements-timesfm.txt"
            ) from exc

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        try:
            model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch")
            model = model_cls.from_pretrained(model_id)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=max_context,
                    max_horizon=max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            return model
        except Exception as exc:  # noqa: BLE001
            raise TimesFMUnavailable(f"Failed to load TimesFM model {model_id}: {exc}") from exc


def run_backtest(args: argparse.Namespace) -> dict:
    symbols = _validate_symbols(args.symbols)
    bars_by_symbol = fetch_daily_bars(symbols, period=args.period, timeout=args.timeout)
    forecaster = TimesFMForecaster(args.model, args.context, args.horizon)

    trades: list[BacktestTrade] = []
    data_status = {}
    for symbol in symbols:
        bars = _clean_bars(bars_by_symbol.get(symbol, []))
        data_status[symbol] = {"bar_count": len(bars), "first_date": bars[0].date if bars else None, "last_date": bars[-1].date if bars else None}
        trades.extend(_backtest_symbol(symbol, bars, forecaster, args))

    summary = _summary(trades)
    return {
        "schema_version": "timesfm_backtest_v0.1",
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
            "costs": "not included",
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
        if action == "NO_TRADE" and not args.include_no_trade:
            continue
        if entry_bar.close is None or exit_bar.close is None:
            continue

        realized = (float(exit_bar.close) / float(entry_bar.close) - 1.0) * 100.0
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
                realized_return_pct=round(realized, 4),
                rule_status=rule_status,
            )
        )
    return output


def _candidate_from_forecast(symbol: str, forecast: ForecastResult, args: argparse.Namespace) -> tuple[str, float, str]:
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


def _quantile_values(quantile_forecast: Any, horizon: int) -> dict[str, float]:
    values = quantile_forecast[0][horizon - 1]
    length = len(values)
    if length >= 10:
        return {"q10": float(values[1]), "q50": float(values[5]), "q90": float(values[9])}
    if length >= 9:
        return {"q10": float(values[0]), "q50": float(values[4]), "q90": float(values[8])}
    return {}


def _probability_positive(q_values: dict[str, float], last_log: float) -> Optional[float]:
    quantiles = [(0.10, q_values.get("q10")), (0.50, q_values.get("q50")), (0.90, q_values.get("q90"))]
    clean = [(prob, value) for prob, value in quantiles if value is not None]
    if not clean:
        return None
    threshold = last_log
    if clean[0][1] >= threshold:
        return 0.9
    if clean[-1][1] <= threshold:
        return 0.1
    for (left_p, left_v), (right_p, right_v) in zip(clean, clean[1:]):
        if left_v <= threshold <= right_v and right_v != left_v:
            cdf = left_p + (right_p - left_p) * ((threshold - left_v) / (right_v - left_v))
            return max(0.0, min(1.0, 1.0 - cdf))
    return None


def _log_forecast_to_return_pct(value: Optional[float], last_log: float) -> float:
    if value is None:
        return 0.0
    return (math.exp(float(value) - last_log) - 1.0) * 100.0


def _confidence(return_pct: float, probability: Optional[float]) -> float:
    prob = probability if probability is not None else 0.5
    raw = 0.35 + min(abs(return_pct), 8.0) / 20.0 + max(0.0, prob - 0.5)
    return round(max(0.0, min(0.9, raw)), 4)


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
