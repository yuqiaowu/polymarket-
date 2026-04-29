import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from market_system.timesfm_strategy import TimesFMForecaster
from market_system.vix_squeeze_strategy import VixSqueezeEngine


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
START_DATE = datetime(2022, 1, 1)
CONTEXT_DAYS = 256
HORIZON_DAYS = 5
TRANSACTION_COST = 0.0005


def main():
    forecaster = TimesFMForecaster(MODEL_ID, CONTEXT_DAYS, HORIZON_DAYS)
    engine = VixSqueezeEngine(high_vix_threshold=25.0, low_vix_threshold=15.0)
    df = _load_data(START_DATE, datetime.now())
    signals = _build_signals(df, forecaster, engine)

    partial = _run_backtest(df, signals, spike_mode="partial")
    no_spike = _run_backtest(df, signals, spike_mode="off")
    benchmark = pd.Series(partial["benchmark"], index=df.index[: len(partial["benchmark"])])

    partial_equity = pd.Series(partial["equity"], index=df.index[: len(partial["equity"])])
    no_spike_equity = pd.Series(no_spike["equity"], index=df.index[: len(no_spike["equity"])])

    rows = [
        ("Partial Spike Reduction", partial_equity, partial["events"]),
        ("No Spike Protection", no_spike_equity, no_spike["events"]),
        ("TQQQ Buy & Hold", benchmark, []),
    ]
    print("\n=== VIX SQUEEZE SPIKE MODE COMPARISON ===")
    for name, equity, events in rows:
        stats = _metrics(equity)
        print(
            f"{name}: {stats['multiple']:.2f}x | "
            f"Return {stats['return_pct']:.1f}% | "
            f"CAGR {stats['cagr_pct']:.1f}% | "
            f"MDD {stats['mdd_pct']:.1f}% | "
            f"Sharpe {stats['sharpe']:.2f} | "
            f"Events {len(events)}"
        )

    path = "reports/vix_squeeze_spike_mode_comparison.png"
    _save_plot(
        {
            "Partial Spike Reduction": partial_equity,
            "No Spike Protection": no_spike_equity,
            "TQQQ Buy & Hold": benchmark,
        },
        path,
    )
    print(f"Chart saved to: {os.path.abspath(path)}")


def _build_signals(df, forecaster, engine):
    signals = {}
    for i in range(1, len(df)):
        if i + 1 < CONTEXT_DAYS:
            continue
        vix_history = df["vix"].iloc[max(0, i - 50): i + 1].tolist()
        context_window = df["vix"].iloc[max(0, i - CONTEXT_DAYS + 1): i + 1].tolist()
        forecast = forecaster.forecast(context_window, HORIZON_DAYS)
        signals[i] = engine.analyze_timesfm(vix_history, forecast)
    return signals


def _run_backtest(df, signals, spike_mode):
    equity = [1.0]
    benchmark = [1.0]
    exposure = 0.0
    events = []

    for i in range(1, len(df)):
        ret = _safe_return(df["tqqq_return"].iloc[i])
        equity.append(equity[-1] * (1 + exposure * ret))
        benchmark.append(benchmark[-1] * (1 + ret))

        signal = signals.get(i)
        if signal is None:
            continue

        target = _target_exposure(signal, exposure, spike_mode)
        if target >= 0.0 and target != exposure:
            turnover = abs(target - exposure)
            equity[-1] *= 1 - TRANSACTION_COST * turnover
            events.append(
                {
                    "date": df.index[i].date().isoformat(),
                    "from": exposure,
                    "to": target,
                    "action": signal.action,
                    "reason": signal.reason,
                    "vix": signal.vix_current,
                    "q50": signal.vix_forecast_mean,
                    "slope": signal.vix_forecast_slope,
                }
            )
            exposure = target
    return {"equity": equity, "benchmark": benchmark, "events": events}


def _target_exposure(signal, current_exposure, spike_mode):
    if spike_mode == "off" and signal.reason.startswith("SPIKE_"):
        if signal.reason == "SPIKE_FULL_HEDGE":
            return -1.0
        return current_exposure
    if signal.action == "REDUCE_RISK":
        return min(current_exposure, signal.target_exposure)
    return signal.target_exposure


def _load_data(start_date, end_date):
    vix = _close_series(yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False), "^VIX")
    tqqq = _close_series(yf.download("TQQQ", start=start_date, end=end_date, auto_adjust=True, progress=False), "TQQQ")
    df = pd.concat([vix.rename("vix"), tqqq.rename("tqqq")], axis=1).dropna()
    df["tqqq_return"] = df["tqqq"].pct_change()
    if len(df) < CONTEXT_DAYS + 10:
        raise RuntimeError(f"Not enough daily bars for TimesFM context: {len(df)} rows")
    return df


def _close_series(frame, symbol):
    if frame.empty:
        raise RuntimeError(f"No data downloaded for {symbol}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


def _safe_return(value):
    return 0.0 if pd.isna(value) else float(value)


def _metrics(equity):
    returns = equity.pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    sharpe = 0.0 if returns.std() == 0 else returns.mean() / returns.std() * np.sqrt(252)
    return {
        "multiple": equity.iloc[-1],
        "return_pct": (equity.iloc[-1] - 1) * 100,
        "cagr_pct": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "mdd_pct": (equity / equity.cummax() - 1).min() * 100,
        "sharpe": sharpe,
    }


def _save_plot(series_by_label, relative_path):
    os.makedirs(os.path.dirname(relative_path), exist_ok=True)
    plt.figure(figsize=(13, 7))
    for label, series in series_by_label.items():
        plt.plot(series.index, series.values, label=label, linewidth=2 if label != "TQQQ Buy & Hold" else 1.6)
    plt.yscale("log")
    plt.title("TimesFM VIX Squeeze: Partial Spike Reduction vs No Spike Protection")
    plt.ylabel("Equity, log scale")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.35)
    plt.savefig(relative_path)


if __name__ == "__main__":
    main()
