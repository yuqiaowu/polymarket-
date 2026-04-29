import os
import sys
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from market_system.vix_squeeze_qlib_strategy import VixSqueezeQlibEngine


START_DATE = datetime(2022, 1, 1)
TRANSACTION_COST = 0.0005


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the daily TQQQ VIX squeeze backtest with the trained GRU forecaster.")
    parser.add_argument("--model-dir", default="models/vix_gru", help="Directory containing vix_gru_model.pth and scaler files.")
    return parser.parse_args()


def run_qlib_backtest():
    print("Starting Qlib/GRU VIX Squeeze backtest: daily TQQQ, strict T+1 execution")
    df = _load_data(START_DATE, datetime.now())
    args = _parse_args()
    model_path = os.path.join(args.model_dir, "vix_gru_model.pth")
    engine = VixSqueezeQlibEngine(model_path)

    if not engine.forecaster.is_available:
        print("WARNING: Qlib/GRU model files are missing. This run is a neutral fallback, not a real Qlib comparison.")

    equity_strategy = [1.0]
    equity_benchmark = [1.0]
    positions = ["HEDGE"]
    trades = []
    current_position = "HEDGE"

    for i in range(1, len(df)):
        ret = _safe_return(df["tqqq_return"].iloc[i])

        # A. Today's return is earned by yesterday's position.
        next_equity = equity_strategy[-1] if current_position == "HEDGE" else equity_strategy[-1] * (1 + ret)
        equity_strategy.append(next_equity)
        equity_benchmark.append(equity_benchmark[-1] * (1 + ret))
        positions.append(current_position)

        # B. Today's close/VIX data can only decide tomorrow's position.
        vix_history = df["vix"].iloc[: i + 1].tolist()
        signal = engine.analyze(vix_history)
        if signal.action in {"HEDGE", "LONG_TQQQ"} and signal.action != current_position:
            equity_strategy[-1] *= 1 - TRANSACTION_COST
            trades.append(
                {
                    "date": df.index[i].date().isoformat(),
                    "from": current_position,
                    "to": signal.action,
                    "vix": round(float(signal.vix_current), 2),
                    "qlib_score": round(float(signal.qlib_score), 4),
                    "confidence": round(float(signal.confidence), 2),
                }
            )
            current_position = signal.action

    result = _summarize(
        strategy_name="Qlib/GRU VIX Squeeze",
        df=df,
        equity_strategy=equity_strategy,
        equity_benchmark=equity_benchmark,
        positions=positions,
        trades=trades,
        model_available=engine.forecaster.is_available,
    )
    _print_result(result, trades)
    _save_plot(df, equity_strategy, equity_benchmark, "Qlib/GRU VIX Squeeze", "reports/vix_squeeze_qlib_report.png")
    return result


def _load_data(start_date, end_date):
    print(f"Fetching daily VIX and TQQQ data from {start_date.date()} to {end_date.date()}...")
    vix = _close_series(yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False), "^VIX")
    tqqq = _close_series(yf.download("TQQQ", start=start_date, end=end_date, auto_adjust=True, progress=False), "TQQQ")
    df = pd.concat([vix.rename("vix"), tqqq.rename("tqqq")], axis=1).dropna()
    df["tqqq_return"] = df["tqqq"].pct_change()
    if len(df) < 70:
        raise RuntimeError(f"Not enough daily bars for Qlib/GRU context: {len(df)} rows")
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


def _summarize(strategy_name, df, equity_strategy, equity_benchmark, positions, trades, model_available):
    strategy = pd.Series(equity_strategy, index=df.index[: len(equity_strategy)])
    benchmark = pd.Series(equity_benchmark, index=df.index[: len(equity_benchmark)])
    daily_returns = strategy.pct_change().fillna(0.0)
    years = max((strategy.index[-1] - strategy.index[0]).days / 365.25, 1e-9)
    total_return = strategy.iloc[-1] - 1
    benchmark_return = benchmark.iloc[-1] - 1
    cagr = strategy.iloc[-1] ** (1 / years) - 1
    benchmark_cagr = benchmark.iloc[-1] ** (1 / years) - 1
    mdd = _max_drawdown(strategy)
    benchmark_mdd = _max_drawdown(benchmark)
    sharpe = 0.0 if daily_returns.std() == 0 else daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    exposure = sum(1 for position in positions if position == "LONG_TQQQ") / len(positions)
    return {
        "strategy": strategy_name,
        "model_available": model_available,
        "period": f"{strategy.index[0].date()} to {strategy.index[-1].date()}",
        "final_multiple": strategy.iloc[-1],
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe,
        "benchmark_multiple": benchmark.iloc[-1],
        "benchmark_return_pct": benchmark_return * 100,
        "benchmark_cagr_pct": benchmark_cagr * 100,
        "benchmark_max_drawdown_pct": benchmark_mdd * 100,
        "trade_count": len(trades),
        "long_exposure_pct": exposure * 100,
    }


def _max_drawdown(equity):
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def _print_result(result, trades):
    print("\n=== QLIB/GRU VIX SQUEEZE RESULTS ===")
    print(f"Model available: {result['model_available']}")
    print(f"Period: {result['period']}")
    print(
        f"Strategy: {result['final_multiple']:.2f}x | "
        f"Return: {result['total_return_pct']:.1f}% | "
        f"CAGR: {result['cagr_pct']:.1f}% | "
        f"MDD: {result['max_drawdown_pct']:.1f}% | "
        f"Sharpe: {result['sharpe']:.2f}"
    )
    print(
        f"Benchmark TQQQ: {result['benchmark_multiple']:.2f}x | "
        f"Return: {result['benchmark_return_pct']:.1f}% | "
        f"CAGR: {result['benchmark_cagr_pct']:.1f}% | "
        f"MDD: {result['benchmark_max_drawdown_pct']:.1f}%"
    )
    print(f"Trades: {result['trade_count']} | Long exposure: {result['long_exposure_pct']:.1f}%")
    print("Recent trades:")
    for trade in trades[-10:]:
        print(f"  {trade}")


def _save_plot(df, equity_strategy, equity_benchmark, label, relative_path):
    os.makedirs(os.path.dirname(relative_path), exist_ok=True)
    plt.figure(figsize=(12, 7))
    plt.plot(df.index[: len(equity_benchmark)], equity_benchmark, label="TQQQ Buy & Hold", alpha=0.7)
    plt.plot(df.index[: len(equity_strategy)], equity_strategy, label=label, linewidth=2)
    plt.yscale("log")
    plt.title(f"{label} vs TQQQ Buy & Hold")
    plt.ylabel("Equity, log scale")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.savefig(relative_path)
    print(f"Chart saved to: {os.path.abspath(relative_path)}")


if __name__ == "__main__":
    run_qlib_backtest()
