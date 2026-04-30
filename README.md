# TimesFM VIX Squeeze TQQQ Strategy

This repository keeps one strategy:

- Forecast VIX with `google/timesfm-2.5-200m-pytorch`.
- Trade only TQQQ.
- Use strict T+1 execution: today's close/VIX data decides tomorrow's exposure.
- Reduce or exit risk when VIX spikes unless TimesFM strongly confirms mean reversion.

Run:

```bash
python scripts/run_vix_squeeze_backtest.py
```

Latest verified run:

- Period: 2022-01-03 to 2026-04-30
- Strategy: 3.00x
- Return: 199.6%
- CAGR: 28.9%
- Max drawdown: -29.1%
- Sharpe: 1.19
- TQQQ benchmark: 1.51x, return 51.0%, max drawdown -81.0%
- Trades: 17
- Long exposure: 17.4%

The chart is saved to `reports/vix_squeeze_timesfm_report.png`.
