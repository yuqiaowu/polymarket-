# TimesFM VIX Squeeze TQQQ Backtest

Generated from `scripts/run_vix_squeeze_backtest.py` on 2026-04-30.

## Conclusion

This is the retained strategy. It uses TimesFM to forecast VIX, then trades only TQQQ from the VIX squeeze regime.

## Latest Result

| Metric | Strategy | TQQQ Buy & Hold |
|---|---:|---:|
| Period | 2022-01-03 to 2026-04-30 | 2022-01-03 to 2026-04-30 |
| Final multiple | 3.00x | 1.51x |
| Total return | 199.6% | 51.0% |
| CAGR | 28.9% | 10.0% |
| Max drawdown | -29.1% | -81.0% |
| Sharpe | 1.19 | N/A |

Additional strategy stats:

- Trades: 17
- Long exposure: 17.4%
- Execution model: strict T+1 daily close
- Transaction cost: 5 bps per exposure turnover

The chart is `reports/vix_squeeze_timesfm_report.png`.
