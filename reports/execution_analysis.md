# Execution Analysis Report

Database: `results/paper_trades.db`

## A_5pct_60s

### Execution Quality

| Metric | Value |
|--------|-------|
| Trades | 4 |
| Entry Maker rate | 100.0% |
| Exit Maker rate | 100.0% |
| Exit Taker fallback | 0.0% |
| Avg time-to-fill | 3105 ms |
| Median time-to-fill | 3990 ms |
| P90 time-to-fill | 4040 ms |
| Avg spread at entry | 0.0140 bps |
| Signals total | 10 |
| Signal accept rate | 100.0% |
| Rejected (spread) | 0 |
| Order fill rate | 40.0% |
| Order cancel rate | 60.0% |

### PnL Decomposition

| Component | Value (bps) |
|-----------|------------|
| Total Gross PnL | -3.45 |
| Maker cost (entry+exit) | 0.00 |
| Taker penalty (fallback) | 0.00 |
| Slippage (total) | 4.00 |
| **Total Net PnL** | **-7.45** |

### What-If Scenarios

| Scenario | Net PnL (bps) |
|----------|--------------|
| Actual | -7.45 |
| All Maker (best case) | -7.45 |
| All Taker (worst case) | -51.45 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Avg gross bps/trade | -0.861 |
| Avg net bps/trade | -1.861 |
| Win rate | 25.0% |
| Avg win | +5.155 bps |
| Avg loss | -4.200 bps |
| Max win | +5.155 bps |
| Max loss | -10.741 bps |
| Profit factor | 0.41 |
| Sharpe | -0.65 |
| Max drawdown | 11.4 bps |

## B_10pct_120s

### Execution Quality

| Metric | Value |
|--------|-------|
| Trades | 2 |
| Entry Maker rate | 100.0% |
| Exit Maker rate | 100.0% |
| Exit Taker fallback | 0.0% |
| Avg time-to-fill | 3680 ms |
| Median time-to-fill | 3680 ms |
| P90 time-to-fill | 3969 ms |
| Avg spread at entry | 0.0140 bps |
| Signals total | 6 |
| Signal accept rate | 100.0% |
| Rejected (spread) | 0 |
| Order fill rate | 33.3% |
| Order cancel rate | 66.7% |

### PnL Decomposition

| Component | Value (bps) |
|-----------|------------|
| Total Gross PnL | -10.94 |
| Maker cost (entry+exit) | 0.00 |
| Taker penalty (fallback) | 0.00 |
| Slippage (total) | 2.00 |
| **Total Net PnL** | **-12.94** |

### What-If Scenarios

| Scenario | Net PnL (bps) |
|----------|--------------|
| Actual | -12.94 |
| All Maker (best case) | -12.94 |
| All Taker (worst case) | -34.94 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Avg gross bps/trade | -5.468 |
| Avg net bps/trade | -6.468 |
| Win rate | 0.0% |
| Avg win | +0.000 bps |
| Avg loss | -6.468 bps |
| Max win | -0.874 bps |
| Max loss | -12.063 bps |
| Profit factor | 0.00 |
| Sharpe | -1.64 |
| Max drawdown | 12.1 bps |
