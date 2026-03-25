# Execution Analysis Report

Database: `results/paper_trades.db`

## A_5pct_60s

### Execution Quality

| Metric | Value |
|--------|-------|
| Trades | 15 |
| Entry Maker rate | 100.0% |
| Exit Maker rate | 100.0% |
| Exit Taker fallback | 0.0% |
| Avg time-to-fill | 2864 ms |
| Median time-to-fill | 2841 ms |
| P90 time-to-fill | 4041 ms |
| Avg spread at entry | 0.0140 bps |
| Signals total | 45 |
| Signal accept rate | 100.0% |
| Rejected (spread) | 0 |
| Order fill rate | 33.3% |
| Order cancel rate | 66.7% |

### PnL Decomposition

| Component | Value (bps) |
|-----------|------------|
| Total Gross PnL | -20.36 |
| Maker cost (entry+exit) | 0.00 |
| Taker penalty (fallback) | 0.00 |
| Slippage (total) | 15.00 |
| **Total Net PnL** | **-35.36** |

### What-If Scenarios

| Scenario | Net PnL (bps) |
|----------|--------------|
| Actual | -35.36 |
| All Maker (best case) | -35.36 |
| All Taker (worst case) | -200.36 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Avg gross bps/trade | -1.357 |
| Avg net bps/trade | -2.357 |
| Win rate | 20.0% |
| Avg win | +3.439 bps |
| Avg loss | -3.806 bps |
| Max win | +5.155 bps |
| Max loss | -10.741 bps |
| Profit factor | 0.23 |
| Sharpe | -2.24 |
| Max drawdown | 39.3 bps |

### Adverse Selection Analysis

Unfilled (cancelled) orders: **30** vs filled: **15**

What if we had chased unfilled signals with Taker entry?

| Metric | Filled (actual) | Unfilled (Taker chase) |
|--------|----------------|----------------------|
| Avg net bps | -2.357 | -11.358 |
| Win rate | 20.0% | 0.0% |
| Total net bps | -35.4 | -329.4 |

No adverse selection detected. Filled trades are comparable to or better
than what unfilled signals would have produced.

## B_10pct_120s

### Execution Quality

| Metric | Value |
|--------|-------|
| Trades | 6 |
| Entry Maker rate | 100.0% |
| Exit Maker rate | 100.0% |
| Exit Taker fallback | 0.0% |
| Avg time-to-fill | 2320 ms |
| Median time-to-fill | 1780 ms |
| P90 time-to-fill | 3680 ms |
| Avg spread at entry | 0.0140 bps |
| Signals total | 26 |
| Signal accept rate | 100.0% |
| Rejected (spread) | 0 |
| Order fill rate | 23.1% |
| Order cancel rate | 76.9% |

### PnL Decomposition

| Component | Value (bps) |
|-----------|------------|
| Total Gross PnL | -20.32 |
| Maker cost (entry+exit) | 0.00 |
| Taker penalty (fallback) | 0.00 |
| Slippage (total) | 6.00 |
| **Total Net PnL** | **-26.32** |

### What-If Scenarios

| Scenario | Net PnL (bps) |
|----------|--------------|
| Actual | -26.32 |
| All Maker (best case) | -26.32 |
| All Taker (worst case) | -92.32 |

### Trade Statistics

| Metric | Value |
|--------|-------|
| Avg gross bps/trade | -3.386 |
| Avg net bps/trade | -4.386 |
| Win rate | 16.7% |
| Avg win | +2.132 bps |
| Avg loss | -5.690 bps |
| Max win | +2.132 bps |
| Max loss | -12.063 bps |
| Profit factor | 0.07 |
| Sharpe | -2.24 |
| Max drawdown | 25.4 bps |

### Adverse Selection Analysis

Unfilled (cancelled) orders: **20** vs filled: **6**

What if we had chased unfilled signals with Taker entry?

| Metric | Filled (actual) | Unfilled (Taker chase) |
|--------|----------------|----------------------|
| Avg net bps | -4.386 | -10.455 |
| Win rate | 16.7% | 5.0% |
| Total net bps | -26.3 | -209.1 |

No adverse selection detected. Filled trades are comparable to or better
than what unfilled signals would have produced.
