# Walk-Forward Analysis Report

## Data
- BTCUSDT: 69,221 snapshots, 3.17 hours
- Walk-forward: 9 rolling windows (train=0.75h, test=0.25h)

## Key Finding

**The previously confirmed 30s horizon is NOT robust in walk-forward.**
60s and 120s horizons are significantly more stable.

## Walk-Forward Results: Top Configurations

### VIP Maker (RT 1.0 bps)

| Config | Trades | **WF Net** | Win | PF | Positive Windows |
|--------|--------|-----------|-----|-----|------------------|
| **10%_120s_vol0** | 25 | **+2.598** | 68.0% | 3.40 | 4/9 |
| 5%_180s_vol0 | 15 | +1.307 | 60.0% | 1.72 | 2/9 |
| 5%_120s_vol0 | 25 | +1.094 | 64.0% | 2.00 | 3/9 |
| **5%_60s_vol0** | **50** | **+0.736** | 58.0% | 2.45 | **4/9** |
| 10%_60s_vol0 | 54 | +0.231 | 59.3% | 1.84 | 4/9 |
| 15%_30s_vol50 | 59 | +0.150 | 71.2% | 2.20 | **5/9** |

### Standard Maker (RT 3.0 bps)

| Config | Trades | **WF Net** | Win | PF | Status |
|--------|--------|-----------|-----|-----|--------|
| **10%_120s_vol0** | 25 | **+0.598** | 68.0% | 3.40 | **PROFITABLE** |
| 5%_120s_vol0 | 25 | -0.906 | 64.0% | 2.00 | Close |
| 5%_180s_vol0 | 15 | -0.693 | 60.0% | 1.72 | Close |

**Standard Maker profitability exists at 120s horizon with 10% threshold.**

## Phase Comparison: Single OOS vs Walk-Forward

| Config | Single OOS (Phase 6) | Walk-Forward (Phase 7) |
|--------|---------------------|----------------------|
| 5%_30s VIP | +0.783 (n=15) | **-0.383** (n=95) |
| 20%_60s VIP | +0.301 (n=27) | -0.872 (n=57) |
| 5%_60s VIP | -0.771 (n=25) | **+0.736** (n=50) |

**Critical lesson:** Single-split OOS can be misleading. Walk-forward with multiple windows gives a more honest picture. The 30s config that passed Phase 6 failed Phase 7.

## Revised Architecture

Based on walk-forward evidence, the optimal config shifts:

```
Old (Phase 6): 5% threshold, 30s horizon, LR filter
New (Phase 7): 5-10% threshold, 60-120s horizon, no vol filter

Best trade-off: 5%_60s (50 trades, net +0.736, 4/9 windows positive)
Best per-trade: 10%_120s (25 trades, net +2.598, standard-maker viable)
```

## Window-by-Window Consistency (5%_60s_VIP)

The strategy is NOT consistently profitable across all windows.
4 of 9 windows are positive, meaning ~44% of the time the strategy makes money.
This is marginal — a truly robust strategy should have >60% positive windows.

## Verdict

| Criterion | 30s (old) | 60s (new) | 120s (new) |
|-----------|-----------|-----------|------------|
| WF net (VIP) | -0.383 | **+0.736** | **+2.598** |
| WF net (STD) | -2.383 | -1.264 | **+0.598** |
| Trades | 95 | 50 | 25 |
| Positive windows | 2/9 | 4/9 | 4/9 |
| Statistical power | Marginal | Marginal | Low |

**Recommendation:**
- 60s with VIP: **Cautious proceed** (4/9 positive, net >0, 50 trades)
- 120s with Standard: **Promising but n=25** — needs more data
- 30s: **Demote** — walk-forward negative despite single-split positive
- All configs need 24h+ data to reach statistical significance
