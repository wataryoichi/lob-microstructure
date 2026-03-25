# Phase 8: Full-Cycle Verification

## BTCUSDT: 74,063 snaps, 3.31h
Price: 70763.85 - 71484.95 (range: 1.013%)

### Task 1: Walk-Forward (train=0.75h, test=0.25h)

| Architecture | Trades | Net bps | Win Rate | PF | Sharpe | Max DD | +Windows |
|-------------|--------|---------|----------|-----|--------|--------|----------|
| Primary (5%_60s_VIP) | 59 | +0.763 | 57.6% | 2.49 | 1.10 | 41.1 | 5/10 |
| Aggressive (10%_120s_VIP) | 29 | +1.945 | 62.1% | 2.58 | 1.20 | 13.0 | 4/10 |
| Standard (10%_120s_STD) | 29 | -0.055 | 62.1% | 2.58 | -0.03 | 37.0 | 3/10 |

### Task 2: PnL Distribution (Aggressive (10%_120s_VIP))

#### Trade-level PnL (net, bps)

| Stat | Value |
|------|-------|
| Count | 29 |
| Mean | +1.945 |
| Median | +0.568 |
| Std | 8.741 |
| Max win | +27.557 |
| Max loss | -13.005 |
| P25 | -4.252 |
| P75 | +5.511 |

#### Outlier Dependence

- Total net PnL: +56.4 bps
- Top 3 trades contribute: +60.2 bps (107% of total)
- Bottom 3 trades: -33.3 bps
- Profit is reasonably distributed across trades.

#### Per-Window Net PnL

| Window | Trades | Net bps/trade | Total Net | Win Rate |
|--------|--------|-------------|-----------|----------|
| 4 | 3 | +0.219 | +0.7 | 100.0% |
| 5 | 7 | +3.349 | +23.4 | 42.9% |
| 6 | 5 | +5.685 | +28.4 | 60.0% |
| 7 | 5 | +2.538 | +12.7 | 100.0% |
| 8 | 5 | -0.054 | -0.3 | 60.0% |
| 9 | 4 | -2.140 | -8.6 | 25.0% |

#### Regime Analysis (Walk-Forward Trades)

**vol_regime:**

| Regime | Trades | Net bps | Win Rate |
|--------|--------|---------|----------|
| high | 10 | +2.486 | 70.0% |
| low | 14 | +2.463 | 50.0% |
| med | 5 | -0.589 | 60.0% |

**spread_regime:**

| Regime | Trades | Net bps | Win Rate |
|--------|--------|---------|----------|
| med | 5 | +1.630 | 60.0% |
| tight | 5 | +9.129 | 80.0% |
| wide | 19 | +0.137 | 52.6% |

### Task 3: Volatility Filter Optimization

Goal: increase positive-window rate above 60%

| Vol Filter | Trades | Net bps | Win | +Windows | Sharpe |
|-----------|--------|---------|-----|----------|--------|
| p0 | 59 | +0.763 | 57.6% | 5/10 (50%) | 1.10 |
| p30 | 50 | -1.821 | 44.0% | 1/10 (10%) | -2.14 |
| p50 | 35 | -1.776 | 51.4% | 2/10 (20%) | -1.78 |
| p60 | 29 | -1.973 | 51.7% | 3/10 (30%) | -1.71 |
| p70 | 23 | -2.336 | 43.5% | 2/10 (20%) | -2.23 |
| p80 | 16 | -0.118 | 62.5% | 4/10 (40%) | -0.09 |

#### 120s Aggressive + Vol Filter

| Vol Filter | Trades | Net bps | Win | +Windows | Sharpe |
|-----------|--------|---------|-----|----------|--------|
| p0 | 29 | +1.945 | 62.1% | 4/10 (40%) | 1.20 |
| p30 | 27 | -0.378 | 59.3% | 4/10 (40%) | -0.27 |
| p50 | 19 | -1.851 | 47.4% | 2/10 (20%) | -1.06 |
| p60 | 18 | -0.862 | 38.9% | 1/10 (10%) | -0.46 |
| p70 | 12 | -1.807 | 33.3% | 2/10 (20%) | -0.94 |

## ETHUSDT: 49,008 snaps, 1.41h
Price: 2163.70 - 2190.36 (range: 1.225%)

### Task 1: Walk-Forward (train=0.4h, test=0.15h)

| Architecture | Trades | Net bps | Win Rate | PF | Sharpe | Max DD | +Windows |
|-------------|--------|---------|----------|-----|--------|--------|----------|
| Primary (5%_60s_VIP) | 27 | -0.379 | 40.7% | 1.36 | -0.37 | 32.8 | 2/6 |
| Aggressive (10%_120s_VIP) | 12 | -1.990 | 41.7% | 0.71 | -0.81 | 46.4 | 2/6 |
| Standard (10%_120s_STD) | 12 | -3.990 | 41.7% | 0.71 | -1.62 | 68.4 | 1/6 |

### Task 2: PnL Distribution (Primary (5%_60s_VIP))

#### Trade-level PnL (net, bps)

| Stat | Value |
|------|-------|
| Count | 27 |
| Mean | -0.379 |
| Median | -1.368 |
| Std | 5.387 |
| Max win | +15.070 |
| Max loss | -9.207 |
| P25 | -3.228 |
| P75 | +2.167 |

#### Outlier Dependence

- Total net PnL: -10.2 bps
- Top 3 trades contribute: +31.9 bps (312% of total)
- Bottom 3 trades: -24.2 bps
- **WARNING: Profit is heavily concentrated in a few trades.**

#### Per-Window Net PnL

| Window | Trades | Net bps/trade | Total Net | Win Rate |
|--------|--------|-------------|-----------|----------|
| 0 | 5 | +1.509 | +7.5 | 40.0% |
| 1 | 5 | -0.751 | -3.8 | 60.0% |
| 2 | 5 | -0.512 | -2.6 | 40.0% |
| 3 | 4 | -2.182 | -8.7 | 25.0% |
| 4 | 4 | -1.872 | -7.5 | 25.0% |
| 5 | 4 | +1.189 | +4.8 | 50.0% |

#### Regime Analysis (Walk-Forward Trades)

**vol_regime:**

| Regime | Trades | Net bps | Win Rate |
|--------|--------|---------|----------|
| high | 10 | +0.171 | 50.0% |
| low | 15 | -0.488 | 33.3% |
| med | 2 | -2.309 | 0.0% |

**spread_regime:**

| Regime | Trades | Net bps | Win Rate |
|--------|--------|---------|----------|
| med | 5 | +1.469 | 80.0% |
| tight | 17 | -0.372 | 35.3% |
| wide | 5 | -2.248 | 0.0% |

### Task 3: Volatility Filter Optimization

Goal: increase positive-window rate above 60%

| Vol Filter | Trades | Net bps | Win | +Windows | Sharpe |
|-----------|--------|---------|-----|----------|--------|
| p0 | 27 | -0.379 | 40.7% | 2/6 (33%) | -0.37 |
| p30 | 23 | -1.574 | 39.1% | 0/6 (0%) | -1.84 |
| p50 | 19 | -1.453 | 52.6% | 1/6 (17%) | -1.71 |
| p60 | 14 | -1.809 | 50.0% | 1/6 (17%) | -2.01 |
| p70 | 12 | -3.360 | 33.3% | 2/6 (33%) | -2.11 |
| p80 | 9 | -1.923 | 44.4% | 3/6 (50%) | -1.37 |

#### 120s Aggressive + Vol Filter

| Vol Filter | Trades | Net bps | Win | +Windows | Sharpe |
|-----------|--------|---------|-----|----------|--------|
| p0 | 12 | -1.990 | 41.7% | 2/6 (33%) | -0.81 |
| p30 | 11 | -1.970 | 45.5% | 3/6 (50%) | -0.83 |
| p50 | 11 | -3.827 | 27.3% | 1/6 (17%) | -2.79 |
| p60 | 10 | -5.331 | 10.0% | 1/6 (17%) | -4.37 |
| p70 | 9 | -4.535 | 33.3% | 2/6 (33%) | -2.23 |

---

## GO / NO GO Decision

### Summary Table

| Symbol | Best Config | WF Net | Trades | Win | +Windows | Verdict |
|--------|------------|--------|--------|-----|----------|---------|
| **BTC** | 10%_120s VIP | **+1.945** | 29 | 62% | 4/10 (40%) | **Conditional GO** |
| **BTC** | 5%_60s VIP | **+0.763** | 59 | 58% | 5/10 (50%) | **Conditional GO** |
| BTC | 10%_120s STD | -0.055 | 29 | 62% | 3/10 | NO GO |
| ETH | 5%_60s VIP | -0.379 | 27 | 41% | 2/6 | **NO GO** |

### BTC Verdict: **CONDITIONAL GO (Paper Trading)**

Justification:
1. Walk-forward net is positive: +0.763 bps (60s, n=59) and +1.945 bps (120s, n=29)
2. Win rate >55% and profit factor >2.4 on both configs
3. 120s config shows +27.6 bps max win vs -13.0 bps max loss (positive skew)

Conditions:
1. **VIP Maker only** — Standard Maker is breakeven at best (-0.055)
2. **Paper trading first** — positive-window rate is 40-50%, below the 60% target
3. **Monitor regime** — edge is concentrated in tight-spread periods (+9.1 bps vs wide +0.1 bps)
4. **No vol filter** — adding vol filter consistently worsens results (surprising)

Key risks:
- Top 3 trades contribute 107% of total PnL (outlier-dependent)
- Only 3.31h of data; time-of-day coverage limited to 04:00-12:00 UTC
- 40-50% positive windows = significant drawdown periods expected

### ETH Verdict: **NO GO**

- All configs negative in walk-forward
- Outlier-dependent (top 3 trades = 312% of total PnL)
- Wide spread regime is catastrophic (net -2.2 bps, 0% win)
- Insufficient data (1.4h) but even with that, no edge visible

### Standard Maker Verdict: **NO GO**

- 120s/10% at standard: -0.055 bps (essentially zero)
- The 2 bps additional cost completely eliminates the edge
- Would need consistently >5 bps gross moves, not achievable in current low-vol period

### Deployment Recommendation

```
Phase: Paper Trading (NOT live capital)
Symbol: BTCUSDT only
Configs to monitor in parallel:
  A) 5%_60s VIP — more trades, lower per-trade edge
  B) 10%_120s VIP — fewer trades, higher per-trade edge
Duration: 1 week minimum
Promote to live if:
  - 7-day walk-forward net > 0
  - Positive window rate > 50%
  - No single day with >50 bps drawdown
Kill if:
  - 7-day net < 0
  - Drawdown exceeds 100 bps
```
