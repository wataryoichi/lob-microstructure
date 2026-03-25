# LOB Microstructure: Findings (Trading-First)

## Executive Summary

| Metric | Best Result |
|--------|-------------|
| **Net PnL (VIP maker)** | **+0.76 bps/trade** |
| Win rate | 77.8% |
| Profit factor | 2.93 |
| Strategy | Extreme imbalance (top/bot 20%), 30s hold |
| Sample | 9 trades (5k snapshots, ~26 min, low-vol) |
| Statistical confidence | **Low** -- needs 24h+ data |

The signal is **order imbalance**, not ML. ML is secondary.

---

## 1. What Matters (Ranked by Impact)

| Rank | Factor | Impact | Evidence |
|------|--------|--------|----------|
| **#1** | **Preprocessing (SG filter)** | +0.139 F1 | 3-4x model choice |
| **#2** | **Horizon selection** | -0.93 vs -3.82 net | 60s vs 500ms cost drag |
| **#3** | **Direct signal vs ML** | +0.76 vs +0.18 net | Imbalance beats classifier |
| #4 | Fee tier | 3x cost difference | VIP vs standard |
| #5 | Label scheme | +0.064 F1 | Binary > ternary |
| #6 | Model complexity | +0.036 F1 | XGB vs LR (marginal) |

---

## 2. Imbalance Strategy (Primary)

### Parameter Sweep (64 configs, real Bybit data)

**Top 5 by Net PnL (VIP maker, non-overlapping):**

| Threshold | Horizon | Vol Filter | Trades | Gross | **Net** | Win | PF |
|-----------|---------|------------|--------|-------|---------|-----|-----|
| 20% | 30s | none | 9 | +1.76 | **+0.76** | 78% | 2.93 |
| 10% | 10s | p50 | 10 | +1.24 | **+0.24** | 60% | 3.08 |
| 15% | 30s | none | 9 | +1.14 | **+0.14** | 67% | 1.94 |
| 10% | 10s | none | 24 | +0.82 | -0.19 | 58% | 2.33 |
| 5% | 10s | none | 19 | +0.38 | -0.62 | 58% | 1.47 |

### Key Observations

1. **30s horizon > 10s** for net profitability (bigger moves, same RT cost)
2. **20% threshold** (extreme extremes) has highest edge per trade but fewer trades
3. **Vol filter p50** improves 10s horizon but data too sparse for 30s
4. **Standard maker (3 bps RT) kills all configs** -- VIP is required
5. **Profit factor > 2.9** on best config -- strong when positive

### Critical Limitation
**n=9 trades is not statistically significant.** 24h+ data collection is in progress (WebSocket, BTC + ETH). The 30s horizon will generate ~2,880 non-overlapping windows per day vs 9 in the current sample.

---

## 3. ML Classification (Secondary, Layer 1)

### Binary F1 Scores (LR + XGBoost, real data)

| Config | Raw | Kalman | SG |
|--------|-----|--------|----|
| 100ms, d5 | 0.50 | 0.43 | **0.53** |
| 500ms, d40 | 0.50 | 0.47 | **0.77** |
| 1000ms, d40 | 0.48 | 0.45 | **0.71** |

### Why Classification Accuracy Doesn't Trade

| Horizon | Avg Move | Maker RT | Edge/Cost |
|---------|----------|----------|-----------|
| 500ms | 0.21 bps | 3.0 bps | 7% |
| 10s | 2.30 bps | 3.0 bps | 77% |
| 60s | ~5.0 bps | 3.0 bps | 167% |

**90.4% accuracy at 500ms produces only +0.18 bps/trade gross.**
The price moves are 14x smaller than fees.

---

## 4. Regime Analysis

Current data (26 min) is too short for meaningful regime slicing.
Expected with 24h data:

| Dimension | Expected Insight |
|-----------|-----------------|
| Vol regime | High-vol should amplify edge 2-5x |
| Time of day | NY/London open should show bigger moves |
| Spread regime | Tight spread = lower slippage |
| Symbol (BTC vs ETH) | ETH may have wider moves |

---

## 5. Next Steps

### Immediate (24h data arrives)
- [ ] Run imbalance sweep on 24h BTC + ETH data
- [ ] Regime analysis with statistical power
- [ ] Validate 30s horizon edge across time-of-day
- [ ] Compare BTC vs ETH edge

### P1 (after 24h validation)
- [ ] ML filter on imbalance candidates (Stage 2)
- [ ] Walk-forward evaluation
- [ ] Execution assumptions (queue, fill uncertainty)

### Kill Criterion
If 24h data at VIP maker shows avg_net_bps < 0 across all configs and regimes: **pivot to different pair or exchange.**

---

## 6. Leaderboard (Auto-Generated)

See `results/leaderboard.json` for full machine-readable data.

Current top entry:
```
Strategy: imbalance_direct
Config: 20% threshold, 30s horizon, VIP maker
Net: +0.76 bps/trade
Win: 77.8%, PF: 2.93
Status: NEEDS 24h VALIDATION
```
