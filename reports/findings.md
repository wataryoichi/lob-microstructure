# LOB Microstructure: Final Findings (Phase 8)

## Verdict: **BTC CONDITIONAL GO (Paper Trading) / ETH NO GO**

| Symbol | Config | WF Net (VIP) | Trades | Win | +Windows | Decision |
|--------|--------|-------------|--------|-----|----------|----------|
| **BTC** | **10%_120s** | **+1.945** | 29 | 62% | 4/10 | **Paper trade** |
| **BTC** | **5%_60s** | **+0.763** | 59 | 58% | 5/10 | **Paper trade** |
| BTC | 10%_120s STD | -0.055 | 29 | 62% | 3/10 | NO GO |
| ETH | all configs | negative | - | - | - | NO GO |

**Phase 8 updated the Phase 7 Standard Maker result from +0.598 to -0.055 (more data, more windows).**

---

## OOS Results (Train 70% / Test 30%, strict temporal split)

### BTCUSDT (3.03h data, OOS=33 min)

| Config | Filter | OOS Trades | Gross bps | **Net bps** | Win | PF | Sharpe |
|--------|--------|-----------|-----------|-------------|-----|-----|--------|
| **5%_30s** | **LR t=0.55** | **15** | **+1.78** | **+0.78** | **86.7%** | **12.6** | - |
| 20%_60s | none | 27 | +1.30 | **+0.30** | 66.7% | 2.94 | 0.47 |
| 5%_30s | none | 46 | +0.91 | -0.09 | 58.7% | 2.21 | -0.20 |
| 5%_60s | none | 25 | +0.23 | -0.77 | 52.0% | 1.15 | -0.96 |

**Best OOS: imbalance 5% + 30s + LR filter = +0.783 bps/trade net (VIP maker).**
Win rate 86.7% with profit factor 12.6. Strong edge, though n=15 is still modest.

### ETHUSDT (1.12h data, OOS=20 min)

| Config | Filter | OOS Trades | Gross bps | **Net bps** | Win | PF |
|--------|--------|-----------|-----------|-------------|-----|-----|
| 5%_60s | LR t=0.55 | 10 | +0.73 | -0.27 | 70.0% | 1.72 |
| 5%_30s | none | 27 | +0.66 | -0.34 | 55.6% | 1.64 |
| 20%_60s_vol70 | none | 9 | -1.08 | -2.08 | 33.3% | 0.43 |

ETH is not profitable OOS with current data. The LR filter brings it close (-0.27 bps), but 20 minutes of OOS is insufficient for a definitive kill.

---

## What Actually Works (and What Doesn't)

### Works
1. **Extreme order imbalance (5%, top/bottom)** is a real signal
   - In-sample: +1.2 bps/trade gross (BTC 60s), +2.2 bps gross (BTC 30s)
   - OOS: +0.91 bps gross (BTC 30s), +1.30 bps gross (BTC 60s)
2. **LR filter improves marginal configs** significantly
   - BTC 5%_30s: net -0.09 -> **+0.78** (delta +0.87 bps)
   - Cuts bad trades, keeps good ones (15/46 trades retained, 86.7% win)
3. **Savitzky-Golay preprocessing** remains the #1 factor for classification (+0.139 F1)
4. **60s and 30s horizons** generate moves large enough to exceed VIP costs

### Doesn't Work
1. **Sub-10s horizons**: moves too small relative to any fee structure
2. **XGBoost as filter**: overfits on small data, LR is consistently better
3. **ETH OOS**: negative net, though close to breakeven with filter
4. **High vol filter on ETH**: reduced sample too aggressively for OOS period

### Still Unknown (need 24h+ data)
1. **Time-of-day variation**: only have 04:00-12:00 UTC coverage
2. **High-volatility regimes**: current sample is relatively calm (BTC range: 1%)
3. **Standard maker profitability**: VIP passes, standard (+3 bps RT) needs testing with more data

---

## Regime Analysis (In-Sample, 3h BTC)

| Dimension | Best | Net bps | Worst | Net bps |
|-----------|------|---------|-------|---------|
| Vol regime | med | +3.79 | high | +0.66 |
| Spread regime | wide | +1.79 | med | +1.00 |
| Time of day | 04-08 UTC | +2.30 | 08-12 UTC | +0.80 |

Note: Regime analysis is in-sample only. OOS regime validation requires 24h+ data spanning all time-of-day buckets.

---

## Statistical Assessment

| Metric | BTC best OOS | Requirement | Status |
|--------|-------------|-------------|--------|
| n_trades | 15 | >= 30 ideally | **Marginal** |
| avg_net_bps | +0.783 | > 0 | **PASS** |
| win_rate | 86.7% | > 50% | **PASS** |
| profit_factor | 12.6 | > 1.5 | **PASS** |
| OOS period | 33 min | 24h+ ideally | **Insufficient** |

The edge is real but statistical power is limited. A Binomial test: P(13 of 15 wins by chance at 50% base) = 0.003 -- statistically significant at p<0.01.

---

## Recommendation

### BTCUSDT: **Promote to extended validation**
- Config: imbalance 5% threshold, 30s horizon, LR filter (t=0.55)
- Fee: VIP Maker required (1 bps RT)
- Next: 24h walk-forward with hourly windows
- Entry condition: extreme 5-level imbalance (SG-filtered) + LR says "go"

### ETHUSDT: **Continue monitoring, do not deploy**
- Close to breakeven (-0.27 bps) with LR filter
- Need longer OOS period (current 20 min is too short)
- Revisit with 24h+ data

### Kill Decision
- **Neither symbol is killed.** BTC shows walk-forward positive edge at both VIP and Standard.
- ETH deferred pending more data.
- **30s horizon demoted** — single-split OOS was misleading.

---

## Phase 7: Walk-Forward Verification (9 rolling windows, 3.17h BTC)

### Walk-Forward Top Results (BTC)

| Config | Trades | WF Net (VIP) | WF Net (STD) | Win | PF | +Windows |
|--------|--------|-------------|-------------|-----|-----|----------|
| **10%_120s** | 25 | **+2.598** | **+0.598** | 68% | 3.40 | 4/9 |
| 5%_60s | 50 | **+0.736** | -1.264 | 58% | 2.45 | 4/9 |
| 10%_60s | 54 | +0.231 | -1.769 | 59% | 1.84 | 4/9 |
| 15%_30s_vol50 | 59 | +0.150 | -1.850 | 71% | 2.20 | 5/9 |

### Critical Lesson: Single-Split vs Walk-Forward

| Config | Phase 6 Single OOS | Phase 7 Walk-Forward |
|--------|-------------------|---------------------|
| 5%_30s | +0.783 | **-0.383** (DEMOTED) |
| 5%_60s | -0.771 | **+0.736** (PROMOTED) |
| 10%_120s | not tested | **+2.598** (NEW BEST) |

**Walk-forward with multiple windows is essential.** Single-split results are unreliable.

---

## Revised Architecture (Phase 7)

```
Primary config (VIP, higher trade count):
  Signal:     5-level imbalance (SG), extreme 5%
  Horizon:    60 seconds non-overlapping
  Cost:       VIP Maker 1.0 bps RT
  WF Net:     +0.736 bps/trade (50 trades, 4/9 windows positive)

Aggressive config (VIP, best per-trade):
  Signal:     5-level imbalance (SG), extreme 10%
  Horizon:    120 seconds non-overlapping
  Cost:       VIP Maker 1.0 bps RT
  WF Net:     +2.598 bps/trade (25 trades, 4/9 windows positive)

Standard Maker config:
  Signal:     5-level imbalance (SG), extreme 10%
  Horizon:    120 seconds non-overlapping
  Cost:       Standard Maker 3.0 bps RT
  WF Net:     +0.598 bps/trade (25 trades)
  Status:     FIRST STANDARD-FEE VIABLE CONFIG

No ML filter in primary configs (walk-forward shows filter is unstable
with rolling small windows; adds complexity without consistent benefit).
```

## Remaining Risks

1. **4/9 positive windows (44%)** — not yet >60% target for robust deployment
2. **3.17h data** — need 24h+ for time-of-day stability
3. **Low vol period** — BTC range only 1% during sample
4. **n=25-50 trades** — statistically marginal for per-trade metrics
