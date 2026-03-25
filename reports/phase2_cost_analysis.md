# Phase 2: Cost Analysis Report

## Executive Summary

**Classification accuracy is high (F1=0.774, Acc=90.4%), but the strategy is NOT profitable at current exchange fees.**

The average 500ms price move in BTC/USDT is only **0.21 bps**, while the minimum round-trip cost (VIP maker) is **1.0 bps**. This makes the breakeven accuracy mathematically impossible to achieve for sub-second strategies.

## Data

- Source: Bybit BTC/USDT, live LOB snapshots
- Period: 2026-03-25, ~26 minutes of continuous data
- Snapshots: 5,000 at 40-level depth
- BTC price range: $71,049 - $71,244 (range: 0.27%)

## Best Classification Results

| Config | Model | Filter | Accuracy | F1 |
|--------|-------|--------|----------|----|
| Binary, 40-level, 500ms | XGBoost | Savitzky-Golay | 90.4% | 0.774 |
| Ternary, 40-level, 500ms | CatBoost | Savitzky-Golay | 81.1% | 0.711 |
| Binary, 40-level, 1000ms | XGBoost | Savitzky-Golay | 83.8% | 0.709 |

## Price Move Distribution

| Horizon | Mean |abs| (bps) | Median |abs| (bps) | Std (bps) | Moves > 3bps |
|---------|---------------------|---------------------|-----------:|--------------|
| 100ms   | 0.054 | 0.000 | 0.248 | 0.0% |
| 500ms   | 0.250 | 0.000 | 0.618 | 0.4% |
| 1000ms  | 0.466 | 0.028 | 0.907 | 1.1% |
| 5000ms  | 1.546 | 1.124 | 2.178 | 14.2% |
| 10000ms | 2.301 | 1.715 | 3.120 | 30.6% |

## Cost Structure (Bybit)

| Fee Tier | Maker | Taker | RT (Maker) | RT (Taker) |
|----------|-------|-------|-----------|-----------|
| Standard | 1.0 bps | 5.5 bps | 3.0 bps | 13.0 bps |
| VIP 1 | 0.6 bps | 4.5 bps | 2.2 bps | 11.0 bps |
| VIP Maker | 0.0 bps | 3.0 bps | 1.0 bps | 7.0 bps |

## Breakeven Analysis

For the best config (XGBoost+SG, 500ms, 40-level):
- Average price move: 0.21 bps
- Gross PnL: +0.180 bps/trade (correct predictions earn ~1 bps, losses ~-0.4 bps)
- Net PnL (standard maker): **-2.82 bps/trade** (catastrophic)
- Net PnL (VIP maker): **-0.82 bps/trade** (still negative)

**The problem is structural**: 500ms moves are 10-15x smaller than round-trip costs.

## Strategies Explored

### 1. Confidence-Based Filtering
Filtering by model confidence (prob > 0.6, 0.7, 0.8, 0.9) increases accuracy but does NOT increase gross PnL per trade because the model is mostly predicting the majority class correctly, not finding high-alpha trades.

### 2. Longer Horizons (5s, 10s, 20s, 50s)
- 10s, volatility-filtered (p50): gross = 1.085 bps -> net = -1.915 bps (closest to breakeven)
- 50s, all trades: gross = 1.775 bps -> net = -1.225 bps
- Still negative, but approaching feasibility at longer horizons

### 3. Pure Market-Making
BTC/USDT spread is typically 1 tick (0.10 USDT = 0.015 bps). Maker fee alone (1 bps) is 70x the half-spread. **Not viable without fee rebates.**

### 4. Volatility Regime Filtering
Trading only during high-volatility periods (p50+) improves gross PnL from 0.4 to 1.1 bps at 10s horizon. Promising direction but insufficient with current fee levels.

## Root Cause

The paper achieves high classification accuracy, but:

1. **Class imbalance**: 89% of 500ms intervals show "Down" (majority class). The model gets high accuracy by mostly predicting Down.
2. **Tiny price moves**: BTC/USDT in this period moved only 0.27% in 26 minutes. The per-interval moves are sub-basis-point.
3. **Fee structure**: Even Bybit's maker fee (1 bps) is 5x the average 500ms move.

## Path to Profitability

| Approach | Feasibility | Expected Net Edge |
|----------|-------------|-------------------|
| 1-5 min horizon + volatility filter | Medium | Need >3 bps avg move |
| VIP/Maker rebate programs | High | Reduce RT to ~1 bps |
| Cross-exchange arb | High complexity | Spread > 2 bps between venues |
| Higher vol regime (news/events) | Medium | 5-50x normal volatility |
| Different pair (smaller cap) | Medium | Wider spreads + bigger moves |
| Direct market access (no exchange fee) | Requires infra | Sub-bps cost possible |

## Conclusion

**Paper's classification results are reproducible and academically valid**, but the gap between classification accuracy and profitable trading is enormous at current BTC/USDT fee levels. The paper does not address this gap.

**Recommendation**: Pivot to longer horizons (1-5 min) and/or pairs with higher volatility. The preprocessing insight (SG filter > raw > Kalman) remains valid and valuable regardless of trading horizon.
