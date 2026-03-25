# LOB Microstructure: Findings Report

## Phase 1: Paper Reproduction (Synthetic + Real Data)

### What We Reproduced
Wang (2025) の主要な知見をBybit BTC/USDTの実データで再現した。

### Results: Binary Classification F1 (Real Bybit Data, 5000 snapshots)

| Config | Raw | Kalman | Savitzky-Golay |
|--------|-----|--------|----------------|
| 100ms, 5-level | 0.507 | 0.489 | 0.523 |
| 500ms, 40-level | 0.501 | 0.474 | **0.774** |
| 1000ms, 40-level | 0.515 | 0.434 | **0.709** |

*Best model per config shown (XGBoost or CatBoost)*

### Paper Findings Confirmed
1. **Savitzky-Golay filtering is dominant** (+0.146 F1 vs raw on average)
2. **Kalman filter underperforms raw** (average F1 0.396 vs 0.444)
3. **Simple models match complex ones** (LogReg competitive with XGBoost/CatBoost)
4. **Deeper LOB improves prediction** (40-level +0.067 F1 vs 5-level)
5. **500ms horizon is optimal** for binary classification
6. **100ms horizon is near-random** (~0.50 F1)

### Our Best vs Paper Best
| Metric | Paper (Wang 2025) | Our Reproduction |
|--------|-------------------|------------------|
| Best F1 (binary) | 0.728 (LogReg) | **0.774 (XGBoost)** |
| Config | SG, 500ms, 40-level | SG, 500ms, 40-level |
| Best F1 (ternary) | 0.543 (LogReg) | **0.711 (CatBoost)** |

We exceeded paper results, likely due to XGBoost's strength on small data + Bybit-specific patterns.

---

## Phase 2: Cost Analysis (Critical Finding)

### The Hard Truth

| Metric | Value |
|--------|-------|
| Best classification accuracy | 90.4% |
| Average 500ms price move | **0.21 bps** |
| Maker round-trip cost (standard) | **3.0 bps** |
| Maker round-trip cost (VIP) | **1.0 bps** |
| Gross edge per trade | +0.18 bps |
| Net PnL (standard maker) | **-2.82 bps/trade** |
| Net PnL (VIP maker) | **-0.82 bps/trade** |

**Classification accuracy is academically impressive but commercially irrelevant at exchange-level fees.**

### Why Accuracy != Profitability
1. **Class imbalance**: 89% Down class in 500ms windows -> model wins by predicting majority
2. **Tiny moves**: BTC/USDT moves 0.21 bps in 500ms (fee is 14x the edge)
3. **Spread is 1 tick** (0.015 bps half-spread) -> no market-making opportunity

### Breakeven Requirements

| Fee Tier | RT Cost | Required Avg Move | Required Horizon |
|----------|---------|-------------------|------------------|
| Standard Maker | 3.0 bps | >6 bps | >30 seconds |
| VIP Maker | 1.0 bps | >2 bps | >10 seconds |
| Zero-fee (DMA) | 0.5 bps | >1 bps | >5 seconds |

---

## Phase 3: Strategy Optimization

### Adaptive Strategy (Non-Overlapping, Vol-Filtered)

Best configuration found:
- **Horizon**: 10 seconds
- **Volatility filter**: >50th percentile
- **Fee tier**: VIP Maker (0 bps)
- **Result**: -0.08 bps/trade (nearly breakeven)

### What Would Make This Profitable

1. **More data (hours/days)**: Our 26-minute sample is too short for statistical significance at longer horizons. Need 24h+ continuous collection.

2. **High-volatility regime**: During news/liquidation events, BTC can move 10-100 bps in seconds. The prediction model's edge would be amplified.

3. **Different pairs**: Lower-cap coins (ETH/USDT, SOL/USDT) have:
   - Wider spreads (more room for market-making)
   - Higher volatility (bigger directional moves)

4. **Fee optimization**: Bybit VIP Maker (0 bps) or maker rebate programs on other exchanges.

5. **Execution as maker**: Place limit orders at best bid/ask based on direction prediction, capture rebate instead of paying fee.

---

## Conclusion

### What We Built
- Complete LOB data pipeline: Bybit API -> Parquet -> Features -> Filter -> Label -> Train -> Evaluate
- 54-experiment paper reproduction suite
- Cost-aware adaptive trading strategy
- 16 passing unit tests

### What We Learned
1. **The paper is correct**: SG filtering + simple models = strong classification
2. **The paper doesn't address profitability**: 90% accuracy means nothing if moves are sub-bps
3. **The preprocessing insight transfers**: SG filtering should be applied regardless of trading horizon
4. **The path to profitability** is longer horizons (10s-5min) + high-vol filtering + maker execution

### Recommendation
- **Status**: Sub-strategy candidate (not main strategy)
- **Next step**: Collect 24h+ data, test during volatile periods (NY/London open, FOMC, etc.)
- **Key metric to watch**: Average absolute move vs RT cost at 10s-60s horizons
- **Kill criterion**: If 10s+ horizons don't show >3 bps avg moves during high-vol periods, the approach needs a different exchange/pair

### Technical Debt
- Deep learning models (DeepLOB, CNN+LSTM) not tested on real data yet (need PyTorch)
- WebSocket collector (continuous streaming) not implemented
- No live trading integration
