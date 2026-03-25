# LOB Microstructure Trading System

**Paper**: Wang (2025) "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer"

**Thesis**: Short-term LOB price prediction gains come primarily from data preprocessing and feature engineering, not from model complexity. We reproduce this claim, then test whether the resulting edge survives transaction costs.

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Fetch LOB data from Bybit (or generate synthetic)
python -m src.cli fetch-data --synthetic -n 100000

# 2. Run focused experiment grid (paper reproduction)
python -m src.cli run-experiments --config configs/paper_reproduction.yaml

# 3. View results
cat reports/experiment_report.md
```

## Project Layers

This project has three distinct layers. Each builds on the previous, and they should not be mixed.

### Layer 1: Paper Reproduction (`configs/paper_reproduction.yaml`)

Reproduce Wang (2025) Table 1-4 as faithfully as possible.

- **Data**: Bybit BTC/USDT LOB, 100ms snapshots
- **Comparison grid** (fixed):
  - Preprocessing: raw / Savitzky-Golay / Kalman
  - LOB depth: 5 / 40 levels
  - Horizon: 100ms / 500ms / 1000ms
  - Labels: binary (up/down) / ternary (up/flat/down)
  - Models: Logistic Regression / XGBoost (paper shows these match or beat deep models)
- **Metric**: F1 score (macro), per-class F1
- **Split**: 80/20 temporal (no shuffle), 20% of train as validation
- **Goal**: Confirm or refute that SG filtering + simple models >= deep models

### Layer 2: Research Extensions (`configs/extended.yaml`)

Extend the paper's findings toward practical trading.

- **Longer horizons**: 5s / 10s / 30s / 60s (where price moves may exceed costs)
- **Volatility regime filtering**: trade only when rolling vol > threshold
- **Order imbalance as direct signal**: extreme quantile (top/bottom 10%) trading
- **Models**: Add CatBoost; DeepLOB only after baselines are solid
- **Metric**: Gross PnL (bps/trade), accuracy, win rate

### Layer 3: Trading Simulation (`configs/trading.yaml`)

Cost-aware evaluation of strategies from Layer 2.

- **Cost assumptions**:
  - Standard Maker: 1.0 bps one-way, 3.0 bps round-trip
  - VIP Maker: 0.0 bps one-way, 1.0 bps round-trip (slippage only)
  - Taker: 5.5 bps one-way, 13.0 bps round-trip
- **Execution model**: non-overlapping trades only (no autocorrelation inflation)
- **Metric**: Net PnL (bps/trade), Sharpe, max drawdown, breakeven analysis
- **Kill criterion**: If net PnL < 0 at VIP maker on 24h+ data across vol regimes, the approach needs a different pair

## Evaluation Design

### Data Split
```
|---- Train (64%) ----|-- Val (16%) --|---- Test (20%) ----|
                                       ^ all metrics here
```
- Temporal split only (no shuffle)
- All thresholds and parameters chosen on Train+Val
- Test is touched once per configuration

### Label Definitions
- **Binary**: Up (1) if mid-price increases over horizon, Down (0) otherwise
- **Ternary**: Up (2) / Flat (1) / Down (0), with epsilon auto-tuned for ~33% per class
- **Horizon**: Number of 100ms steps ahead (e.g., horizon=50 means 5 seconds)

### Signal-to-Position Conversion
- **Classification output**: predicted class (0/1 or 0/1/2)
- **Position**: Long (+1) if predicted Up, Short (-1) if predicted Down, Flat (0) if predicted Flat
- **Size**: Fixed 1 unit per trade (no confidence-based sizing in Layer 1-2)
- **Duration**: Hold for exactly `horizon` steps, then close

### Cost Model
```
round_trip_cost = 2 * (exchange_fee + slippage)
net_return = gross_return - round_trip_cost
breakeven_accuracy = 0.5 + cost / (2 * avg_move)
```

## Project Structure

```
configs/
  base.yaml                 # Shared defaults
  paper_reproduction.yaml   # Layer 1: paper grid
  extended.yaml             # Layer 2: longer horizons, imbalance
  trading.yaml              # Layer 3: cost-aware backtest

src/
  cli.py              # CLI: fetch-data, build-features, train, evaluate, run-experiments
  config.py            # YAML config loader with inheritance
  constants.py         # Symbols, depths, horizons, fee schedules
  data_collector.py    # Bybit REST API + synthetic data generator
  data_loader.py       # Parquet I/O, temporal split, depth filtering
  features.py          # LOB features: mid-price, imbalance, spread, weighted mid (Eq. 1-4)
  filters.py           # Savitzky-Golay (Eq. 5-7), Kalman (Eq. 8-10)
  labeling.py          # Binary/ternary labels, auto-epsilon, class weights
  models.py            # LR, XGBoost, CatBoost, DeepLOB, CNN+LSTM (ABC base)
  metrics.py           # Classification metrics + trading PnL metrics
  experiments.py       # Full experiment grid runner
  strategy.py          # Adaptive strategy: vol filter, confidence, non-overlapping
  backtest.py          # Single/multi model evaluation
  cost_model.py        # Round-trip cost, breakeven, slippage estimation
  report.py            # Markdown report generator

tests/
  test_basics.py       # 16 tests: features, filters, labels, config, cost model

reports/
  experiment_report.md     # Auto-generated from run-experiments
  phase2_cost_analysis.md  # Cost analysis findings
  findings.md              # Comprehensive findings (Phases 1-3)
```

## Key Design Decisions

1. **Filtering first**: Savitzky-Golay / Kalman applied before feature extraction
2. **Label design matters**: Binary vs ternary, epsilon sensitivity, class balance
3. **Simple models first**: LR and XGBoost only; add deep models after baselines are solid
4. **Cost-aware from day one**: Every strategy evaluated gross AND net
5. **Non-overlapping trades**: Avoid autocorrelation in PnL estimation
6. **Temporal split only**: No shuffle, no future information leakage

## Current Results (Real Bybit Data)

### Layer 1: Paper Reproduction (5,000 snapshots, ~26 min)

| Config (binary) | Raw | Kalman | **Savitzky-Golay** |
|------------------|------|--------|---------------------|
| 100ms, 5-level  | 0.507 | 0.489 | 0.523 |
| 500ms, 40-level | 0.501 | 0.474 | **0.774** |
| 1000ms, 40-level | 0.515 | 0.434 | **0.709** |

**SG filtering is the dominant factor** (+0.146 F1 vs raw on average).

### Layer 3: Cost Reality

| Metric | Value |
|--------|-------|
| Avg 500ms price move | 0.21 bps |
| Standard maker RT cost | 3.0 bps |
| Best gross edge (classification) | +0.18 bps/trade |
| Best gross edge (imbalance signal) | **+0.82 bps/trade** |
| Gap to VIP breakeven | **0.18 bps** |

## References

- [SSRN Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331939)
- [arXiv](https://arxiv.org/abs/2506.05764)
