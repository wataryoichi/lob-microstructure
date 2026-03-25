# LOB Microstructure Trading System

**Paper**: Wang (2025) "Better Inputs Matter More Than Stacking Another Hidden Layer"

**Thesis**: The main edge in LOB trading comes from input design (preprocessing, labeling, cost modeling), not model complexity. Our primary strategy is **extreme order imbalance**, with ML as an optional filter.

## Quick Start

```bash
pip install -e ".[dev]"

# Data collection (24h+ WebSocket)
python -m src.cli collect-ws --symbols BTCUSDT,ETHUSDT --duration 25

# Or synthetic data for development
python -m src.cli fetch-data --synthetic -n 100000

# Run imbalance strategy sweep
python -m src.cli run-imbalance --threshold 0.10 --horizon 100

# View leaderboard (trading-first)
python -m src.cli show-leaderboard

# Paper reproduction (Layer 1)
python -m src.cli run-experiments --config configs/paper_reproduction.yaml
```

## Primary Strategy: Extreme Order Imbalance

The core trading signal is **not an ML model** -- it is the raw 5-level order imbalance at extreme quantiles (top/bottom 10-20%), with Savitzky-Golay smoothing.

| Metric | Value (5k sample) |
|--------|-------------------|
| Best net PnL (VIP maker) | **+0.76 bps/trade** |
| Win rate | 77.8% |
| Profit factor | 2.93 |
| Holding period | 30 seconds |
| Signal | Imbalance top/bot 20% |
| Sample size | 9 trades (needs 24h+ validation) |

ML is used as a **filter** (adopt/reject candidate trades), not as the primary signal generator.

## Project Layers

| Layer | Purpose | Primary Metric | Config |
|-------|---------|----------------|--------|
| 1. Paper Reproduction | Confirm SG > raw > Kalman | F1 (macro) | `paper_reproduction.yaml` |
| 2. Research Extensions | Longer horizons, direct signals | Gross bps/trade | `extended.yaml` |
| 3. Trading Simulation | Cost-aware non-overlapping PnL | **Net bps/trade** | `trading.yaml` |

## Evaluation Design

### Primary Metrics (Trading-First)
1. **avg_net_bps** - Net PnL per trade after costs
2. **avg_gross_bps** - Gross PnL per trade
3. **breakeven_gap_bps** - Distance to net=0
4. **win_rate** / **profit_factor**
5. **n_trades** - Statistical significance

### Secondary Metrics (Classification)
- F1 (macro), accuracy - for Layer 1 paper reproduction only

### Cost Assumptions

| Tier | Maker | Taker | Round-Trip (Maker) |
|------|-------|-------|--------------------|
| Standard | 1.0 bps | 5.5 bps | 3.0 bps |
| VIP | 0.0 bps | 3.0 bps | 1.0 bps |

### Data Split
```
|---- Train (64%) ----|-- Val (16%) --|---- Test (20%) ----|
```
Temporal only. No shuffle. Percentile thresholds computed on rolling trailing window.

## Project Structure

```
src/
  cli.py                # CLI entry points
  ws_collector.py       # 24h+ WebSocket LOB collector (BTC + ETH)
  imbalance_strategy.py # PRIMARY: model-free extreme imbalance strategy
  regime.py             # Regime analysis (vol/spread/time slicing)
  leaderboard.py        # Auto-generated trading-first leaderboard
  strategy.py           # Adaptive strategy (ML-based, secondary)
  features.py           # LOB features (Eq. 1-4)
  filters.py            # Savitzky-Golay, Kalman (Eq. 5-10)
  labeling.py           # Binary/ternary labels
  models.py             # LR, XGBoost (primary); DeepLOB (research only)
  metrics.py            # Trading + classification metrics
  experiments.py        # Experiment grid runner
  backtest.py           # Model evaluation
  cost_model.py         # Exchange cost model
  data_collector.py     # REST API + synthetic data
  data_loader.py        # Parquet I/O
  config.py             # YAML config
  report.py             # Report generation

configs/
  paper_reproduction.yaml  # Layer 1
  extended.yaml            # Layer 2
  trading.yaml             # Layer 3

reports/
  findings.md              # Main findings (trading-first)
  experiment_report.md     # Layer 1 results
  regime_analysis.md       # Regime breakdown
  leaderboard.md           # Auto-generated ranking
```

## Current Status

- 25 passing tests
- WS collection running (BTC + ETH, 24h+)
- 64 imbalance strategy configurations swept
- 3 configurations showing positive net PnL at VIP rates
- Awaiting 24h data for statistical validation

## Key Findings So Far

1. **SG preprocessing is #1 factor** (+0.139 F1, 3-4x impact of model choice)
2. **500ms moves (0.21 bps) are too small** for standard fees (3 bps RT)
3. **30s imbalance signal at VIP rates** shows **+0.76 bps/trade** net
4. **Simple models (LR) beat XGBoost** at 30s+ horizons
5. **Deep models are not needed** -- the edge is in the signal, not the model

## References

- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331939) / [arXiv](https://arxiv.org/abs/2506.05764)
