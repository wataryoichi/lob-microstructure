# LOB Microstructure Trading System

**Paper**: Wang (2025) "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer"

**Thesis**: Data preprocessing and feature engineering matter more than model complexity for short-term LOB price prediction.

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Fetch LOB data from Bybit
python -m src.cli fetch-data --config configs/base.yaml

# 2. Build features & labels
python -m src.cli build-features --config configs/base.yaml

# 3. Train models (paper reproduction)
python -m src.cli train --config configs/paper_reproduction.yaml

# 4. Evaluate
python -m src.cli evaluate --config configs/paper_reproduction.yaml

# 5. Backtest
python -m src.cli backtest --config configs/paper_reproduction.yaml
```

## Project Structure

```
├── configs/           # YAML configurations
├── src/
│   ├── cli.py         # CLI entry points
│   ├── config.py      # Configuration management
│   ├── constants.py   # Constants and parameters
│   ├── data_collector.py  # Bybit LOB data collection
│   ├── data_loader.py     # Load & parse LOB snapshots
│   ├── features.py    # LOB feature engineering
│   ├── filters.py     # Kalman, Savitzky-Golay filters
│   ├── labeling.py    # Binary/ternary price labels
│   ├── models.py      # ML models (LR, XGBoost, DeepLOB, etc.)
│   ├── metrics.py     # Classification & trading metrics
│   ├── backtest.py    # PnL simulation
│   └── cost_model.py  # Transaction cost model
├── tests/
├── data/              # LOB data (gitignored)
├── results/           # Model outputs
└── reports/           # Analysis reports
```

## Key Design Decisions

1. **Filtering first**: Savitzky-Golay / Kalman before feature extraction
2. **Label design matters**: Binary vs ternary, threshold sensitivity
3. **Simple models first**: Logistic Regression -> XGBoost -> Deep Learning
4. **Cost-aware from day one**: Maker/taker fees, slippage, latency

## References

- [SSRN Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331939)
- [arXiv](https://arxiv.org/abs/2506.05764)
