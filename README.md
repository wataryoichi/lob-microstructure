# LOB Microstructure Trading System

**Paper**: Wang (2025) "Better Inputs Matter More Than Stacking Another Hidden Layer"

**Thesis**: The main edge in LOB trading comes from input design (preprocessing, labeling, cost modeling), not model complexity. Our primary strategy is **extreme order imbalance** with Maker execution.

**Status**: Paper Trading (7-day forward test in progress)

## Reports

| Report | Audience | Content |
|--------|----------|---------|
| [Technical Report](reports/technical_report.md) | Researchers | Full Phase 1-13 methodology and results |
| [Executive Summary](reports/executive_summary.md) | Management | Business-oriented findings and risk assessment |
| [Phase 8 Analysis](reports/phase8_full_cycle_analysis.md) | Quant | Walk-forward results and GO/NO GO |
| [Execution Analysis](reports/execution_analysis.md) | Quant | Maker fill rates and adverse selection |

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/                         # 50 tests

# Paper Trading (7-day forward test)
python -m src.cli run-paper-trading --symbol BTCUSDT --duration 168

# Monitor
python -m src.cli paper-status
python -m src.cli analyze-execution

# Backtesting
python -m src.cli fetch-data --synthetic -n 100000
python -m src.cli sweep-imbalance --symbols BTCUSDT,ETHUSDT
python -m src.cli show-leaderboard
```

## Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Preprocessing > Model complexity** | SG filter +0.139 F1 (3.9x model choice) |
| 2 | **Classification accuracy ≠ Profit** | 90.4% acc → only +0.18 bps/trade gross |
| 3 | **Direct signal > ML prediction** | Imbalance: +0.76 bps net vs ML: +0.18 bps |
| 4 | **60-120s is the optimal horizon** | Walk-forward positive; <30s = cost-negative |
| 5 | **Single OOS is unreliable** | 30s: +0.78 single → -0.38 walk-forward |
| 6 | **VIP Maker required** | Standard fee kills all configs |
| 7 | **No adverse selection** | Unfilled Taker chase: -10.5 bps (much worse) |

## Architecture

```
Strategy:  5-level order imbalance (SG-filtered), extreme 5-10%
Horizon:   60s (primary) / 120s (aggressive)
Execution: Post-Only Maker → Taker fallback on timeout
Cost:      VIP Maker (1.0 bps RT) required
Safety:    100 bps daily DD limit, 5 consecutive loss halt

29 modules | 8,116 lines | 50 tests | 27 commits
```

## Project Phases

| Phase | Focus | Key Result |
|-------|-------|------------|
| 1 | Paper reproduction | SG filter = #1 factor (+0.139 F1) |
| 2 | Cost analysis | 500ms moves too small (0.21 vs 3.0 bps) |
| 3-4 | Strategy pivot | Imbalance direct signal > ML classifier |
| 5-6 | OOS verification | BTC PASS at VIP maker |
| 7 | Walk-forward | 30s demoted, 60s/120s promoted |
| 8 | Full-cycle GO/NO GO | BTC Conditional GO (Paper Trading) |
| 9 | Paper Trading engine | Real-time WS + Maker fill simulation |
| 10 | Execution analytics | 100% Maker fill, no adverse selection |
| 11 | Risk Manager | DD limits, consecutive loss halt |
| 12 | Exchange API + Notifier | Bybit V5, Discord/Telegram |
| 13 | Order Manager + Testnet | Taker fallback, state reconciliation |

## GO / NO GO Criteria (after 7-day Paper Trading)

| Criterion | GO | KILL |
|-----------|-----|------|
| Total net PnL | > 0 bps | < 0 bps |
| Positive-day rate | > 50% | < 40% |
| Max single-day DD | < 100 bps | > 100 bps |
| Trade count | > 50 per config | < 20 |

## References

- [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5331939) / [arXiv](https://arxiv.org/abs/2506.05764)
- [Template: quant-paper-kit](https://github.com/wataryoichi/quant-paper-kit)
