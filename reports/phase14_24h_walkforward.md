# Phase 14: 24h Walk-Forward Results

Data: BTC 848k snaps (24.4h), ETH 843k snaps (24.6h)
Windows: train=4h, test=1h, 20 rolling windows per symbol

## Results Summary

| Config | Symbol | Trades | **Net bps** | Win | PF | Sharpe | **+Windows** |
|--------|--------|--------|-------------|-----|-----|--------|--------------|
| **B: 10%_120s VIP** | **BTC** | **515** | **+0.685** | **57.9%** | **1.69** | **1.56** | **13/20 (65%)** |
| A: 5%_60s VIP | BTC | 958 | -0.678 | 53.5% | 1.15 | -3.10 | 3/20 (15%) |
| B: 10%_120s STD | BTC | 515 | -1.315 | 57.9% | 1.69 | -2.99 | 4/20 (20%) |
| A: 5%_60s VIP | ETH | 954 | -0.594 | 52.8% | 1.17 | -2.40 | 8/20 (40%) |
| B: 10%_120s VIP | ETH | 512 | -0.565 | 50.8% | 1.13 | -1.22 | 8/20 (40%) |
| B: 10%_120s STD | ETH | 512 | -2.565 | 50.8% | 1.13 | -5.54 | 1/20 (5%) |

## Key Finding

**Config B (10%_120s VIP) on BTC is the ONLY profitable configuration across 24 hours.**

- Net: **+0.685 bps/trade** (515 trades = +353 bps total)
- **65% of windows are positive** (exceeds 60% stability target!)
- Sharpe: 1.56 (statistically significant)
- Profit factor: 1.69

All other configurations are negative in 24h walk-forward.

## Config A (5%_60s) is KILLED

- BTC: -0.678 net, only 3/20 windows positive (15%)
- The 60s horizon does not survive 24h validation
- Paper Trading results (-1.83 bps/trade) are consistent with this

## Config B per-window breakdown (BTC, VIP)

| Window | Trades | Net bps | Win | Verdict |
|--------|--------|---------|-----|---------|
| W0 | 27 | -0.049 | 48% | Marginal |
| W1 | 27 | -0.636 | 56% | Loss |
| W2 | 27 | +3.030 | 59% | **Strong win** |
| W3 | 27 | +2.773 | 74% | **Strong win** |
| W4 | 27 | +2.592 | 56% | **Strong win** |
| W5 | 26 | +0.044 | 42% | Breakeven |
| W6 | 26 | -0.815 | 50% | Loss |
| W7 | 26 | +1.460 | 62% | Win |
| W8 | 25 | +0.542 | 56% | Win |
| W9 | 24 | -1.682 | 46% | Loss |
| W10 | 25 | +2.104 | 84% | **Strong win** |
| W11 | 25 | +0.117 | 60% | Marginal win |
| W12 | 25 | +1.911 | 72% | **Strong win** |
| W13 | 24 | +1.685 | 67% | Win |
| W14 | 25 | -0.866 | 44% | Loss |
| W15 | 25 | -0.826 | 56% | Loss |
| W16 | 26 | +0.176 | 46% | Marginal win |
| W17 | 26 | -0.145 | 65% | Marginal loss |
| W18 | 26 | +0.223 | 50% | Marginal win |
| W19 | 26 | +1.759 | 65% | Win |

## Revised Recommendation

| Config | 3h WF (Phase 7) | 24h WF (Phase 14) | Paper Trading | Final |
|--------|-----------------|-------------------|---------------|-------|
| A: 5%_60s VIP | +0.736 | **-0.678** | -1.83 | **KILL** |
| B: 10%_120s VIP | +2.598 → +1.945 | **+0.685** | -1.58 | **CONTINUE** |
| B: 10%_120s STD | +0.598 → -0.055 | **-1.315** | N/A | **KILL** |

- Config A: KILLED (negative in both 24h WF and Paper Trading)
- Config B VIP: Only survivor. WF positive (+0.685, 65% windows)
- Config B STD: KILLED (clearly negative)
- ETH: KILLED (all configs negative)

## Implication for Paper Trading

The currently running Paper Trader has Config A at -517.8 bps and Config B at -232.5 bps.

Config A should be disabled. Config B (-1.58 bps/trade in paper) is worse than the 24h WF prediction (+0.685). Possible explanations:
1. Paper Trader's Maker fill simulation may be too optimistic/pessimistic
2. Percentile calibration differs between backtest and real-time
3. 22h of paper trading may not yet reflect the 65% positive-window rate

**Recommendation: Continue Config B paper trading for the full 7 days. If still negative at day 7, KILL the entire approach.**
