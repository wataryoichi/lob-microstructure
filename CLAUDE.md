# LOB Microstructure Trading System

## Overview
Wang (2025) "Better Inputs Matter More" の再現→実務検証→Paper Tradingプロジェクト。
BTC/USDT LOBデータで前処理・直接シグナル・コスト込み評価を徹底し、Phase 13まで完了。

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/                                    # 50 tests
python -m src.cli run-paper-trading --symbol BTCUSDT  # Paper Trading
python -m src.cli paper-status                   # Status check
python -m src.cli analyze-execution              # Execution analysis
python -m src.cli sweep-imbalance                # Imbalance parameter sweep
python -m src.cli show-leaderboard               # Trading-first leaderboard
```

## Project Status: Phase 13 Complete
- Paper Trading: 7日間稼働中 (Config A: 5%_60s + Config B: 10%_120s)
- Live Trading基盤: 全モジュール完成 (API, WS, OrderManager, RiskManager, Notifier)
- Testnet E2E: APIキー設定待ち (docs/testnet_e2e_checklist.md 参照)

## Confirmed Architecture
```
Signal:     5-level imbalance (SG-filtered), extreme 5-10%
Horizon:    60s (primary) / 120s (aggressive), non-overlapping
Execution:  Post-Only Maker → Taker fallback on timeout
Cost:       VIP Maker required (1.0 bps RT)
Safety:     100 bps daily DD limit, 5 consecutive loss halt
```

## Key Findings (Phase 1-13)
1. SG前処理がモデル選択の3.9倍効く (+0.139 F1)
2. 500ms分類精度90.4%でもnet -2.82 bps (手数料14倍)
3. Direct imbalance signal > ML prediction (実戦)
4. 60-120sが最適ホライズン (WF net +0.76〜+1.95)
5. Walk-forwardで30s設定は過学習と判明
6. 逆選択なし — Maker fill filtering は正しく機能
7. VIP Maker必須 — Standardでは全設定赤字

## 評価原則
- **主指標: net bps/trade** (F1は補助)
- Walk-forward (10窓以上) なしに判定を下さない
- 正の窓率 > 50% を安定性基準とする
- 単一OOS分割を信頼しない

## 禁止事項
- Mainnet APIキーをコードにハードコードしない
- Paper Tradingプロセスに干渉しない
- Gross PnLだけで判断しない
- 未来情報を混ぜない
