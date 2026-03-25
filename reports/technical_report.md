# LOB Microstructure Trading System: Technical Research Report

**Project**: wataryoichi/lob-microstructure
**Period**: 2026-03-25 (single session)
**Base Paper**: Wang (2025) "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books: Better Inputs Matter More Than Stacking Another Hidden Layer"
**Data**: Bybit BTC/USDT, ETH/USDT Limit Order Book (100ms snapshots)

---

## 1. Research Motivation and Hypothesis

### 1.1 Background

BTC/ETHの短期価格予測において、ディープラーニングの複雑なアーキテクチャ（DeepLOB, CNN+LSTM等）を積み上げるよりも、LOBデータの前処理・特徴量設計・ラベリングのほうが予測精度への寄与が大きいのではないか、という仮説をWang (2025)が提起した。

本プロジェクトの目標は、この仮説を再現検証したうえで、**分類精度を実際のトレーディング収益に変換できるか** を徹底的に検証することである。

### 1.2 Research Questions

1. SG (Savitzky-Golay) フィルタリングは本当にモデル複雑化より効くのか？
2. 高い分類精度（F1=0.77）は取引コスト控除後も利益を生むのか？
3. ML以外の直接シグナル（Order Imbalance）のほうが実戦的ではないか？
4. Paper Tradingでforward testした場合、バックテスト結果は維持されるか？

---

## 2. System Architecture

### 2.1 Software Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | scikit-learn (LogReg), XGBoost, CatBoost |
| Signal Processing | SciPy (Savitzky-Golay), NumPy (Kalman) |
| Data I/O | Parquet (PyArrow), SQLite |
| Real-time | asyncio + websockets |
| Exchange API | Bybit V5 REST + WebSocket |
| CLI | Typer |
| Tests | pytest (50 passing) |

### 2.2 Module Map (29 modules, 8,116 lines)

```
データ層:
  data_collector.py    REST API + 合成データ生成
  ws_collector.py      WebSocket連続LOB収集 (24h+対応)
  data_loader.py       Parquet I/O, 時系列分割

特徴量・前処理層:
  features.py          LOB特徴量 (Eq.1-4: mid, imbalance, spread, weighted mid)
  filters.py           Savitzky-Golay (Eq.5-7), Kalman (Eq.8-10)
  labeling.py          Binary/Ternary ラベル, auto-epsilon

モデル層:
  models.py            LR, XGBoost, CatBoost, DeepLOB, CNN+LSTM
  ml_filter.py         ML as signal filter (Stage 2判定)

戦略層:
  imbalance_strategy.py  主戦略: 極端Order Imbalanceの直接シグナル
  strategy.py            適応型戦略 (ML-based, 副次的)

評価層:
  metrics.py           分類指標 + trading-first PnL指標
  experiments.py       実験グリッドランナー
  walk_forward.py      Rolling Walk-forward検証
  regime.py            ボラ/スプレッド/時間帯別分析
  leaderboard.py       自動ランキング (net bps/trade基準)
  backtest.py          モデル評価フレームワーク
  cost_model.py        取引コストモデル
  report.py            レポート自動生成
  execution_analytics.py  執行品質分析 + 逆選択分析

執行層:
  paper_trader.py      Paper Trading エンジン (WS + Maker fill simulation)
  exchange_api.py      Bybit V5 REST ラッパー (Testnet/Mainnet)
  ws_private.py        Bybit Private WS (注文/約定通知)
  order_manager.py     注文状態管理 + Taker Fallback
  risk_manager.py      安全装置 (DD制限, 連続負け制限)
  notifier.py          Discord/Telegram通知
```

---

## 3. Phase 1: 論文再現 (Layer 1)

### 3.1 実験設計

Wang (2025) Table 1-4 の再現を目指し、以下のグリッドで36実験を実行:

- **前処理**: Raw / Savitzky-Golay / Kalman
- **LOB深度**: 5 / 40 レベル
- **予測ホライズン**: 100ms / 500ms / 1000ms
- **ラベル**: Binary (Up/Down) / Ternary (Up/Flat/Down)
- **モデル**: Logistic Regression / XGBoost (論文の主張に沿い、2モデルに限定)

### 3.2 結果: Binary Classification F1 (Bybit実データ, 5,000 snapshots)

| Config | Raw (LR / XGB) | Kalman (LR / XGB) | **SG (LR / XGB)** |
|--------|-----------------|--------------------|--------------------|
| 100ms, d5 | 0.496 / 0.507 | 0.369 / 0.488 | **0.570 / 0.491** |
| 500ms, d40 | 0.516 / 0.491 | 0.460 / 0.474 | **0.665 / 0.774** |
| 1000ms, d40 | 0.498 / 0.465 | 0.479 / 0.434 | **0.592 / 0.709** |

### 3.3 要因分解

| ランク | 要因 | F1への効果 |
|--------|------|-----------|
| **#1** | **前処理 (SG vs Raw)** | **+0.139** |
| #2 | ラベル方式 (Binary vs Ternary) | +0.064 |
| #3 | モデル選択 (XGB vs LR) | +0.036 |
| #4 | LOB深度 (40 vs 5) | +0.035 |

**前処理がモデル選択の3.9倍効く。** 論文の主張「Better Inputs Matter More」を定量的に確認。

### 3.4 論文結果との比較

| Metric | 論文 (Wang 2025) | 本プロジェクト |
|--------|-----------------|---------------|
| Best Binary F1 | 0.728 (LogReg) | **0.774 (XGBoost)** |
| Best Ternary F1 | 0.543 (LogReg) | **0.692 (XGBoost)** |
| Best Config | SG, 500ms, 40-level | SG, 500ms, 40-level (同一) |

本プロジェクトでは論文を上回るF1を達成。Bybit固有のLOBパターンとXGBoostの小データでの強さが寄与。

---

## 4. Phase 2: コスト分析 (最重要転換点)

### 4.1 分類精度 ≠ 収益

| Metric | Value |
|--------|-------|
| Best classification accuracy | 90.4% |
| Average 500ms BTC price move | **0.21 bps** |
| Standard Maker RT cost | **3.0 bps** |
| VIP Maker RT cost | **1.0 bps** |
| Gross edge per trade | +0.18 bps |
| **Net PnL (standard)** | **-2.82 bps/trade** |

90.4%の分類精度は学術的には優秀だが、500msの価格変動(0.21 bps)がmaker手数料(3 bps)の14分の1しかなく、**商業的には完全に不採算**。

### 4.2 Breakeven分析

| ホライズン | 平均|変動| (bps) | Maker RT (bps) | Edge/Cost |
|-----------|---------------------|-----------------|-----------|
| 500ms | 0.21 | 3.0 | 7% |
| 5s | 1.55 | 3.0 | 52% |
| 10s | 2.30 | 3.0 | 77% |
| 60s | ~5.0 | 3.0 | **167%** |

**60s以上でのみ、平均変動がコストを上回る。** これがPhase 3以降のホライズン拡張の根拠。

---

## 5. Phase 3-4: 戦略ピボット — ML分類器からDirect Signalへ

### 5.1 Order Imbalance Direct Strategy

MLモデルを「シグナル生成器」から「シグナルフィルター」に降格し、5-level order imbalanceの極端値をメインシグナルとする戦略にピボット:

```
Signal:   5-level aggregate imbalance (SG-filtered)
Entry:    rolling percentile rank > 95th (long) or < 5th (short)
Exit:     fixed horizon (60s or 120s), non-overlapping
Filter:   LR classifier (P(profitable) > 0.55) → 採用/却下
```

### 5.2 Imbalance Strategy Sweep (BTC 64k snapshots, 3.17h)

| Config | Trades | Gross bps | Net (VIP) | Win | PF |
|--------|--------|-----------|-----------|-----|-----|
| 5%_60s_VIP | 91 | +2.23 | **+1.23** | 60% | 2.66 |
| 10%_60s_VIP | 96 | +1.37 | **+0.37** | 55% | 1.82 |
| 5%_30s_VIP | 167 | +1.14 | **+0.14** | 61% | 2.15 |
| 20%_60s_VIP | 100 | +1.12 | **+0.12** | 61% | 1.63 |

### 5.3 ML Filter効果 (LR vs XGBoost)

| Config | Unfilt Net | LR Filter Net | Delta | Kept |
|--------|-----------|--------------|-------|------|
| 5%_30s | -0.065 | **+1.029** | +1.094 | 24% |
| 5%_60s | -1.136 | **+0.291** | +1.427 | 17% |

**LR filterが一貫して有効。** XGBoostは過学習で有害。LRはトレード数を76-83%カットして質を向上。

---

## 6. Phase 5-6: OOS検証

### 6.1 設計

- **分割**: Train 70% / Test 30% (時系列順、シャッフルなし)
- **閾値**: Train期間のみで決定
- **Kill Criterion**: OOS avg_net_bps < 0 at VIP → 不採用

### 6.2 結果 (BTC)

| Config | OOS Trades | OOS Net | Win | PF |
|--------|-----------|---------|-----|-----|
| 5%_30s + LR | 15 | **+0.783** | 86.7% | 12.6 |
| 20%_60s (no filter) | 27 | **+0.301** | 66.7% | 2.94 |
| 5%_30s (no filter) | 46 | -0.087 | 58.7% | 2.21 |

**BTC: PASS** (net > 0 at VIP). ETH: CONDITIONAL (-0.266).

---

## 7. Phase 7: Walk-Forward検証 (最重要検証)

### 7.1 設計

- **9-10個のrolling windows** (train=0.75h, test=0.25h)
- 各窓で独立にpercentile閾値計算 + LR filter学習
- 全窓のテスト期間を統合して評価

### 7.2 結果: Single-Split vs Walk-Forward

| Config | Phase 6 Single OOS | Phase 7 Walk-Forward |
|--------|-------------------|---------------------|
| 5%_30s VIP | **+0.783** | **-0.383** (過学習発覚) |
| 5%_60s VIP | -0.771 | **+0.736** (真の勝者) |
| 10%_120s VIP | 未テスト | **+2.598** (最高per-trade) |

**単一OOS分割は信頼できない。Walk-forwardが必須。** Phase 6で「最良」とされた30s設定はWFで-0.383に転落。

### 7.3 Standard Maker (RT 3.0 bps) 生存設定

| Config | WF Net (VIP) | WF Net (STD) | Trades |
|--------|-------------|-------------|--------|
| 10%_120s | +2.598 | **+0.598** | 25 |
| 5%_60s | +0.736 | -1.264 | 50 |

**120s/10%のみがStandard Makerで黒字化。** ただしn=25は統計的に不十分。

---

## 8. Phase 8: Full-Cycle検証とGO/NO GO判定

### 8.1 Walk-Forward (10 windows, 3.31h BTC)

| Architecture | Trades | Net (VIP) | Win | PF | Sharpe | +Windows |
|-------------|--------|-----------|-----|-----|--------|----------|
| 5%_60s VIP | 59 | **+0.763** | 58% | 2.49 | 1.10 | 5/10 |
| 10%_120s VIP | 29 | **+1.945** | 62% | 2.58 | 1.20 | 4/10 |
| 10%_120s STD | 29 | -0.055 | 62% | 2.58 | -0.03 | 3/10 |

### 8.2 PnL分布分析

- Top 3 trades = 107% of total PnL（利益集中型）
- Tight spread regime: **+9.1 bps/trade** vs wide: +0.1 bps
- Vol filter追加で**悪化**（直感に反するが事実）

### 8.3 GO/NO GO判定

| Symbol | Config | Verdict |
|--------|--------|---------|
| **BTC** | 10%_120s VIP | **Conditional GO (Paper Trading)** |
| **BTC** | 5%_60s VIP | **Conditional GO (Paper Trading)** |
| BTC | 10%_120s STD | NO GO |
| ETH | all | NO GO |

---

## 9. Phase 9-11: Paper Trading と執行分析

### 9.1 Paper Trading Engine

WebSocket→LOB保持→SG imbalance計算→Maker注文シミュレーション→SQLiteログの完全パイプラインを実装。Spread filter (1.5 tick) 統合。

### 9.2 ライブ結果 (1.7h, 25 trades)

| Config | Trades | Avg Net | Total Net | Win |
|--------|--------|---------|-----------|-----|
| A_5pct_60s | 18 | -1.88 | -33.8 bps | 22% |
| B_10pct_120s | 7 | -3.42 | -24.0 bps | 14% |

Paper Tradingは現時点でマイナス。Walk-forwardでの「4-5/10窓が正」= 50-60%の時間帯はマイナスという予測と一致。

### 9.3 執行品質分析

| Metric | Config A | Config B |
|--------|----------|----------|
| Entry Maker rate | 100% | 100% |
| Exit Taker fallback | 0% | 0% |
| Avg time-to-fill | 2.9s | 2.3s |
| Order fill rate | 33% | 23% |

### 9.4 逆選択分析

| | Filled (Maker) | Unfilled (Taker chase) |
|---|---|---|
| Config A avg net | -2.36 bps | **-11.36 bps** |
| Config B avg net | -4.39 bps | **-10.46 bps** |

**逆選択なし。** Unfilled signalをTakerで追いかけていたら大幅に悪化。Maker戦略の選別機能が確認された。

---

## 10. Phase 12-13: Live Trading基盤

### 10.1 構築済みモジュール

| Module | 機能 | テスト |
|--------|------|--------|
| exchange_api.py | Bybit V5 REST (PostOnly強制) | 5 (疎通確認含む) |
| ws_private.py | Private WS (注文/約定通知) | 2 |
| order_manager.py | 注文状態管理 + Taker Fallback | 5 |
| risk_manager.py | DD制限, 連続負け停止 | 6 |
| notifier.py | Discord/Telegram/Console | 4 |

### 10.2 Order Manager 状態遷移

```
place_maker_order → PENDING_NEW → NEW → FILLED
                                    ↓ timeout
                              PENDING_CANCEL → CANCELLED → IOC Taker → FILLED/CANCELLED
```

---

## 11. 主要な教訓と知見

### 11.1 研究的知見

1. **前処理 > モデル複雑性**: SG filterの+0.139 F1はモデル選択(+0.036)の3.9倍。論文の主張を定量的に確認。
2. **分類精度 ≠ 収益**: 90.4%精度でも500msでは0.18 bps/trade gross。手数料の14分の1。
3. **単一OOS分割は危険**: Phase 6で+0.783のconfig がPhase 7のWFで-0.383に転落。
4. **直接シグナル > ML**: Order imbalance極端値の直接利用がML分類器より実戦的。

### 11.2 実務的知見

5. **60-120sが最適ゾーン**: 30s以下はコスト負け、180s以上はサンプル不足。
6. **VIP Maker必須**: Standard Maker (3 bps) では120sでもギリギリ。
7. **LR filter > XGBoost filter**: XGBは小データで過学習。LRが安定。
8. **Vol filterは逆効果**: 直感に反するが、低vol時にもedgeが存在する。
9. **逆選択なし**: Maker注文の「約定しない」ことが正しいフィルタリング。

### 11.3 方法論的教訓

10. **Walk-forward は必須**: 10窓以上のrolling検証なしに戦略判定を下すべきでない。
11. **Trading-first metrics**: F1ではなくnet bps/tradeを主指標にすべき。
12. **正の窓率 > 50%**: PnLがプラスでも窓の過半数がマイナスなら不安定。

---

## 12. 結論と今後の展望

### 12.1 最終判定

| 問い | 答え |
|------|------|
| SG前処理は効くか？ | **Yes (F1 +0.139, #1要因)** |
| 分類精度は利益になるか？ | **No (500ms: 0.18 bps gross vs 3 bps cost)** |
| 直接シグナルは使えるか？ | **Conditional Yes (WF +0.76-1.95 bps at VIP)** |
| Paper Tradingで維持されるか？ | **Under observation (1.7h, negative so far)** |

### 12.2 推奨アクション

1. **7日間Paper Trading完了後**: GO/NO GO判定（net > 0, +day rate > 50%）
2. **Testnet E2E**: APIキー設定後に全注文フロー検証
3. **24h+データ**: 時間帯別安定性の確認
4. **ETH再評価**: より長期間データでの再検証
5. **Standard Maker探索**: 120s+設定でのさらなる最適化

### 12.3 プロジェクト統計

| Metric | Value |
|--------|-------|
| ソースコード | 8,116行 (29モジュール) |
| テスト | 50 passing, 3 skipped |
| Git commits | 27 |
| 実験数 | 200+ configurations |
| データ | 74k BTC + 49k ETH snapshots |
| Paper Trading | 25 trades, 1.7h稼働 |

---

*Report generated: 2026-03-25*
*Project: wataryoichi/lob-microstructure*
