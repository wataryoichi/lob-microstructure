# LOB Microstructure Trading System

## Overview
Wang (2025) "Better Inputs Matter More Than Stacking Another Hidden Layer" の再現・実務検証。
BTC/USDT LOBデータ（Bybit）で、前処理・ラベリングの効果をモデル複雑性と比較し、
コスト込みで実戦可能かを判定する。

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli fetch-data --synthetic -n 100000
python -m src.cli run-experiments --config configs/paper_reproduction.yaml
```

## 3レイヤー構造（厳守）

### Layer 1: 論文再現 (`configs/paper_reproduction.yaml`)
- 論文の主張を忠実に再現する。独自改造しない
- 比較軸: preprocessing(3) x depth(2) x horizon(3) x labels(2) x models(2) = 72通り
- モデルは **Logistic Regression と XGBoost のみ**
- 評価: F1 (macro), per-class F1

### Layer 2: 研究拡張 (`configs/extended.yaml`)
- Layer 1の知見をもとに、実戦向きの設定を探索
- 長いホライズン (5s, 10s, 30s, 60s)
- Order imbalance直接シグナル
- ボラティリティレジーム別分析
- 評価: Gross PnL (bps/trade), win rate

### Layer 3: 実戦シミュレーション (`configs/trading.yaml`)
- Layer 2の有望な設定をコスト込みで評価
- 非重複トレードのみ（自己相関排除）
- コスト: Standard Maker 3.0bps RT, VIP Maker 1.0bps RT
- 評価: Net PnL (bps/trade), Sharpe, max drawdown
- Kill criterion: VIP maker で Net < 0 on 24h+ data → 不採用

## 評価設計

### データ分割
```
|---- Train (64%) ----|-- Val (16%) --|---- Test (20%) ----|
```
- 時系列順のみ。シャッフルしない
- 閾値・パラメータはTrain+Valで決定
- Testは各設定で1回だけ触る

### ラベル定義
- Binary: mid_price(t+h) > mid_price(t) → Up(1), else Down(0)
- Ternary: |change| > epsilon → Up(2)/Down(0), else Flat(1)
  - epsilon: auto-tune for ~33% per class

### コスト仮定 (Bybit BTC/USDT)
| Tier | Maker | Taker | RT (Maker) | RT (Taker) |
|------|-------|-------|-----------|-----------|
| Standard | 1.0 bps | 5.5 bps | 3.0 bps | 13.0 bps |
| VIP | 0.0 bps | 3.0 bps | 1.0 bps | 7.0 bps |

### Signal → Position 変換
- Up予測 → Long(+1), Down予測 → Short(-1), Flat予測 → Flat(0)
- サイズ: 固定1単位（Layer 1-2）
- 保持: horizonステップ後にクローズ
- 非重複: 次のトレードはhorizon経過後

## 禁止事項
- Layer 1が完了する前にLayer 2-3に進まない
- モデルを3つ以上同時に比較しない（LR + XGBoost + 1つだけ）
- Gross成績だけで判断しない
- パラメータ最適化と最終評価を同じ期間でやらない
- 未来情報を混ぜない

## 技術要件
- Python, 型ヒント, 関数分割
- ロジック/設定(YAML)分離, seed固定
- 出力: CSV/JSON/Markdown
- LOBデータはParquet推奨

## 過去の知見
- SG前処理が最も効く（+0.146 F1 vs raw, Kalmanは効かない）
- 500ms BTC/USDT変動 = 0.21 bps（手数料の1/14）
- Order imbalance極端値 (top/bot 10%) のgross edge = 0.82 bps (10s)
- VIP maker breakeven まであと 0.18 bps
