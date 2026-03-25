# LOB Microstructure Trading System

## Overview
Wang (2025) "Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books:
Better Inputs Matter More Than Stacking Another Hidden Layer" の再現・実務検証プロジェクト。

BTC/USDT LOBデータ（Bybit）を用い、前処理・特徴量設計・ラベリングの違いが
短期価格方向予測にどう影響するかを検証し、実戦可能なトレードシステムを構築する。

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli fetch-data --config configs/base.yaml
python -m src.cli build-features --config configs/base.yaml
python -m src.cli train --config configs/paper_reproduction.yaml
python -m src.cli backtest --config configs/paper_reproduction.yaml
```

## 開発原則

### 検証の順序（厳守）
1. **論文を素直に再現する**（独自改造しない）
2. **コスト分析を直後にやる**（Gross成績だけで深掘りに進まない）
3. 感度分析（何がロバストで何が脆いかを把握）
4. ファクター分解（αの源泉を特定）
5. 実務化設計（執行改善・取引タイミング選別）
6. 最終判断（数値基準で明確に判定）

### 禁止事項
- 論文再現前に独自改造を大量に入れない
- 未来情報を混ぜない（ローリングウィンドウは過去のみ使用）
- Grossの見た目だけで判断しない（必ずコスト込みで評価）
- パラメータ最適化と最終評価を同じ期間でやらない

### 技術要件
- Python, 型ヒント, 関数分割
- ロジックと設定（YAML）を分離
- 乱数はseed固定
- 出力はCSV/JSON/Markdown/PNG

### 期間分割（厳守）
- Train: 80% of LOB snapshots (first ~80,000 snapshots)
- Test: 20% of LOB snapshots (last ~20,000 snapshots)
- すべての閾値・パラメータ選択はTrain内のデータのみで決定

## 対象データ
- Exchange: Bybit
- Pair: BTC/USDT (primary), ETH/USDT (extension)
- LOB snapshots: 100ms intervals
- Depth: 5, 10, 20, 40 levels

## 論文デフォルトパラメータ
- Savitzky-Golay: cubic polynomial, window_size=21
- Kalman: random walk model
- Classification: binary (up/down) and ternary (up/flat/down)
- LOB depth: 5-level (baseline), 40-level (extended)
- Sequence length: T=1 (baseline), T=10 (extended)
- Models: Logistic Regression, XGBoost, CatBoost, DeepLOB, Conv1D+LSTM

## 判断基準
- メイン戦略候補: Test Accuracy > 55%, Net Sharpe > 1.0
- サブ戦略候補: Test Accuracy > 52%, Net Sharpe > 0.5
- 不採用: 上記未達

## 過去PJの教訓
- コスト分析はPhase2でやる。後回しにしない
- 入力設計（前処理・特徴量）がモデル選択より重要
- Maker寄り・低コスト前提で設計する
- LOBデータは巨大なので効率的なI/Oが重要（Parquet推奨）
