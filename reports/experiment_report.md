# LOB Microstructure Experiment Report

**Total experiments**: 36 (36 successful, 0 skipped/failed)

## Binary Classification (F1 Scores)

|                                        |   logistic_regression |   xgboost |
|:---------------------------------------|----------------------:|----------:|
| ('1000ms, 40-level', 'kalman')         |                0.4794 |    0.4341 |
| ('1000ms, 40-level', 'raw')            |                0.498  |    0.4649 |
| ('1000ms, 40-level', 'savitzky_golay') |                0.5924 |    0.7091 |
| ('100ms, 5-level', 'kalman')           |                0.3687 |    0.4882 |
| ('100ms, 5-level', 'raw')              |                0.496  |    0.5072 |
| ('100ms, 5-level', 'savitzky_golay')   |                0.5696 |    0.4909 |
| ('500ms, 40-level', 'kalman')          |                0.4596 |    0.4742 |
| ('500ms, 40-level', 'raw')             |                0.5157 |    0.4907 |
| ('500ms, 40-level', 'savitzky_golay')  |                0.6653 |    0.7742 |

**Best binary**: xgboost + savitzky_golay (40-level, 500ms) -> F1=0.7742, Acc=0.9037

## Ternary Classification (F1 Scores)

|                                        |   logistic_regression |   xgboost |
|:---------------------------------------|----------------------:|----------:|
| ('1000ms, 40-level', 'kalman')         |                0.3909 |    0.3983 |
| ('1000ms, 40-level', 'raw')            |                0.4316 |    0.3615 |
| ('1000ms, 40-level', 'savitzky_golay') |                0.4317 |    0.6268 |
| ('100ms, 5-level', 'kalman')           |                0.216  |    0.3341 |
| ('100ms, 5-level', 'raw')              |                0.3386 |    0.3043 |
| ('100ms, 5-level', 'savitzky_golay')   |                0.4203 |    0.3867 |
| ('500ms, 40-level', 'kalman')          |                0.3093 |    0.3274 |
| ('500ms, 40-level', 'raw')             |                0.3608 |    0.4053 |
| ('500ms, 40-level', 'savitzky_golay')  |                0.4842 |    0.6918 |

**Best ternary**: xgboost + savitzky_golay (40-level, 500ms) -> F1=0.6918, Acc=0.8023

## Filter Comparison (average F1 across all configs)

| filter         |   mean |    std |   count |
|:---------------|-------:|-------:|--------:|
| kalman         | 0.39   | 0.083  |      12 |
| raw            | 0.4312 | 0.0747 |      12 |
| savitzky_golay | 0.5703 | 0.127  |      12 |

## Model Comparison (average F1 across all configs)

| model               |   mean |    std |   count |
|:--------------------|-------:|-------:|--------:|
| logistic_regression | 0.446  | 0.1079 |      18 |
| xgboost             | 0.4816 | 0.137  |      18 |

## Depth Comparison (average F1 across all configs)

|   depth |   mean |    std |   count |
|--------:|-------:|-------:|--------:|
|       5 | 0.4101 | 0.1031 |      12 |
|      40 | 0.4907 | 0.1251 |      24 |

## Horizon Comparison (average F1 across all configs)

|   horizon_ms |   mean |    std |   count |
|-------------:|-------:|-------:|--------:|
|          100 | 0.4101 | 0.1031 |      12 |
|          500 | 0.4965 | 0.1467 |      12 |
|         1000 | 0.4849 | 0.1055 |      12 |

## Training Time (seconds)

| model               |   mean |   min |   max |
|:--------------------|-------:|------:|------:|
| logistic_regression |   0.08 |  0.01 |  0.2  |
| xgboost             |   3.22 |  1.28 | 18.53 |

## Key Findings

1. **Savitzky-Golay filtering improves F1 by +0.1390** on average vs raw
2. LogReg avg F1=0.4460 vs XGBoost avg F1=0.4816 (XGBoost wins)
3. Depth 40 avg F1=0.4907 vs Depth 5 avg F1=0.4101 (diff: +0.0807)
