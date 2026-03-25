# LOB Microstructure Experiment Report

**Total experiments**: 54 (54 successful, 0 skipped/failed)

## Binary Classification (F1 Scores)

|                                        |   catboost |   logistic_regression |   xgboost |
|:---------------------------------------|-----------:|----------------------:|----------:|
| ('1000ms, 40-level', 'kalman')         |     0.4199 |                0.4794 |    0.4341 |
| ('1000ms, 40-level', 'raw')            |     0.5151 |                0.498  |    0.4649 |
| ('1000ms, 40-level', 'savitzky_golay') |     0.6131 |                0.5924 |    0.7091 |
| ('100ms, 5-level', 'kalman')           |     0.4892 |                0.3687 |    0.4882 |
| ('100ms, 5-level', 'raw')              |     0.5072 |                0.496  |    0.5072 |
| ('100ms, 5-level', 'savitzky_golay')   |     0.5228 |                0.5696 |    0.4909 |
| ('500ms, 40-level', 'kalman')          |     0.47   |                0.4596 |    0.4742 |
| ('500ms, 40-level', 'raw')             |     0.5011 |                0.5157 |    0.4907 |
| ('500ms, 40-level', 'savitzky_golay')  |     0.7531 |                0.6653 |    0.7742 |

**Best binary**: xgboost + savitzky_golay (40-level, 500ms) -> F1=0.7742, Acc=0.9037

## Ternary Classification (F1 Scores)

|                                        |   catboost |   logistic_regression |   xgboost |
|:---------------------------------------|-----------:|----------------------:|----------:|
| ('1000ms, 40-level', 'kalman')         |     0.3618 |                0.3909 |    0.3983 |
| ('1000ms, 40-level', 'raw')            |     0.4159 |                0.4316 |    0.3615 |
| ('1000ms, 40-level', 'savitzky_golay') |     0.6451 |                0.4317 |    0.6268 |
| ('100ms, 5-level', 'kalman')           |     0.3769 |                0.216  |    0.3341 |
| ('100ms, 5-level', 'raw')              |     0.4308 |                0.3386 |    0.3043 |
| ('100ms, 5-level', 'savitzky_golay')   |     0.521  |                0.4203 |    0.3867 |
| ('500ms, 40-level', 'kalman')          |     0.321  |                0.3093 |    0.3274 |
| ('500ms, 40-level', 'raw')             |     0.4444 |                0.3608 |    0.4053 |
| ('500ms, 40-level', 'savitzky_golay')  |     0.7111 |                0.4842 |    0.6918 |

**Best ternary**: catboost + savitzky_golay (40-level, 500ms) -> F1=0.7111, Acc=0.8113

## Filter Comparison (average F1 across all configs)

| filter         |   mean |    std |   count |
|:---------------|-------:|-------:|--------:|
| kalman         | 0.3955 | 0.076  |      18 |
| raw            | 0.4438 | 0.0672 |      18 |
| savitzky_golay | 0.5894 | 0.1179 |      18 |

## Model Comparison (average F1 across all configs)

| model               |   mean |    std |   count |
|:--------------------|-------:|-------:|--------:|
| catboost            | 0.5011 | 0.1169 |      18 |
| logistic_regression | 0.446  | 0.1079 |      18 |
| xgboost             | 0.4816 | 0.137  |      18 |

## Depth Comparison (average F1 across all configs)

|   depth |   mean |    std |   count |
|--------:|-------:|-------:|--------:|
|       5 | 0.4316 | 0.0942 |      18 |
|      40 | 0.4986 | 0.128  |      36 |

## Horizon Comparison (average F1 across all configs)

|   horizon_ms |   mean |    std |   count |
|-------------:|-------:|-------:|--------:|
|          100 | 0.4316 | 0.0942 |      18 |
|          500 | 0.5088 | 0.1495 |      18 |
|         1000 | 0.4883 | 0.1055 |      18 |

## Training Time (seconds)

| model               |   mean |   min |   max |
|:--------------------|-------:|------:|------:|
| catboost            |   2.9  |  1.21 |  9.53 |
| logistic_regression |   0.09 |  0.01 |  0.28 |
| xgboost             |   2.79 |  1.2  |  7.51 |

## Key Findings

1. **Savitzky-Golay filtering improves F1 by +0.1456** on average vs raw
2. LogReg avg F1=0.4460 vs XGBoost avg F1=0.4816 (XGBoost wins)
3. Depth 40 avg F1=0.4986 vs Depth 5 avg F1=0.4316 (diff: +0.0670)
