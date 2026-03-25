# Paper Specification: Cryptocurrency LOB Microstructure Prediction

## Paper Information
- **Title:** (Cryptocurrency LOB short-term price prediction benchmarking study)
- **URL:** https://arxiv.org/html/2506.05764v2
- **Data Source:** Bybit BTC/USDT, single trading day 2025-01-30, 100ms sampling interval
- **LOB Depth Available:** Top 200 bid + 200 ask levels (price and quantity per level)

---

## 1. LOB Feature Definitions

### Equation 1 -- Previous Mid-Price
```
m_{t-1} = (a^1_{t-1} + b^1_{t-1}) / 2
```
where `a^1_{t-1}` = level-1 ask price, `b^1_{t-1}` = level-1 bid price.

### Equation 2 -- First-Level Order Imbalance
```
I^1_t = (Q^b_{t,1} - Q^a_{t,1}) / (Q^b_{t,1} + Q^a_{t,1})
```
where `Q^b_{t,1}` and `Q^a_{t,1}` are bid and ask quantities at level 1.

### Equation 3 -- Five-Level Aggregate Imbalance
```
I^5_t = (sum_{i=1}^{5} Q^b_{t,i} - sum_{i=1}^{5} Q^a_{t,i}) / (sum_{i=1}^{5} Q^b_{t,i} + sum_{i=1}^{5} Q^a_{t,i})
```

### Equation 4 -- Weighted Mid-Price Change
```
Delta_m^w_t = sum_{i=1}^{3} w_i * (m^i_t - m^i_{t-1})
```
Weights: `w_i proportional to 1/i`, summing to one. For 3 levels:
- w_1 = (1/1) / (1 + 1/2 + 1/3) = 6/11 ~ 0.5455
- w_2 = (1/2) / (1 + 1/2 + 1/3) = 3/11 ~ 0.2727
- w_3 = (1/3) / (1 + 1/2 + 1/3) = 2/11 ~ 0.1818

where `m^i_t` is the mid-price at level i:
```
m^i_t = (a^i_t + b^i_t) / 2
```

### Additional Hand-Crafted Features (mentioned but not formalized)
- Bid-ask spread
- Recent mid-price returns
- Rolling window volatility
- Order flow imbalance across multiple depth levels
- Principal component summaries of the LOB snapshot
- Top k bid/ask price and volume levels (raw LOB data)

---

## 2. Filtering / Denoising

### 2.1 Savitzky-Golay Filter

**Parameters:**
- Polynomial degree: d = 3 (cubic)
- Window size: 2m+1 = 21 samples (m = 10)

**Equation 5 -- Optimization Problem:**
```
min_{a_0, a_1, a_2, a_3} sum_{j=-10}^{10} [v_{t+j} - (a_0 + a_1*j + a_2*j^2 + a_3*j^3)]^2
```

**Equation 6 -- Normal Equations:**
```
(A^T * A) * a = A^T * v
where A_{j,l} = j^l,  j in {-10, ..., 10},  l in {0, ..., 3}
```

**Equation 7 -- Smoothed Output:**
```
v_hat_t = sum_{j=-10}^{10} c_j * v_{t+j}
```
where convolution weights {c_j} are precomputed from the normal equations solution.

### 2.2 Kalman Filter

One-dimensional Kalman filter applied to each feature series independently, treating it as a noisy observation of a latent random-walk state.

**Equation 8 -- State and Observation Model:**
```
x_t = x_{t-1} + w_{t-1},    w_{t-1} ~ N(0, Q)    (process noise)
v_t = x_t + eps_t,           eps_t ~ N(0, R)       (measurement noise)
```

**Equation 9 -- Kalman Gain and State Update:**
```
K_t = (P_{t-1} + Q) / (P_{t-1} + Q + R)
x_hat_t = x_hat_{t-1} + K_t * (v_t - x_hat_{t-1})
```

**Equation 10 -- Variance Update:**
```
P_t = (1 - K_t) * (P_{t-1} + Q)
```

**Parameters:** Q and R are NOT specified in the paper. The authors note: "the Kalman filter's parameters are more rigid and sensitive" and that a "limited grid search over a smaller sample" was used. Kalman filtering consistently underperformed raw data in experiments.

---

## 3. Labeling Schemes

### 3.1 Binary Classification
- **Classes:** Up (1) / Down (0)
- **Definition:** Based on mid-price change over the prediction horizon
- **Threshold:** Not explicitly defined (presumably Delta_p > 0 => Up, else Down)

### 3.2 Ternary Classification
- **Classes:** Up / Flat (Stationary) / Down
- **Definition:** Stationary range: -eps <= Delta_p/p <= +eps
- **Threshold eps:** NOT specified numerically. Paper states: "we tune eps to achieve roughly equal class frequencies"
- **Class imbalance handling:** Inverse-frequency weighting in the loss function

---

## 4. Model Architectures

### 4.1 CNN + CatBoost
1. Two Conv1D layers with ReLU activations
2. Batch Normalization after each conv layer
3. GlobalMaxPooling
4. Dense projection to 64-dimensional embedding
5. CatBoost classifier with inverse-frequency class weights

### 4.2 DeepLOB
1. Input reshape: (T x F) -> (T x F x 1) tensor
2. Three Conv2D blocks:
   - Kernel size: 1 x 3
   - Activation: LeakyReLU
   - Batch Normalization after each block
3. LSTM(64) layer processing sequences of up to 100 timesteps
4. Trained with focal loss and class weights
5. Filter counts, strides, padding: NOT specified

### 4.3 Simpler CNN + LSTM
1. Single Conv2D layer (instead of three in DeepLOB)
2. Batch Normalization
3. SpatialDropout2D (dropout rate NOT specified)
4. Bidirectional LSTM(32)
5. Output reshaped to (T, features)

### 4.4 XGBoost (Standalone)
1. Input: (T x F) flattened to single vector of length T*F
2. XGBClassifier with `multi:softprob` objective
3. Sample weights to correct class imbalance
4. Early stopping on validation set
5. Grid search over n_estimators and learning_rate (ranges NOT specified)

### 4.5 CNN + XGBoost
1. Identical CNN backbone to CNN+CatBoost (two Conv1D layers, ReLU, BN, GlobalMaxPooling, Dense->64)
2. XGBoost classifier on the 64-dim embeddings

### 4.6 Logistic Regression
- Hand-crafted features from LOB
- No further architecture details provided

---

## 5. Data Preprocessing Pipeline

1. **Source data:** Bybit BTC/USDT, 2025-01-30, 100ms intervals, top 200 bid/ask levels
2. **Missing data:** Snapshots with missing entries (NaN) for required depth levels are dropped
   - "Drop any snapshot missing one of the 40 bid-ask levels"
3. **Normalization:** Per-day z-score normalization, using previous-day statistics to avoid lookahead bias
4. **Filtering:** Applied after data cleaning -- Savitzky-Golay (window=21, degree=3) or Kalman or Raw
5. **Sequence construction:**
   - T=1: single 100ms snapshot
   - T=10: ten consecutive 100ms snapshots concatenated (spanning 1 second)

---

## 6. Training Details

### Train/Test Splits
- **Tables 1-3 (main experiments):** 100,000 snapshots total
  - Training: 80,000 rows
  - Test: 20,000 rows
  - Validation: 20% of training set (i.e., ~16,000 rows)
  - After filtering for 40-level completeness: reduced to 5,442 test samples
- **Table 4 (sequence length experiments):** 1,000,000 snapshots total
  - Training: 800,000 rows
  - Test: 200,000 rows
  - Only 148,340 rows (74% coverage) had complete >= 5 bid/ask levels
  - Test support: 73,953 (class 0) + 74,387 (class 1)

### Training Hyperparameters (mostly NOT specified)
- **Optimizer:** Not stated
- **Learning rate:** Not provided
- **Batch size:** Not disclosed
- **Number of epochs:** Not specified
- **Early stopping:** Used for XGBoost (criteria not detailed)
- **Loss function:** Focal loss for DeepLOB; inverse-frequency weighted loss for ternary; standard for others
- **Focal loss parameters (gamma, alpha):** NOT specified
- **Dropout rates:** NOT specified
- **Weight decay / L2 regularization:** NOT specified

---

## 7. LOB Depth Levels Tested

- **5 levels** (top 5 bid + 5 ask)
- **10 levels** (top 10 bid + 10 ask)
- **40 levels** (top 40 bid + 40 ask)

---

## 8. Prediction Horizons

- **100 ms** (1 timestep ahead)
- **500 ms** (5 timesteps ahead)
- **1000 ms** (10 timesteps ahead)

Fixed millisecond intervals regardless of T value. When T=1, a single 100ms snapshot predicts the mid-price change over the next h milliseconds.

---

## 9. Results Tables

### Evaluation Metric
**F1 Score (Equations 11-12):**
```
F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
```

### Table 1: Ternary Classification, T=1

| Horizon & Depth | Filter | CatBoost | DeepLOB | CNN+LSTM | CNN+XGB | XGBoost | LogReg |
|---|---|---|---|---|---|---|---|
| 100ms, 5-level | Raw | 0.3922 | 0.4080 | 0.3987 | 0.4029 | 0.3939 | 0.3777 |
| 100ms, 5-level | Kalman | 0.3651 | 0.3838 | 0.3594 | 0.3609 | 0.3875 | 0.3732 |
| 100ms, 5-level | Savitzky-Golay | 0.4135 | 0.4281 | 0.4207 | 0.4179 | 0.4173 | 0.4101 |
| 500ms, 40-level | Raw | 0.4066 | 0.4286 | 0.4160 | 0.4159 | 0.4457 | 0.4356 |
| 500ms, 40-level | Kalman | 0.3216 | 0.3044 | 0.3255 | 0.3211 | 0.4114 | 0.2993 |
| 500ms, 40-level | Savitzky-Golay | 0.5189 | 0.5237 | 0.5252 | 0.5282 | 0.5393 | 0.5434 |
| 1000ms, 40-level | Raw | 0.4046 | 0.4099 | 0.4142 | 0.4118 | 0.4357 | 0.4901 |
| 1000ms, 40-level | Kalman | 0.3191 | 0.3618 | 0.3603 | 0.3397 | 0.4084 | 0.3421 |
| 1000ms, 40-level | Savitzky-Golay | 0.4611 | 0.4762 | 0.4936 | 0.4604 | 0.4999 | 0.5382 |

### Table 2: Binary Classification, T=1

| Horizon & Depth | Filter | CatBoost | DeepLOB | CNN+LSTM | CNN+XGB | XGBoost | LogReg |
|---|---|---|---|---|---|---|---|
| 100ms, 5-level | Raw | 0.5257 | 0.5215 | 0.5235 | 0.5211 | 0.5296 | 0.5216 |
| 100ms, 5-level | Kalman | 0.5129 | 0.5047 | 0.5043 | 0.5148 | 0.5268 | 0.5128 |
| 100ms, 5-level | Savitzky-Golay | 0.5271 | 0.5296 | 0.5317 | 0.5327 | 0.5338 | 0.5336 |
| 500ms, 40-level | Raw | 0.5941 | 0.6333 | 0.6323 | 0.6320 | 0.6542 | 0.6517 |
| 500ms, 40-level | Kalman | 0.5046 | 0.4825 | 0.5139 | 0.5716 | 0.6301 | 0.4882 |
| 500ms, 40-level | Savitzky-Golay | 0.6260 | 0.7189 | 0.7189 | 0.7130 | 0.7281 | 0.7284 |
| 1000ms, 40-level | Raw | 0.5799 | 0.6448 | 0.6372 | 0.6276 | 0.6509 | 0.6515 |
| 1000ms, 40-level | Kalman | 0.5109 | 0.4908 | 0.4680 | 0.5536 | 0.6115 | 0.4930 |
| 1000ms, 40-level | Savitzky-Golay | 0.6140 | 0.6947 | 0.6888 | 0.6876 | 0.7150 | 0.7089 |

### Table 3: LOB Depth Impact (XGBoost, Binary, Savitzky-Golay, 1000ms, T=1)

| Depth | Accuracy | F1(class 0) | F1(class 1) | Test Support |
|---|---|---|---|---|
| 40 levels | 0.7150 | 0.7115 | 0.7184 | 5,442 |
| 10 levels | 0.5837 | 0.5732 | 0.5937 | 17,471 |
| 5 levels | 0.5797 | 0.5560 | 0.6009 | 18,336 |

### Table 4: Sequence Length Effects (Savitzky-Golay, Binary, 1M snapshots)

| Model | T | Accuracy | F1(0) | F1(1) | Support 0 | Support 1 | Runtime |
|---|---|---|---|---|---|---|---|
| XGBoost | 1 | 0.5732 | 0.5757 | 0.5707 | 73,953 | 74,387 | 1m 36s |
| XGBoost | 10 | 0.5949 | 0.5989 | 0.5909 | 73,951 | 74,387 | 7m 04s |
| LogReg | 1 | 0.5551 | 0.5442 | 0.5656 | 73,953 | 74,387 | 1m 11s |
| LogReg | 10 | 0.5716 | 0.5654 | 0.5777 | 73,951 | 74,387 | 9m 13s |

---

## 10. Code / Pseudocode

No code or pseudocode provided in the paper. Experiments conducted "offline in Python."

---

## Key Findings Summary

1. **Savitzky-Golay filtering consistently improved** prediction accuracy across all models and configurations
2. **Kalman filtering often performed worse than raw data**, requiring more parameter tuning
3. **Simpler models (XGBoost, LogReg) matched or exceeded deep learning models** (DeepLOB, CNN+LSTM) when given proper preprocessing
4. **Best overall result: 0.7284 F1** (Logistic Regression, binary, Savitzky-Golay, 500ms horizon, 40-level LOB)
5. **Deeper LOB (40 levels) substantially outperformed shallower depths** but reduced data coverage
6. **Longer sequences (T=10 vs T=1) provided ~2% accuracy gain** at significant computational cost
7. **100ms prediction horizon was near-random** (~0.53 binary accuracy); 500ms and 1000ms were more predictable

## Parameters NOT Specified in Paper (implementation choices needed)
- Kalman filter Q and R values
- Ternary classification threshold epsilon
- Conv layer filter counts, strides, and padding
- Focal loss gamma and alpha parameters
- Optimizer type, learning rate, batch size, epochs
- Dropout rates
- XGBoost grid search ranges
- Exact feature vector composition for each model
- Normalization parameters beyond "per-day z-score"
