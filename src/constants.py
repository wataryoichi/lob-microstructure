"""Constants for LOB Microstructure project."""

from __future__ import annotations

# --- Trading pairs ---
PRIMARY_SYMBOL: str = "BTCUSDT"
EXTENSION_SYMBOLS: list[str] = ["ETHUSDT"]
ALL_SYMBOLS: list[str] = [PRIMARY_SYMBOL] + EXTENSION_SYMBOLS

# --- LOB parameters ---
MAX_LOB_DEPTH: int = 200  # Bybit provides up to 200 levels
DEPTH_LEVELS: list[int] = [5, 10, 40]  # tested in paper
DEFAULT_DEPTH: int = 40

# --- Timing ---
SNAPSHOT_INTERVAL_MS: int = 100  # 100ms between snapshots
PREDICTION_HORIZONS_MS: list[int] = [100, 500, 1000]
SEQUENCE_LENGTHS: list[int] = [1, 10]

# --- Weighted mid-price change weights (Eq. 4) ---
# w_i proportional to 1/i for i=1,2,3; normalized to sum to 1
WEIGHTED_MID_WEIGHTS: list[float] = [6 / 11, 3 / 11, 2 / 11]

# --- Filter types ---
FILTER_TYPES: list[str] = ["raw", "savitzky_golay", "kalman"]

# --- Label schemes ---
LABEL_SCHEMES: list[str] = ["binary", "ternary"]

# --- Model types ---
MODEL_TYPES: list[str] = [
    "logistic_regression",
    "xgboost",
    "catboost",
    "deeplob",
    "cnn_lstm",
    "cnn_xgboost",
    "cnn_catboost",
]

# --- Bybit API ---
BYBIT_REST_BASE: str = "https://api.bybit.com"
BYBIT_WS_PUBLIC: str = "wss://stream.bybit.com/v5/public/linear"

# --- Cost parameters (Bybit, as of 2025) ---
BYBIT_MAKER_FEE_BPS: float = 1.0
BYBIT_TAKER_FEE_BPS: float = 5.5
