"""Comprehensive experiment runner for paper reproduction.

Runs all combinations of:
- Filter types: raw, savitzky_golay, kalman
- LOB depths: 5, 10, 40
- Prediction horizons: 1, 5, 10 (100ms, 500ms, 1000ms)
- Label schemes: binary, ternary
- Models: logistic_regression, xgboost, catboost
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_single_backtest
from .data_loader import filter_complete_snapshots, get_mid_prices, train_test_split_temporal
from .features import build_feature_matrix, normalize_features
from .filters import filter_feature_dataframe
from .labeling import compute_class_weights, create_labels
from .models import get_model

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    filter_type: str
    depth: int
    horizon_steps: int
    label_scheme: str
    model_type: str


def run_single_experiment(
    raw_df: pd.DataFrame,
    exp: ExperimentConfig,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    normalize_window: int = 1000,
    sg_window: int = 21,
    sg_polyorder: int = 3,
    kalman_q: float = 0.001,
    kalman_r: float = 0.1,
) -> dict:
    """Run a single experiment configuration and return metrics."""
    exp_name = f"{exp.model_type}_{exp.label_scheme}_{exp.filter_type}_d{exp.depth}_h{exp.horizon_steps}"
    logger.info(f"Running: {exp_name}")

    start_time = time.time()

    # Filter complete snapshots for this depth
    df = filter_complete_snapshots(raw_df.copy(), depth=exp.depth)
    if len(df) < 1000:
        logger.warning(f"Skip {exp_name}: only {len(df)} complete snapshots")
        return {"experiment": exp_name, "status": "skipped", "reason": "insufficient_data"}

    # Build features
    feat_df = build_feature_matrix(df, depth=exp.depth)

    # Apply filter
    feat_df = filter_feature_dataframe(
        feat_df,
        filter_type=exp.filter_type,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
        kalman_q=kalman_q,
        kalman_r=kalman_r,
    )

    # Normalize
    feat_df = normalize_features(feat_df, window=normalize_window)

    # Create labels
    mid_prices = get_mid_prices(df)
    labels = create_labels(mid_prices, scheme=exp.label_scheme, horizon=exp.horizon_steps)
    feat_df = feat_df.copy()
    feat_df["label"] = labels
    feat_df["mid_price_raw"] = mid_prices

    # Filter valid rows
    feat_df = feat_df[feat_df["label"] >= 0].dropna().reset_index(drop=True)
    if len(feat_df) < 500:
        logger.warning(f"Skip {exp_name}: only {len(feat_df)} valid samples after filtering")
        return {"experiment": exp_name, "status": "skipped", "reason": "insufficient_valid_data"}

    # Split
    exclude = ["timestamp", "label", "mid_price", "mid_price_raw"]
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    train_df, test_df = train_test_split_temporal(feat_df, train_ratio=train_ratio)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    # Validation split
    val_size = int(len(X_train) * val_ratio)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]

    n_classes = len(set(y_train))
    mid_prices_test = test_df["mid_price_raw"].values if "mid_price_raw" in test_df.columns else np.zeros(len(test_df))

    # Build model
    model = get_model(
        exp.model_type,
        n_features=X_train.shape[1],
        n_classes=n_classes,
        seed=seed,
    )

    # Run backtest
    result = run_single_backtest(
        model=model,
        X_train=X_train_fit, y_train=y_train_fit,
        X_test=X_test, y_test=y_test,
        mid_prices_test=mid_prices_test,
        X_val=X_val, y_val=y_val,
        horizon=exp.horizon_steps,
        model_name=exp_name,
    )

    total_time = time.time() - start_time

    return {
        "experiment": exp_name,
        "status": "ok",
        "model": exp.model_type,
        "filter": exp.filter_type,
        "depth": exp.depth,
        "horizon_steps": exp.horizon_steps,
        "horizon_ms": exp.horizon_steps * 100,
        "label_scheme": exp.label_scheme,
        "n_train": len(X_train_fit),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "accuracy": result.classification_metrics["accuracy"],
        "f1_macro": result.classification_metrics["f1_macro"],
        **{k: v for k, v in result.classification_metrics.items() if k.startswith("f1_class_")},
        "train_time_sec": result.train_time_sec,
        "total_time_sec": total_time,
        "hit_ratio": result.trading_metrics.get("hit_ratio", 0),
        "avg_gross_bps": result.trading_metrics.get("avg_gross_bps", 0),
        "avg_net_maker_bps": result.trading_metrics.get("avg_net_maker_bps", 0),
    }


def run_paper_reproduction(
    raw_df: pd.DataFrame,
    output_dir: str | Path = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full paper reproduction experiment suite.

    Matches Table 1 & 2 from Wang (2025):
    - Filters: raw, savitzky_golay, kalman
    - Depths: 5, 40
    - Horizons: 1 (100ms), 5 (500ms), 10 (1000ms)
    - Labels: binary, ternary
    - Models: logistic_regression, xgboost, catboost
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Focused experiment grid: preprocessing x horizon x labels x models
    # Models: LR + XGBoost only (paper shows these match or beat deep models)
    filters = ["raw", "savitzky_golay", "kalman"]
    depths = [5, 40]
    horizons = [1, 5, 10]
    labels = ["binary", "ternary"]
    models = ["logistic_regression", "xgboost"]

    experiments = []
    for filt, depth, horizon, label, model in product(filters, depths, horizons, labels, models):
        # Key combos from paper tables:
        # 100ms+5level, 500ms+40level, 1000ms+40level
        if depth == 5 and horizon != 1:
            continue
        if depth == 40 and horizon == 1:
            continue
        experiments.append(ExperimentConfig(
            filter_type=filt,
            depth=depth,
            horizon_steps=horizon,
            label_scheme=label,
            model_type=model,
        ))

    logger.info(f"Running {len(experiments)} experiments...")

    results = []
    for i, exp in enumerate(experiments):
        logger.info(f"[{i+1}/{len(experiments)}] {exp}")
        try:
            result = run_single_experiment(raw_df, exp, seed=seed)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {exp} -> {e}")
            results.append({
                "experiment": f"{exp.model_type}_{exp.label_scheme}_{exp.filter_type}_d{exp.depth}_h{exp.horizon_steps}",
                "status": "error",
                "error": str(e),
            })

        # Save intermediate results
        if (i + 1) % 10 == 0:
            _save_results(results, output_dir)

    _save_results(results, output_dir)
    df = pd.DataFrame(results)
    logger.info(f"\nCompleted {len(results)} experiments")
    return df


def run_depth_comparison(
    raw_df: pd.DataFrame,
    output_dir: str | Path = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Run Table 3: LOB depth comparison (XGBoost, binary, SG, 1000ms)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for depth in [5, 10, 40]:
        exp = ExperimentConfig(
            filter_type="savitzky_golay",
            depth=depth,
            horizon_steps=10,
            label_scheme="binary",
            model_type="xgboost",
        )
        result = run_single_experiment(raw_df, exp, seed=seed)
        results.append(result)

    _save_results(results, output_dir, filename="depth_comparison.json")
    return pd.DataFrame(results)


def run_extended_horizons(
    raw_df: pd.DataFrame,
    output_dir: str | Path = "results",
    seed: int = 42,
) -> pd.DataFrame:
    """Layer 2: Longer horizons where price moves may exceed costs.

    Grid:
    - Horizons: 50 (5s), 100 (10s), 300 (30s), 600 (60s)
    - Filters: raw, savitzky_golay
    - Labels: binary only
    - Models: logistic_regression, xgboost
    - Depth: 40 only
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for filt in ["raw", "savitzky_golay"]:
        for horizon in [50, 100, 300, 600]:
            for model in ["logistic_regression", "xgboost"]:
                experiments.append(ExperimentConfig(
                    filter_type=filt,
                    depth=40,
                    horizon_steps=horizon,
                    label_scheme="binary",
                    model_type=model,
                ))

    logger.info(f"Running {len(experiments)} extended horizon experiments...")

    results = []
    for i, exp in enumerate(experiments):
        logger.info(f"[{i+1}/{len(experiments)}] {exp}")
        try:
            result = run_single_experiment(raw_df, exp, seed=seed)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {exp} -> {e}")
            results.append({
                "experiment": f"{exp.model_type}_{exp.filter_type}_d{exp.depth}_h{exp.horizon_steps}",
                "status": "error",
                "error": str(e),
            })

    _save_results(results, output_dir, filename="extended_results.json")
    return pd.DataFrame(results)


def _save_results(results: list[dict], output_dir: Path, filename: str = "experiment_results.json"):
    """Save results to JSON."""
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


def format_paper_table(
    df: pd.DataFrame,
    label_scheme: str = "binary",
) -> str:
    """Format results as a paper-style comparison table."""
    sub = df[(df["status"] == "ok") & (df["label_scheme"] == label_scheme)].copy()
    if sub.empty:
        return "No results for this label scheme."

    # Pivot: rows = (horizon_ms, depth, filter), columns = model
    sub["config"] = sub.apply(
        lambda r: f"{r['horizon_ms']}ms, {r['depth']}-level", axis=1
    )

    pivot = sub.pivot_table(
        index=["config", "filter"],
        columns="model",
        values="f1_macro",
        aggfunc="first",
    )

    title = f"## {'Binary' if label_scheme == 'binary' else 'Ternary'} Classification F1 Scores\n"
    return title + pivot.round(4).to_markdown()
