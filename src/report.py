"""Report generation for experiment results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_experiment_report(
    results_dir: str | Path = "results",
    output_dir: str | Path = "reports",
) -> str:
    """Generate a markdown report from experiment results."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = results_dir / "experiment_results.json"
    if not results_path.exists():
        return "No experiment results found."

    with open(results_path) as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()

    if ok.empty:
        return "No successful experiments."

    lines = [
        "# LOB Microstructure Experiment Report",
        "",
        f"**Total experiments**: {len(df)} ({len(ok)} successful, {len(df)-len(ok)} skipped/failed)",
        "",
    ]

    # Paper Table 2: Binary Classification
    binary = ok[ok["label_scheme"] == "binary"]
    if not binary.empty:
        lines.append("## Binary Classification (F1 Scores)")
        lines.append("")
        lines.append(_make_pivot_table(binary))
        lines.append("")

        # Best binary result
        best = binary.loc[binary["f1_macro"].idxmax()]
        lines.append(
            f"**Best binary**: {best['model']} + {best['filter']} "
            f"({best['depth']}-level, {best['horizon_ms']}ms) "
            f"-> F1={best['f1_macro']:.4f}, Acc={best['accuracy']:.4f}"
        )
        lines.append("")

    # Paper Table 1: Ternary Classification
    ternary = ok[ok["label_scheme"] == "ternary"]
    if not ternary.empty:
        lines.append("## Ternary Classification (F1 Scores)")
        lines.append("")
        lines.append(_make_pivot_table(ternary))
        lines.append("")

        best = ternary.loc[ternary["f1_macro"].idxmax()]
        lines.append(
            f"**Best ternary**: {best['model']} + {best['filter']} "
            f"({best['depth']}-level, {best['horizon_ms']}ms) "
            f"-> F1={best['f1_macro']:.4f}, Acc={best['accuracy']:.4f}"
        )
        lines.append("")

    # Filter comparison
    lines.append("## Filter Comparison (average F1 across all configs)")
    lines.append("")
    filter_avg = ok.groupby("filter")["f1_macro"].agg(["mean", "std", "count"])
    lines.append(filter_avg.round(4).to_markdown())
    lines.append("")

    # Model comparison
    lines.append("## Model Comparison (average F1 across all configs)")
    lines.append("")
    model_avg = ok.groupby("model")["f1_macro"].agg(["mean", "std", "count"])
    lines.append(model_avg.round(4).to_markdown())
    lines.append("")

    # Depth comparison
    lines.append("## Depth Comparison (average F1 across all configs)")
    lines.append("")
    depth_avg = ok.groupby("depth")["f1_macro"].agg(["mean", "std", "count"])
    lines.append(depth_avg.round(4).to_markdown())
    lines.append("")

    # Horizon comparison
    lines.append("## Horizon Comparison (average F1 across all configs)")
    lines.append("")
    horizon_avg = ok.groupby("horizon_ms")["f1_macro"].agg(["mean", "std", "count"])
    lines.append(horizon_avg.round(4).to_markdown())
    lines.append("")

    # Training time comparison
    lines.append("## Training Time (seconds)")
    lines.append("")
    time_avg = ok.groupby("model")["train_time_sec"].agg(["mean", "min", "max"])
    lines.append(time_avg.round(2).to_markdown())
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Does SG beat raw?
    if "savitzky_golay" in ok["filter"].values and "raw" in ok["filter"].values:
        sg_f1 = ok[ok["filter"] == "savitzky_golay"]["f1_macro"].mean()
        raw_f1 = ok[ok["filter"] == "raw"]["f1_macro"].mean()
        diff = sg_f1 - raw_f1
        if diff > 0:
            lines.append(f"1. **Savitzky-Golay filtering improves F1 by +{diff:.4f}** on average vs raw")
        else:
            lines.append(f"1. Savitzky-Golay filtering {diff:+.4f} vs raw (no improvement)")

    # Does simple beat complex?
    if "logistic_regression" in ok["model"].values and "xgboost" in ok["model"].values:
        lr_f1 = ok[ok["model"] == "logistic_regression"]["f1_macro"].mean()
        xgb_f1 = ok[ok["model"] == "xgboost"]["f1_macro"].mean()
        lines.append(
            f"2. LogReg avg F1={lr_f1:.4f} vs XGBoost avg F1={xgb_f1:.4f} "
            f"({'LogReg wins' if lr_f1 >= xgb_f1 else 'XGBoost wins'})"
        )

    # Depth impact
    if 5 in ok["depth"].values and 40 in ok["depth"].values:
        d5_f1 = ok[ok["depth"] == 5]["f1_macro"].mean()
        d40_f1 = ok[ok["depth"] == 40]["f1_macro"].mean()
        lines.append(f"3. Depth 40 avg F1={d40_f1:.4f} vs Depth 5 avg F1={d5_f1:.4f} (diff: {d40_f1-d5_f1:+.4f})")

    lines.append("")

    report = "\n".join(lines)

    # Save
    report_path = output_dir / "experiment_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    return report


def _make_pivot_table(df: pd.DataFrame) -> str:
    """Create a pivot table matching paper format."""
    df = df.copy()
    df["config"] = df.apply(
        lambda r: f"{int(r['horizon_ms'])}ms, {int(r['depth'])}-level", axis=1
    )

    pivot = df.pivot_table(
        index=["config", "filter"],
        columns="model",
        values="f1_macro",
        aggfunc="first",
    )

    return pivot.round(4).to_markdown()
