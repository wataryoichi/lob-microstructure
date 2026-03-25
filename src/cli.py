"""CLI interface for LOB Microstructure Trading System."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(help="LOB Microstructure: Better Inputs Matter More")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def fetch_data(
    config: str = typer.Option("configs/base.yaml", "--config", "-c"),
    n_snapshots: int = typer.Option(0, "--n-snapshots", "-n", help="Override n_snapshots (0=use config)"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate synthetic data for testing"),
) -> None:
    """Fetch LOB data from Bybit or generate synthetic data."""
    from .config import load_config

    cfg = load_config(config)
    output_dir = Path(cfg.data.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        from .data_collector import load_sample_data
        n = n_snapshots if n_snapshots > 0 else cfg.training.n_snapshots
        df = load_sample_data(n_snapshots=n, depth=cfg.data.lob_depth)
        path = output_dir / f"{cfg.data.symbol}_synthetic_{n}.parquet"
        df.to_parquet(path, index=False)
        typer.echo(f"Synthetic data saved to {path} ({len(df)} snapshots)")
    else:
        from .data_collector import collect_snapshots
        n = n_snapshots if n_snapshots > 0 else cfg.training.n_snapshots
        df = collect_snapshots(
            symbol=cfg.data.symbol,
            n_snapshots=n,
            interval_ms=cfg.data.snapshot_interval_ms,
            depth=cfg.data.lob_depth,
            output_dir=output_dir,
        )
        typer.echo(f"Collected {len(df)} snapshots")


@app.command()
def build_features(
    config: str = typer.Option("configs/base.yaml", "--config", "-c"),
    depth: int = typer.Option(0, "--depth", "-d", help="Override LOB depth (0=use config)"),
) -> None:
    """Build features and labels from raw LOB data."""
    from .config import load_config
    from .data_loader import filter_complete_snapshots, get_mid_prices, load_lob_data
    from .features import build_feature_matrix, normalize_features
    from .filters import filter_feature_dataframe
    from .labeling import create_labels

    cfg = load_config(config)
    lob_depth = depth if depth > 0 else cfg.data.lob_depth

    # Load raw data
    df = load_lob_data(cfg.data.raw_dir, cfg.data.symbol)
    if df.empty:
        typer.echo("No data found. Run fetch-data first.")
        raise typer.Exit(1)

    # Filter complete snapshots
    df = filter_complete_snapshots(df, depth=lob_depth)
    if df.empty:
        typer.echo(f"No complete snapshots at depth={lob_depth}")
        raise typer.Exit(1)

    # Build features
    feat_df = build_feature_matrix(
        df, depth=lob_depth,
        include_raw_lob=cfg.features.include_raw_lob,
        include_imbalance=cfg.features.include_imbalance,
        include_spread=cfg.features.include_spread,
        include_weighted_mid=cfg.features.include_weighted_mid,
    )

    # Apply filter
    feat_df = filter_feature_dataframe(
        feat_df,
        filter_type=cfg.preprocessing.filter_type,
        sg_window=cfg.preprocessing.sg_window,
        sg_polyorder=cfg.preprocessing.sg_polyorder,
        kalman_q=cfg.preprocessing.kalman_q,
        kalman_r=cfg.preprocessing.kalman_r,
    )

    # Normalize
    if cfg.preprocessing.normalize:
        feat_df = normalize_features(feat_df, window=cfg.preprocessing.normalize_window)

    # Create labels
    mid_prices = get_mid_prices(df)
    labels = create_labels(
        mid_prices,
        scheme=cfg.labeling.scheme,
        horizon=cfg.labeling.horizon_steps,
        ternary_epsilon=cfg.labeling.ternary_epsilon,
    )
    feat_df = feat_df.copy()  # defragment
    feat_df["label"] = labels

    # Save
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_{cfg.preprocessing.filter_type}_d{lob_depth}_h{cfg.labeling.horizon_steps}.parquet"
    feat_df.to_parquet(out_path, index=False)
    typer.echo(f"Features saved to {out_path} ({len(feat_df)} rows, {len(feat_df.columns)} columns)")


@app.command()
def train(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
    model_type: str = typer.Option("", "--model", "-m", help="Override model type"),
) -> None:
    """Train a model on processed features."""
    from .config import load_config
    from .data_loader import train_test_split_temporal
    from .models import get_model

    import pandas as pd

    cfg = load_config(config)
    mtype = model_type if model_type else cfg.model.type

    # Find processed features
    proc_dir = Path(cfg.data.processed_dir)
    feature_files = sorted(proc_dir.glob("features_*.parquet"))
    if not feature_files:
        typer.echo("No processed features found. Run build-features first.")
        raise typer.Exit(1)

    feat_df = pd.read_parquet(feature_files[-1])
    typer.echo(f"Loaded {feature_files[-1].name}: {len(feat_df)} rows")

    # Remove invalid labels and NaN rows
    feat_df = feat_df[feat_df["label"] >= 0].dropna().reset_index(drop=True)
    typer.echo(f"After filtering: {len(feat_df)} valid samples")

    # Split
    exclude = ["timestamp", "label", "mid_price"]
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    train_df, test_df = train_test_split_temporal(feat_df, train_ratio=cfg.training.train_ratio)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    # Validation split from training set
    val_size = int(len(X_train) * cfg.training.val_ratio)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]

    n_classes = len(set(y_train))
    # Verify all splits have all classes
    train_classes = set(y_train_fit)
    val_classes = set(y_val)
    test_classes = set(y_test)
    typer.echo(f"Train: {len(X_train_fit)} (classes: {train_classes}), Val: {len(X_val)} (classes: {val_classes}), Test: {len(X_test)} (classes: {test_classes})")

    # Train
    model = get_model(
        mtype,
        n_features=X_train.shape[1],
        n_classes=n_classes,
        seed=cfg.model.seed,
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        early_stopping_rounds=cfg.model.early_stopping_rounds,
    )

    from .backtest import run_single_backtest

    mid_prices_test = test_df["mid_price"].values if "mid_price" in test_df.columns else np.zeros(len(test_df))
    timestamps_test = test_df["timestamp"].values if "timestamp" in test_df.columns else None

    result = run_single_backtest(
        model=model,
        X_train=X_train_fit, y_train=y_train_fit,
        X_test=X_test, y_test=y_test,
        mid_prices_test=mid_prices_test,
        timestamps_test=timestamps_test,
        X_val=X_val, y_val=y_val,
        horizon=cfg.labeling.horizon_steps,
        maker_fee_bps=cfg.cost.maker_fee_bps if cfg.cost.enabled else 0,
        taker_fee_bps=cfg.cost.taker_fee_bps if cfg.cost.enabled else 0,
        slippage_bps=cfg.cost.slippage_bps if cfg.cost.enabled else 0,
        model_name=mtype,
    )

    # Save results
    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": mtype,
        "classification": result.classification_metrics,
        "trading": result.trading_metrics,
        "train_time_sec": result.train_time_sec,
    }
    out_path = results_dir / f"{mtype}_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    typer.echo(f"\nResults saved to {out_path}")

    # Print summary
    typer.echo(f"\n{'='*50}")
    typer.echo(f"Model: {mtype}")
    typer.echo(f"Accuracy: {result.classification_metrics['accuracy']:.4f}")
    typer.echo(f"F1 (macro): {result.classification_metrics['f1_macro']:.4f}")
    for k, v in result.classification_metrics.items():
        if k.startswith("f1_class_"):
            typer.echo(f"  {k}: {v:.4f}")
    typer.echo(f"Train time: {result.train_time_sec:.1f}s")
    typer.echo(f"Avg gross: {result.trading_metrics['avg_gross_bps']:.2f} bps/trade")
    typer.echo(f"Avg net (maker): {result.trading_metrics['avg_net_maker_bps']:.2f} bps/trade")
    typer.echo(f"Hit ratio: {result.trading_metrics['hit_ratio']:.4f}")
    typer.echo(f"Sharpe: {result.trading_metrics['sharpe']:.2f}")


@app.command()
def evaluate(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Compare all models (paper reproduction Table 2)."""
    from .config import load_config
    from .data_loader import train_test_split_temporal
    from .models import get_model
    from .backtest import run_model_comparison, results_to_dataframe

    import pandas as pd

    cfg = load_config(config)

    # Load processed features
    proc_dir = Path(cfg.data.processed_dir)
    feature_files = sorted(proc_dir.glob("features_*.parquet"))
    if not feature_files:
        typer.echo("No processed features found. Run build-features first.")
        raise typer.Exit(1)

    feat_df = pd.read_parquet(feature_files[-1])
    feat_df = feat_df[feat_df["label"] >= 0].dropna().reset_index(drop=True)

    exclude = ["timestamp", "label", "mid_price"]
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    train_df, test_df = train_test_split_temporal(feat_df, train_ratio=cfg.training.train_ratio)
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values.astype(int)

    val_size = int(len(X_train) * cfg.training.val_ratio)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_fit = X_train[:-val_size]
    y_train_fit = y_train[:-val_size]

    n_classes = len(set(y_train))
    n_feat = X_train.shape[1]

    mid_prices_test = test_df["mid_price"].values if "mid_price" in test_df.columns else np.zeros(len(test_df))

    # Build model configs for comparison
    model_types = ["logistic_regression", "xgboost", "catboost"]
    model_configs = []
    for mt in model_types:
        model = get_model(mt, n_features=n_feat, n_classes=n_classes, seed=cfg.model.seed)
        model_configs.append({"name": mt, "model": model})

    results = run_model_comparison(
        model_configs=model_configs,
        X_train=X_train_fit, y_train=y_train_fit,
        X_test=X_test, y_test=y_test,
        mid_prices_test=mid_prices_test,
        X_val=X_val, y_val=y_val,
        horizon=cfg.labeling.horizon_steps,
    )

    summary_df = results_to_dataframe(results)

    results_dir = Path(cfg.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "model_comparison.csv", index=False)

    typer.echo("\n" + summary_df.to_string(index=False))
    typer.echo(f"\nResults saved to {results_dir}/model_comparison.csv")


@app.command()
def backtest(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run full backtest with cost analysis."""
    typer.echo("Running backtest with cost analysis...")
    # Delegate to train with cost enabled
    from .config import load_config
    cfg = load_config(config)
    if not cfg.cost.enabled:
        typer.echo("Note: cost.enabled=false in config. Set to true for realistic backtest.")
    train(config=config)


@app.command()
def run_experiments(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
    mode: str = typer.Option("full", "--mode", "-m", help="full / depth / quick"),
) -> None:
    """Run comprehensive experiment suite (paper reproduction)."""
    from .config import load_config
    from .data_loader import load_lob_data
    from .experiments import (
        format_paper_table,
        run_depth_comparison,
        run_paper_reproduction,
    )

    cfg = load_config(config)

    raw_df = load_lob_data(cfg.data.raw_dir, cfg.data.symbol)
    if raw_df.empty:
        typer.echo("No data found. Run fetch-data first.")
        raise typer.Exit(1)

    if mode == "depth":
        df = run_depth_comparison(raw_df, output_dir=cfg.output.results_dir, seed=cfg.model.seed)
    elif mode == "extended":
        from .experiments import run_extended_horizons
        df = run_extended_horizons(raw_df, output_dir=cfg.output.results_dir, seed=cfg.model.seed)
    else:
        df = run_paper_reproduction(raw_df, output_dir=cfg.output.results_dir, seed=cfg.model.seed)

    ok_df = df[df["status"] == "ok"]
    if ok_df.empty:
        typer.echo("No successful experiments.")
        return

    # Print binary table
    if "binary" in ok_df["label_scheme"].values:
        typer.echo("\n" + format_paper_table(ok_df, "binary"))
    if "ternary" in ok_df["label_scheme"].values:
        typer.echo("\n" + format_paper_table(ok_df, "ternary"))

    # Best result
    best = ok_df.loc[ok_df["f1_macro"].idxmax()]
    typer.echo(f"\nBest: {best['experiment']} -> F1={best['f1_macro']:.4f}, Acc={best['accuracy']:.4f}")


@app.command()
def collect_ws(
    symbols: str = typer.Option("BTCUSDT,ETHUSDT", "--symbols", "-s"),
    duration_hours: float = typer.Option(25.0, "--duration", "-d"),
    interval_ms: int = typer.Option(100, "--interval"),
    depth: int = typer.Option(200, "--depth"),
    output_dir: str = typer.Option("data/raw", "--output-dir"),
) -> None:
    """Collect LOB data via WebSocket (24h+ continuous)."""
    from .ws_collector import collect_ws_sync

    symbol_list = [s.strip() for s in symbols.split(",")]
    typer.echo(f"Starting WS collection: {symbol_list}, {duration_hours}h, {interval_ms}ms")
    typer.echo(f"Output: {output_dir}/")
    typer.echo("Press Ctrl+C to stop early.")

    collect_ws_sync(
        symbols=symbol_list,
        duration_hours=duration_hours,
        interval_ms=interval_ms,
        depth=depth,
        output_dir=output_dir,
    )


@app.command()
def run_imbalance(
    config: str = typer.Option("configs/trading.yaml", "--config", "-c"),
    threshold: float = typer.Option(0.10, "--threshold", "-t", help="Quantile threshold (e.g. 0.10 = top/bot 10%)"),
    horizon: int = typer.Option(100, "--horizon", "-h", help="Holding period in timesteps"),
) -> None:
    """Run imbalance direct strategy."""
    from .config import load_config
    from .data_loader import load_lob_data, filter_complete_snapshots
    from .imbalance_strategy import ImbalanceStrategyConfig, run_imbalance_strategy

    cfg = load_config(config)
    raw_df = load_lob_data(cfg.data.raw_dir, cfg.data.symbol)
    if raw_df.empty:
        typer.echo("No data found.")
        raise typer.Exit(1)

    df = filter_complete_snapshots(raw_df, depth=cfg.data.lob_depth)
    strat_cfg = ImbalanceStrategyConfig(
        long_threshold_pct=1.0 - threshold,
        short_threshold_pct=threshold,
        horizon_steps=horizon,
        maker_fee_bps=cfg.cost.maker_fee_bps,
        slippage_bps=cfg.cost.slippage_bps,
    )
    result = run_imbalance_strategy(df, strat_cfg)

    typer.echo(f"\n{'='*50}")
    typer.echo(f"Imbalance Strategy: thresh={threshold}, horizon={horizon}")
    typer.echo(f"Trades: {result.n_trades}")
    typer.echo(f"Avg gross: {result.avg_gross_bps:.3f} bps/trade")
    typer.echo(f"Avg net: {result.avg_net_bps:.3f} bps/trade")
    typer.echo(f"Total net: {result.net_pnl_bps:.1f} bps")
    typer.echo(f"Win rate: {result.win_rate:.1%}")
    typer.echo(f"Sharpe: {result.sharpe:.2f}")
    typer.echo(f"Max DD: {result.max_drawdown_bps:.1f} bps")


@app.command()
def show_leaderboard(
    sort_by: str = typer.Option("avg_net_bps", "--sort", "-s"),
    top_n: int = typer.Option(20, "--top", "-n"),
) -> None:
    """Show strategy leaderboard (trading-first)."""
    from .leaderboard import load_leaderboard, render_leaderboard

    entries = load_leaderboard()
    if not entries:
        typer.echo("No entries. Run experiments or strategies first.")
        return

    typer.echo(render_leaderboard(entries, sort_by=sort_by, top_n=top_n))


@app.command()
def sweep_imbalance(
    symbols: str = typer.Option("BTCUSDT,ETHUSDT", "--symbols", "-s"),
    data_dir: str = typer.Option("data/raw", "--data-dir"),
    depth: int = typer.Option(5, "--depth", "-d"),
) -> None:
    """Full imbalance parameter sweep + regime analysis + leaderboard update."""
    import glob

    from .data_loader import filter_complete_snapshots
    from .imbalance_strategy import ImbalanceStrategyConfig, run_imbalance_strategy, sweep_imbalance_params
    from .leaderboard import add_entry, entry_from_imbalance_result, load_leaderboard, render_leaderboard
    from .regime import add_regime_labels, generate_regime_report

    import pandas as pd

    symbol_list = [s.strip() for s in symbols.split(",")]

    for sym in symbol_list:
        # Load all data for this symbol
        files = sorted(glob.glob(f"{data_dir}/{sym}*.parquet"))
        files = [f for f in files if "synthetic" not in f]
        if not files:
            typer.echo(f"No data for {sym}")
            continue

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        df = filter_complete_snapshots(df, depth=depth)
        if len(df) < 1000:
            typer.echo(f"{sym}: only {len(df)} snapshots, skipping")
            continue

        mid = (df["bid_p1"] + df["ask_p1"]) / 2
        duration_s = (df["timestamp"].max() - df["timestamp"].min()) / 1000 if "timestamp" in df.columns else 0
        typer.echo(f"\n{'='*50}")
        typer.echo(f"{sym}: {len(df)} snaps, {duration_s/60:.1f} min")
        typer.echo(f"  Price: {mid.min():.2f} - {mid.max():.2f}")

        sweep = sweep_imbalance_params(
            df,
            thresholds=[0.05, 0.10, 0.15, 0.20],
            horizons=[50, 100, 300, 600],
            vol_percentiles=[0, 50, 70],
            maker_fees=[1.0, 0.0],
        )

        if sweep.empty:
            typer.echo("  No valid configurations")
            continue

        typer.echo(f"  {len(sweep)} valid configs")
        typer.echo(f"  Top 5:")
        for _, row in sweep.head(5).iterrows():
            typer.echo(f"    t={row['threshold']:.2f} h={row['horizon_s']:.0f}s vol={row['vol_pct']:.0f}% "
                       f"{row['fee_tier']}: n={row['n_trades']:.0f} "
                       f"gross={row['avg_gross_bps']:.3f} net={row['avg_net_bps']:.3f} "
                       f"win={row['win_rate']:.1%}")

        # Add top configs to leaderboard
        data_label = f"real_ws_{len(df)//1000}k" if duration_s > 0 else f"real_rest_{len(df)//1000}k"
        for _, row in sweep.head(5).iterrows():
            cfg = ImbalanceStrategyConfig(
                long_threshold_pct=1.0 - row["threshold"],
                short_threshold_pct=row["threshold"],
                horizon_steps=int(row["horizon_s"] * 10),
                min_vol_percentile=row["vol_pct"],
                maker_fee_bps=0.0 if row["fee_tier"] == "VIP" else 1.0,
            )
            r = run_imbalance_strategy(df, cfg)
            e = entry_from_imbalance_result(r, symbol=sym, data_source=data_label)
            add_entry(e)

        # Regime analysis
        best = sweep.iloc[0]
        best_cfg = ImbalanceStrategyConfig(
            long_threshold_pct=1.0 - best["threshold"],
            short_threshold_pct=best["threshold"],
            horizon_steps=int(best["horizon_s"] * 10),
            min_vol_percentile=best["vol_pct"],
            maker_fee_bps=0.0 if best["fee_tier"] == "VIP" else 1.0,
        )
        result = run_imbalance_strategy(df, best_cfg)
        if result.trades:
            df_regime = add_regime_labels(df)
            for trade in result.trades:
                idx = trade["index"]
                if idx < len(df_regime):
                    for col in ["vol_regime", "spread_regime", "hour_bucket"]:
                        if col in df_regime.columns:
                            trade[col] = str(df_regime.iloc[idx][col])

            report = generate_regime_report(result.trades, title=f"{sym} Regime Analysis")
            report_path = f"reports/{sym.lower()}_regime.md"
            with open(report_path, "w") as f:
                f.write(report)
            typer.echo(f"  Regime report: {report_path}")

        sweep.to_csv(f"results/{sym.lower()}_imbalance_sweep.csv", index=False)

    # Show final leaderboard
    entries = load_leaderboard()
    typer.echo(f"\n{render_leaderboard(entries, top_n=15)}")


@app.command()
def run_paper_trading(
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s"),
    duration_hours: float = typer.Option(168.0, "--duration", "-d", help="Duration in hours (default 7 days)"),
    max_spread_ticks: float = typer.Option(1.5, "--max-spread", help="Max spread in ticks for entry"),
    db_path: str = typer.Option("results/paper_trades.db", "--db"),
) -> None:
    """Run Paper Trading (forward test). No real orders placed.

    Monitors LOB via WebSocket, generates imbalance signals, simulates
    Maker fills, logs to SQLite. Run for 7+ days for GO/NO GO validation.

    GO criteria (7 days):
      - Total net PnL > 0
      - Positive-day rate > 50%
      - Max drawdown < 100 bps

    KILL criteria:
      - Total net PnL < 0 after 7 days
      - Any single-day drawdown > 100 bps
    """
    from .paper_trader import run_paper_trader

    typer.echo(f"Paper Trading: {symbol}")
    typer.echo(f"Duration: {duration_hours}h ({duration_hours/24:.1f} days)")
    typer.echo(f"Max spread: {max_spread_ticks} ticks")
    typer.echo(f"DB: {db_path}")
    typer.echo(f"Configs: A(5%_60s), B(10%_120s)")
    typer.echo(f"")
    typer.echo(f"GO criteria (after 7 days):")
    typer.echo(f"  - Net PnL > 0")
    typer.echo(f"  - Positive-day rate > 50%")
    typer.echo(f"  - Max drawdown < 100 bps")
    typer.echo(f"")
    typer.echo(f"Starting... (Ctrl+C to stop)")

    run_paper_trader(
        symbol=symbol,
        duration_hours=duration_hours,
        db_path=db_path,
        max_spread_ticks=max_spread_ticks,
    )


@app.command()
def paper_status(
    db_path: str = typer.Option("results/paper_trades.db", "--db"),
) -> None:
    """Show Paper Trading status and summary."""
    import sqlite3
    from pathlib import Path

    if not Path(db_path).exists():
        typer.echo("No paper trading database found. Run paper-trading first.")
        raise typer.Exit(1)

    conn = sqlite3.connect(db_path)

    # Trades summary per config
    rows = conn.execute("""
        SELECT config_name,
               COUNT(*) as n,
               AVG(net_bps) as avg_net,
               SUM(net_bps) as total_net,
               AVG(CASE WHEN net_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
               MIN(net_bps) as worst,
               MAX(net_bps) as best,
               AVG(gross_bps) as avg_gross
        FROM trades GROUP BY config_name
    """).fetchall()

    if not rows:
        typer.echo("No trades recorded yet.")
        conn.close()
        return

    typer.echo(f"\n{'='*60}")
    typer.echo("Paper Trading Summary")
    typer.echo(f"{'='*60}")

    for r in rows:
        name, n, avg_net, total_net, wr, worst, best, avg_gross = r
        typer.echo(f"\n  {name}:")
        typer.echo(f"    Trades: {n}")
        typer.echo(f"    Avg gross: {avg_gross:+.3f} bps")
        typer.echo(f"    Avg net:   {avg_net:+.3f} bps")
        typer.echo(f"    Total net: {total_net:+.1f} bps")
        typer.echo(f"    Win rate:  {wr:.1%}")
        typer.echo(f"    Best/Worst: {best:+.2f} / {worst:+.2f} bps")

    # Signals summary
    sig_rows = conn.execute("""
        SELECT config_name,
               COUNT(*) as total,
               SUM(accepted) as accepted,
               COUNT(*) - SUM(accepted) as rejected
        FROM signals GROUP BY config_name
    """).fetchall()

    if sig_rows:
        typer.echo(f"\n  Signals:")
        for name, total, accepted, rejected in sig_rows:
            typer.echo(f"    {name}: {total} signals, {accepted} accepted, {rejected} rejected")

    conn.close()


if __name__ == "__main__":
    app()
