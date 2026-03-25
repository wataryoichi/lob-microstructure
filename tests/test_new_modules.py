"""Tests for P0 sprint modules: imbalance strategy, regime, leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestImbalanceStrategy:
    """Test model-free imbalance strategy."""

    def _make_sample_df(self, n=5000, depth=5):
        from src.data_collector import load_sample_data
        return load_sample_data(n_snapshots=n, depth=depth)

    def test_run_basic(self):
        from src.imbalance_strategy import ImbalanceStrategyConfig, run_imbalance_strategy
        df = self._make_sample_df()
        cfg = ImbalanceStrategyConfig(horizon_steps=50, percentile_window=500)
        result = run_imbalance_strategy(df, cfg)
        assert result.n_trades >= 0
        assert isinstance(result.trades, list)
        assert isinstance(result.avg_gross_bps, float)

    def test_trades_non_overlapping(self):
        from src.imbalance_strategy import ImbalanceStrategyConfig, run_imbalance_strategy
        df = self._make_sample_df()
        cfg = ImbalanceStrategyConfig(horizon_steps=50, percentile_window=500)
        result = run_imbalance_strategy(df, cfg)
        if result.n_trades >= 2:
            indices = [t["index"] for t in result.trades]
            for i in range(1, len(indices)):
                assert indices[i] - indices[i - 1] >= cfg.horizon_steps

    def test_sweep_returns_dataframe(self):
        from src.imbalance_strategy import sweep_imbalance_params
        df = self._make_sample_df()
        sweep_df = sweep_imbalance_params(
            df, thresholds=[0.10], horizons=[50], vol_percentiles=[0], maker_fees=[1.0]
        )
        assert isinstance(sweep_df, pd.DataFrame)

    def test_config_affects_trades(self):
        from src.imbalance_strategy import ImbalanceStrategyConfig, run_imbalance_strategy
        df = self._make_sample_df()
        # Tight threshold = fewer trades
        tight = ImbalanceStrategyConfig(
            long_threshold_pct=0.95, short_threshold_pct=0.05,
            horizon_steps=50, percentile_window=500,
        )
        loose = ImbalanceStrategyConfig(
            long_threshold_pct=0.80, short_threshold_pct=0.20,
            horizon_steps=50, percentile_window=500,
        )
        r_tight = run_imbalance_strategy(df, tight)
        r_loose = run_imbalance_strategy(df, loose)
        assert r_tight.n_trades <= r_loose.n_trades


class TestRegime:
    """Test regime analysis."""

    def test_add_regime_labels(self):
        from src.data_collector import load_sample_data
        from src.regime import add_regime_labels
        df = load_sample_data(n_snapshots=2000, depth=5)
        result = add_regime_labels(df, vol_window=200)
        assert "vol_regime" in result.columns
        assert "spread_regime" in result.columns

    def test_analyze_empty_trades(self):
        from src.regime import analyze_trades_by_regime
        result = analyze_trades_by_regime([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestLeaderboard:
    """Test leaderboard."""

    def test_entry_creation(self):
        from src.leaderboard import LeaderboardEntry
        entry = LeaderboardEntry(
            run_id="test_001",
            strategy_type="imbalance_direct",
            symbol="BTCUSDT",
            data_source="synthetic",
            timestamp="2026-01-01T00:00:00",
            n_trades=50,
            avg_gross_bps=1.5,
            avg_net_bps=0.5,
        )
        assert entry.run_id == "test_001"
        assert entry.avg_net_bps == 0.5

    def test_save_load_roundtrip(self, tmp_path):
        from src.leaderboard import LeaderboardEntry, load_leaderboard, save_leaderboard
        path = str(tmp_path / "lb.json")
        entries = [
            LeaderboardEntry(
                run_id="test_001", strategy_type="imbalance_direct",
                symbol="BTCUSDT", data_source="synthetic", timestamp="2026-01-01",
                avg_net_bps=0.5,
            ),
        ]
        save_leaderboard(entries, path)
        loaded = load_leaderboard(path)
        assert len(loaded) == 1
        assert loaded[0].run_id == "test_001"

    def test_render_not_empty(self):
        from src.leaderboard import LeaderboardEntry, render_leaderboard
        entries = [
            LeaderboardEntry(
                run_id="test_001", strategy_type="imbalance_direct",
                symbol="BTCUSDT", data_source="synthetic", timestamp="2026-01-01",
                n_trades=50, avg_net_bps=0.5,
            ),
        ]
        output = render_leaderboard(entries)
        assert "Leaderboard" in output
        assert "test_001" in output
