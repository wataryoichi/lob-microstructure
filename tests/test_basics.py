"""Basic tests for LOB Microstructure system."""

from __future__ import annotations

import numpy as np
import pytest


class TestFeatures:
    """Test LOB feature computations."""

    def test_mid_price(self):
        from src.features import compute_mid_price
        bid = np.array([99.0, 98.0, 100.0])
        ask = np.array([101.0, 102.0, 100.0])
        mid = compute_mid_price(bid, ask)
        np.testing.assert_allclose(mid, [100.0, 100.0, 100.0])

    def test_level1_imbalance(self):
        from src.features import compute_level1_imbalance
        bid_q = np.array([10.0, 5.0, 7.5])
        ask_q = np.array([10.0, 15.0, 2.5])
        imb = compute_level1_imbalance(bid_q, ask_q)
        np.testing.assert_allclose(imb, [0.0, -0.5, 0.5])

    def test_aggregate_imbalance(self):
        from src.features import compute_aggregate_imbalance
        bid_v = np.array([[10.0, 5.0], [10.0, 10.0]])
        ask_v = np.array([[5.0, 10.0], [10.0, 10.0]])
        imb = compute_aggregate_imbalance(bid_v, ask_v, depth=2)
        np.testing.assert_allclose(imb, [0.0, 0.0])

    def test_spread(self):
        from src.features import compute_spread
        bid = np.array([99.0, 98.0])
        ask = np.array([101.0, 102.0])
        spread = compute_spread(bid, ask)
        np.testing.assert_allclose(spread, [2.0, 4.0])


class TestFilters:
    """Test signal filters."""

    def test_savitzky_golay_smooths(self):
        from src.filters import apply_savitzky_golay
        rng = np.random.default_rng(42)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        noisy = signal + rng.normal(0, 0.3, 100)
        smoothed = apply_savitzky_golay(noisy, window_size=21, polyorder=3)
        assert np.std(smoothed - signal) < np.std(noisy - signal)

    def test_kalman_filter_basic(self):
        from src.filters import apply_kalman_filter
        signal = np.array([1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0])
        filtered = apply_kalman_filter(signal, q=0.01, r=0.1)
        assert filtered.shape == signal.shape
        assert not np.any(np.isnan(filtered))

    def test_raw_filter_identity(self):
        from src.filters import apply_filter
        data = np.array([1.0, 2.0, 3.0])
        result = apply_filter(data, filter_type="raw")
        np.testing.assert_array_equal(result, data)


class TestLabeling:
    """Test labeling schemes."""

    def test_binary_labels(self):
        from src.labeling import label_binary
        prices = np.array([100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0])
        labels = label_binary(prices, horizon=1)
        assert labels[0] == 1  # 100 -> 101: up
        assert labels[3] == 0  # 101 -> 100: down
        assert labels[-1] == -1  # invalid (no future)

    def test_ternary_labels_have_three_classes(self):
        from src.labeling import label_ternary
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.01, 1000))
        labels = label_ternary(prices, horizon=5)
        valid = labels[labels >= 0]
        assert len(set(valid)) == 3

    def test_class_weights(self):
        from src.labeling import compute_class_weights
        labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        weights = compute_class_weights(labels)
        assert 0 in weights and 1 in weights
        assert weights[0] > weights[1]


class TestSyntheticData:
    """Test synthetic data generation."""

    def test_sample_data_shape(self):
        from src.data_collector import load_sample_data
        df = load_sample_data(n_snapshots=100, depth=5)
        assert len(df) == 100
        assert "timestamp" in df.columns
        assert "bid_p1" in df.columns
        assert "ask_q5" in df.columns

    def test_sample_data_valid(self):
        from src.data_collector import load_sample_data
        df = load_sample_data(n_snapshots=100, depth=5)
        assert (df["bid_p1"] < df["ask_p1"]).all()
        assert (df["bid_q1"] > 0).all()


class TestConfig:
    """Test configuration loading."""

    def test_load_base_config(self):
        from src.config import load_config
        cfg = load_config("configs/base.yaml")
        assert cfg.data.symbol == "BTCUSDT"
        assert cfg.preprocessing.filter_type == "savitzky_golay"
        assert cfg.labeling.scheme == "binary"

    def test_load_paper_config(self):
        from src.config import load_config
        cfg = load_config("configs/paper_reproduction.yaml")
        assert cfg.data.lob_depth == 40
        assert cfg.cost.enabled is False


class TestCostModel:
    """Test cost model."""

    def test_round_trip_cost(self):
        from src.cost_model import compute_round_trip_cost
        cost = compute_round_trip_cost(maker_fee_bps=1.0, slippage_bps=1.0, is_maker=True)
        assert cost == 4.0

    def test_breakeven_accuracy(self):
        from src.cost_model import compute_breakeven_accuracy
        acc = compute_breakeven_accuracy(avg_move_bps=5.0, cost_bps=4.0)
        assert abs(acc - 0.9) < 1e-10
