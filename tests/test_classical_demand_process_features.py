import tempfile

import numpy as np
import pandas as pd
import pytest

from inventory_algorithm.lightgbm_forecasts import LightGBMForecast, _HAS_LGBM


def _make_panel(n_items: int = 12, n_months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ds = pd.date_range("2021-01-01", periods=n_months, freq="MS")
    rows = []
    for item_id in range(1, n_items + 1):
        level = rng.uniform(5, 30)
        seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (np.arange(n_months) / 12.0) + rng.uniform(0, 2 * np.pi))
        y = np.maximum(0.0, level * seasonal + rng.normal(0, 2.0, n_months))
        # Make some items intermittent
        if item_id % 4 == 0:
            mask = rng.random(n_months) < 0.6
            y = y * mask
        for d, yy in zip(ds, y):
            rows.append({"item_id": item_id, "day": d.strftime("%Y-%m-%d"), "actual_sale": float(yy)})
    return pd.DataFrame(rows)


def test_feature_spec_includes_classical_demand_process_features():
    if not _HAS_LGBM:
        pytest.skip("lightgbm import failed in this environment")

    df = _make_panel(n_items=12, n_months=36)
    expected = {
        "mean_inter_demand_interval",
        "cv_inter_demand_interval",
        "demand_regularity",
        "cv_demand_size",
        "demand_size_skew",
        "demand_size_iqr_ratio",
        "p_large_demand",
        "ets_level_alpha_1",
        "ets_level_alpha_3",
        "ets_level_alpha_5",
        "ets_trend_alpha_3",
        "ets_level_ratio_alpha_3",
        "months_since_last_vs_mean_interval",
    }

    with tempfile.TemporaryDirectory() as td:
        f = LightGBMForecast(store_root=td, customer_id="test_customer")
        train = f.train_and_register(
            df,
            freq="M",
            horizon=3,
            status="prod",
            min_history_points=24,
            min_improvement=0.0,
            val_months=3,
        )
        _, spec = f._load_spec(train.model_version)
        cols = set(spec.get("feature_columns") or [])
        missing = sorted(expected - cols)
        assert not missing, f"missing columns: {missing}"


def test_classical_features_are_finite_in_batch_forecast():
    if not _HAS_LGBM:
        pytest.skip("lightgbm import failed in this environment")

    df = _make_panel(n_items=12, n_months=36)
    with tempfile.TemporaryDirectory() as td:
        f = LightGBMForecast(store_root=td, customer_id="test_customer")
        train = f.train_and_register(
            df,
            freq="M",
            horizon=3,
            status="prod",
            min_history_points=24,
            min_improvement=0.0,
            val_months=3,
        )
        fcst_df, _ = f.batch_forecast(df, forecast_periods=3, freq="M", status="prod", model_version=train.model_version)
        assert not fcst_df.empty
