import tempfile

import numpy as np
import pandas as pd
import pytest

from inventory_algorithm.lightgbm_forecasts import LightGBMForecast, _HAS_LGBM


def _make_monthly_panel(n_items: int = 10, n_months: int = 36) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ds = pd.date_range("2021-01-01", periods=n_months, freq="MS")
    rows = []
    for item_id in range(1, n_items + 1):
        level = rng.uniform(5, 20)
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (np.arange(n_months) / 12.0))
        y = np.maximum(0.0, level * seasonal + rng.normal(0, 1.0, n_months))
        for d, yy in zip(ds, y):
            rows.append({"item_id": item_id, "day": d.strftime("%Y-%m-%d"), "actual_sale": float(yy)})
    return pd.DataFrame(rows)


def test_feature_spec_includes_horizon_features():
    if not _HAS_LGBM:
        pytest.skip("lightgbm import failed in this environment")

    df = _make_monthly_panel(n_items=10, n_months=36)
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
        artifact_root, spec = f._load_spec(train.model_version)
        cols = set(spec.get("feature_columns") or [])
        assert "horizon" in cols
        assert "horizon_log" in cols

