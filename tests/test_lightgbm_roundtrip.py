import tempfile

import numpy as np
import pandas as pd
import pytest

from inventory_algorithm.lightgbm_forecasts import LightGBMForecast, _HAS_LGBM


def _make_monthly_panel(n_items: int = 30, n_months: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ds = pd.date_range('2021-01-01', periods=n_months, freq='MS')

    # global support series (Option A)
    total_index = (100.0 + np.linspace(0, 10, n_months) + rng.normal(0, 1.0, n_months)).astype(float)

    rows = []
    for item_id in range(1, n_items + 1):
        level = rng.uniform(10, 50)
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (np.arange(n_months) / 12.0) + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0, 2.0, n_months)
        y = np.maximum(0.0, level * seasonal + noise)

        for d, yy, ti in zip(ds, y, total_index):
            rows.append(
                {
                    'item_id': item_id,
                    'day': d.strftime('%Y-%m-%d'),
                    'actual_sale': float(yy),
                    'total_index': float(ti),
                }
            )

    return pd.DataFrame(rows)


def test_lightgbm_train_and_batch_forecast_roundtrip():
    if not _HAS_LGBM:
        pytest.skip("lightgbm import failed in this environment (often missing libomp on macOS)")

    df = _make_monthly_panel(n_items=30, n_months=48)

    with tempfile.TemporaryDirectory() as td:
        f = LightGBMForecast(store_root=td, customer_id='test_customer')

        train = f.train_and_register(
            df,
            freq='M',
            horizon=3,
            status='prod',
            min_history_points=24,
            min_improvement=0.0,
            val_months=3,
        )

        # Verify registry sees prod version
        assert f.store.get_active_model_version(status='prod') == train.model_version

        fcst_df, meta = f.batch_forecast(df, forecast_periods=3, freq='M', status='prod')
        assert meta['strategy'] == 'direct'
        assert meta['freq'] == 'MS'
        assert meta['model_version'] == train.model_version

        # We should have forecasts for most items and for each horizon step
        assert not fcst_df.empty
        # batch_forecast returns point forecast + upper quantiles (upper_70, upper_90, upper_95)
        expected_cols = {'item_id', 'day', 'yhat', 'upper_70', 'upper_90', 'upper_95'}
        assert set(fcst_df.columns) == expected_cols

        # Basic sanity: non-negative
        assert (fcst_df['yhat'] >= 0).all()

        # At least one forecast per item in this synthetic dataset
        assert fcst_df['item_id'].nunique() >= 20
