"""Regression: arbitrary monthly calendar days must snap to month-start before pd.date_range(freq='MS')."""

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    ClassicalForecasts,
    canonical_forecaster_freq,
    _normalize_monthly_ds_to_period_anchor,
    _regularize_panel_time_index,
)


def test_to_statsforecast_df_snaps_monthly_to_month_start():
    hist = pd.DataFrame(
        {
            "item_id": ["a"],
            "day": [pd.Timestamp("2024-12-31")],
            "actual_sale": [100.0],
        }
    )
    cf = ClassicalForecasts(mode="local", local_model="naive", season_length=12, freq="M")
    df, _ = cf._to_statsforecast_df(hist)
    assert df["ds"].iloc[0] == pd.Timestamp("2024-12-01")
    assert float(df["y"].iloc[0]) == 100.0


def test_canonical_forecaster_freq_monthly_always_ms():
    assert canonical_forecaster_freq('M') == 'MS'
    assert canonical_forecaster_freq('MS') == 'MS'
    assert canonical_forecaster_freq('me') == 'MS'


def test_normalize_maps_month_end_sales_to_month_start():
    me = pd.date_range("2024-01-31", periods=12, freq="ME")
    y = np.zeros(12)
    y[11] = 3000.0  # December
    df = pd.DataFrame({"unique_id": ["sku"] * 12, "ds": me, "y": y})
    out = _normalize_monthly_ds_to_period_anchor(df, "MS")
    dec = out[(out["ds"].dt.year == 2024) & (out["ds"].dt.month == 12)]
    assert len(dec) == 1
    assert dec["ds"].iloc[0] == pd.Timestamp("2024-12-01")
    assert float(dec["y"].iloc[0]) == 3000.0


def test_normalize_maps_mid_month_to_month_start():
    df = pd.DataFrame({"unique_id": ["sku"], "ds": [pd.Timestamp("2024-12-15")], "y": [500.0]})
    out = _normalize_monthly_ds_to_period_anchor(df, "MS")
    assert out["ds"].iloc[0] == pd.Timestamp("2024-12-01")
    assert float(out["y"].iloc[0]) == 500.0


def test_regularize_preserves_december_peak_after_month_end_input():
    me = pd.date_range("2023-01-31", periods=24, freq="ME")
    y = np.random.default_rng(0).random(24) * 10
    y[[11, 23]] = 2500.0  # Dec 2023 and Dec 2024 on ME stamps
    df = pd.DataFrame({"unique_id": ["sku"] * 24, "ds": me, "y": y})
    df = _normalize_monthly_ds_to_period_anchor(df, "MS")
    reg = _regularize_panel_time_index(df, "MS")
    dec_2024 = reg[(reg["ds"].dt.year == 2024) & (reg["ds"].dt.month == 12)]
    assert len(dec_2024) == 1
    assert dec_2024["ds"].iloc[0] == pd.Timestamp("2024-12-01")
    assert float(dec_2024["y"].iloc[0]) == 2500.0
