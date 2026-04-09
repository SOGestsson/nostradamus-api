"""End-to-end: monthly seasonal phase must not shift when ds uses ME or mid-month stamps.

Regression for mis-anchored pd.date_range(freq='MS') that moved December demand into November forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _december_spike_panel_ms() -> pd.DataFrame:
    """Strong December seasonality; pad through current month-start so auto_model regularization
    does not append a long synthetic zero tail (which forces Naive and yhat=0).
    """
    ms = pd.date_range("2020-01-01", "2025-11-01", freq="MS")
    y = np.where(ms.month == 12, 1000.0, 1.0).astype(float)
    end_anchor = pd.Timestamp.now(tz=None).to_period("M").to_timestamp(how="start")
    last = ms.max()
    if end_anchor > last:
        extra = pd.date_range(start=last + pd.offsets.MonthBegin(1), end=end_anchor, freq="MS")
        ms = pd.DatetimeIndex(np.concatenate([ms.to_numpy(), extra.to_numpy()]))
        y = np.concatenate([y, np.ones(len(extra), dtype=float)])
    return pd.DataFrame({"item_id": "sku_x", "day": ms, "actual_sale": y})


def _panel_with_ds_style(style: str) -> pd.DataFrame:
    base = _december_spike_panel_ms()
    if style == "MS":
        return base
    if style == "ME":
        base = base.copy()
        me_raw = pd.to_datetime(base["day"]).dt.to_period("M").dt.to_timestamp(how="end")
        base["day"] = pd.to_datetime(me_raw).dt.normalize()
        return base
    if style == "mid_month":
        base = base.copy()
        base["day"] = pd.to_datetime(base["day"]) + pd.Timedelta(days=14)
        return base
    raise ValueError(style)


def _assert_forecast_ds_month_starts(out: pd.DataFrame) -> None:
    ds = pd.to_datetime(out["ds"])
    assert (ds.dt.day == 1).all(), "all forecast ds must be first of month"


def _assert_december_dominates_horizon(out: pd.DataFrame) -> None:
    """December yhat should dwarf a typical off-season month in the same horizon."""
    ds = pd.to_datetime(out["ds"])
    dec = out.loc[ds.dt.month == 12, "yhat"].astype(float)
    feb = out.loc[ds.dt.month == 2, "yhat"].astype(float)
    assert not dec.empty and not feb.empty, "h=12 horizon should include both Dec and Feb"
    assert float(dec.min()) > float(feb.max()) * 5.0, (
        f"December forecasts should exceed February max by wide margin; dec.min={dec.min()} feb.max={feb.max()}"
    )


def test_auto_model_monthly_ms_me_mid_month_produce_identical_forecasts() -> None:
    """After normalization, dirty monthly timestamps must not change selected model path vs clean MS.

    Auto-model CV may pick a flat forecaster on synthetic panels; this still guards the
    original bug (ME/mid-month shifting the entire seasonal phase vs MS).
    """
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode="local", local_model="auto_model", season_length=12, freq="M")
    outs = []
    for style in ("MS", "ME", "mid_month"):
        o = forecaster.auto_model_forecast_panel(
            _panel_with_ds_style(style),
            h=12,
            metric="robust",
            n_windows=2,
            cv_h=3,
        )
        sku = o[o["unique_id"] == "sku_x"].sort_values("ds").reset_index(drop=True)
        assert len(sku) == 12
        _assert_forecast_ds_month_starts(sku)
        outs.append(sku)

    base_ds = pd.to_datetime(outs[0]["ds"])
    for alt in outs[1:]:
        assert (pd.to_datetime(alt["ds"]) == base_ds).all()
        np.testing.assert_allclose(
            alt["yhat"].to_numpy(dtype=float),
            outs[0]["yhat"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-6,
            err_msg="yhat must match across ds conventions (same regularized history)",
        )


@pytest.mark.parametrize("ds_style", ["MS", "ME", "mid_month"])
def test_auto_ets_monthly_december_peak_preserved(ds_style: str) -> None:
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    hist = _panel_with_ds_style(ds_style)
    forecaster = ClassicalForecasts(mode="local", local_model="auto_ets", season_length=12, freq="M")
    fcst = forecaster._local_forecast_path(hist, h=12)
    assert len(fcst) == 12
    _assert_forecast_ds_month_starts(fcst)
    _assert_december_dominates_horizon(fcst)


@pytest.mark.parametrize("ds_style", ["MS", "ME"])
def test_daily_path_auto_ets_month_starts_and_december_peak(ds_style: str) -> None:
    """Public daily_path wraps _local_forecast_path; path order matches MS grid after last hist month."""
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    hist = _panel_with_ds_style(ds_style)
    forecaster = ClassicalForecasts(mode="local", local_model="auto_ets", season_length=12, freq="M")
    path, _qs = forecaster.daily_path(hist, 12)
    assert path.shape == (12,)
    last = pd.to_datetime(hist["day"]).max()
    last_ms = pd.to_datetime(last).to_period("M").to_timestamp(how="start")
    future = pd.date_range(start=last_ms + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    check = pd.DataFrame({"ds": future, "yhat": path})
    _assert_forecast_ds_month_starts(check)
    _assert_december_dominates_horizon(check)
