"""Tests for ``_looks_event_seasonal`` (nested inside ``auto_model_forecast_panel``).

The detector is not exposed as a module-level symbol. To exercise it directly
we call ``auto_model_forecast_panel`` and check the resulting model picks —
event-seasonal items take the ``SeasonalNaive`` short-history path for n_obs
in the 13–23m range, or land in the ``'seasonal'`` CV bucket with MA family
dropped on longer histories. Either signal confirms the detector fired.

These tests specifically cover the v5 recurrence gate — the top-volume month
must appear in ≥ 2 distinct years, otherwise a sparse 3-sale series would
trip the concentration gate by accident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import ClassicalForecasts


def _make_panel(uid: str, dates: list[pd.Timestamp], ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({'item_id': uid, 'day': dates, 'actual_sale': ys})


def _pick(out: pd.DataFrame, uid: str) -> str:
    return str(
        out.loc[out['unique_id'] == uid, 'model_used']
        .drop_duplicates().iloc[0]
    )


def _dense_monthly(start_year: int, n_months: int) -> pd.DatetimeIndex:
    """Dense monthly DatetimeIndex starting at ``start_year-01-01``."""
    return pd.date_range(
        start=pd.Timestamp(year=start_year, month=1, day=1),
        periods=n_months,
        freq='MS',
    )


def test_event_seasonal_fires_on_recurring_december_peaks():
    """Regression check: strong Dec concentration across 4 years → fires.

    This is the canonical event-seasonal pattern. Dropping the detector would
    cause a regression here, so this test guards against future over-eager
    tightening of the recurrence rule.
    """
    n = 48
    ds = _dense_monthly(2020, n)
    y = np.zeros(n)
    # Dec 2020, 2021, 2022, 2023 — 4 recurring peaks.
    for yr_off in range(4):
        y[yr_off * 12 + 11] = 100.0 + yr_off  # distinct non-zero vals
    df = _make_panel('xmas', list(ds), list(y))

    f = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = f.auto_model_forecast_panel(df, h=3, metric='wape_bias', n_windows=2)
    pick = _pick(out, 'xmas')
    # Event-seasonal items keep SN (the lag IS the signal); explicitly assert
    # that the MA/level family did not override.
    assert pick == 'SeasonalNaive', f"expected SeasonalNaive for recurring Dec peaks, got {pick}"


def test_event_seasonal_does_not_fire_on_three_sparse_sales_in_different_months():
    """Three sales in three different calendar months across 3 years.

    Volume concentration is high (each month is 1/3 of total → 33%, below 45%
    for top1 but still reaches the top2=60% or top3=75% gates on a very sparse
    history). Without the recurrence check, these items were tripping the
    event-seasonal detector and getting SeasonalNaive — but SN has no
    repeating peak to exploit. The recurrence gate blocks this path.
    """
    n = 36
    ds = _dense_monthly(2023, n)
    y = np.zeros(n)
    y[4] = 10.0   # May 2023
    y[18] = 12.0  # Jul 2024
    y[31] = 11.0  # Aug 2025
    df = _make_panel('sparse', list(ds), list(y))

    f = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = f.auto_model_forecast_panel(df, h=3, metric='wape_bias', n_windows=2)
    pick = _pick(out, 'sparse')
    # Not event-seasonal → must NOT be SeasonalNaive. The intermittent-demand
    # pool (Croston/ADIDA) is the correct bucket for a sparse 3-sale series.
    assert pick != 'SeasonalNaive', (
        f"sparse 3-sale series incorrectly classified as event-seasonal (got {pick})"
    )
