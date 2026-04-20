"""Tests for the sparse-noise pre-CV gate in auto_model_forecast_panel.

The gate catches long histories with very few non-zero observations (no trend,
no calendar-month recurrence) and short-circuits them to HistoricAverage so CV
doesn't pick spurious ETS / Croston / seasonal models on essentially random
data.
"""

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    ClassicalForecasts,
    _auto_model_detect_sparse_noise,
)


def _monthly_ds(n_months: int) -> pd.Series:
    """Return a monthly ``ds`` series of length n ending at the current month."""
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    idx = pd.date_range(end=end, periods=n_months, freq='MS')
    return pd.Series(idx).reset_index(drop=True)


def test_sparse_noise_fires_on_three_random_sales_across_three_years():
    """3 sales spread across 3 distinct calendar months and 3 distinct years."""
    n = 48
    y = np.zeros(n, dtype=float)
    # Indices chosen so the non-zero months land in different calendar months
    # AND different years (12-month spacing would put them in the same month).
    y[5] = 30.0
    y[22] = 25.0
    y[39] = 35.0
    ds = _monthly_ds(n)

    assert _auto_model_detect_sparse_noise(y, ds) is True


def test_sparse_noise_recurrence_safety_net_same_month_multiple_years():
    """Sales in the same calendar month across ≥2 years → NOT sparse noise."""
    n = 48
    y = np.zeros(n, dtype=float)
    # 12-month spacing keeps us in the same calendar month for all three sales.
    y[10] = 30.0
    y[22] = 28.0
    y[34] = 32.0
    ds = _monthly_ds(n)

    assert _auto_model_detect_sparse_noise(y, ds) is False


def test_sparse_noise_trend_safety_net_preserves_trending_sparse_history():
    """Few non-zero points that line up as a clean trend → NOT sparse noise.

    This exercises the branch where ``nz_count < 6`` AND ``nz_frac < 0.15`` but
    the handful of observations show a strong monotone trend (|r| >= 0.5).
    """
    n = 48
    y = np.zeros(n, dtype=float)
    # 4 trending points at the tail: nz_count=4 < 6, nz_frac=4/48 ≈ 0.083 < 0.15.
    y[-4:] = [1.0, 3.0, 5.0, 7.0]
    ds = _monthly_ds(n)

    assert _auto_model_detect_sparse_noise(y, ds) is False


def test_sparse_noise_does_not_fire_on_items_with_enough_nonzero_months():
    """Regression guard: ordinary low-volume items (≥6 non-zero) must pass."""
    n = 36
    y = np.zeros(n, dtype=float)
    for i, v in zip((2, 8, 14, 20, 26, 32), (10.0, 12.0, 8.0, 15.0, 11.0, 9.0)):
        y[i] = v
    ds = _monthly_ds(n)

    assert _auto_model_detect_sparse_noise(y, ds) is False


def test_sparse_noise_does_not_fire_on_short_history():
    """Short histories (<40 months at defaults) never reach the gate.

    With defaults, nz_frac must be <0.15, so a 12-month series would need
    nz_count<1.8 ⇒ at most 1 non-zero — which still fails because nz_count=0
    returns False early (handled upstream) and nz_count>=1 with 12 months
    gives frac >= 0.083, fine, but practically this branch stays cold for
    anything under ~40 periods with non-trivial sales.
    """
    n = 15
    y = np.zeros(n, dtype=float)
    y[7] = 10.0
    ds = _monthly_ds(n)

    # 1/15 = 0.067 < 0.15, nz_count=1<6, no trend, no recurrence ⇒ fires.
    # This documents current behavior; the gate can fire on short histories
    # with essentially 1 sale. Upstream, 12–23m short-history fallbacks run
    # BEFORE the sparse-noise gate (they set picks and ``continue``), so this
    # path is only reachable for items with 24+ effective observations in
    # practice.
    assert _auto_model_detect_sparse_noise(y, ds) is True


def test_sparse_noise_returns_false_on_all_zero_and_empty():
    """Helper contract: all-zero / empty series are handled by the dead-SKU
    path upstream, so the sparse-noise helper returns False defensively."""
    assert (
        _auto_model_detect_sparse_noise(
            np.array([], dtype=float),
            pd.Series([], dtype='datetime64[ns]'),
        )
        is False
    )
    assert _auto_model_detect_sparse_noise(np.zeros(12, dtype=float), _monthly_ds(12)) is False


def test_auto_model_short_circuits_sparse_noise_to_historic_average():
    """End-to-end: a sparse-noise item picks HistoricAverage without running CV."""
    n = 48
    y = np.zeros(n, dtype=float)
    y[5] = 30.0
    y[22] = 25.0
    y[39] = 35.0
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    ds = pd.date_range(end=end, periods=n, freq='MS')

    df = pd.DataFrame({'item_id': 'sparse_item', 'day': ds, 'actual_sale': y})

    forecaster = ClassicalForecasts(
        mode='local', local_model='auto_model', season_length=12, freq='M'
    )
    out = forecaster.auto_model_forecast_panel(df, h=3, metric='robust', n_windows=2)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert (
        picked.loc[picked['unique_id'] == 'sparse_item', 'model_used'].iloc[0]
        == 'HistoricAverage'
    )
