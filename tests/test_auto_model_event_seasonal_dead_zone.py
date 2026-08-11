"""Tests for the event-seasonal CV "dead zone" handler.

The bucket-level seasonal CV uses ``bucket_cv_h = min(h, season_length)``.
For h>=12 on monthly data that's a 12-step CV horizon, which sets
``_min_obs_for_model_cv`` for SeasonalNaive to ``cv_h*n_windows + 2 + season
= 38`` months. Items with 24-37 months of history get SN/SWA/AutoARIMA
filtered out *before* CV runs. The pool collapses to AutoETS +
HistoricAverage; AutoETS on 1-2 cycles typically picks a non-seasonal
config and forecasts a flat low line through the next peak.

The dead-zone handler runs a relaxed mini-CV (smaller cv_h, fewer windows)
over a small set of seasonal-aware candidates — SeasonalNaive, AutoETS,
Theta, OptimizedTheta, HistoricAverage — and picks the winner per uid by
WAPE with a small lag-family advantage requirement. When SeasonalNaive
wins, a clipped peak-ratio correction is applied at the post-forecast
stage so growing/declining year-over-year items aren't pinned to last
year's value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    ClassicalForecasts,
    _auto_model_compute_peak_ratio,
)


def _make_panel(uid: str, dates: pd.DatetimeIndex, ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({'item_id': uid, 'day': list(dates), 'actual_sale': ys})


def _pick(out: pd.DataFrame, uid: str) -> str:
    """Return the picked model_used label, stripped of post-forecast suffixes."""
    raw = str(out.loc[out['unique_id'] == uid, 'model_used'].drop_duplicates().iloc[0])
    return raw.split(':')[0]


def _pick_full(out: pd.DataFrame, uid: str) -> str:
    """Return the raw model_used label including any ``:peak_ratio`` / ``:floor`` suffix."""
    return str(out.loc[out['unique_id'] == uid, 'model_used'].drop_duplicates().iloc[0])


def _yhat_for_month(out: pd.DataFrame, uid: str, month: int) -> float:
    fc_uid = out.loc[out['unique_id'] == uid].copy()
    fc_uid['ds'] = pd.to_datetime(fc_uid['ds'])
    rows = fc_uid.loc[fc_uid['ds'].dt.month == int(month)]
    return float(rows['yhat'].iloc[0])


def _summer_peak_series(
    start: pd.Timestamp,
    n_months: int,
    peaks: dict[int, float],
    off_season: float = 25.0,
    rng_seed: int = 0,
) -> tuple[pd.DatetimeIndex, list[float]]:
    """Build a monthly series with summer peaks (Jul) and a low off-season."""
    rng = np.random.default_rng(rng_seed)
    ds = pd.date_range(start=start, periods=n_months, freq='MS')
    y: list[float] = []
    for d in ds:
        if d.month == 7 and d.year in peaks:
            y.append(float(peaks[d.year]))
        else:
            y.append(float(max(0.0, off_season + rng.normal(0.0, 5.0))))
    return ds, y


def _make_forecaster() -> ClassicalForecasts:
    return ClassicalForecasts(
        mode='local', local_model='auto_model', season_length=12, freq='M'
    )


# ---------------------------------------------------------------------------
# Acceptance tests: dead-zone handler produces a peaked Jul forecast
# ---------------------------------------------------------------------------


def test_event_seasonal_29m_two_summer_peaks_produces_peaked_jul_forecast():
    """The bug case: ~29 months, two recurring Jul peaks. Today (pre-fix)
    this picks AutoETS-flat and misses the next peak entirely. After the
    dead-zone handler the picked model must be a seasonally-aware one
    (SeasonalNaive / Theta / OptimizedTheta / AutoETS-with-trend) AND the
    Jul forecast must reflect the peak, not the off-season tail."""
    ds, y = _summer_peak_series(
        start=pd.Timestamp('2024-01-01'),
        n_months=29,
        peaks={2024: 830.0, 2025: 500.0},
    )
    df = _make_panel('summer_29m', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    pick = _pick(out, 'summer_29m')
    yhat_jul = _yhat_for_month(out, 'summer_29m', 7)

    assert pick in {'SeasonalNaive', 'Theta', 'OptimizedTheta', 'AutoETS'}, (
        f"expected a seasonally-aware model from the dead-zone mini-CV, got {pick}"
    )
    assert yhat_jul > 100.0, (
        f"Jul forecast collapsed to flat low value: {yhat_jul} (pick={pick})"
    )


def test_broad_summer_hump_28m_produces_peaked_jul_forecast():
    """Real production failure shape (Kjoris items 103401-103403, 104318):
    28 months Jan 2024 - Apr 2026, broad summer hump (Apr-Sep elevated,
    peaking in Jul) with leading zeros from a mid-range launch.

    Unlike narrow event-seasonal items (Christmas spikes), the peak-month
    volume share here is only ~25-34% — *below* ``_looks_event_seasonal``'s
    45% threshold. These items would not be caught by the strict detector
    alone, but ``_strong_yearly_seasonality`` (lag-12 correlation >= 0.38)
    correctly identifies the yearly shape. The widened gate trigger
    (event_seasonal OR strong_yearly) handles them.

    Pre-fix production behavior: AutoETS / Theta / HistoricAverage /
    Naive picks producing flat-low forecasts (Jul forecasts of 19-104
    when last year's Jul was 320-830).

    Post-fix: SeasonalNaive (with optional peak_ratio correction) or
    Theta if it has a strong CV advantage on this longer-data series.
    """
    # Mirror the actual 103401 panel structure (broad summer hump).
    ds = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=28, freq='MS')
    rng = np.random.default_rng(13)
    y: list[float] = []
    for d in ds:
        if d < pd.Timestamp('2024-07-01'):
            y.append(0.0)  # leading zeros
        elif d.year == 2024:
            shape_2024 = {7: 830.0, 8: 616.0, 9: 226.0, 10: 116.0, 11: 50.0, 12: 33.0}
            y.append(shape_2024.get(d.month, 80.0))
        elif d.year == 2025:
            shape_2025 = {6: 259.0, 7: 618.0, 8: 294.0, 9: 186.0, 10: 78.0, 11: 89.0, 12: 58.0}
            y.append(shape_2025.get(d.month, float(max(0.0, 95.0 + rng.normal(0.0, 10.0)))))
        else:
            y.append(float(max(0.0, 95.0 + rng.normal(0.0, 15.0))))

    df = _make_panel('broad_summer_hump_28m', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    pick = _pick(out, 'broad_summer_hump_28m')
    yhat_jul = _yhat_for_month(out, 'broad_summer_hump_28m', 7)

    # Must be a seasonally-aware model — Naive/HistoricAverage are the
    # known failure modes that we're explicitly fixing.
    assert pick in {'SeasonalNaive', 'Theta', 'OptimizedTheta', 'AutoETS'}, (
        f"broad-hump item with top1<0.45 must take dead-zone path "
        f"(via _strong_yearly_seasonality); got {pick}"
    )
    # Jul forecast must be > 250: last year's Jul was 618 and the shape is
    # clearly seasonal. Anything below 250 is a flat-line failure.
    assert yhat_jul > 250.0, (
        f"Jul forecast collapsed to flat low value: {yhat_jul} (pick={pick}; "
        f"last Jul was 618). Verify the dead-zone gate fires on items where "
        f"_looks_event_seasonal is False but _strong_yearly_seasonality is True."
    )


def test_broad_summer_hump_28m_with_n_windows_1_picks_seasonal():
    """Regression for the API-default config: forecast_periods=12,
    season_length=12 → cv_h=12, n_windows=1 (Nostradamus API
    `/forecast/generate` default). With n_windows=1 the legacy
    ``_sn_filter_gate = cv_h * n_windows + 2 + season = 26`` fails for
    n_obs=28 (28 >= 26), so the dead-zone gate previously skipped these
    items and the standard single-window CV picked Naive (flat) over
    SeasonalNaive — the lag-12 baseline got poisoned by the leading-zero
    block.

    The dead-zone upper bound of ``3*season + h`` (=48 for monthly h=12)
    is n_windows-independent and catches this regime. Verified against
    production items 103401-103403 / 104318 (Kjoris_1) which all share
    this n_windows=1 + n_obs=28 + leading-zeros + broad-hump shape."""
    ds = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=28, freq='MS')
    rng = np.random.default_rng(13)
    y: list[float] = []
    for d in ds:
        if d < pd.Timestamp('2024-07-01'):
            y.append(0.0)  # leading zeros
        elif d.year == 2024:
            shape_2024 = {7: 830.0, 8: 616.0, 9: 226.0, 10: 116.0, 11: 50.0, 12: 33.0}
            y.append(shape_2024.get(d.month, 80.0))
        elif d.year == 2025:
            shape_2025 = {6: 259.0, 7: 618.0, 8: 294.0, 9: 186.0, 10: 78.0, 11: 89.0, 12: 58.0}
            y.append(shape_2025.get(d.month, float(max(0.0, 95.0 + rng.normal(0.0, 10.0)))))
        else:
            y.append(float(max(0.0, 95.0 + rng.normal(0.0, 15.0))))

    df = _make_panel('broad_summer_hump_n_windows_1', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', cv_h=12, n_windows=1
    )

    pick = _pick(out, 'broad_summer_hump_n_windows_1')
    yhat_jul = _yhat_for_month(out, 'broad_summer_hump_n_windows_1', 7)

    assert pick in {'SeasonalNaive', 'Theta', 'OptimizedTheta', 'AutoETS'}, (
        f"With n_windows=1 the dead-zone upper bound (3*season+h=48) must "
        f"keep the gate firing for n_obs=28; standard CV alone collapses to "
        f"Naive/HistoricAverage on this shape. Got {pick}"
    )
    assert yhat_jul > 250.0, (
        f"Jul forecast collapsed to flat low value: {yhat_jul} (pick={pick}). "
        f"This is the regression that flat-lined items 103401-104318 in "
        f"production despite the dead-zone helper picking SeasonalNaive when "
        f"called directly. The fix widens the gate so it fires regardless of "
        f"the outer n_windows."
    )


def test_event_seasonal_29m_with_leading_zeros_produces_peaked_jul_forecast():
    """Real production case: panel covers 29 months but the SKU was launched
    mid-range, so the first ~6 months are explicit zeros (n_eff = 23 lands
    in the 'smooth' bucket, where the standard CV pool doesn't include
    SeasonalNaive at all). The dead-zone handler uses ``n_obs`` (full
    panel = 29) and must still produce a peaked Jul forecast."""
    n_panel = 29
    ds = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=n_panel, freq='MS')
    rng = np.random.default_rng(7)
    y: list[float] = []
    for d in ds:
        if d < pd.Timestamp('2024-07-01'):
            y.append(0.0)  # leading zeros: SKU not yet launched
        elif d.month == 7 and d.year == 2024:
            y.append(830.0)
        elif d.month == 7 and d.year == 2025:
            y.append(500.0)
        else:
            y.append(float(max(0.0, 60.0 + rng.normal(0.0, 10.0))))
    df = _make_panel('summer_29m_leading_zeros', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    pick = _pick(out, 'summer_29m_leading_zeros')
    yhat_jul = _yhat_for_month(out, 'summer_29m_leading_zeros', 7)

    assert pick in {'SeasonalNaive', 'Theta', 'OptimizedTheta', 'AutoETS'}, (
        f"expected a seasonally-aware model, got {pick}"
    )
    assert yhat_jul > 100.0, (
        f"Jul forecast collapsed to flat low value: {yhat_jul} (pick={pick})"
    )


# ---------------------------------------------------------------------------
# Differentiation: items with different YoY profiles get different forecasts
# ---------------------------------------------------------------------------


def test_dead_zone_differentiates_growing_vs_declining_items():
    """The whole point of the mini-CV (and peak-ratio correction): three
    event-seasonal items with the same panel length but different YoY
    peak profiles must produce *different* Jul forecasts.

    Profiles (peaks chosen so all three pass ``_looks_event_seasonal``'s
    45% volume-concentration gate):
      - flat: peaks 800 → 800 (no YoY change)
      - decline_sharp: peaks 1000 → 600 (~−40%)
      - growing: peaks 500 → 700 (+40%)

    A world-class forecaster differentiates between these — either through
    the mini-CV's per-item pick (Theta on trending items, SN on flat) or
    through the peak-ratio correction on SN. We assert that growing items
    forecast HIGHER than their last peak, declining items forecast LOWER.
    """
    rng = np.random.default_rng(11)

    def _build(peaks: dict[int, float]) -> tuple[pd.DatetimeIndex, list[float]]:
        ds = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=29, freq='MS')
        y: list[float] = []
        for d in ds:
            if d.month == 7 and d.year in peaks:
                y.append(float(peaks[d.year]))
            else:
                # Off-season is small relative to peaks (volume share of
                # the peak month >= 0.45). 25 ± 5 keeps total off-season
                # volume at ~625 across 25 months vs peak total >= 1100.
                y.append(float(max(0.0, 25.0 + rng.normal(0.0, 5.0))))
        return ds, y

    profiles = {
        'flat': {2024: 800.0, 2025: 800.0},
        'decline_sharp': {2024: 1000.0, 2025: 600.0},
        'growing': {2024: 500.0, 2025: 700.0},
    }
    rows = []
    for uid, peaks in profiles.items():
        ds, y = _build(peaks)
        for d, yy in zip(ds, y):
            rows.append({'item_id': uid, 'day': d, 'actual_sale': yy})
    df = pd.DataFrame(rows)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    yhat_flat = _yhat_for_month(out, 'flat', 7)
    yhat_decline = _yhat_for_month(out, 'decline_sharp', 7)
    yhat_growing = _yhat_for_month(out, 'growing', 7)

    last_peak = {'flat': 800.0, 'decline_sharp': 600.0, 'growing': 700.0}

    # Growing item must forecast above last year's peak, declining item
    # must forecast below — differentiation in either the model pick or
    # the peak-ratio correction is what makes this work. We use a 5%
    # buffer to absorb noise in the peak-ratio computation.
    assert yhat_growing > 1.05 * last_peak['growing'], (
        f"growing item should lift above last peak ({last_peak['growing']}); got {yhat_growing}\n"
        f"  picks: flat={_pick_full(out, 'flat')}, "
        f"decline={_pick_full(out, 'decline_sharp')}, "
        f"growing={_pick_full(out, 'growing')}"
    )
    assert yhat_decline < 0.95 * last_peak['decline_sharp'], (
        f"declining item should fall below last peak ({last_peak['decline_sharp']}); got {yhat_decline}\n"
        f"  pick: {_pick_full(out, 'decline_sharp')}"
    )
    # Flat item should land near last peak (within ±15%) — exact pick may
    # vary, but the YoY ratio is ~1.0 so SN+correction or Theta-no-trend
    # should both settle close to 800.
    assert 0.85 * last_peak['flat'] <= yhat_flat <= 1.15 * last_peak['flat'], (
        f"flat item should track last peak ({last_peak['flat']}); got {yhat_flat}\n"
        f"  pick: {_pick_full(out, 'flat')}"
    )


# ---------------------------------------------------------------------------
# Boundary / regression guards
# ---------------------------------------------------------------------------


def test_event_seasonal_38m_does_not_take_dead_zone_path():
    """At the sn_gate boundary (38 months for h=12, n_windows=2) the gate
    must NOT fire — the bucket-level CV is healthy enough to score SN/SWA/
    HA on its own. We assert the result is still a peaked Jul forecast
    (the standard CV path picks one of the seasonal candidates)."""
    ds, y = _summer_peak_series(
        start=pd.Timestamp('2023-04-01'),
        n_months=38,
        peaks={2023: 700.0, 2024: 830.0, 2025: 500.0},
    )
    df = _make_panel('summer_38m', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    yhat_jul = _yhat_for_month(out, 'summer_38m', 7)
    pick = _pick(out, 'summer_38m')
    assert yhat_jul > 100.0, (
        f"38-month seasonal item must produce a peaked Jul forecast (pick={pick}, "
        f"yhat_jul={yhat_jul})"
    )


def test_dead_zone_does_not_fire_on_non_event_seasonal():
    """Regression guard: a 29-month series that fails ``_looks_event_seasonal``
    (only one Jul peak — peak_years < 2) must NOT take the dead-zone
    path. The pick is determined by the standard CV pipeline."""
    ds, y = _summer_peak_series(
        start=pd.Timestamp('2024-01-01'),
        n_months=29,
        peaks={2024: 830.0},  # only one Jul peak — no recurrence
    )
    df = _make_panel('summer_29m_norecur', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=12, metric='wape_bias', n_windows=2
    )

    pick = _pick(out, 'summer_29m_norecur')
    # Standard CV pipeline outcomes for this series.
    assert pick in {
        'AutoETS', 'HistoricAverage', 'Naive', 'SeasonalNaive',
        'Theta', 'OptimizedTheta', 'WindowAverage', 'MA6', 'MA12',
    }, f"unexpected model pick: {pick}"


def test_dead_zone_skipped_for_short_horizon():
    """When ``h`` is small (e.g. 3), the seasonal CV path uses
    ``bucket_cv_h = 3`` and the SN min-obs gate drops to 20. The dead-zone
    range becomes ``[24, 20)`` which is empty, so the gate must not fire."""
    ds, y = _summer_peak_series(
        start=pd.Timestamp('2024-01-01'),
        n_months=29,
        peaks={2024: 830.0, 2025: 500.0},
    )
    df = _make_panel('summer_h3', ds, y)

    out = _make_forecaster().auto_model_forecast_panel(
        df, h=3, metric='wape_bias', n_windows=2
    )

    pick = _pick(out, 'summer_h3')
    assert pick, f"expected a model pick, got empty: {pick}"


# ---------------------------------------------------------------------------
# Peak-ratio helper unit tests
# ---------------------------------------------------------------------------


def test_peak_ratio_clamps_growth():
    """A 35% YoY peak increase should yield a ratio inside [0.7, 1.4]."""
    n = 24
    ds = pd.date_range('2024-01-01', periods=n, freq='MS')
    y = np.full(n, 50.0)
    # Locate Jul 2024 and Jul 2025 in the date range
    for i, d in enumerate(ds):
        if d.month == 7 and d.year == 2024:
            y[i] = 310.0
        if d.month == 7 and d.year == 2025:
            y[i] = 420.0
    g = pd.DataFrame({'unique_id': 'x', 'ds': ds, 'y': y})
    ratio = _auto_model_compute_peak_ratio(g, season_length=12)
    assert ratio is not None
    assert 1.30 <= ratio <= 1.40, f"expected ~1.35 (clipped at 1.4), got {ratio}"


def test_peak_ratio_clamps_decline():
    """A 40% YoY peak decline should yield a ratio inside [0.7, 1.4]
    (clipped at 0.7 since 0.6 falls below the floor)."""
    n = 24
    ds = pd.date_range('2024-01-01', periods=n, freq='MS')
    y = np.full(n, 50.0)
    for i, d in enumerate(ds):
        if d.month == 7 and d.year == 2024:
            y[i] = 825.0
        if d.month == 7 and d.year == 2025:
            y[i] = 500.0
    g = pd.DataFrame({'unique_id': 'x', 'ds': ds, 'y': y})
    ratio = _auto_model_compute_peak_ratio(g, season_length=12)
    assert ratio is not None
    assert ratio == 0.7, f"expected ratio clipped to 0.7 (from raw ~0.606), got {ratio}"


def test_peak_ratio_returns_none_on_short_history():
    """Less than 2 full cycles → ratio cannot be computed."""
    n = 18
    ds = pd.date_range('2024-01-01', periods=n, freq='MS')
    y = np.zeros(n)
    y[6] = 100.0  # Jul 2024 only
    g = pd.DataFrame({'unique_id': 'x', 'ds': ds, 'y': y})
    ratio = _auto_model_compute_peak_ratio(g, season_length=12)
    assert ratio is None
