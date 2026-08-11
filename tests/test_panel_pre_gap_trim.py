"""Tests for ``_trim_panel_pre_gap``: detect and trim multi-month data gaps.

Real production catalogs frequently contain calendar gaps in their monthly
series (data not tracked, SKU paused, import quirks). After
``_regularize_panel_time_index`` fills those gaps with zeros, the panel has
a regular grid but with multi-month zero blocks that:

  * Break SeasonalNaive's lag-12 alignment (lag lookups straddle the gap).
  * Drag down HistoricAverage by averaging in zero months.
  * Confuse Theta / AutoETS into degenerate (often flat) configurations.

Production observation (Kjoris_1, 706 items): 451 (64%) had calendar gaps,
321 (46%) had 12+ month gaps. Item 106501 had all of 2023 missing. Without
trimming, the bucket-level CV scored SeasonalNaive at WAPE 0.86 vs 0.18
when run on the post-gap block alone — a complete misranking.

This test file covers both that the gap-trim fires when expected AND that
it leaves leading zeros untouched (those are launch phase, not a gap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    ClassicalForecasts,
    _trim_panel_pre_gap,
)


def _panel(uid: str, dates: pd.DatetimeIndex, ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({'unique_id': uid, 'ds': list(dates), 'y': ys})


def _make_panel_input(uid: str, dates: pd.DatetimeIndex, ys: list[float]) -> pd.DataFrame:
    """Input shape that ``auto_model_forecast_panel`` accepts."""
    return pd.DataFrame({'item_id': uid, 'day': list(dates), 'actual_sale': ys})


# ---------------------------------------------------------------------------
# Unit tests for _trim_panel_pre_gap
# ---------------------------------------------------------------------------


def test_trim_no_gap_passes_panel_through_unchanged():
    """Series with no zero runs of ``>= gap_threshold`` is left alone."""
    n = 36
    ds = pd.date_range('2023-01-01', periods=n, freq='MS')
    rng = np.random.default_rng(1)
    y = list(50.0 + rng.normal(0.0, 5.0, n))
    df = _panel('clean', ds, y)
    out = _trim_panel_pre_gap(df, freq='MS')
    assert len(out) == n
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_trim_leading_zeros_are_not_treated_as_gap():
    """6 months of leading zeros (launch phase) must NOT trigger the trim.

    Critical contract: leading zeros represent "SKU not yet launched", and
    are handled by the downstream ``y_eff = y_full[first_pos:]`` trim. The
    gap-trim only fires on inner zero runs flanked by real data on both
    sides. Without this contract, items 103401-104318 would have their
    panels reduced below the 24-month dead-zone gate threshold.
    """
    n = 28
    ds = pd.date_range('2024-01-01', periods=n, freq='MS')
    y = [0.0] * 6 + [100.0 + i for i in range(n - 6)]  # 6 leading zeros, then real
    df = _panel('leading', ds, y)
    out = _trim_panel_pre_gap(df, freq='MS')
    assert len(out) == n, f"leading zeros must not trigger trim; got {len(out)} rows"


def test_trim_inner_12month_gap_drops_pre_gap_data():
    """A 12-month inner zero run must trim everything before it."""
    # Year 1 (12 months real) + Year 2 (12 months zero gap) + Year 3 (12 months real)
    ds = pd.date_range('2023-01-01', periods=36, freq='MS')
    y = (
        [100.0 + i for i in range(12)]  # Year 1: real
        + [0.0] * 12                    # Year 2: gap
        + [200.0 + i for i in range(12)]  # Year 3: real
    )
    df = _panel('gap_y2', ds, y)
    out = _trim_panel_pre_gap(df, freq='MS')
    # Trim should drop Year 1 + Year 2 (cutoff = first real obs after gap = index 24).
    assert len(out) == 12
    assert out['y'].iloc[0] == 200.0
    assert out['ds'].iloc[0] == pd.Timestamp('2025-01-01')


def test_trim_inner_11month_gap_does_not_fire():
    """An 11-month inner zero run is *just below* the threshold and is kept."""
    ds = pd.date_range('2023-01-01', periods=35, freq='MS')
    y = (
        [100.0 + i for i in range(12)]
        + [0.0] * 11
        + [200.0 + i for i in range(12)]
    )
    df = _panel('gap_short', ds, y)
    out = _trim_panel_pre_gap(df, freq='MS')
    assert len(out) == 35


def test_trim_picks_most_recent_gap_when_multiple_exist():
    """If two gaps appear, only data after the most recent gap is kept."""
    ds = pd.date_range('2022-01-01', periods=48, freq='MS')
    y = (
        [100.0 + i for i in range(6)]  # 6 months real (2022 H1)
        + [0.0] * 12                   # 12-month gap (2022 H2 + 2023 H1)
        + [150.0 + i for i in range(6)]  # 6 months real (2023 H2)
        + [0.0] * 12                   # second 12-month gap (2024 H1 + 2024 H2)
        + [200.0 + i for i in range(12)]  # 12 months real (2025)
    )
    df = _panel('two_gaps', ds, y)
    out = _trim_panel_pre_gap(df, freq='MS')
    # Should keep only after the most recent gap = last 12 months.
    assert len(out) == 12
    assert out['y'].iloc[0] == 200.0


def test_trim_per_uid_isolation():
    """Trimming one series doesn't affect another uid in the same panel."""
    ds = pd.date_range('2023-01-01', periods=36, freq='MS')
    y_clean = [100.0 + i for i in range(36)]
    y_gap = (
        [100.0 + i for i in range(12)] + [0.0] * 12 + [200.0 + i for i in range(12)]
    )
    df = pd.concat([
        _panel('clean', ds, y_clean),
        _panel('gap', ds, y_gap),
    ], ignore_index=True)
    out = _trim_panel_pre_gap(df, freq='MS')
    assert len(out[out['unique_id'] == 'clean']) == 36
    assert len(out[out['unique_id'] == 'gap']) == 12


def test_trim_skips_non_monthly_frequencies():
    """Daily / weekly panels are returned unchanged (gap detection is monthly)."""
    n = 100
    ds = pd.date_range('2024-01-01', periods=n, freq='D')
    y = [10.0 + (i % 7) for i in range(n)]
    df = _panel('daily', ds, y)
    out = _trim_panel_pre_gap(df, freq='D')
    assert len(out) == n


def test_trim_all_zero_series_passes_through():
    """A series with no positive values has nothing to trim against."""
    ds = pd.date_range('2024-01-01', periods=24, freq='MS')
    df = _panel('dead', ds, [0.0] * 24)
    out = _trim_panel_pre_gap(df, freq='MS')
    assert len(out) == 24


# ---------------------------------------------------------------------------
# Integration test: end-to-end forecast with a year-gap item
# ---------------------------------------------------------------------------


def test_year_gap_summer_seasonal_item_recovers_seasonal_pick():
    """Mirrors production item Kjoris_1/106501: ~12 months in 2022 + entire
    2023 missing + 12 months in 2024 + 12 months in 2025 + partial 2026.
    Total panel length 41 raw rows, 53 after regularization.

    Pre-fix: bucket-level CV scored SN poorly because lag-12 lookups
    straddled the 2023 zero block. HistoricAverage won by default,
    producing a flat Jul forecast around the 53-month mean.

    Post-fix: ``_trim_panel_pre_gap`` drops 2022 (pre-gap) and 2023 (gap),
    leaving 29 months (Jan 2024 – May 2026). Falls into the dead-zone
    gate range [24, 38), where the mini-CV picks a seasonally-aware model
    (typically SeasonalNaive) and forecasts a peaked Jul.
    """
    rng = np.random.default_rng(42)

    def _seasonal_value(d: pd.Timestamp, peak_july: float) -> float:
        # Broad summer hump: Apr-Sep elevated, peak Jul, off-season ~10% of peak.
        m = d.month
        if m == 7:
            return float(peak_july)
        if m == 6 or m == 8:
            return float(peak_july * 0.6)
        if m in (5, 9):
            return float(peak_july * 0.35)
        if m in (4, 10):
            return float(peak_july * 0.20)
        return float(max(0.0, peak_july * 0.08 + rng.normal(0.0, peak_july * 0.02)))

    rows = []
    # 2022 (full year, real)
    for d in pd.date_range('2022-01-01', periods=12, freq='MS'):
        rows.append({'item_id': 'gap_item', 'day': d, 'actual_sale': _seasonal_value(d, 3636.0)})
    # 2023: SKIPPED entirely (no rows -> _regularize_panel_time_index will fill with 0)
    # 2024 (full year, real)
    for d in pd.date_range('2024-01-01', periods=12, freq='MS'):
        rows.append({'item_id': 'gap_item', 'day': d, 'actual_sale': _seasonal_value(d, 3795.0)})
    # 2025 (full year, real)
    for d in pd.date_range('2025-01-01', periods=12, freq='MS'):
        rows.append({'item_id': 'gap_item', 'day': d, 'actual_sale': _seasonal_value(d, 3920.0)})
    # 2026 partial (5 months)
    for d in pd.date_range('2026-01-01', periods=5, freq='MS'):
        rows.append({'item_id': 'gap_item', 'day': d, 'actual_sale': _seasonal_value(d, 3920.0)})
    df = pd.DataFrame(rows)

    f = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = f.auto_model_forecast_panel(df, h=12, metric='wape_bias', n_windows=2)
    out['ds'] = pd.to_datetime(out['ds'])

    pick = str(out['model_used'].iloc[0]).split(':')[0]
    yhat_jul = float(out[out['ds'].dt.month == 7]['yhat'].iloc[0])

    # Must NOT pick a flat-forecast model. HistoricAverage on a 53-month
    # zero-padded series is the known failure mode.
    assert pick in {'SeasonalNaive', 'Theta', 'OptimizedTheta', 'AutoETS'}, (
        f"item with year-long gap must reach a seasonal pick after trim; got {pick}"
    )
    # Jul forecast must reflect the peak (last year's Jul = 3920). Anything
    # below 1500 is a clear flatline failure (the pre-fix HA pick produced
    # 1349 — that's our bright-line threshold).
    assert yhat_jul > 1500.0, (
        f"Jul forecast collapsed despite gap-trim: {yhat_jul} (pick={pick}); "
        f"expected ~3920 from SN+peak_ratio"
    )
