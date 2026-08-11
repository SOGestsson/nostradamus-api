"""Tests for AutoModel upper_70/90/95 estimation and nesting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    ClassicalForecasts,
    _attach_upper_quantiles,
    _conformal_corrected_quantile,
    _conformal_forecast_n_windows,
    _excess_quantiles_from_values,
    _event_month_limits,
    _event_probability,
    _historical_excess_quantiles,
    _is_spiky_intermittent,
)


def _spike_year(nov: float, dec: float = 0.0) -> list[float]:
    """One year of an event item: sales only in Nov/Dec."""
    return [0.0] * 10 + [nov, dec]


def _make_panel(uid: str, dates: pd.DatetimeIndex, ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({'item_id': uid, 'day': list(dates), 'actual_sale': ys})


def _make_forecaster() -> ClassicalForecasts:
    return ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')


def _assert_nesting(out: pd.DataFrame, uid: str | None = None) -> None:
    fc = out if uid is None else out.loc[out['unique_id'] == uid]
    yhat = fc['yhat'].to_numpy(dtype=float)
    u70 = fc['upper_70'].to_numpy(dtype=float)
    u90 = fc['upper_90'].to_numpy(dtype=float)
    u95 = fc['upper_95'].to_numpy(dtype=float)
    assert np.all(u95 >= yhat - 1e-9)
    assert np.all(u90 >= yhat - 1e-9)
    assert np.all(u70 >= yhat - 1e-9)
    assert np.all(u70 <= u90 + 1e-9)
    assert np.all(u90 <= u95 + 1e-9)
    assert np.all(np.isfinite(u95))


def test_conformal_n_windows_adaptive():
    assert _conformal_forecast_n_windows(25, 12) == 1
    assert _conformal_forecast_n_windows(40, 12) == 2
    assert _conformal_forecast_n_windows(14, 12) == 1
    # More history -> more calibration windows (capped at 5)
    assert _conformal_forecast_n_windows(40, 6) == 5
    assert _conformal_forecast_n_windows(200, 6) == 5


def test_conformal_corrected_quantile_finite_sample():
    # With few points the required rank exceeds n -> clip to max (conservative).
    arr = np.array([1.0, 2.0, 3.0])
    assert _conformal_corrected_quantile(arr, 0.95) == 3.0
    # With 19 points, ceil(20*0.95)=19 -> the max (19th smallest).
    arr19 = np.arange(1.0, 20.0)
    assert _conformal_corrected_quantile(arr19, 0.95) == 19.0
    # q70 of 12 points: ceil(13*0.7)=10 -> 10th smallest.
    arr12 = np.arange(1.0, 13.0)
    assert _conformal_corrected_quantile(arr12, 0.70) == 10.0


def test_attach_upper_quantiles_residuals_backstop_thin_conformal():
    """When conformal columns exist but are tighter than the CV residual band,
    the residual band wins (elementwise max)."""
    yhat = np.array([10.0, 10.0])
    fcst = pd.DataFrame({
        'Naive': yhat,
        'Naive-hi-70': [10.5, 10.5],
        'Naive-hi-90': [11.0, 11.0],
        'Naive-hi-95': [11.5, 20.0],  # second row wider than residual band
    })
    cv_q = {'item': {'q70': 2.0, 'q90': 4.0, 'q95': 6.0}}
    u70, u90, u95 = _attach_upper_quantiles(
        yhat,
        fcst,
        'Naive',
        uid_series=pd.Series(['item', 'item']),
        cv_excess_by_uid=cv_q,
    )
    assert u95[0] == 16.0  # residual band (10+6) beats thin conformal (11.5)
    assert u95[1] == 20.0  # conformal kept where it is wider
    assert u90[0] == 14.0
    assert u70[0] == 12.0


def test_attach_upper_quantiles_uses_cv_residuals():
    yhat = np.array([10.0, 12.0])
    fcst = pd.DataFrame({'Naive': yhat})
    cv_q = {'item': {'q70': 2.0, 'q90': 4.0, 'q95': 6.0}}
    u70, u90, u95 = _attach_upper_quantiles(
        yhat,
        fcst,
        'Naive',
        uid_series=pd.Series(['item', 'item']),
        cv_excess_by_uid=cv_q,
    )
    assert u95[0] == 16.0
    assert u90[0] == 14.0
    assert u70[0] == 12.0
    _assert_nesting(pd.DataFrame({'yhat': yhat, 'upper_70': u70, 'upper_90': u90, 'upper_95': u95}))


def test_auto_model_long_history_uses_conformal_not_multiplier():
    """36+ months should get conformal intervals (not crude 1.5× fallback)."""
    rng = np.random.default_rng(42)
    ds = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1), periods=40, freq='MS')
    y = [50.0 + 10.0 * np.sin(i / 6.0) + rng.normal(0, 3) for i in range(40)]
    df = _make_panel('long_hist', ds, [max(5.0, v) for v in y])
    out = _make_forecaster().auto_model_forecast_panel(df, h=6, metric='wape', n_windows=2)
    uid_out = out.loc[out['unique_id'] == 'long_hist']
    _assert_nesting(uid_out, 'long_hist')
    yhat = uid_out['yhat'].to_numpy(dtype=float)
    u95 = uid_out['upper_95'].to_numpy(dtype=float)
    crude = np.maximum(yhat * 1.5, yhat + 1.0)
    assert float(yhat.max()) > 0.0
    assert not np.allclose(u95, crude, rtol=1e-6, atol=1e-6)
    assert float(u95.max()) > float(yhat.max())


def test_auto_model_short_history_uses_cv_residual_fallback():
    """~28 months: conformal gate may fail; CV residuals should still lift upper_95."""
    ds = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1), periods=28, freq='MS')
    y = []
    for d in ds:
        if d.month == 7:
            y.append(120.0)
        else:
            y.append(20.0)
    df = _make_panel('short_hist', ds, y)
    out = _make_forecaster().auto_model_forecast_panel(df, h=6, metric='wape', n_windows=2)
    uid_out = out.loc[out['unique_id'] == 'short_hist']
    _assert_nesting(uid_out, 'short_hist')
    yhat = uid_out['yhat'].to_numpy(dtype=float)
    u95 = uid_out['upper_95'].to_numpy(dtype=float)
    crude = np.maximum(yhat * 1.5, yhat + 1.0)
    assert float(u95.max()) >= float(yhat.max())
    assert not np.allclose(u95, crude, rtol=1e-6, atol=1e-6)


def test_auto_model_dead_sku_zeros_all_uppers():
    rows = [{'item_id': 'dead', 'day': f'2020-{m:02d}-01', 'actual_sale': float(m)} for m in range(1, 7)]
    df = pd.DataFrame(rows)
    out = _make_forecaster().auto_model_forecast_panel(df, h=6, metric='wape_bias', n_windows=2)
    assert float(out['yhat'].max()) <= 0.0
    assert float(out['upper_95'].max()) <= 0.0
    assert float(out['upper_90'].max()) <= 0.0
    assert float(out['upper_70'].max()) <= 0.0


def test_peak_ratio_scales_upper_quantiles():
    from inventory_algorithm.classical_forecasts import _auto_model_compute_peak_ratio

    ds = pd.date_range('2021-01-01', periods=30, freq='MS')
    peaks = {2022: 100.0, 2023: 130.0, 2024: 160.0}
    y = []
    for d in ds:
        if d.month == 7 and d.year in peaks:
            y.append(peaks[d.year])
        else:
            y.append(25.0)
    df = _make_panel('peak_uid', ds, y)
    out = _make_forecaster().auto_model_forecast_panel(df, h=12, metric='wape_bias', n_windows=1)
    uid_out = out.loc[out['unique_id'] == 'peak_uid'].copy()
    _assert_nesting(uid_out, 'peak_uid')
    if 'peak_ratio' in str(uid_out['model_used'].iloc[0]):
        ratio = _auto_model_compute_peak_ratio(
            pd.DataFrame({'unique_id': 'peak_uid', 'ds': ds, 'y': y})
        ).get('peak_uid', 1.0)
        if ratio and ratio > 1.01:
            july = uid_out.loc[pd.to_datetime(uid_out['ds']).dt.month == 7]
            if not july.empty:
                assert float(july['upper_95'].iloc[0]) >= float(july['yhat'].iloc[0])


def test_excess_quantiles_include_relative_when_baselines_given():
    values = [0.0, 5.0, 10.0, 50.0]
    baselines = [20.0, 20.0, 20.0, 200.0]
    q = _excess_quantiles_from_values(values, baselines)
    assert 'q95' in q and 'q95_rel' in q
    # Absolute q95 dominated by the 50 excess; relative one by 10/20 = 0.5.
    assert q['q95'] == 50.0
    assert q['q95_rel'] == 0.5


def test_relative_band_caps_off_season_inflation():
    """A peak-month absolute error must not inflate low-forecast months by the
    same absolute amount; the relative quantile caps it proportionally."""
    yhat = np.array([300.0, 50.0])  # peak month, off-season month
    fcst = pd.DataFrame({'Naive': yhat})
    # Peak error of 300 absolute but only ~1x relative.
    cv_q = {'item': {
        'q70': 100.0, 'q90': 200.0, 'q95': 300.0,
        'q70_rel': 0.4, 'q90_rel': 0.7, 'q95_rel': 1.0,
    }}
    u70, u90, u95 = _attach_upper_quantiles(
        yhat,
        fcst,
        'Naive',
        uid_series=pd.Series(['item', 'item']),
        cv_excess_by_uid=cv_q,
    )
    # Peak month: min(300 abs, 1.0*300 rel) = 300 -> upper 600.
    assert u95[0] == 600.0
    # Off-season: min(300 abs, 1.0*50 rel) = 50 -> upper 100, NOT 350.
    assert u95[1] == 100.0
    assert u90[1] == 50.0 + 0.7 * 50.0
    assert u70[1] == 50.0 + 0.4 * 50.0
    _assert_nesting(pd.DataFrame({'yhat': yhat, 'upper_70': u70, 'upper_90': u90, 'upper_95': u95}))


def test_recurring_event_month_limited_by_its_own_magnitudes():
    """>=2 years of Nov sales -> Nov band comes from November's own magnitudes,
    so an inflated CV band can't run away; off-season is unaffected."""
    ds = pd.Series(pd.to_datetime(['2026-07-01', '2026-11-01']))
    yhat = np.array([100.0, 100.0])
    fcst = pd.DataFrame({'HistoricAverage': yhat})
    # Inflated CV band: abs q95 way above anything seen in November.
    cv_q = {'item': {'q70': 0.0, 'q90': 0.0, 'q95': 3000.0, 'q95_hi': 3000.0, 'q95_rel': 30.0}}
    # Two full years ending Dec 2025; Nov 2024=900, Nov 2025=1000 (event recurred).
    hist_ds = pd.date_range('2024-01-01', periods=24, freq='MS')
    y_arr = np.array(_spike_year(900.0) + _spike_year(1000.0), dtype=float)
    y_hist = {'item': y_arr}
    ds_hist = {'item': pd.Series(hist_ds)}
    u70, u90, u95 = _attach_upper_quantiles(
        yhat,
        fcst,
        'HistoricAverage',
        uid_series=pd.Series(['item', 'item']),
        ds_series=ds,
        cv_excess_by_uid=cv_q,
        historical_excess_by_uid=None,
        y_hist_by_uid=y_hist,
        ds_hist_by_uid=ds_hist,
    )
    # November covers the event but cannot exceed its max plus that month's own
    # growth (1000 * 1.111), despite the 3000 CV band.
    growth = (1000.0 - 900.0) / 900.0
    assert float(u95[1]) >= 900.0
    assert float(u95[1]) <= 1000.0 * (1.0 + growth) + 1e-6
    _assert_nesting(pd.DataFrame({'yhat': yhat, 'upper_70': u70, 'upper_90': u90, 'upper_95': u95}))


def test_limits_never_exceed_month_history_when_month_is_stable():
    """A stable month gets only its own observed growth as headroom, so no level
    can exceed what that month has ever produced by more than that."""
    hist_ds = pd.Series(pd.date_range('2023-01-01', periods=36, freq='MS'))
    y = np.array(
        _spike_year(1400.0) + _spike_year(1380.0) + _spike_year(1418.0), dtype=float
    )
    limits, ramp = _event_month_limits(y, 650.0, ds_hist=hist_ds, forecast_month=11)
    assert ramp == 0.0 and limits is not None
    ceiling = 1418.0 * (1.0 + (1418.0 - 1380.0) / 1380.0)
    assert limits[95] <= ceiling + 1e-6
    assert limits[95] < 1.1 * 1418.0
    # Levels stay distinct and ordered.
    assert limits[70] < limits[90] < limits[95]
    # The event is near-certain here, so even upper_70 must cover it.
    assert limits[70] > 1300.0


def test_declining_event_magnitude_pulls_limits_down():
    """Nov 2735 then 921: recency-weighted magnitudes mean the band tracks the
    recent event size, and a declining month gets no headroom above its max."""
    hist_ds = pd.Series(pd.date_range('2023-01-01', periods=24, freq='MS'))
    y = np.array(_spike_year(2735.0) + _spike_year(921.0), dtype=float)
    limits, _ = _event_month_limits(y, 100.0, ds_hist=hist_ds, forecast_month=11)
    assert limits is not None
    assert limits[95] <= 2735.0 + 1e-6
    # upper_70 is dragged toward the recent, smaller event.
    assert limits[70] < 2735.0

    # Same two magnitudes the other way round -> a growing month sits higher.
    y_growing = np.array(_spike_year(921.0) + _spike_year(2735.0), dtype=float)
    limits_growing, _ = _event_month_limits(y_growing, 100.0, ds_hist=hist_ds, forecast_month=11)
    assert limits_growing[70] > limits[70]


def test_levels_switch_off_in_turn_as_event_probability_decays():
    """The mixture's tail is the event itself, so upper_95 stays near the event
    size while upper_70 (then upper_90) collapses to the forecast as misses
    accumulate. That spread is the safety-stock signal."""
    yhat = 100.0
    prev = None
    for misses in (0, 1, 3, 6, 10):
        years = [_spike_year(1000.0) for _ in range(4)] + [_spike_year(0.0) for _ in range(misses)]
        y = np.array([v for yr in years for v in yr], dtype=float)
        hist_ds = pd.Series(pd.date_range('2010-01-01', periods=len(y), freq='MS'))
        limits, _ = _event_month_limits(y, yhat, ds_hist=hist_ds, forecast_month=11)
        assert limits is not None
        assert limits[70] <= limits[90] <= limits[95] + 1e-9
        if prev is not None:
            # More misses never widens the band.
            assert limits[70] <= prev[70] + 1e-9
            assert limits[95] <= prev[95] + 1e-9
        prev = limits
    # After many missed seasons upper_70 has switched off entirely...
    assert prev[70] == yhat
    # ...while upper_95 still protects against the event returning.
    assert prev[95] > 500.0


def test_event_probability_discounts_old_evidence():
    """A recent miss counts for more than an old one, and the estimate stays
    strictly inside (0, 1) — an item still being forecast is never hopeless."""
    recent_miss = _event_probability(np.array([0.0, 1.0, 1.0, 1.0, 1.0]))
    old_miss = _event_probability(np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
    assert recent_miss < old_miss
    all_hits = _event_probability(np.ones(5))
    all_misses = _event_probability(np.zeros(5))
    assert 0.0 < all_misses < recent_miss < all_hits < 1.0


def test_continuous_item_keeps_residual_band_without_month_limits():
    """Regular month-in month-out demand must not be clamped to same-month
    history: ordinary noise legitimately exceeds last year's value."""
    rng = np.random.default_rng(0)
    y = np.abs(rng.normal(150.0, 40.0, size=60))
    hist_ds = pd.Series(pd.date_range('2021-01-01', periods=60, freq='MS'))
    assert not _is_spiky_intermittent(np.asarray(y))
    limits, ramp = _event_month_limits(y, 150.0, ds_hist=hist_ds, forecast_month=11)
    assert limits is None
    assert ramp == 0.0

    # An event item with the same length is still gated in.
    y_spike = np.array(_spike_year(1000.0) * 5, dtype=float)
    assert _is_spiky_intermittent(y_spike)
    limits_spike, _ = _event_month_limits(y_spike, 100.0, ds_hist=hist_ds, forecast_month=11)
    assert limits_spike is not None


def test_trailing_pad_zeros_do_not_make_item_look_intermittent():
    """Panel regularization pads months with no data; those must not turn a
    continuously selling item into a 'spiky' one."""
    y = np.concatenate([np.full(48, 150.0), np.zeros(12)])
    assert not _is_spiky_intermittent(y)


def test_single_year_spike_uses_growing_band_not_seasonality():
    """One Nov observation is not recurrence: no month limits at Nov; instead a
    horizon-growing band toward the observed max."""
    hist_ds = pd.Series(pd.date_range('2025-01-01', periods=14, freq='MS'))
    y = np.zeros(14)
    y[10] = 2522.0  # Nov 2025, single spike
    y[11] = 249.0
    # November forecast month but only 1 year of Nov data -> sparse ramp path.
    lim_e, early = _event_month_limits(
        y, 130.0, ds_hist=hist_ds, forecast_month=11, horizon_pos=1, horizon_len=12
    )
    lim_l, late = _event_month_limits(
        y, 130.0, ds_hist=hist_ds, forecast_month=11, horizon_pos=12, horizon_len=12
    )
    assert lim_e is None and lim_l is None
    # Growing with horizon; late horizon reaches the observed max.
    assert 0.0 < early < late
    assert late >= (2522.0 - 130.0) - 1e-6
    assert abs(early - (2522.0 - 130.0) / 12.0) < 1e-6
    # A quiet month (July) gets the same ramp: timing of the next spike unknown.
    _, jul = _event_month_limits(
        y, 130.0, ds_hist=hist_ds, forecast_month=7, horizon_pos=6, horizon_len=12
    )
    assert abs(jul - (2522.0 - 130.0) * 0.5) < 1e-6


def test_near_zero_cv_does_not_hide_seasonal_month_band():
    """CV holdouts on quiet months yield ~0 excess; a two-year Nov spike level
    must still lift upper_95 for November."""
    ds = pd.Series(pd.to_datetime(['2026-11-01']))
    yhat = np.array([80.0])
    fcst = pd.DataFrame({'HistoricAverage': yhat})
    cv_q = {'item': {'q70': 0.0, 'q90': 0.0, 'q95': 0.0, 'q95_rel': 0.0}}
    hist_ds = pd.date_range('2024-01-01', periods=24, freq='MS')
    y_arr = np.array(_spike_year(850.0) + _spike_year(900.0), dtype=float)
    y_hist = {'item': y_arr}
    ds_hist = {'item': pd.Series(hist_ds)}
    u70, u90, u95 = _attach_upper_quantiles(
        yhat,
        fcst,
        'HistoricAverage',
        uid_series=pd.Series(['item']),
        ds_series=ds,
        cv_excess_by_uid=cv_q,
        y_hist_by_uid=y_hist,
        ds_hist_by_uid=ds_hist,
    )
    assert float(u95[0]) >= 900.0 - 1e-9


def test_historical_excess_quantiles_nonempty():
    ds = pd.date_range('2020-01-01', periods=24, freq='MS')
    y = np.array([10.0 if m % 12 == 6 else 30.0 for m in range(24)], dtype=float)
    q = _historical_excess_quantiles(y, pd.Series(ds))
    assert 'q95' in q
    assert q['q95'] >= 0.0
