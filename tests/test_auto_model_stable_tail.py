"""Stable-tail and level-shift heuristics for auto_model SeasonalNaive / CV selection."""
from __future__ import annotations

import numpy as np

from inventory_algorithm.classical_forecasts import (
    _auto_model_exclude_seasonal_naive,
    _auto_model_maybe_prefer_level_under_stable_tail,
    _recent_tail_stable_level,
    _seasonal_naive_lag_regime_mismatch,
)


def test_recent_tail_stable_level_detects_flat_positive_tail():
    y = np.array([1000.0] * 20 + [120.0, 118.0, 122.0, 119.0, 121.0, 120.0, 118.0, 120.0])
    ok, m = _recent_tail_stable_level(y, tail_n=8)
    assert ok is True
    assert 115.0 < m < 125.0


def test_recent_tail_stable_level_rejects_volatile_tail():
    y = np.array([1000.0] * 20 + [10.0, 200.0, 5.0, 180.0, 20.0, 150.0, 8.0, 160.0])
    ok, _ = _recent_tail_stable_level(y, tail_n=8, max_cv=0.55)
    assert ok is False


def test_exclude_seasonal_naive_yoy_collapse_excludes_even_with_stable_tail():
    """A YoY collapse + stable new level is the *strongest* signal that SN is wrong.

    SN reproduces absolute past values (y[t-s]); after a 60%+ level shift down to
    a stable floor it would still forecast last year's old peaks. Demand has
    settled at a new run rate, and the old seasonal values are no longer
    representative — the level-prefer reranker should then steer to a level
    model. Previously this branch incorrectly *kept* SN eligible when the tail
    was stable; that bug was the Mjúkís/Jarðarberja failure mode.
    """
    s = 12
    y = np.concatenate([np.full(12, 1500.0), np.full(4, 200.0), np.full(8, 100.0)])
    assert len(y) == 24
    assert _recent_tail_stable_level(y, tail_n=8)[0] is True
    assert _auto_model_exclude_seasonal_naive(
        y, season_length=s, event_seasonal=False, stable_recent_level=False
    ) is True
    assert _auto_model_exclude_seasonal_naive(
        y, season_length=s, event_seasonal=False, stable_recent_level=True
    ) is True


def _empty_scores():
    return {'a': {}}


def test_maybe_prefer_level_swaps_adaptive_for_competitive_historic_average():
    """Reranker demotes adaptive picks to a level model that beats them on WAPE.

    In ``smooth`` / ``trend`` buckets the demote threshold is zero margin.
    Level-model targets are HA, MA6, MA12 and legacy WindowAverage —
    SeasonalWindowAverage is in the demote-FROM set, not a target.
    """
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'smooth'},
        wape_scores_map={'a': {
            'AutoETS': 0.20,
            'HistoricAverage': 0.18,
            'WindowAverage': 0.19,
            'SeasonalWindowAverage': 0.17,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'HistoricAverage'


def test_maybe_prefer_level_demotes_seasonal_window_average_under_stable_tail():
    """SWA is in the demote-FROM set; a beating WindowAverage takes over."""
    from inventory_algorithm.classical_forecasts import (
        _auto_model_maybe_prefer_level_under_stable_tail,
    )
    best_by_uid = {'a': 'SeasonalWindowAverage'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'smooth'},
        wape_scores_map={'a': {
            'SeasonalWindowAverage': 0.20,
            'HistoricAverage': 0.22,
            'WindowAverage': 0.18,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'WindowAverage'


def test_maybe_prefer_level_skips_lag_family_for_event_seasonal():
    """Event-seasonal items keep SN/SWA — the lag IS the forecast signal there."""
    from inventory_algorithm.classical_forecasts import (
        _auto_model_maybe_prefer_level_under_stable_tail,
    )
    best_by_uid = {'a': 'SeasonalNaive'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {
            'SeasonalNaive': 0.20,
            'HistoricAverage': 0.10,
            'WindowAverage': 0.10,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
        event_seasonal_uid={'a': True},
    )
    assert best_by_uid['a'] == 'SeasonalNaive'


def test_maybe_prefer_level_works_for_robust_metric():
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map=_empty_scores(),
        rmse_scores_map={'a': {'AutoETS': 100.0, 'HistoricAverage': 95.0}},
        mae_scores_map={'a': {'AutoETS': 80.0, 'HistoricAverage': 78.0}},
        metric_name='robust',
    )
    assert best_by_uid['a'] == 'HistoricAverage'


def test_maybe_prefer_level_noop_when_level_much_worse():
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {'AutoETS': 0.10, 'HistoricAverage': 0.50}},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'AutoETS'


def test_seasonal_naive_lag_regime_mismatch_when_recent_above_last_year_same_months():
    s = 12
    y = np.zeros(24)
    y[12:16] = 50.0
    y[-8:] = 100.0
    assert _seasonal_naive_lag_regime_mismatch(y, s, tail_n=8) is True


def test_seasonal_naive_lag_regime_mismatch_when_lag_much_higher_than_recent():
    s = 12
    # Last year same months were ~500, recent stable ~100 → SN would overshoot
    y = np.full(24, 500.0)
    y[-8:] = 100.0
    assert _seasonal_naive_lag_regime_mismatch(y, s, tail_n=8) is True


def test_seasonal_naive_lag_regime_mismatch_no_fire_when_similar():
    s = 12
    y = np.full(24, 100.0)
    y[-8:] = 95.0
    assert _seasonal_naive_lag_regime_mismatch(y, s, tail_n=8) is False
