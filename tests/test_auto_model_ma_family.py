"""MA family (MA3/MA6/MA12) candidate set, factories, and reranker behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from inventory_algorithm.classical_forecasts import (
    MA_ALL_WINDOWS,
    _auto_model_maybe_prefer_level_under_stable_tail,
    _build_candidate_model_factories,
    _build_ma_factories,
    _build_model_factories_for_keys,
    _ma_window_from_alias,
    _ma_windows_for_history,
    _min_obs_for_model_cv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_scores():
    return {'a': {}}


# ---------------------------------------------------------------------------
# _ma_windows_for_history: history gating
# ---------------------------------------------------------------------------


def test_ma_windows_for_history_short():
    # n=5: only MA3 has enough data (needs >= 4.5).
    assert _ma_windows_for_history(5) == [3]


def test_ma_windows_for_history_medium():
    # n=20: all three windows are valid.
    assert _ma_windows_for_history(20) == [3, 6, 12]


def test_ma_windows_for_history_no_upper_gate():
    """No upper gate on MA12 — on non-stationary long histories MA12 ≠ HA and
    is typically the correct level model (old regime dominates HA's average).
    """
    assert _ma_windows_for_history(48) == [3, 6, 12]
    assert _ma_windows_for_history(60) == [3, 6, 12]


def test_ma_windows_for_history_too_short_for_any():
    # int(1.5 * 3) == 4, so MA3 requires n >= 4.
    assert _ma_windows_for_history(3) == []
    assert _ma_windows_for_history(0) == []


# ---------------------------------------------------------------------------
# Alias / factory wiring
# ---------------------------------------------------------------------------


def test_ma_window_from_alias_parses_ma_aliases():
    assert _ma_window_from_alias('MA3') == 3
    assert _ma_window_from_alias('MA12') == 12
    assert _ma_window_from_alias('HistoricAverage') is None
    assert _ma_window_from_alias('WindowAverage') is None
    assert _ma_window_from_alias('MA') is None
    assert _ma_window_from_alias('') is None


def test_build_ma_factories_produces_distinct_aliases():
    specs = _build_ma_factories([3, 6, 12])
    aliases = [alias for alias, _ in specs]
    assert aliases == ['MA3', 'MA6', 'MA12']
    # Each factory builds a fresh instance with its own alias attribute.
    instances = [fac() for _, fac in specs]
    assert {getattr(m, 'alias', None) for m in instances} == {'MA3', 'MA6', 'MA12'}


def test_build_candidate_model_factories_registers_ma_family():
    specs = _build_candidate_model_factories(season_length=12)
    aliases = [alias for alias, _ in specs]
    for w in MA_ALL_WINDOWS:
        assert f'MA{w}' in aliases
    # Legacy class-name WindowAverage is still present for back-compat.
    assert 'WindowAverage' in aliases


def test_build_model_factories_for_keys_maps_ma_keys():
    specs = _build_model_factories_for_keys(['ma3', 'ma6', 'historic_average'], season_length=12)
    names = [alias for alias, _ in specs]
    assert names == ['MA3', 'MA6', 'HistoricAverage']


def test_min_obs_for_model_cv_includes_ma_window():
    """MA{k} requires k observations in the smallest CV training window."""
    base = _min_obs_for_model_cv('HistoricAverage', season_length=12, cv_h=6, n_windows=2)
    ma3 = _min_obs_for_model_cv('MA3', season_length=12, cv_h=6, n_windows=2)
    ma12 = _min_obs_for_model_cv('MA12', season_length=12, cv_h=6, n_windows=2)
    assert ma3 == base + 3
    assert ma12 == base + 12


# ---------------------------------------------------------------------------
# Level-prefer reranker
# ---------------------------------------------------------------------------


def test_reranker_picks_ma_when_it_beats_historic_average():
    """Reranker picks the lowest-WAPE level model. MA3 is excluded from the
    target pool (too reactive), so on this score table MA6 wins over HA."""
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'smooth'},
        wape_scores_map={'a': {
            'AutoETS': 0.25,
            'HistoricAverage': 0.22,
            'MA3': 0.10,  # best WAPE but excluded — too reactive
            'MA6': 0.20,
            'MA12': 0.23,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'MA6'


def test_reranker_excludes_ma3_from_target_pool():
    """Even if MA3 has the best WAPE, the reranker never picks it — MA3 is a
    reactive short-window model, not the stable level anchor the reranker
    needs (v3 A/B: MA3 bias median −28%, several full collapses)."""
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'smooth'},
        wape_scores_map={'a': {
            'AutoETS': 0.25,
            'HistoricAverage': 0.25,
            'MA3': 0.05,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    # Pool collapses to [pick, HA] with HA tied → HA wins (lower ordering hint
    # than AutoETS, and AutoETS isn't a level target).
    assert best_by_uid['a'] == 'HistoricAverage'


def test_reranker_tie_break_prefers_longer_ma_window():
    """On WAPE ties prefer LONGER windows — they're more stable, which is
    exactly what the reranker is trying to achieve when demoting a jittery
    adaptive pick. Tie-break order: MA12 > WindowAverage > MA6 > HA."""
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'smooth'},
        wape_scores_map={'a': {
            'AutoETS': 0.25,
            'HistoricAverage': 0.20,
            'MA6': 0.20,
            'MA12': 0.20,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'MA12'


def test_reranker_seasonal_bucket_requires_1pp_margin():
    """Seasonal bucket: an MA12 edge <1pp isn't enough to demote the adaptive pick.

    A 1 pp margin filters CV noise (typically 1–2 pp between candidates on
    monthly seasonal series) without blocking legitimate level-model wins.
    """
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {
            'AutoETS': 0.205,
            'HistoricAverage': 0.200,  # 0.5pp edge — below 1pp margin
            'MA12': 0.201,              # 0.4pp edge — below 1pp margin
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'AutoETS'

    # Same table, but now MA12 beats AutoETS by exactly 1 pp: demotion allowed.
    best_by_uid2 = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid2,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {
            'AutoETS': 0.22,
            'HistoricAverage': 0.215,
            'MA12': 0.21,   # 1pp edge (exactly at the margin)
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid2['a'] == 'MA12'


def test_reranker_seasonal_bucket_lets_2pp_ma12_win_v5():
    """v5 regression guard: 2 pp MA12 edge now demotes AutoETS in seasonal bucket.

    v4's 3 pp margin blocked this scenario; v5's 1 pp margin lets it through,
    recovering ~20 MA12 wins that were blocked in v4 A/B testing.
    """
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {
            'AutoETS': 0.22,
            'HistoricAverage': 0.21,
            'MA12': 0.20,   # 2 pp edge — blocked in v4, allowed in v5
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'MA12'


def test_reranker_allows_trend_bucket_with_stable_tail():
    """Mild-trend series that settled into a stable tail should now be reranked."""
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'trend'},
        wape_scores_map={'a': {
            'AutoETS': 0.25,
            'HistoricAverage': 0.20,
            'MA6': 0.18,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'MA6'


def test_reranker_skips_strong_trend_even_on_stable_tail():
    """A strong-trend flag vetoes level-model demotion — a flat level can't follow a trend."""
    best_by_uid = {'a': 'AutoETS'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'trend'},
        wape_scores_map={'a': {
            'AutoETS': 0.25,
            'HistoricAverage': 0.15,
            'MA6': 0.15,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
        strong_trend_uid={'a': True},
    )
    assert best_by_uid['a'] == 'AutoETS'


def test_reranker_demotes_lag_family_to_ma():
    """SN/SWA are in the demote-FROM set; a beating MA6 takes over."""
    best_by_uid = {'a': 'SeasonalNaive'}
    _auto_model_maybe_prefer_level_under_stable_tail(
        best_by_uid=best_by_uid,
        stable_tail_uid={'a': True},
        bucket_by_uid={'a': 'seasonal'},
        wape_scores_map={'a': {
            'SeasonalNaive': 0.30,
            'HistoricAverage': 0.25,
            'MA6': 0.20,
        }},
        rmse_scores_map=_empty_scores(),
        mae_scores_map=_empty_scores(),
        metric_name='wape_bias',
    )
    assert best_by_uid['a'] == 'MA6'


# ---------------------------------------------------------------------------
# End-to-end: MA selection on a step-down cohort via auto_model_forecast_panel
# ---------------------------------------------------------------------------


def test_auto_model_level_shift_cohort_prefers_ma_family():
    """Stable-at-new-level cohort: auto_model should land on a level-family pick
    (HA or any MA{k}) rather than SeasonalNaive / AutoETS / Theta."""
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    ds = pd.date_range(end=end, periods=36, freq='MS')
    rng = np.random.default_rng(0)
    # 12 months at high level, then 24 months at a stable lower level with mild noise.
    y = np.concatenate([np.full(12, 100.0), np.full(24, 20.0) + rng.normal(0, 1.5, 24)])
    df = pd.DataFrame({'item_id': 'step_a', 'day': ds, 'actual_sale': y})

    from inventory_algorithm.classical_forecasts import ClassicalForecasts
    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df, h=3, metric='wape_bias', n_windows=2)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    model = picked.loc[picked['unique_id'] == 'step_a', 'model_used'].iloc[0]
    # Accept any level-family pick. The important property is that the adaptive
    # and lag-family models do NOT win on a stable-at-new-level series.
    assert (
        model == 'HistoricAverage'
        or _ma_window_from_alias(str(model)) is not None
    ), f"expected HA/MA{{k}}, got {model}"
