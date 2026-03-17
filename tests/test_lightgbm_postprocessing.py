import numpy as np
import pandas as pd

from inventory_algorithm.lightgbm_forecasts import (
    _apply_caps,
    _compute_candidates,
    _select_forecast,
)


def test_postprocess_stable_prefers_recent_level_anchor():
    adjustments = {"archetype": "stable", "month_rate_same": 0.1}
    candidates = _compute_candidates(
        archetype="stable",
        p_effective=0.4,
        amount_used=10.0,
        floor=2.0,
        mean_nonzero_12=0.0,  # disables nonzero_level_anchor
        recent_level=20.0,  # enables recent_level_anchor
        month_rate_same=0.1,
        cv_val=0.2,
        ramp_up=False,
        m_stats={"mean": 5.0, "max": 15.0, "nonzero_rate": 0.1},
        last_nonzero_age=2.0,
        overall_nonzero_rate=0.9,
        nonzero_count_last_12=12,
        croston_mean=3.0,
        adida_mean=3.0,
    )
    yhat, pri = _select_forecast(candidates, adjustments)
    assert pri == 1
    assert adjustments["winning_source"] == "recent_level_anchor"
    assert yhat >= 0.0


def test_postprocess_seasonal_prefers_peak_anchor():
    adjustments = {"archetype": "seasonal", "month_rate_same": 0.5}
    candidates = _compute_candidates(
        archetype="seasonal",
        p_effective=0.6,
        amount_used=10.0,
        floor=1.0,
        mean_nonzero_12=5.0,
        recent_level=4.0,
        month_rate_same=0.5,
        cv_val=0.4,
        ramp_up=False,
        m_stats={"mean": 10.0, "max": 40.0, "nonzero_rate": 0.5},
        last_nonzero_age=1.0,
        overall_nonzero_rate=0.8,
        nonzero_count_last_12=9,
        croston_mean=2.0,
        adida_mean=2.0,
    )
    yhat, pri = _select_forecast(candidates, adjustments)
    assert pri == 3
    assert adjustments["winning_source"] == "seasonal_peak_anchor"
    assert yhat >= 0.0


def test_postprocess_intermittent_prefers_croston_floor():
    adjustments = {"archetype": "intermittent", "month_rate_same": 0.2}
    candidates = _compute_candidates(
        archetype="intermittent",
        p_effective=0.2,
        amount_used=1.0,
        floor=0.0,
        mean_nonzero_12=0.0,
        recent_level=1.0,
        month_rate_same=0.2,
        cv_val=0.7,
        ramp_up=False,
        m_stats={"mean": 1.0, "max": 2.0, "nonzero_rate": 0.2},
        last_nonzero_age=3.0,
        overall_nonzero_rate=0.2,
        nonzero_count_last_12=2,
        croston_mean=10.0,  # makes croston_floor big
        adida_mean=1.0,
    )
    yhat, pri = _select_forecast(candidates, adjustments)
    assert pri == 2
    assert adjustments["winning_source"] == "croston_floor"
    assert yhat >= 0.0


def test_postprocess_noisy_prefers_classical_override_and_respects_cap():
    adjustments = {"archetype": "noisy", "month_rate_same": 0.1}
    candidates = _compute_candidates(
        archetype="noisy",
        p_effective=0.1,
        amount_used=0.0,
        floor=0.0,
        mean_nonzero_12=0.0,
        recent_level=10.0,
        month_rate_same=0.1,
        cv_val=0.2,  # enables classical_override via confidence and cv threshold
        ramp_up=False,
        m_stats={"mean": 1.0, "max": 1.0, "nonzero_rate": 0.1},
        last_nonzero_age=2.0,
        overall_nonzero_rate=0.9,
        nonzero_count_last_12=12,
        croston_mean=10.0,
        adida_mean=10.0,
    )
    yhat, pri = _select_forecast(candidates, adjustments)
    assert pri == 2
    assert adjustments["winning_source"] == "classical_override"

    # Apply a hard cap
    capped = _apply_caps(
        yhat=float(yhat),
        uid="1",
        forecast_ds=pd.Timestamp("2026-01-01"),
        month_caps={("1", 1): {"max_y": 1.0, "nonzero_rate": 0.0}},
        archetype="noisy",
        month_rate_same=0.1,
        cap_multiplier=2.0,
        cap_small_floor=0.0,
        cap_nonzero_threshold=0.25,
        adjustments=adjustments,
    )
    assert capped <= adjustments["cap"]
    assert capped >= 0.0

