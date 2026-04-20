import pandas as pd


def test_wape_bias_penalty_prefers_lower_bias_when_wape_close():
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # Two models with nearly identical WAPE; bias should decide via penalty.
    wape = pd.Series({"m1": 0.100, "m2": 0.101})  # within abs_eps=0.005
    bias_pct = pd.Series({"m1": 35.0, "m2": 12.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "m2"


def test_wape_bias_penalty_allows_small_bias_within_ok_zone():
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # If both biases are within bias_ok, it should pick the best WAPE.
    wape = pd.Series({"m1": 0.100, "m2": 0.102})
    bias_pct = pd.Series({"m1": 8.0, "m2": 1.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "m1"


def test_wape_bias_penalty_disables_bias_when_wape_is_absolute_error_scale():
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # When WAPE values are large (e.g., denom~0 fallback), bias_pct isn't a percent.
    # In that case we pick lowest WAPE (tie-break by abs(bias) deterministically).
    wape = pd.Series({"m1": 12.0, "m2": 10.0})
    bias_pct = pd.Series({"m1": 1000.0, "m2": 1.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "m2"


def test_wape_bias_penalty_prefers_adaptive_over_seasonal_naive_when_close():
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # SeasonalNaive is slightly better on WAPE, but not materially better.
    wape = pd.Series({"SeasonalNaive": 0.100, "Theta": 0.101})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "Theta": 0.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "Theta"


def test_wape_bias_penalty_keeps_seasonal_naive_when_materially_better():
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # SeasonalNaive is materially better than every non-lag alternative
    # (advantage > seasonal_naive_min_wape_advantage, default 0.08).
    wape = pd.Series({"SeasonalNaive": 0.05, "Theta": 0.20})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "Theta": 0.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "SeasonalNaive"


def test_wape_bias_penalty_demotes_seasonal_naive_when_advantage_below_threshold():
    """SN's lead must clear the threshold even when no other model is in close band."""
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    # 5pp lead is below the default 8pp threshold → demote to best non-lag.
    wape = pd.Series({"SeasonalNaive": 0.10, "AutoETS": 0.15, "HistoricAverage": 0.18})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "AutoETS": 0.0, "HistoricAverage": 0.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "AutoETS"


def test_wape_bias_penalty_demotes_seasonal_window_average_too():
    """SWA shares SN's lag-family failure mode and is subject to the same guard."""
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    wape = pd.Series({"SeasonalWindowAverage": 0.10, "WindowAverage": 0.14})
    bias_pct = pd.Series({"SeasonalWindowAverage": 0.0, "WindowAverage": 0.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "WindowAverage"


def test_wape_bias_penalty_doubles_advantage_required_for_unstable_lag():
    """If the lag pick's per-window WAPE varies a lot, demand a bigger advantage."""
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    wape = pd.Series({"SeasonalNaive": 0.05, "AutoETS": 0.18})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "AutoETS": 0.0})

    # 13pp lead: above default 8pp, below doubled 16pp threshold.
    wape_std = pd.Series({"SeasonalNaive": 0.30, "AutoETS": 0.05})
    picked = _pick_model_wape_bias_penalty(
        wape, bias_pct, wape_std_by_model=wape_std,
    )
    assert picked == "AutoETS"

    # Same WAPE values but stable SN → advantage above threshold → keep SN.
    wape_std_stable = pd.Series({"SeasonalNaive": 0.02, "AutoETS": 0.05})
    picked2 = _pick_model_wape_bias_penalty(
        wape, bias_pct, wape_std_by_model=wape_std_stable,
    )
    assert picked2 == "SeasonalNaive"


def test_wape_bias_penalty_event_seasonal_keeps_lag_family():
    """``prefer_seasonal_naive=True`` (event-seasonal) skips the demotion guard."""
    from inventory_algorithm.classical_forecasts import _pick_model_wape_bias_penalty

    wape = pd.Series({"SeasonalNaive": 0.10, "AutoETS": 0.11})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "AutoETS": 0.0})

    picked = _pick_model_wape_bias_penalty(
        wape, bias_pct, prefer_seasonal_naive=True,
    )
    assert picked == "SeasonalNaive"
