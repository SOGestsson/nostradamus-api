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

    # SeasonalNaive is materially better (advantage > seasonal_naive_min_wape_advantage, default 0.06).
    wape = pd.Series({"SeasonalNaive": 0.080, "Theta": 0.150})
    bias_pct = pd.Series({"SeasonalNaive": 0.0, "Theta": 0.0})

    picked = _pick_model_wape_bias_penalty(wape, bias_pct)
    assert picked == "SeasonalNaive"
