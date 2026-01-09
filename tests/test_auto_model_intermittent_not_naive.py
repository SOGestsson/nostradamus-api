import pandas as pd


def test_auto_model_intermittent_series_not_naive():
    """Intermittent demand should prefer Croston/ADIDA/SeasonalNaive over Naive."""
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    n_days = 200
    ds = pd.date_range("2024-01-01", periods=n_days, freq="D")

    # Mostly zeros with occasional spikes.
    y = [0.0] * n_days
    for i in [10, 40, 70, 100, 130, 160, 190]:
        y[i] = 25.0

    df = pd.DataFrame({"item_id": ["X"] * n_days, "day": ds, "actual_sale": y})

    # Use a realistic daily season length so the series isn't treated as "short".
    forecaster = ClassicalForecasts(mode="local", local_model="auto_model", season_length=7, freq="D")
    panel = forecaster.auto_model_forecast_panel(df, h=30, metric="robust", n_windows=1)

    model_used = str(panel["model_used"].iloc[0])
    assert model_used != "Naive"
