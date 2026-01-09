import pandas as pd


def _make_monthly_series(uid: str, n_months: int) -> pd.DataFrame:
    # Month-start convention
    ds = pd.date_range('2020-01-01', periods=n_months, freq='MS')
    # Simple seasonal-ish + trend signal; doesn't matter much because these tests
    # exercise the short/insufficient fallback path.
    y = (pd.Series(range(n_months)) * 0.2 + (pd.Series(range(n_months)) % 12) * 0.5).astype(float)
    return pd.DataFrame({'unique_id': uid, 'ds': ds, 'y': y.values})


def test_auto_model_monthly_12_to_23_months_fallback_is_seasonal_window_average():
    df = _make_monthly_series('item_15', 15)

    # Force the short/insufficient-length branch by asking for more CV windows than possible.
    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df.rename(columns={'unique_id': 'item_id', 'ds': 'day', 'y': 'actual_sale'}), h=3, metric='robust', n_windows=5)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_15', 'model_used'].iloc[0] == 'SeasonalWindowAverage'


def test_auto_model_monthly_under_12_months_fallback_is_historic_average():
    df = _make_monthly_series('item_6', 6)

    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df.rename(columns={'unique_id': 'item_id', 'ds': 'day', 'y': 'actual_sale'}), h=3, metric='robust', n_windows=5)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_6', 'model_used'].iloc[0] == 'HistoricAverage'
