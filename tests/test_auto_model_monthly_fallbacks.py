import pandas as pd


def _make_monthly_series(uid: str, n_months: int) -> pd.DataFrame:
    # End at current month so panel regularization does not add years of synthetic zeros.
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    ds = pd.date_range(end=end, periods=n_months, freq='MS')
    # Simple seasonal-ish + trend signal; doesn't matter much because these tests
    # exercise the short/insufficient fallback path.
    y = (pd.Series(range(n_months)) * 0.2 + (pd.Series(range(n_months)) % 12) * 0.5).astype(float)
    return pd.DataFrame({'unique_id': uid, 'ds': ds, 'y': y.values})


def test_auto_model_monthly_12_to_23_months_flat_stable_fallback_is_naive():
    """12–23 month flat demand → stable-tail Naive (repeats current run rate).

    SeasonalWindowAverage with the new ``window_size=3`` requires at least three
    complete seasonal cycles (36 months for monthly), so it can no longer fit
    on this short-history branch.
    """
    df = _make_monthly_series('item_15', 15)
    df['y'] = 1.0

    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(
        df.rename(columns={'unique_id': 'item_id', 'ds': 'day', 'y': 'actual_sale'}),
        h=3, metric='robust', n_windows=5,
    )

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_15', 'model_used'].iloc[0] == 'Naive'


def test_auto_model_monthly_12_to_23_months_volatile_fallback_is_historic_average():
    """12–23 month series without trend AND without stable tail → HistoricAverage.

    MA6 was tried on this branch but regressed on items with latent monthly
    seasonality (<24m is too short to detect seasonality, so a forecast origin
    in the off-season projected low into the next peak). HA averages across
    more of the cycle and is safer as a blind short-history default.
    """
    import numpy as np
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    ds = pd.date_range(end=end, periods=15, freq='MS')
    y = np.array([5.0, 80.0, 2.0, 90.0, 3.0, 70.0, 8.0, 60.0, 4.0, 95.0, 6.0, 85.0, 3.0, 75.0, 9.0])
    df = pd.DataFrame({'item_id': 'item_vol_15', 'day': ds, 'actual_sale': y})

    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df, h=3, metric='robust', n_windows=5)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_vol_15', 'model_used'].iloc[0] == 'HistoricAverage'


def test_auto_model_monthly_under_12_months_stable_tail_fallback_is_naive():
    """Short series with stable recent demand use Naive (repeats last value)."""
    df = _make_monthly_series('item_6', 6)

    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df.rename(columns={'unique_id': 'item_id', 'ds': 'day', 'y': 'actual_sale'}), h=3, metric='robust', n_windows=5)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_6', 'model_used'].iloc[0] == 'Naive'


def test_auto_model_monthly_under_12_months_volatile_fallback_is_historic_average():
    """Short series with volatile demand still use HistoricAverage."""
    import numpy as np
    end = pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')
    ds = pd.date_range(end=end, periods=6, freq='MS')
    y = np.array([5.0, 80.0, 2.0, 90.0, 3.0, 70.0])
    df = pd.DataFrame({'item_id': 'item_vol', 'day': ds, 'actual_sale': y})

    from inventory_algorithm.classical_forecasts import ClassicalForecasts

    forecaster = ClassicalForecasts(mode='local', local_model='auto_model', season_length=12, freq='M')
    out = forecaster.auto_model_forecast_panel(df, h=3, metric='robust', n_windows=5)

    picked = out[['unique_id', 'model_used']].drop_duplicates()
    assert picked.loc[picked['unique_id'] == 'item_vol', 'model_used'].iloc[0] == 'HistoricAverage'
