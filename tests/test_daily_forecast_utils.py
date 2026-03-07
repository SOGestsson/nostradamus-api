"""Unit tests for api.daily_forecast_utils.monthly_to_daily."""
import pytest
import pandas as pd
from api.daily_forecast_utils import monthly_to_daily


def test_monthly_two_months_expansion():
    """Two months (Jan 31 days, Feb 28 days) -> 59 daily rows."""
    dates = ['2025-01-31', '2025-02-28']
    forecast = [62.0, 56.0]
    u70 = [70.0, 64.0]
    u90 = [80.0, 72.0]
    u95 = [90.0, 82.0]
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, u70, u90, u95)
    assert len(daily_dates) == 31 + 28
    assert len(daily_fcst) == len(daily_dates)
    assert len(daily_var) == len(daily_dates)
    assert daily_dates[0] == '2025-01-01'
    assert daily_dates[30] == '2025-01-31'
    assert daily_dates[31] == '2025-02-01'
    assert daily_dates[-1] == '2025-02-28'
    # Jan: 62/31 ≈ 2.0
    assert abs(daily_fcst[0] - 2.0) < 1e-9
    assert abs(daily_fcst[30] - 2.0) < 1e-9
    # Feb: 56/28 = 2.0
    assert abs(daily_fcst[31] - 2.0) < 1e-9
    # All variance non-negative
    assert all(v >= 0 for v in daily_var)


def test_monthly_fallback_no_quantiles():
    """When no quantiles: daily variance = (daily_forecast * 0.2)^2."""
    dates = ['2025-01-31']
    forecast = [62.0]
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, None)
    assert len(daily_dates) == 31
    assert abs(daily_fcst[0] - 62.0 / 31) < 1e-9
    expected_var = (62.0 / 31 * 0.2) ** 2
    assert abs(daily_var[0] - expected_var) < 1e-9
    assert all(v >= 0 for v in daily_var)


def test_monthly_fallback_yhat_zero():
    """When yhat=0 and no quantiles: variance = 0 (daily_mean*0.2)^2 = 0."""
    dates = ['2025-01-31']
    forecast = [0.0]
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, None)
    assert len(daily_dates) == 31
    assert daily_fcst[0] == 0.0
    assert daily_var[0] == 0.0


def test_variance_from_quantiles_average():
    """Variance from quantiles: sigma = mean of (upper_p - mu)/z_p, then var_month/n_days."""
    # One month Jan: yhat=100, upper_95=116.45 -> sigma = 16.45/1.645 = 10, var_month=100, var_day=100/31
    dates = ['2025-01-31']
    forecast = [100.0]
    u95 = [116.45]  # 100 + 1.645*10
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, u95)
    assert len(daily_dates) == 31
    assert abs(daily_fcst[0] - 100.0 / 31) < 1e-9
    expected_sigma = 10.0
    expected_var_day = (expected_sigma ** 2) / 31
    assert abs(daily_var[0] - expected_var_day) < 0.01
    assert all(v >= 0 for v in daily_var)


def test_negative_variance_guard_quantile_below_mean():
    """If upper quantile < mean (bad data), clamp gives sigma>=0, variance>=0."""
    dates = ['2025-01-31']
    forecast = [100.0]
    u95 = [90.0]  # below mean -> (90-100) clamped to 0
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, u95)
    assert len(daily_dates) == 31
    # sigma from u95 would be negative; we use max(0, u) so u=0 -> sigma=0 -> var=0
    assert daily_var[0] >= 0
    # With only one quantile and it clamped to 0, sigma_estimates = [0], mean=0, var=0
    assert daily_var[0] == 0.0


def test_already_daily_input():
    """When forecast_dates are 1 day apart, treat as daily: one day per period, no expansion."""
    dates = ['2025-02-01', '2025-02-02', '2025-02-03']
    forecast = [10.0, 20.0, 15.0]
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, None)
    assert len(daily_dates) == 3
    assert daily_dates == ['2025-02-01', '2025-02-02', '2025-02-03']
    assert daily_fcst == [10.0, 20.0, 15.0]
    # Fallback variance per day
    assert abs(daily_var[0] - (10.0 * 0.2) ** 2) < 1e-9
    assert abs(daily_var[1] - (20.0 * 0.2) ** 2) < 1e-9
    assert all(v >= 0 for v in daily_var)


def test_empty_dates():
    """Empty input -> empty output."""
    daily_dates, daily_fcst, daily_var = monthly_to_daily([], [], None, None, None)
    assert daily_dates == []
    assert daily_fcst == []
    assert daily_var == []


def test_dates_forecast_length_mismatch_raises():
    """Mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        monthly_to_daily(['2025-01-31'], [1.0, 2.0], None, None, None)


def test_month_start_dates():
    """Month-start dates (e.g. 2025-01-01) still expand to full month."""
    dates = ['2025-01-01', '2025-02-01']
    forecast = [31.0, 28.0]  # 1 per day
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, None)
    assert len(daily_dates) == 31 + 28
    assert daily_dates[0] == '2025-01-01'
    assert daily_dates[30] == '2025-01-31'
    assert abs(daily_fcst[0] - 1.0) < 1e-9
    assert abs(daily_fcst[31] - 1.0) < 1e-9


def test_leap_year_february():
    """February in leap year has 29 days."""
    dates = ['2024-02-29']  # 2024 is leap year
    forecast = [29.0]
    daily_dates, daily_fcst, daily_var = monthly_to_daily(dates, forecast, None, None, None)
    assert len(daily_dates) == 29
    assert daily_dates[-1] == '2024-02-29'
    assert abs(daily_fcst[0] - 1.0) < 1e-9


# --- Integration test for POST /api/v1/forecast/generate_daily ---


def _make_monthly_history(item_id: int, n_months: int, start: str = '2022-01-01'):
    """Build sim_input_his list for one item, month-start convention."""
    import pandas as pd
    dr = pd.date_range(start=start, periods=n_months, freq='MS')
    return [
        {'item_id': item_id, 'actual_sale': 10.0 + (i % 12), 'day': d.strftime('%Y-%m-%d')}
        for i, d in enumerate(dr)
    ]


def test_generate_daily_endpoint_monthly_returns_daily():
    """POST /generate_daily with monthly freq returns daily forecast_dates, forecast, variance."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    his = _make_monthly_history(100, 24)
    payload = {
        'sim_input_his': his,
        'forecast_periods': 2,
        'mode': 'local',
        'local_model': 'auto_ets',
        'season_length': 12,
        'freq': 'M',
    }
    resp = client.post('/api/v1/forecast/generate_daily', json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert 'forecasts' in data
    assert data['total_items'] == 1
    assert data['frequency'] in ('M', 'MS', 'ME')
    fc = data['forecasts'][0]
    assert 'item_id' in fc
    assert 'forecast_dates' in fc
    assert 'forecast' in fc
    assert 'variance' in fc
    assert 'model_used' in fc
    assert len(fc['forecast_dates']) == len(fc['forecast']) == len(fc['variance'])
    # 2 months -> Jan (31) + Feb (28) = 59 days (or 31+31 if first month is different)
    n_days = len(fc['forecast_dates'])
    assert n_days >= 28 and n_days <= 62
    assert all(isinstance(x, (int, float)) and not (x != x) for x in fc['forecast'])
    assert all(isinstance(x, (int, float)) and x >= 0 and not (x != x) for x in fc['variance'])


def test_generate_daily_endpoint_auto_model():
    """POST /generate_daily with auto_model returns daily expansion."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    his = _make_monthly_history(101, 36)
    payload = {
        'sim_input_his': his,
        'forecast_periods': 3,
        'mode': 'local',
        'local_model': 'auto_model',
        'season_length': 12,
        'freq': 'M',
        'auto_model_n_windows': 2,
    }
    resp = client.post('/api/v1/forecast/generate_daily', json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data['forecasts']) == 1
    fc = data['forecasts'][0]
    assert 'error' not in fc or not fc.get('error')
    assert len(fc['forecast_dates']) == len(fc['forecast']) == len(fc['variance'])
    # 3 months -> roughly 90 days (Jan+Feb+Mar or similar)
    assert len(fc['forecast_dates']) >= 28
    assert all(v >= 0 for v in fc['variance'])


def test_generate_daily_endpoint_missing_columns_400():
    """POST /generate_daily with missing required columns returns 400."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    resp = client.post('/api/v1/forecast/generate_daily', json={
        'sim_input_his': [{'item_id': 1}],
        'forecast_periods': 1,
        'freq': 'M',
    })
    assert resp.status_code == 400


def test_generate_daily_endpoint_two_items():
    """POST /generate_daily with two items returns one result per item."""
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    his_1 = _make_monthly_history(1, 24)
    his_2 = _make_monthly_history(2, 24)
    combined = his_1 + his_2
    payload = {
        'sim_input_his': combined,
        'forecast_periods': 1,
        'mode': 'local',
        'local_model': 'naive',
        'season_length': 12,
        'freq': 'M',
    }
    resp = client.post('/api/v1/forecast/generate_daily', json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data['total_items'] == 2
    assert len(data['forecasts']) == 2
    item_ids = {fc['item_id'] for fc in data['forecasts']}
    assert item_ids == {1, 2} or item_ids == {'1', '2'}
    for fc in data['forecasts']:
        assert len(fc['forecast_dates']) == len(fc['forecast']) == len(fc['variance'])
        assert all(v >= 0 for v in fc['variance'])
