"""
Temporary module: expand monthly classical forecast results to daily level
with daily forecast and daily variance (from quantiles or fallback).

Used only by the /generate_daily endpoint. No changes to core forecasting.

When no quantiles are available, daily variance is set to (daily_forecast * 0.2)^2
(approx. 20% CV per day). All variance values are clamped to be non-negative.
"""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

# z-scores for upper quantiles (standard normal)
Z_70 = 0.524  # P(Z <= z) = 0.70
Z_90 = 1.282  # P(Z <= z) = 0.90
Z_95 = 1.645  # P(Z <= z) = 0.95


def monthly_to_daily(
    forecast_dates: list[str],
    forecast: list[float],
    upper_70: list[float] | None = None,
    upper_90: list[float] | None = None,
    upper_95: list[float] | None = None,
) -> tuple[list[str], list[float], list[float]]:
    """
    Expand monthly forecast and quantiles to daily dates, daily forecast, and daily variance.

    - Daily forecast = monthly value / number of days in that month.
    - Daily variance: fit a normal from available upper quantiles (average of sigma estimates
      from 70/90/95), then variance_month = sigma^2, daily_variance = variance_month / n_days.
      When no quantiles are available, daily variance = (daily_forecast * 0.2)^2
      (approx. 20% CV per day). See module docstring.
    - All variances are clamped to be non-negative.

    Returns:
        (daily_dates, daily_forecast, daily_variance) as three lists of the same length.
    """
    daily_dates: list[str] = []
    daily_forecast: list[float] = []
    daily_variance: list[float] = []

    n = len(forecast_dates)
    if n != len(forecast):
        raise ValueError("forecast_dates and forecast must have the same length")

    # If periods are already one day apart, treat as daily (no expansion).
    is_daily = False
    if n >= 2:
        try:
            d0 = pd.to_datetime(forecast_dates[0])
            d1 = pd.to_datetime(forecast_dates[1])
            if (d1 - d0).days <= 1:
                is_daily = True
        except Exception:
            pass

    def _n_days_in_month(d: pd.Timestamp) -> int:
        return calendar.monthrange(int(d.year), int(d.month))[1]

    def _days_in_month(d: pd.Timestamp) -> list[str]:
        n_days = _n_days_in_month(d)
        start = d.replace(day=1)
        return [start.replace(day=day).strftime("%Y-%m-%d") for day in range(1, n_days + 1)]

    for i in range(n):
        try:
            period_end = pd.to_datetime(forecast_dates[i])
        except Exception:
            period_end = pd.to_datetime(forecast_dates[i], errors="coerce")
        if pd.isna(period_end):
            continue
        if is_daily:
            month_dates = [period_end.strftime("%Y-%m-%d")]
            n_days = 1
        else:
            month_start = period_end.replace(day=1)
            n_days = _n_days_in_month(month_start)
            month_dates = _days_in_month(month_start)

        yhat_month = float(forecast[i]) if i < len(forecast) else 0.0
        daily_mean = yhat_month / n_days if n_days else 0.0

        # Variance: from quantiles (average of sigma estimates) or fallback
        sigma_estimates: list[float] = []
        mu = yhat_month
        if upper_95 and i < len(upper_95):
            u = max(0.0, float(upper_95[i]) - mu)
            sigma_estimates.append(u / Z_95)
        if upper_90 and i < len(upper_90):
            u = max(0.0, float(upper_90[i]) - mu)
            sigma_estimates.append(u / Z_90)
        if upper_70 and i < len(upper_70):
            u = max(0.0, float(upper_70[i]) - mu)
            sigma_estimates.append(u / Z_70)

        if sigma_estimates:
            sigma_month = max(0.0, float(np.mean(sigma_estimates)))
            variance_month = sigma_month * sigma_month
            var_day = variance_month / n_days if n_days else 0.0
            var_day = max(0.0, var_day)
        else:
            # When no quantiles are available, daily variance = (daily_forecast * 0.2)^2
            # (approx. 20% CV per day). Documented fallback.
            var_day = max(0.0, (daily_mean * 0.2) ** 2)

        for d in month_dates:
            daily_dates.append(d)
            daily_forecast.append(daily_mean)
            daily_variance.append(var_day)

    return daily_dates, daily_forecast, daily_variance
