from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable
import math

import numpy as np
import pandas as pd

from services.duckdb_model_store import DuckDBModelStore, ModelVersionRow
from inventory_algorithm.classical_forecasts import simple_croston_mean, simple_adida_mean


try:
    import lightgbm as lgb  # type: ignore

    _HAS_LGBM = True
    _LGBM_IMPORT_ERROR: str | None = None
except Exception as e:
    lgb = None
    _HAS_LGBM = False
    _LGBM_IMPORT_ERROR = repr(e)

try:
    import optuna  # type: ignore
    from optuna.integration import LightGBMPruningCallback  # type: ignore

    _HAS_OPTUNA = True
except Exception:
    optuna = None
    LightGBMPruningCallback = None
    _HAS_OPTUNA = False

# Version marker for debugging code loading issues
_LIGHTGBM_FORECASTS_VERSION = "2026-02-06-fourier-v2"


# -------------------------
# Metrics
# -------------------------

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-12:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


# -------------------------
# Hyperparameter Tuning
# -------------------------

def _tune_lgbm_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    objective: str = "regression",
    n_trials: int = 30,
    timeout_seconds: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune LightGBM hyperparameters using Optuna.

    Optimizes: learning_rate, num_leaves, feature_fraction, bagging_fraction,
               min_data_in_leaf, lambda_l1, lambda_l2.

    Returns the best hyperparameters found within the given constraints.
    Falls back to default parameters if Optuna is not available.
    """
    if not _HAS_OPTUNA or optuna is None or lgb is None:
        # Return sensible defaults if Optuna not available
        return {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
        }

    # Suppress Optuna logs for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_fn(trial: Any) -> float:
        params = {
            "objective": objective,
            "metric": "l2",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        if objective == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            )

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        if LightGBMPruningCallback is not None:
            callbacks.append(LightGBMPruningCallback(trial, "l2"))

        try:
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=callbacks,
            )
            preds = model.predict(X_val)
            mse = float(np.mean((y_val - preds) ** 2))
            return mse
        except Exception:
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )

    # Handle case where no trials completed successfully
    try:
        best_params = study.best_params
    except ValueError:
        # No completed trials - return defaults
        return {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
        }
    # Add fixed params that weren't tuned
    best_params["bagging_freq"] = 1
    return best_params


# -------------------------
# Monthly normalization
# -------------------------

def _to_month_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")


def _normalize_monthly_history(df_his: pd.DataFrame) -> pd.DataFrame:
    """Normalize any input history to monthly, month-start timestamps.

    - Parses `day` to datetime
    - Aggregates sales within a month (sum)
    - Outputs columns: item_id, ds, y (+ any extra wide columns aggregated by mean)

    Extra wide columns (drivers/support series) are aggregated as mean within month.
    """

    df = df_his.copy()
    if "day" not in df.columns:
        raise ValueError("Missing required column: day")
    if "item_id" not in df.columns:
        raise ValueError("Missing required column: item_id")
    if "actual_sale" not in df.columns:
        raise ValueError("Missing required column: actual_sale")

    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day", "item_id"])
    # Treat item_id as an opaque identifier (often non-numeric in real systems).
    df["item_id"] = df["item_id"].astype(str)

    df["ds"] = _to_month_start(df["day"])
    df["y"] = pd.to_numeric(df["actual_sale"], errors="coerce").fillna(0.0).astype(float)
    # Clamp negatives (e.g., returns) to zero for demand modeling.
    df.loc[df["y"] < 0, "y"] = 0.0

    extra_cols = [c for c in df.columns if c not in ("item_id", "day", "actual_sale", "ds", "y")]

    agg: dict[str, str] = {"y": "sum"}
    for c in extra_cols:
        agg[c] = "mean"

    out = df.groupby(["item_id", "ds"], as_index=False, sort=True).agg(agg)
    out["unique_id"] = out["item_id"].astype(str)
    return out


def _pivot_long_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Convert long drivers (day, driver_name, driver_value[, item_id]) to wide."""
    if drivers is None or drivers.empty:
        return pd.DataFrame()
    df = drivers.copy()
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df["ds"] = _to_month_start(df["day"])
    if "driver_name" not in df.columns or "driver_value" not in df.columns:
        return pd.DataFrame()

    idx = ["ds"] + (["item_id"] if "item_id" in df.columns else [])
    wide = df.pivot_table(index=idx, columns="driver_name", values="driver_value", aggfunc="mean")
    wide = wide.reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def _build_item_month_caps(
    base: pd.DataFrame,
    *,
    lookback_years: int = 4,
) -> dict[tuple[str, int], dict[str, float]]:
    """Compute per-item month-of-year caps from history.

    Returns {(unique_id, month): {"max_y": ..., "mean_y": ..., "nonzero_rate": ...}}.
    """
    if base is None or base.empty or "unique_id" not in base.columns or "ds" not in base.columns or "y" not in base.columns:
        return {}

    df = base[["unique_id", "ds", "y"]].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df[df["ds"].notna()].copy()
    if df.empty:
        return {}

    max_ds = df["ds"].max()
    if pd.isna(max_ds):
        return {}
    start_ds = max_ds - pd.DateOffset(years=int(max(1, lookback_years)))
    df = df[df["ds"] >= start_ds].copy()
    if df.empty:
        return {}

    df["month"] = df["ds"].dt.month
    df["nonzero"] = (pd.to_numeric(df["y"], errors="coerce").fillna(0.0) > 0.0).astype(float)

    agg = (
        df.groupby(["unique_id", "month"], as_index=False)
        .agg(max_y=("y", "max"), mean_y=("y", "mean"), nonzero_rate=("nonzero", "mean"))
    )

    out: dict[tuple[str, int], dict[str, float]] = {}
    for _, r in agg.iterrows():
        key = (str(r["unique_id"]), int(r["month"]))
        out[key] = {
            "max_y": float(r.get("max_y", 0.0)),
            "mean_y": float(r.get("mean_y", 0.0)),
            "nonzero_rate": float(r.get("nonzero_rate", 0.0)),
        }
    return out


# -------------------------
# Feature engineering (direct strategy) PP
# -------------------------

def _month_sin_cos(month: int) -> tuple[float, float]:
    # Use 0-based month index for cyclic encoding (Jan=0,...,Dec=11)
    ang = 2.0 * np.pi * ((float(month) - 1.0) / 12.0)
    return float(np.sin(ang)), float(np.cos(ang))


def _fourier_features(month: int, k: int = 3) -> dict[str, float]:
    """Generate k Fourier pairs for month seasonality.

    For k=3 (default), produces 6 features:
    - month_sin_1, month_cos_1 (fundamental frequency, same as _month_sin_cos)
    - month_sin_2, month_cos_2 (2nd harmonic - captures semi-annual patterns)
    - month_sin_3, month_cos_3 (3rd harmonic - captures quarterly patterns)

    Higher harmonics allow the model to fit more complex seasonal patterns
    while maintaining smooth periodic behavior.
    """
    feats: dict[str, float] = {}
    for i in range(1, k + 1):
        ang = 2.0 * np.pi * i * (float(month) - 1.0) / 12.0
        feats[f"month_sin_{i}"] = float(np.sin(ang))
        feats[f"month_cos_{i}"] = float(np.cos(ang))
    return feats


def _quarter_sin_cos(quarter: int) -> tuple[float, float]:
    ang = 2.0 * np.pi * ((float(quarter) - 1.0) / 4.0)
    return float(np.sin(ang)), float(np.cos(ang))


def _get_easter_date(year: int) -> tuple[int, int]:
    """Calculate Easter Sunday for a given year using the Anonymous Gregorian algorithm.

    Returns (month, day) tuple.
    Easter can fall between March 22 and April 25.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return month, day


def _easter_features(ds: pd.Timestamp, year_easter_cache: dict[int, tuple[int, int]] | None = None) -> dict[str, float]:
    """Generate Easter-related temporal features for a given date.

    Returns dict with:
    - days_to_easter: days until Easter (negative if after)
    - is_easter_month: 1 if same month as Easter, 0 otherwise
    - is_pre_easter_4w: 1 if within 4 weeks before Easter
    - easter_proximity: 1.0 at Easter, decaying to 0 over 4 weeks
    """
    year = ds.year
    month = ds.month
    day = ds.day

    # Use cache if provided, otherwise compute
    if year_easter_cache is not None and year in year_easter_cache:
        easter_month, easter_day = year_easter_cache[year]
    else:
        easter_month, easter_day = _get_easter_date(year)
        if year_easter_cache is not None:
            year_easter_cache[year] = (easter_month, easter_day)

    # Calculate days to Easter (positive = before Easter, negative = after)
    from datetime import date
    try:
        easter_date = date(year, easter_month, easter_day)
        current_date = date(year, month, day)
        days_to_easter = (easter_date - current_date).days
    except ValueError:
        days_to_easter = 0

    # Features
    is_easter_month = 1.0 if month == easter_month else 0.0
    is_pre_easter_4w = 1.0 if 0 <= days_to_easter <= 28 else 0.0

    # Easter proximity: peaks at Easter, decays over 4 weeks before and after
    if abs(days_to_easter) <= 28:
        easter_proximity = max(0.0, 1.0 - abs(days_to_easter) / 28.0)
    else:
        easter_proximity = 0.0

    return {
        "days_to_easter": float(days_to_easter),
        "is_easter_month": is_easter_month,
        "is_pre_easter_4w": is_pre_easter_4w,
        "easter_proximity": easter_proximity,
    }


def _nonzero_run_length(values: np.ndarray) -> int:
    # months since last non-zero ending at current index
    for i in range(len(values) - 1, -1, -1):
        if float(values[i]) != 0.0:
            return int(len(values) - 1 - i)
    return int(len(values))


def _rolling_slope(y: np.ndarray) -> float:
    """Slope of the last 12 points (or fewer) using a simple linear fit."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    window = y[-12:] if len(y) >= 12 else y
    x = np.arange(len(window), dtype=float)
    if float(np.nanstd(window)) <= 1e-12:
        return 0.0
    slope, _ = np.polyfit(x, window, deg=1)
    return float(slope)


def _recent_level(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return 0.0
    window = y[-12:] if len(y) >= 12 else y
    nz = window[window > 0.0]
    if len(nz) >= 3:
        return float(np.median(nz))
    return float(np.mean(window))


def _cv(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("inf")
    mean = float(np.mean(y))
    if mean <= 1e-9:
        return float("inf")
    return float(np.std(y) / mean)


def _archetype(nonzero_rate: float, cv: float, seasonal_strength: float) -> str:
    if seasonal_strength >= 0.8 and nonzero_rate <= 0.5:
        return "seasonal"
    if 0.1 <= nonzero_rate <= 0.5:
        return "intermittent"
    if cv >= 1.0:
        return "noisy"
    return "stable"


def _compute_global_monthly_effects(
    base: pd.DataFrame,
    *,
    lookback_months: int = 36,
) -> dict[int, float]:
    """Compute global month-of-year effects across all items.
    
    Returns a dict mapping month (1-12) to median demand level across all items.
    Uses robust aggregation (median) to handle outliers and sparse data.
    
    Args:
        base: DataFrame with columns ['ds', 'y'] (or 'y_orig' if available)
        lookback_months: Number of recent months to use for aggregation
        
    Returns:
        dict[int, float]: {month: median_demand_level} for months 1-12
    """
    if base is None or base.empty or "ds" not in base.columns:
        return {m: 0.0 for m in range(1, 13)}
    
    # Use y_orig if available (original scale), otherwise y
    y_col = "y_orig" if "y_orig" in base.columns else "y"
    if y_col not in base.columns:
        return {m: 0.0 for m in range(1, 13)}
    
    df = base.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    if df.empty:
        return {m: 0.0 for m in range(1, 13)}
    
    # Sort by date and take last N months (by date, not by row count)
    df = df.sort_values("ds", kind="mergesort")
    if len(df) > 0:
        max_ds = df["ds"].max()
        if pd.notna(max_ds):
            # Calculate cutoff date: lookback_months before the latest date
            cutoff_ds = pd.Timestamp(max_ds) - pd.DateOffset(months=lookback_months)
            df = df[df["ds"] >= cutoff_ds].copy()
    
    df["month"] = df["ds"].dt.month
    y_vals = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    
    # Compute median per month across all items (robust to outliers)
    global_monthly: dict[int, float] = {}
    for m in range(1, 13):
        month_data = y_vals[df["month"] == m]
        # Use median of non-zero values, or 0 if all zeros
        non_zero = month_data[month_data > 0.0]
        if len(non_zero) > 0:
            global_monthly[m] = float(np.median(non_zero))
        else:
            global_monthly[m] = 0.0
    
    return global_monthly


def _month_index(ds: pd.Timestamp, start_ds: pd.Timestamp) -> int:
    """Integer number of months between two month-start timestamps."""
    ds = pd.Timestamp(ds)
    start_ds = pd.Timestamp(start_ds)
    return int((ds.year * 12 + ds.month) - (start_ds.year * 12 + start_ds.month))


def _fit_trend_params(ds: np.ndarray, y: np.ndarray, method: str) -> dict[str, Any]:
    """Fit a simple per-item trend model.

    Supported methods:
    - 'none': no detrending
    - 'linear': y ~= a*t + b
    - 'log1p_linear': log1p(y) ~= a*t + b  (i.e., exponential trend on y)
    """

    method = (method or "none").strip().lower()
    if method not in {"none", "linear", "log1p_linear"}:
        raise ValueError("detrend_method must be one of: none, linear, log1p_linear")

    ds0 = pd.Timestamp(ds[0])
    x = np.array([_month_index(pd.Timestamp(d), ds0) for d in ds], dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "none":
        return {"method": "none", "start_ds": ds0.strftime("%Y-%m-%d"), "slope": 0.0, "intercept": 0.0}

    if method == "log1p_linear":
        yt = np.log1p(np.maximum(0.0, y))
    else:
        yt = y

    if len(yt) < 2 or float(np.nanstd(yt)) <= 1e-12:
        slope = 0.0
        intercept = float(np.nanmean(yt)) if len(yt) else 0.0
    else:
        slope, intercept = np.polyfit(x, yt, deg=1)

    return {
        "method": method,
        "start_ds": ds0.strftime("%Y-%m-%d"),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _resolve_detrend_method(
    base: pd.DataFrame,
    requested: str | None,
    *,
    zero_ratio_threshold: float = 0.5,
    min_nonzero: int = 6,
) -> tuple[str, dict[str, float]]:
    """Resolve detrend method with a simple intermittency heuristic.

    If requested != 'auto', return it verbatim. For 'auto', choose:
    - 'none' for intermittent/low-volume series
    - 'linear' otherwise
    """

    requested = (requested or "auto").strip().lower()
    if requested not in {"auto", "none", "linear", "log1p_linear"}:
        requested = "auto"

    if requested != "auto":
        return requested, {}

    if base is None or base.empty or "unique_id" not in base.columns or "y" not in base.columns:
        return "linear", {}

    stats: list[tuple[float, float]] = []
    for uid, grp in base.groupby("unique_id", sort=False):
        y = pd.to_numeric(grp["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(y) == 0:
            continue
        last12 = y[-12:] if len(y) >= 12 else y
        zero_ratio = float(np.mean(last12 == 0.0)) if len(last12) else 1.0
        nonzero_count = float(np.sum(y > 0.0))
        stats.append((zero_ratio, nonzero_count))

    if not stats:
        return "linear", {}

    zero_ratios = np.array([s[0] for s in stats], dtype=float)
    nonzero_counts = np.array([s[1] for s in stats], dtype=float)
    median_zero_ratio = float(np.median(zero_ratios))
    median_nonzero = float(np.median(nonzero_counts))

    resolved = "none" if (median_zero_ratio >= zero_ratio_threshold or median_nonzero < min_nonzero) else "linear"
    return resolved, {
        "auto_median_zero_ratio_12": median_zero_ratio,
        "auto_median_nonzero_count": median_nonzero,
    }


def _trend_model_value(ds: pd.Timestamp, params: dict[str, Any]) -> float:
    """Trend value in *model space* for a given timestamp.

    - linear: returns trend in y-units
    - log1p_linear: returns trend in log1p(y)-units
    """

    method = str(params.get("method") or "none").strip().lower()
    if method == "none":
        return 0.0
    start_ds = pd.Timestamp(str(params.get("start_ds")))
    t = float(_month_index(pd.Timestamp(ds), start_ds))
    return float(params.get("intercept", 0.0) + params.get("slope", 0.0) * t)


def _build_direct_rows_for_item(
    *,
    unique_id: str,
    ds: np.ndarray,
    y: np.ndarray,
    y_orig: np.ndarray | None,
    trend_params: dict[str, Any] | None,
    exo: dict[str, np.ndarray],
    static: dict[str, Any],
    horizon: int,
    lags: list[int],
    roll_windows: list[int],
    static_interaction_cols: list[str] | None = None,
    global_monthly: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    n = len(y)
    max_needed = max(max(roll_windows) - 1, 1) if roll_windows else 1

    # Precompute per-month historical stats for same-month features (last 36 months).
    same_month_stats: dict[int, dict[str, float]] = {}
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    base_y = y_orig if y_orig is not None else y
    vals_3y = base_y[-36:] if len(base_y) > 36 else base_y
    months_3y = months[-36:] if len(months) > 36 else months
    for m in range(1, 13):
        hist_vals = vals_3y[months_3y == m]
        if len(hist_vals) == 0:
            same_month_stats[m] = {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0}
        else:
            same_month_stats[m] = {
                "mean": float(np.mean(hist_vals)),
                "max": float(np.max(hist_vals)),
                "nonzero_rate": float(np.mean(hist_vals > 0.0)),
            }

    start_ds = pd.Timestamp(ds[0])
    for t in range(0, n - horizon):
        if t < max_needed:
            continue

        forecast_ds = pd.Timestamp(ds[t]) + pd.offsets.MonthBegin(horizon)
        target = float(y[t + horizon])
        if y_orig is not None:
            target_orig = float(y_orig[t + horizon])
        else:
            target_orig = target

        trend_model_at_forecast = 0.0
        if trend_params is not None:
            trend_model_at_forecast = _trend_model_value(forecast_ds, trend_params)

        row: dict[str, Any] = {
            "unique_id": unique_id,
            "decision_ds": pd.Timestamp(ds[t]),
            "forecast_ds": forecast_ds,
            "y": target,
            "y_orig": target_orig,
            "trend_model": float(trend_model_at_forecast),
        }

        m = int(forecast_ds.month)
        row["month"] = m
        row["quarter"] = int(((m - 1) // 3) + 1)
        row["year"] = int(forecast_ds.year)
        # Extended Fourier features (k=3 harmonics)
        fourier_feats = _fourier_features(m, k=3)
        row.update(fourier_feats)
        # Backward compat aliases for single harmonic
        row["month_sin"] = fourier_feats["month_sin_1"]
        row["month_cos"] = fourier_feats["month_cos_1"]
        qs, qc = _quarter_sin_cos(int(row["quarter"]))
        row["quarter_sin"] = qs
        row["quarter_cos"] = qc
        row["year_idx"] = float(forecast_ds.year - start_ds.year)

        # Easter features for paskavara (Easter items)
        easter_feats = _easter_features(forecast_ds)
        row.update(easter_feats)

        # Note: static interaction columns (e.g., item_type_month_sin) are computed by
        # _add_static_interactions() which is called after DataFrame construction.
        # Do not pre-create placeholders here to avoid duplicate columns.

        # Lags are defined as: lag_1 = y[t], lag_2=y[t-1], ...
        for lag in lags:
            idx = t - (lag - 1)
            row[f"lag_{lag}"] = float(y[idx]) if idx >= 0 else 0.0

        if y_orig is not None:
            row["lag_1_orig"] = float(y_orig[t])

        for w in roll_windows:
            start = t - (w - 1)
            window = y[start : t + 1]
            row[f"roll_mean_{w}"] = float(np.mean(window))
            if w >= 2:
                row[f"roll_std_{w}"] = float(np.std(window, ddof=0))

        # Trend-ish
        row["diff1"] = float(y[t] - y[t - 1])
        row["diff12"] = float(y[t] - y[t - 12]) if t >= 12 else 0.0

        # Intermittency
        last12 = y[t - 11 : t + 1]
        row["zero_ratio_12"] = float(np.mean(last12 == 0.0))
        row["nonzero_run_length"] = float(_nonzero_run_length(y[: t + 1]))

        # Conditional-on-nonzero level features (use original scale when available)
        nz_source = y_orig if y_orig is not None else y
        window_nz = nz_source[t - 11 : t + 1]
        nz_vals = window_nz[window_nz > 0.0]
        row["mean_nonzero_12"] = float(np.mean(nz_vals)) if len(nz_vals) else 0.0
        row["median_nonzero_12"] = float(np.median(nz_vals)) if len(nz_vals) else 0.0
        nz_full = nz_source[: t + 1]
        nz_full_vals = nz_full[nz_full > 0.0]
        last_nz_val = float(nz_full_vals[-1]) if len(nz_full_vals) else 0.0
        row["last_nonzero_value"] = float(last_nz_val)

        # Same-month historical features (per item)
        m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
        row["same_month_mean_3y"] = float(m_stats["mean"])
        row["same_month_max_3y"] = float(m_stats["max"])
        row["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
        # Alias: per-item month nonzero rate computed server-side.
        row["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

        mean_last12 = float(np.mean(base_y[t - 11 : t + 1]))
        row["seasonal_amplitude_ratio"] = float(m_stats["max"]) / max(mean_last12, 1.0)

        # Global seasonal components (cross-item learning)
        # Add global month-of-year effect and item vs global ratio
        if global_monthly is not None:
            global_month_level = global_monthly.get(m, 0.0)
            row["global_month_level"] = global_month_level
            # Ratio: how does this item's month pattern compare to global?
            # Use item's same_month_mean vs global median for that month
            item_month_mean = float(m_stats.get("mean", 0.0))
            if global_month_level > 0.0:
                row["item_vs_global_ratio"] = float(item_month_mean / global_month_level)
            else:
                # If global is zero, use item's mean as ratio (or 1.0 if item also zero)
                row["item_vs_global_ratio"] = 1.0 if item_month_mean == 0.0 else float(item_month_mean)
        else:
            # Fallback if global_monthly not provided
            row["global_month_level"] = 0.0
            row["item_vs_global_ratio"] = 1.0

        # Simple decay signal: slope of last 12 months
        row["rolling_12_slope"] = float(_rolling_slope(y[: t + 1]))

        # Exogenous (wide) at time t (known at forecast time)
        for k, arr in exo.items():
            if len(arr) == n:
                row[k] = arr[t]

        # Static
        for k, v in static.items():
            row[k] = v

        rows.append(row)

    return rows


def _encode_categories(
    df: pd.DataFrame,
    cat_cols: list[str],
    mappings: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    out = df.copy()
    out_mappings: dict[str, dict[str, int]] = {} if mappings is None else dict(mappings)

    for col in cat_cols:
        if col not in out.columns:
            continue
        if mappings is None:
            values = out[col].astype(str).fillna("").unique().tolist()
            values = sorted(set(values))
            mapping = {v: i for i, v in enumerate(values)}
            out_mappings[col] = mapping
        else:
            mapping = out_mappings.get(col, {})

        out[col] = out[col].astype(str).fillna("").map(mapping).fillna(-1).astype(int)

    return out, out_mappings


def _add_static_interactions(
    df: pd.DataFrame,
    static_interaction_cols: list[str] | None,
) -> pd.DataFrame:
    if not static_interaction_cols:
        return df
    # Check for Fourier features (new format with harmonics)
    has_fourier_harmonics = "month_sin_1" in df.columns and "month_cos_1" in df.columns
    # Backward compat check
    has_old_format = "month_sin" in df.columns or "month_cos" in df.columns
    if not has_fourier_harmonics and not has_old_format:
        return df
    # Build all new columns first to avoid DataFrame fragmentation
    new_cols: dict[str, pd.Series] = {}
    existing_cols = set(df.columns)
    for col in static_interaction_cols:
        if col not in df.columns:
            continue
        col_vals = df[col].astype(float)
        # Backward compat: old-format interaction columns ({col}_month_sin, {col}_month_cos)
        if "month_sin" in df.columns:
            new_col_name = f"{col}_month_sin"
            if new_col_name not in existing_cols:
                new_cols[new_col_name] = col_vals * df["month_sin"].astype(float)
        if "month_cos" in df.columns:
            new_col_name = f"{col}_month_cos"
            if new_col_name not in existing_cols:
                new_cols[new_col_name] = col_vals * df["month_cos"].astype(float)
        # New format: interactions with all Fourier harmonics (k=3)
        for i in range(1, 4):
            sin_col = f"month_sin_{i}"
            cos_col = f"month_cos_{i}"
            if sin_col in df.columns:
                new_col_name = f"{col}_{sin_col}"
                if new_col_name not in existing_cols:
                    new_cols[new_col_name] = col_vals * df[sin_col].astype(float)
            if cos_col in df.columns:
                new_col_name = f"{col}_{cos_col}"
                if new_col_name not in existing_cols:
                    new_cols[new_col_name] = col_vals * df[cos_col].astype(float)
    # Add all new columns at once using concat to avoid fragmentation
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _add_sample_weights(
    df: pd.DataFrame,
    weight_map: dict[str, float] | None,
) -> pd.DataFrame:
    if df is None or df.empty or not weight_map:
        df["sample_weight"] = 1.0
        return df
    counts = df["unique_id"].astype(str).value_counts().to_dict()
    def _row_weight(uid: str) -> float:
        base = float(weight_map.get(str(uid), 1.0))
        denom = float(counts.get(str(uid), 1.0))
        return base / denom if denom > 0 else base
    df["sample_weight"] = df["unique_id"].astype(str).map(_row_weight).fillna(1.0)
    return df


def _compute_target_encodings(
    base: pd.DataFrame,
    cols: list[str],
    *,
    prior: float = 10.0,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if base is None or base.empty or not cols:
        return {}, base
    df = base.copy()
    global_mean = float(pd.to_numeric(df["y"], errors="coerce").fillna(0.0).mean())
    enc_maps: dict[str, dict[str, Any]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].astype(str).fillna("")
        stats = (
            pd.DataFrame({"key": series, "y": pd.to_numeric(df["y"], errors="coerce").fillna(0.0)})
            .groupby("key", as_index=True)
            .agg(count=("y", "size"), mean=("y", "mean"))
        )
        smooth = (stats["mean"] * stats["count"] + global_mean * float(prior)) / (stats["count"] + float(prior))
        mapping = {str(k): float(v) for k, v in smooth.to_dict().items()}
        enc_maps[col] = {"__global__": global_mean, "mapping": mapping}
        df[f"{col}_te"] = series.map(mapping).fillna(global_mean).astype(float)
    return enc_maps, df


def _apply_target_encodings(
    df: pd.DataFrame,
    te_cols: list[str],
    te_maps: dict[str, Any],
) -> pd.DataFrame:
    if df is None or df.empty or not te_cols or not te_maps:
        return df
    out = df.copy()
    for col in te_cols:
        if col not in out.columns:
            continue
        series = out[col].astype(str).fillna("")
        mapping = dict((te_maps.get(col) or {}).get("mapping") or {})
        global_mean = float((te_maps.get(col) or {}).get("__global__", 0.0))
        out[f"{col}_te"] = series.map(mapping).fillna(global_mean).astype(float)
    return out


def _tokenize_name(value: str) -> list[str]:
    if not value:
        return []
    return [t for t in re.split(r"[^a-z0-9]+", value.lower()) if t]


def _stable_token_bucket(token: str, n_buckets: int) -> int:
    if not token:
        return 0
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % int(n_buckets)


def _build_name_clusters(
    base: pd.DataFrame,
    *,
    name_col: str = "name",
) -> tuple[dict[str, Any], int]:
    if base is None or base.empty or name_col not in base.columns:
        return {"__default__": -1, "mapping": {}}, 0
    names = base[name_col].astype(str).fillna("").unique().tolist()
    names = [n for n in names if n]
    if len(names) < 3:
        return {"__default__": -1, "mapping": {}}, 0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        return {"__default__": -1, "mapping": {}}, 0
    k = int(max(2, min(20, np.sqrt(len(names)))))
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
    X = vectorizer.fit_transform(names)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    mapping = {str(n): int(lbl) for n, lbl in zip(names, labels)}
    return {"__default__": -1, "mapping": mapping}, k


def _apply_name_cluster(
    base: pd.DataFrame,
    cluster_map: dict[str, Any],
    *,
    name_col: str = "name",
) -> pd.DataFrame:
    if base is None or base.empty or name_col not in base.columns:
        return base
    df = base.copy()
    mapping = dict((cluster_map or {}).get("mapping") or {})
    default = int((cluster_map or {}).get("__default__", -1))
    df["name_cluster_id"] = df[name_col].astype(str).fillna("").map(mapping).fillna(default).astype(int)
    return df


def _name_token_bucket_vocab(
    base: pd.DataFrame,
    *,
    name_col: str = "name",
    n_buckets: int = 16,
    top_k: int = 10,
) -> dict[str, list[str]]:
    """Deprecated: token-bucket name features are no longer used.

    Kept for backward compatibility with old specs; returns an empty vocab.
    """
    _ = base, name_col, n_buckets, top_k
    return {}


# ================================================================
# Attribute routing for static item_features
# ================================================================

# Default attribute type classification.
# These are sensible defaults for a typical Icelandic food wholesaler.
DEFAULT_ATTRIBUTE_TYPES: dict[str, list[str]] = {
    "calendar": [
        "jolavara",
        "paskavara",
        "thorravara",
        "bolludagsvara",
        "sprengidagsvara",
        "sumarvara",
    ],
    "trend": [
        "sykurlaus",
        "laktosafritt",
        "glutenfritt",
        "lifraent",
        "vegan",
        "protein",
        "heilsuvara",
    ],
    "segment": [
        "item_group",
        "brand",
        "category",
        "sub_category",
        "supplier_group",
    ],
}


def classify_attribute(
    col_name: str,
    attribute_types: dict[str, list[str]] | None = None,
) -> str:
    """Classify a static attribute into its mechanism type.

    Returns one of: 'calendar', 'trend', 'segment', 'generic'.
    'generic' = treat with standard Fourier interaction (backward compat).
    """
    types = attribute_types or DEFAULT_ATTRIBUTE_TYPES
    for attr_type, cols in types.items():
        if col_name in cols:
            return attr_type
    return "generic"


def route_static_columns(
    static_cols: list[str],
    base: pd.DataFrame,
    attribute_types: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Route all static columns into their mechanism buckets.

    Returns dict with keys: 'calendar', 'trend', 'segment', 'generic',
    each containing a list of column names.

    Columns not present in base are excluded.
    Also excludes internal/derived columns (name_cluster_id, *_te, etc.)
    and the deprecated name_tok_* token buckets.
    """
    types = attribute_types or DEFAULT_ATTRIBUTE_TYPES
    routed: dict[str, list[str]] = {
        "calendar": [],
        "trend": [],
        "segment": [],
        "generic": [],
    }

    skip_prefixes = ("name_tok_",)
    skip_exact = {"name", "flavour", "name_cluster_id"}

    for col in static_cols:
        if col not in base.columns:
            continue
        if any(col.startswith(p) for p in skip_prefixes):
            continue
        if col in skip_exact or col.endswith("_te"):
            continue

        attr_type = classify_attribute(col, types)
        routed[attr_type].append(col)

    return routed


def _calendar_proximity_features(
    forecast_month: int,
    static: dict[str, Any],
    calendar_cols: list[str],
    forecast_ds: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Generate sharp calendar proximity features for seasonal attributes."""
    feats: dict[str, float] = {}
    m = int(forecast_month)

    # Calendar definitions: tuned for Icelandic food wholesale patterns.
    CALENDAR_DEFS: dict[str, dict[str, Any]] = {
        "jolavara": {
            "peak_months": [11, 12],
            "ramp_months": [10],
            "active_window": [10, 11, 12, 1],
            "peak_center": 12,
            "decay_width": 3,
        },
        "paskavara": {
            "peak_months": [3, 4],
            "ramp_months": [2, 3],
            "active_window": [2, 3, 4, 5],
            "peak_center": 4,
            "decay_width": 2,
        },
        "thorravara": {
            "peak_months": [1, 2],
            "ramp_months": [1],
            "active_window": [1, 2],
            "peak_center": 1,
            "decay_width": 1,
        },
        "bolludagsvara": {
            "peak_months": [2],
            "ramp_months": [1, 2],
            "active_window": [1, 2, 3],
            "peak_center": 2,
            "decay_width": 1,
        },
        "sprengidagsvara": {
            "peak_months": [2],
            "ramp_months": [2],
            "active_window": [2, 3],
            "peak_center": 2,
            "decay_width": 1,
        },
        "sumarvara": {
            "peak_months": [6, 7, 8],
            "ramp_months": [5],
            "active_window": [5, 6, 7, 8, 9],
            "peak_center": 7,
            "decay_width": 3,
        },
    }

    for col in calendar_cols:
        val = float(static.get(col, 0) or 0.0)
        if val == 0.0:
            continue

        defn = CALENDAR_DEFS.get(col)
        if defn is None:
            feats[f"{col}_calendar_active"] = val
            continue

        feats[f"{col}_peak"] = val if m in defn["peak_months"] else 0.0
        feats[f"{col}_ramp"] = val if m in defn["ramp_months"] else 0.0
        feats[f"{col}_active_window"] = val if m in defn["active_window"] else 0.0

        peak_center = defn["peak_center"]
        decay_width = defn["decay_width"]
        dist = min(abs(m - peak_center), 12 - abs(m - peak_center))
        proximity = max(0.0, 1.0 - dist / max(decay_width, 1))
        feats[f"{col}_proximity"] = val * proximity

        if dist > decay_width + 1:
            feats[f"{col}_off_season"] = val * 1.0
        else:
            feats[f"{col}_off_season"] = 0.0

    # Easter proximity linked to paskavara flag using existing _easter_features.
    if forecast_ds is not None and float(static.get("paskavara", 0) or 0.0) > 0.0:
        easter_feats = _easter_features(forecast_ds)
        feats["paska_easter_proximity"] = float(static.get("paskavara", 0) or 0.0) * float(
            easter_feats.get("easter_proximity", 0.0)
        )
        feats["paska_pre_easter_4w"] = float(static.get("paskavara", 0) or 0.0) * float(
            easter_feats.get("is_pre_easter_4w", 0.0)
        )

    return feats


def _compute_global_trend_by_attribute(
    base: pd.DataFrame,
    trend_cols: list[str],
    *,
    lookback_months: int = 24,
) -> dict[str, float]:
    """Compute average YoY growth rate for items with each trend attribute."""
    if base is None or base.empty:
        return {f"global_yoy_{c}": 0.0 for c in trend_cols}

    growth_rates: dict[str, float] = {}
    df = base.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    if df.empty:
        return {f"global_yoy_{c}": 0.0 for c in trend_cols}

    max_ds = df["ds"].max()
    cutoff_recent = max_ds - pd.DateOffset(months=12)
    cutoff_prior = max_ds - pd.DateOffset(months=24)

    for col in trend_cols:
        if col not in df.columns:
            growth_rates[f"global_yoy_{col}"] = 0.0
            continue
        flagged = df[pd.to_numeric(df[col], errors="coerce").fillna(0.0) > 0.0]
        if flagged.empty or len(flagged) < 6:
            growth_rates[f"global_yoy_{col}"] = 0.0
            continue

        recent = float(flagged[flagged["ds"] > cutoff_recent]["y"].sum())
        prior = float(
            flagged[(flagged["ds"] > cutoff_prior) & (flagged["ds"] <= cutoff_recent)]["y"].sum()
        )

        if prior > 1.0:
            growth_rates[f"global_yoy_{col}"] = (recent - prior) / prior
        elif recent > 0.0:
            growth_rates[f"global_yoy_{col}"] = 1.0
        else:
            growth_rates[f"global_yoy_{col}"] = 0.0

    return growth_rates


def _trend_attribute_features(
    year_idx: float,
    rolling_12_slope: float,
    static: dict[str, Any],
    trend_cols: list[str],
    global_yoy: dict[str, float] | None = None,
) -> dict[str, float]:
    """Generate trend/lifecycle interaction features for secular attributes."""
    feats: dict[str, float] = {}
    for col in trend_cols:
        val = float(static.get(col, 0) or 0.0)
        if val == 0.0:
            continue

        feats[f"{col}_year_idx"] = val * float(year_idx)
        feats[f"{col}_year_sq"] = val * (float(year_idx) ** 2) * 0.01
        feats[f"{col}_slope"] = val * float(rolling_12_slope)
        if global_yoy is not None:
            yoy_key = f"global_yoy_{col}"
            feats[f"{col}_global_yoy"] = val * float(global_yoy.get(yoy_key, 0.0))
    return feats


def _detect_lifecycle_phase(
    y_orig: np.ndarray,
    *,
    min_history: int = 6,
) -> dict[str, float]:
    """Detect product lifecycle phase from demand history."""
    n = len(y_orig)
    feats: dict[str, float] = {
        "lifecycle_age_months": float(n),
        "lifecycle_is_launch": 0.0,
        "lifecycle_is_growth": 0.0,
        "lifecycle_is_mature": 0.0,
        "lifecycle_is_decline": 0.0,
        "lifecycle_phase": 2.0,
    }

    if n < min_history:
        feats["lifecycle_is_launch"] = 1.0
        feats["lifecycle_phase"] = 0.0
        return feats

    if n <= 6:
        feats["lifecycle_is_launch"] = 1.0
        feats["lifecycle_phase"] = 0.0
        return feats

    if n >= 12:
        recent_6 = y_orig[-6:]
        prior_6 = y_orig[-12:-6]
    else:
        half = n // 2
        recent_6 = y_orig[half:]
        prior_6 = y_orig[:half]

    recent_mean = float(np.mean(recent_6)) if len(recent_6) else 0.0
    prior_mean = float(np.mean(prior_6)) if len(prior_6) else 0.0

    if prior_mean > 1.0:
        change_ratio = (recent_mean - prior_mean) / prior_mean
    elif recent_mean > 0.0:
        change_ratio = 1.0
    else:
        change_ratio = 0.0

    window = y_orig[-12:] if n >= 12 else y_orig
    if len(window) >= 3 and float(np.std(window)) > 1e-12:
        x = np.arange(len(window), dtype=float)
        slope, _ = np.polyfit(x, window, deg=1)
        normalized_slope = slope / max(float(np.mean(window)), 1.0)
    else:
        normalized_slope = 0.0

    if change_ratio > 0.15 and normalized_slope > 0.02:
        feats["lifecycle_is_growth"] = 1.0
        feats["lifecycle_phase"] = 1.0
    elif change_ratio < -0.15 and normalized_slope < -0.02:
        feats["lifecycle_is_decline"] = 1.0
        feats["lifecycle_phase"] = 3.0
    else:
        feats["lifecycle_is_mature"] = 1.0
        feats["lifecycle_phase"] = 2.0

    return feats


def _precompute_segment_monthly_stats(
    base: pd.DataFrame,
    segment_cols: list[str],
    *,
    lookback_months: int = 36,
) -> dict[str, dict[Any, dict[int, dict[str, float]]]]:
    """Precompute per-segment, per-month demand statistics."""
    if base is None or base.empty:
        return {}

    df = base.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    if df.empty:
        return {}

    max_ds = df["ds"].max()
    cutoff = max_ds - pd.DateOffset(months=lookback_months)
    df = df[df["ds"] >= cutoff].copy()
    df["month"] = df["ds"].dt.month
    df["y_num"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

    result: dict[str, dict[Any, dict[int, dict[str, float]]]] = {}
    for col in segment_cols:
        if col not in df.columns:
            continue
        col_stats: dict[Any, dict[int, dict[str, float]]] = {}
        for (seg_val, month), grp in df.groupby([col, "month"], sort=False):
            if pd.isna(seg_val):
                continue
            vals = grp["y_num"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            if seg_val not in col_stats:
                col_stats[seg_val] = {}
            col_stats[seg_val][int(month)] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "count": float(len(vals)),
                "nonzero_rate": float(np.mean(vals > 0.0)),
            }
        result[col] = col_stats
    return result


def _segment_features_for_row(
    forecast_month: int,
    static: dict[str, Any],
    segment_cols: list[str],
    segment_stats: dict[str, dict[Any, dict[int, dict[str, float]]]],
    item_same_month_mean: float,
) -> dict[str, float]:
    """Look up precomputed segment features for a single row."""
    feats: dict[str, float] = {}
    m = int(forecast_month)

    for col in segment_cols:
        val = static.get(col)
        if val is None or pd.isna(val):
            feats[f"{col}_seg_month_mean"] = 0.0
            feats[f"{col}_seg_month_std"] = 0.0
            feats[f"{col}_seg_month_nonzero"] = 0.0
            feats[f"{col}_vs_segment"] = 1.0
            continue
        col_data = segment_stats.get(col, {})
        month_data = col_data.get(val, {}).get(m)
        if month_data is None:
            feats[f"{col}_seg_month_mean"] = 0.0
            feats[f"{col}_seg_month_std"] = 0.0
            feats[f"{col}_seg_month_nonzero"] = 0.0
            feats[f"{col}_vs_segment"] = 1.0
            continue

        seg_mean = float(month_data.get("mean", 0.0))
        feats[f"{col}_seg_month_mean"] = seg_mean
        feats[f"{col}_seg_month_std"] = float(month_data.get("std", 0.0))
        feats[f"{col}_seg_month_nonzero"] = float(month_data.get("nonzero_rate", 0.0))

        if seg_mean > 0.0:
            feats[f"{col}_vs_segment"] = float(item_same_month_mean) / seg_mean
        else:
            feats[f"{col}_vs_segment"] = 1.0

    return feats


def _build_direct_rows_for_item_v2(
    *,
    unique_id: str,
    ds: np.ndarray,
    y: np.ndarray,
    y_orig: np.ndarray | None,
    trend_params: dict[str, Any] | None,
    exo: dict[str, np.ndarray],
    static: dict[str, Any],
    horizon: int,
    lags: list[int],
    roll_windows: list[int],
    calendar_cols: list[str] | None = None,
    trend_cols: list[str] | None = None,
    segment_cols: list[str] | None = None,
    generic_interaction_cols: list[str] | None = None,
    global_monthly: dict[int, float] | None = None,
    global_yoy: dict[str, float] | None = None,
    segment_stats: dict[str, dict[Any, dict[int, dict[str, float]]]] | None = None,
) -> list[dict[str, Any]]:
    """Build supervised rows with three-mechanism attribute routing."""
    rows: list[dict[str, Any]] = []
    calendar_cols = calendar_cols or []
    trend_cols = trend_cols or []
    segment_cols = segment_cols or []
    generic_interaction_cols = generic_interaction_cols or []

    n = len(y)
    max_needed = max(max(roll_windows) - 1, 1) if roll_windows else 1

    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    base_y = y_orig if y_orig is not None else y
    vals_3y = base_y[-36:] if len(base_y) > 36 else base_y
    months_3y = months[-36:] if len(months) > 36 else months
    same_month_stats: dict[int, dict[str, float]] = {}
    for m in range(1, 13):
        hist_vals = vals_3y[months_3y == m]
        if len(hist_vals) == 0:
            same_month_stats[m] = {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0}
        else:
            same_month_stats[m] = {
                "mean": float(np.mean(hist_vals)),
                "max": float(np.max(hist_vals)),
                "nonzero_rate": float(np.mean(hist_vals > 0.0)),
            }

    lifecycle_feats = _detect_lifecycle_phase(base_y if base_y is not None else y)
    start_ds = pd.Timestamp(ds[0])

    for t in range(0, n - horizon):
        if t < max_needed:
            continue

        forecast_ds = pd.Timestamp(ds[t]) + pd.offsets.MonthBegin(horizon)
        target = float(y[t + horizon])
        target_orig = float(y_orig[t + horizon]) if y_orig is not None else target

        trend_model_at_forecast = 0.0
        if trend_params is not None:
            trend_model_at_forecast = _trend_model_value(forecast_ds, trend_params)

        row: dict[str, Any] = {
            "unique_id": unique_id,
            "decision_ds": pd.Timestamp(ds[t]),
            "forecast_ds": forecast_ds,
            "y": target,
            "y_orig": target_orig,
            "trend_model": float(trend_model_at_forecast),
        }

        m = int(forecast_ds.month)
        row["month"] = m
        row["quarter"] = int(((m - 1) // 3) + 1)
        row["year"] = int(forecast_ds.year)

        fourier_feats = _fourier_features(m, k=3)
        row.update(fourier_feats)
        row["month_sin"] = fourier_feats["month_sin_1"]
        row["month_cos"] = fourier_feats["month_cos_1"]
        qs, qc = _quarter_sin_cos(int(row["quarter"]))
        row["quarter_sin"] = qs
        row["quarter_cos"] = qc
        row["year_idx"] = float(forecast_ds.year - start_ds.year)

        easter_feats = _easter_features(forecast_ds)
        row.update(easter_feats)

        if calendar_cols:
            cal_feats = _calendar_proximity_features(
                m,
                static,
                calendar_cols,
                forecast_ds=forecast_ds,
            )
            row.update(cal_feats)

        if trend_cols:
            rolling_slope = float(_rolling_slope(y[: t + 1]))
            trend_feats = _trend_attribute_features(
                year_idx=row["year_idx"],
                rolling_12_slope=rolling_slope,
                static=static,
                trend_cols=trend_cols,
                global_yoy=global_yoy,
            )
            row.update(trend_feats)

        if segment_cols and segment_stats:
            item_month_mean = float(same_month_stats.get(m, {}).get("mean", 0.0))
            seg_feats = _segment_features_for_row(
                m,
                static,
                segment_cols,
                segment_stats,
                item_month_mean,
            )
            row.update(seg_feats)

        row.update(lifecycle_feats)

        for lag in lags:
            idx = t - (lag - 1)
            row[f"lag_{lag}"] = float(y[idx]) if idx >= 0 else 0.0

        if y_orig is not None:
            row["lag_1_orig"] = float(y_orig[t])

        for w in roll_windows:
            start = t - (w - 1)
            window = y[start : t + 1]
            row[f"roll_mean_{w}"] = float(np.mean(window))
            if w >= 2:
                row[f"roll_std_{w}"] = float(np.std(window, ddof=0))

        row["diff1"] = float(y[t] - y[t - 1])
        row["diff12"] = float(y[t] - y[t - 12]) if t >= 12 else 0.0

        last12 = y[t - 11 : t + 1]
        row["zero_ratio_12"] = float(np.mean(last12 == 0.0))
        row["nonzero_run_length"] = float(_nonzero_run_length(y[: t + 1]))

        nz_source = y_orig if y_orig is not None else y
        window_nz = nz_source[t - 11 : t + 1]
        nz_vals = window_nz[window_nz > 0.0]
        row["mean_nonzero_12"] = float(np.mean(nz_vals)) if len(nz_vals) else 0.0
        row["median_nonzero_12"] = float(np.median(nz_vals)) if len(nz_vals) else 0.0
        nz_full = nz_source[: t + 1]
        nz_full_vals = nz_full[nz_full > 0.0]
        last_nz_val = float(nz_full_vals[-1]) if len(nz_full_vals) else 0.0
        row["last_nonzero_value"] = float(last_nz_val)

        m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
        row["same_month_mean_3y"] = float(m_stats["mean"])
        row["same_month_max_3y"] = float(m_stats["max"])
        row["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
        row["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

        mean_last12 = float(np.mean(base_y[t - 11 : t + 1]))
        row["seasonal_amplitude_ratio"] = float(m_stats["max"]) / max(mean_last12, 1.0)

        if global_monthly is not None:
            global_month_level = global_monthly.get(m, 0.0)
            row["global_month_level"] = global_month_level
            item_month_mean = float(m_stats.get("mean", 0.0))
            if global_month_level > 0.0:
                row["item_vs_global_ratio"] = float(item_month_mean / global_month_level)
            else:
                row["item_vs_global_ratio"] = 1.0 if item_month_mean == 0.0 else float(item_month_mean)
        else:
            row["global_month_level"] = 0.0
            row["item_vs_global_ratio"] = 1.0

        row["rolling_12_slope"] = float(_rolling_slope(y[: t + 1]))

        for k, arr in exo.items():
            if len(arr) == n:
                row[k] = arr[t]

        for k, v in static.items():
            row[k] = v

        rows.append(row)

    return rows


def _build_statsforecast_features(
    base: pd.DataFrame,
    *,
    max_h: int,
    season_length: int = 12,
) -> pd.DataFrame:
    """Build per-item StatsForecast features via rolling CV (leak-safe)."""
    if base is None or base.empty:
        return pd.DataFrame()
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, Theta, CrostonOptimized, ADIDA
    except Exception:
        return pd.DataFrame()

    df = base.loc[:, ["unique_id", "ds", "y"]].copy()
    df = df.dropna(subset=["unique_id", "ds"])
    if df.empty:
        return pd.DataFrame()

    counts = df["unique_id"].value_counts()
    eligible = counts[counts >= max(3, int(max_h) + 2)].index.tolist()
    if not eligible:
        return pd.DataFrame()
    df = df[df["unique_id"].isin(eligible)].copy()

    min_obs = int(counts.loc[eligible].min())
    n_windows = max(1, min(12, min_obs - int(max_h)))

    models = [
        CrostonOptimized(),
        ADIDA(),
        AutoETS(season_length=int(season_length)),
        Theta(season_length=int(season_length)),
    ]
    sf = StatsForecast(models=models, freq="MS", n_jobs=1)
    try:
        cv = sf.cross_validation(
            df=df,
            h=int(max_h),
            step_size=1,
            n_windows=int(n_windows),
        )
    except Exception:
        return pd.DataFrame()

    rename = {
        "CrostonOptimized": "sf_croston_optimized",
        "ADIDA": "sf_adida",
        "AutoETS": "sf_auto_ets",
        "Theta": "sf_theta",
    }
    keep = ["unique_id", "ds"] + [c for c in rename.keys() if c in cv.columns]
    out = cv.loc[:, keep].copy()
    out["sf_available"] = 1.0
    out = out.rename(columns=rename)
    out = out.dropna(subset=["unique_id", "ds"]).reset_index(drop=True)
    return out


def _build_statsforecast_future_features(
    base: pd.DataFrame,
    *,
    max_h: int,
    season_length: int = 12,
) -> pd.DataFrame:
    """Build per-item StatsForecast features for future horizons."""
    if base is None or base.empty:
        return pd.DataFrame()
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, Theta, CrostonOptimized, ADIDA
    except Exception:
        return pd.DataFrame()

    df = base.loc[:, ["unique_id", "ds", "y"]].copy()
    df = df.dropna(subset=["unique_id", "ds"])
    if df.empty:
        return pd.DataFrame()

    models = [
        CrostonOptimized(),
        ADIDA(),
        AutoETS(season_length=int(season_length)),
        Theta(season_length=int(season_length)),
    ]
    sf = StatsForecast(models=models, freq="MS", n_jobs=1)
    try:
        fcst = sf.forecast(df=df, h=int(max_h))
    except Exception:
        return pd.DataFrame()

    rename = {
        "CrostonOptimized": "sf_croston_optimized",
        "ADIDA": "sf_adida",
        "AutoETS": "sf_auto_ets",
        "Theta": "sf_theta",
    }
    keep = ["unique_id", "ds"] + [c for c in rename.keys() if c in fcst.columns]
    out = fcst.loc[:, keep].copy()
    out["sf_available"] = 1.0
    out = out.rename(columns=rename)
    out = out.dropna(subset=["unique_id", "ds"]).reset_index(drop=True)
    return out


# -------------------------
# Public interface
# -------------------------


@dataclass
class TrainResult:
    customer_id: str
    model_version: str
    status: str
    artifact_root: str
    feature_spec_hash: str
    items_trained: int
    rows_trained: int
    metrics_summary: dict[str, Any]


class LightGBMForecast:
    """Monthly-only LightGBM forecaster.

    Defaults:
    - Direct strategy (one model per horizon step)
    - Uses wide-format regressors present in `sim_input_his`
    - Support series: Option A (global aggregates) by joining as wide columns
    """

    def __init__(self, store_root: str, customer_id: str):
        if not _HAS_LGBM:
            detail = _LGBM_IMPORT_ERROR or "unknown import error"
            raise RuntimeError(
                "lightgbm could not be imported. "
                "On macOS this is often a missing OpenMP runtime (libomp). "
                f"Import error: {detail}"
            )
        self.store = DuckDBModelStore(store_root=store_root, customer_id=customer_id)

    def train_and_register(
        self,
        hist: pd.DataFrame,
        *,
        item_attributes: pd.DataFrame | None = None,
        drivers: pd.DataFrame | None = None,
        freq: str = "M",
        horizon: int = 12,
        exogenous_columns: list[str] | None = None,
        detrend_method: str = "none",
        model_version: str | None = None,
        status: str = "staging",
        notes: str | None = None,
        min_history_points: int = 6,
        min_improvement: float = 0.0,
        lgbm_min_data_in_leaf: int = 50,
        lgbm_min_data_in_bin: int = 1,
        val_months: int = 6,
        tune_hyperparameters: bool = False,
        optuna_n_trials: int = 30,
        optuna_timeout: int = 300,
        progress_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> TrainResult:
        # Monthly-only for v1
        _ = freq  # kept for signature compatibility

        model_version = model_version or datetime.now(UTC).strftime("v%Y_%m_%d_%H%M%S")

        if progress_hook:
            progress_hook(
                {
                    "phase": "validating",
                    "model_version": model_version,
                    "horizon": int(horizon),
                }
            )

        # Normalize history to monthly (month start)
        if progress_hook:
            progress_hook({"phase": "aggregating_monthly"})
        base = _normalize_monthly_history(hist)

        # Merge item attributes (static)
        static_cols: list[str] = []
        if item_attributes is not None and not item_attributes.empty and "item_id" in item_attributes.columns:
            attrs = item_attributes.copy()
            attrs["item_id"] = attrs["item_id"].astype(str)
            base["item_id"] = base["item_id"].astype(str)
            base = base.merge(attrs, on="item_id", how="left", suffixes=("", "_attr"))
            static_cols = [c for c in attrs.columns if c != "item_id"]

        # Name-based features: TF-IDF cluster id only (token buckets removed)
        name_cluster_map, name_cluster_k = _build_name_clusters(base, name_col="name")
        base = _apply_name_cluster(base, name_cluster_map, name_col="name")
        if "name_cluster_id" not in static_cols:
            static_cols.append("name_cluster_id")


        # Merge long drivers (optional) → wide monthly
        if drivers is not None and not drivers.empty:
            wide = _pivot_long_drivers(drivers)
            if not wide.empty:
                if "item_id" in wide.columns:
                    wide["item_id"] = wide["item_id"].astype(str)
                    base = base.merge(wide, on=["item_id", "ds"], how="left")
                else:
                    base = base.merge(wide, on=["ds"], how="left")

        # Determine exogenous columns from wide data
        base_cols = {"item_id", "unique_id", "ds", "y"}
        if exogenous_columns is None:
            exogenous_columns = [c for c in base.columns if c not in base_cols and c not in set(static_cols)]
        else:
            exogenous_columns = [c for c in exogenous_columns if c in base.columns]

        resolved_detrend_method, auto_stats = _resolve_detrend_method(base, detrend_method)
        detrend_method = resolved_detrend_method

        # Feature recipe (monthly)
        lags = [1, 2, 3, 6, 12, 24]
        roll_windows = [3, 6, 12]

        # Route static attributes into calendar/trend/segment/generic buckets
        attribute_types = DEFAULT_ATTRIBUTE_TYPES
        routed = route_static_columns(static_cols, base, attribute_types)
        calendar_cols = routed["calendar"]
        trend_cols = routed["trend"]
        segment_cols = routed["segment"]
        generic_interaction_cols = routed["generic"]

        # Low-cardinality static categories for month interactions (generic only)
        static_cat_cols = [
            c
            for c in static_cols
            if c in base.columns
            and (pd.api.types.is_object_dtype(base[c]) or pd.api.types.is_categorical_dtype(base[c]))
        ]
        static_interaction_cols: list[str] = []
        for c in generic_interaction_cols:
            if c not in base.columns:
                continue
            unique_count = int(base[c].nunique(dropna=True))
            if unique_count <= 50:
                static_interaction_cols.append(c)

        if progress_hook:
            progress_hook({"phase": "building_features"})

        # Build supervised rows across all items for each horizon
        all_rows_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in range(1, int(horizon) + 1)}

        detrend_method = (detrend_method or "none").strip().lower()
        if detrend_method not in {"none", "linear", "log1p_linear"}:
            raise ValueError("detrend_method must be one of: none, linear, log1p_linear")

        trend_params_by_uid: dict[str, dict[str, Any]] = {}

        base = base.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
        demand_by_uid = (
            base.groupby("unique_id")["y"].sum().astype(float).to_dict()
            if not base.empty else {}
        )
        weight_map = {str(uid): 1.0 / (float(total) + 1.0) for uid, total in demand_by_uid.items()}
        if weight_map:
            weights = np.array(list(weight_map.values()), dtype=float)
            w_med = float(np.median(weights)) if len(weights) else 1.0
            if w_med <= 0:
                w_med = 1.0
            for uid in list(weight_map.keys()):
                weight_map[uid] = float(np.clip(weight_map[uid] / w_med, 0.5, 3.0))

        # Compute global monthly effects across all items (for cross-item learning)
        # This helps static features like item_group, flavour, etc. by providing
        # a baseline seasonal pattern that items can deviate from
        global_monthly_effects = _compute_global_monthly_effects(
            base,
            lookback_months=36,
        )

        # Global trend stats for trend attributes
        global_yoy = _compute_global_trend_by_attribute(base, trend_cols)

        # Segment-level per-month stats
        segment_stats = _precompute_segment_monthly_stats(
            base,
            segment_cols,
            lookback_months=36,
        )

        # Archetype assignment for residual pooling (computed on full history)
        archetype_by_uid: dict[str, str] = {}
        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            y_hist = grp["y"].to_numpy(dtype=float)
            ds_hist = grp["ds"].to_numpy()
            months = np.array([pd.Timestamp(d).month for d in ds_hist], dtype=int)
            vals_3y = y_hist[-36:] if len(y_hist) > 36 else y_hist
            months_3y = months[-36:] if len(months) > 36 else months
            same_month_nonzero = []
            for m in range(1, 13):
                hist_vals = vals_3y[months_3y == m]
                same_month_nonzero.append(float(np.mean(hist_vals > 0.0)) if len(hist_vals) else 0.0)
            seasonal_strength = float(max(same_month_nonzero or [0.0]))
            overall_nonzero_rate = float(np.mean(vals_3y > 0.0)) if len(vals_3y) else 0.0
            cv_val = _cv(vals_3y if len(vals_3y) else y_hist)
            archetype_by_uid[str(uid)] = _archetype(overall_nonzero_rate, cv_val, seasonal_strength)
        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            ds = grp["ds"].to_numpy()
            y_orig = grp["y"].to_numpy(dtype=float)

            trend_params = _fit_trend_params(ds, y_orig, detrend_method)
            trend_params_by_uid[str(uid)] = trend_params

            if detrend_method == "none":
                y_model = y_orig
            elif detrend_method == "linear":
                trend = np.array([_trend_model_value(pd.Timestamp(d), trend_params) for d in ds], dtype=float)
                y_model = y_orig - trend
            else:  # log1p_linear
                yt = np.log1p(np.maximum(0.0, y_orig))
                trend = np.array([_trend_model_value(pd.Timestamp(d), trend_params) for d in ds], dtype=float)
                y_model = yt - trend

            exo = {c: pd.to_numeric(grp[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in exogenous_columns}

            static: dict[str, Any] = {}
            for c in static_cols:
                # take last known value as static
                v = grp[c].iloc[-1] if c in grp.columns else None
                static[c] = v

            for h in range(1, int(horizon) + 1):
                rows = _build_direct_rows_for_item_v2(
                    unique_id=str(uid),
                    ds=ds,
                    y=y_model,
                    y_orig=y_orig,
                    trend_params=trend_params,
                    exo=exo,
                    static=static,
                    horizon=h,
                    lags=lags,
                    roll_windows=roll_windows,
                    calendar_cols=calendar_cols,
                    trend_cols=trend_cols,
                    segment_cols=segment_cols,
                    generic_interaction_cols=generic_interaction_cols,
                    global_monthly=global_monthly_effects,
                    global_yoy=global_yoy,
                    segment_stats=segment_stats,
                )
                all_rows_by_h[h].extend(rows)

        # Add per-item StatsForecast features (Croston/ADIDA/ETS/Theta)
        sf_feats = _build_statsforecast_features(base, max_h=int(horizon), season_length=12)
        if not sf_feats.empty:
            sf_feats = sf_feats.rename(columns={"ds": "forecast_ds"})
            coverage = float(sf_feats["unique_id"].nunique()) / float(base["unique_id"].nunique() or 1)
            for h in range(1, int(horizon) + 1):
                rows = all_rows_by_h.get(h) or []
                if not rows:
                    continue
                df_h = pd.DataFrame(rows)
                if coverage >= 0.3:
                    df_h = df_h.merge(sf_feats, on=["unique_id", "forecast_ds"], how="left")
                    if "sf_available" not in df_h.columns:
                        df_h["sf_available"] = 0.0
                    for c in ["sf_croston_optimized", "sf_adida", "sf_auto_ets", "sf_theta"]:
                        if c in df_h.columns:
                            df_h[c] = df_h[c].where(df_h["sf_available"] > 0.0, 0.0)
                    df_h = df_h.fillna(0.0)
                else:
                    for c in ["sf_croston_optimized", "sf_adida", "sf_auto_ets", "sf_theta", "sf_available"]:
                        df_h[c] = 0.0
                all_rows_by_h[h] = df_h.to_dict(orient="records")

        # Train one model per horizon
        artifact_root = os.path.join(self.store.customer_dir(), "artifacts", model_version)
        os.makedirs(artifact_root, exist_ok=True)

        metrics_summary: dict[str, Any] = {
            "strategy": "direct",
            "horizon": int(horizon),
            "val_months": int(val_months),
            "detrend_method": detrend_method,
        }
        if auto_stats:
            metrics_summary.update(auto_stats)

        models: dict[int, Any] = {}
        feature_cols: list[str] | None = None

        # Only treat truly categorical static columns as categorical.
        cat_cols = ["unique_id", *static_cat_cols]

        # Build stable category mappings once (reused across horizons)
        mappings: dict[str, dict[str, int]] = {}
        for col in cat_cols:
            values = base[col].astype(str).fillna("").unique().tolist()
            values = sorted(set(values))
            mappings[col] = {v: i for i, v in enumerate(values)}

        # Determine global validation cutoff based on forecast_ds
        max_forecast_ds = None
        for h, rows in all_rows_by_h.items():
            if rows:
                dmax = max(r["forecast_ds"] for r in rows)
                if max_forecast_ds is None or dmax > max_forecast_ds:
                    max_forecast_ds = dmax
        if max_forecast_ds is None:
            raise ValueError("No training rows after feature generation (check history length).")

        cutoff = pd.Timestamp(max_forecast_ds) - pd.offsets.MonthBegin(int(val_months - 1))

        # Target-encode high-cardinality static fields (train-split only to avoid leakage)
        te_cols = [c for c in static_cols if c in {"name", "flavour"}]
        te_maps, _ = _compute_target_encodings(base[base["ds"] < cutoff].copy(), te_cols, prior=10.0)
        base = _apply_target_encodings(base, te_cols, te_maps)
        te_feature_cols = [f"{c}_te" for c in te_cols]
        static_cols = static_cols + [c for c in te_feature_cols if c not in static_cols]

        # For eligibility: evaluate horizon 1 per-item
        per_item_eval: dict[str, dict[str, float]] = {}
        cv_residual_rows: list[dict[str, Any]] = []

        # Resolve feature columns once from the first non-empty horizon
        # base_feature_cols: columns from raw rows (used for selecting from raw DataFrames)
        # feature_cols: full list including interaction columns (used for training/spec)
        base_feature_cols: list[str] | None = None
        if feature_cols is None:
            for rows in all_rows_by_h.values():
                if rows:
                    df_spec = pd.DataFrame(rows)
                    excluded = {"y", "decision_ds", "forecast_ds", "y_orig", "trend_model", "lag_1_orig"}
                    base_feature_cols = [c for c in df_spec.columns if c not in excluded]
                    feature_cols = base_feature_cols.copy()
                    break
        if not feature_cols or not base_feature_cols:
            raise ValueError("Could not determine feature columns (no training rows).")

        # Train shared occurrence model (y > 0) across all horizons
        occ_rows: list[dict[str, Any]] = []
        for rows in all_rows_by_h.values():
            if rows:
                occ_rows.extend(rows)
        occ_model: Any | None = None
        if occ_rows:
            occ_df = pd.DataFrame(occ_rows)
            occ_df = _add_sample_weights(occ_df, weight_map)
            y_occ_src = occ_df["y_orig"] if "y_orig" in occ_df.columns else occ_df["y"]
            y_occ = (pd.to_numeric(y_occ_src, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0).astype(int)
            is_val_occ = pd.to_datetime(occ_df["forecast_ds"]) >= cutoff

            X_occ = occ_df[base_feature_cols].copy()
            X_occ["unique_id"] = X_occ["unique_id"].astype(str)
            X_occ, _ = _encode_categories(X_occ, cat_cols, mappings=mappings)
            X_occ = _add_static_interactions(X_occ, static_interaction_cols)
            X_occ = X_occ.fillna(0)

            # IMPORTANT: Update feature_cols to include interaction columns added by _add_static_interactions
            # This ensures the spec's feature_columns matches what the model was actually trained on
            feature_cols = list(X_occ.columns)

            X_occ_train = X_occ[~is_val_occ]
            y_occ_train = y_occ[~is_val_occ]
            X_occ_val = X_occ[is_val_occ]
            y_occ_val = y_occ[is_val_occ]
            w_occ = occ_df["sample_weight"].to_numpy(dtype=float)
            w_occ_train = w_occ[~is_val_occ]
            w_occ_val = w_occ[is_val_occ]

            if len(X_occ_train) >= 50 and y_occ_train.sum() > 0:
                pos = float(y_occ_train.sum())
                neg = float(len(y_occ_train) - pos)
                occ_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "min_data_in_leaf": int(max(1, lgbm_min_data_in_leaf)),
                    "min_data_in_bin": int(max(1, lgbm_min_data_in_bin)),
                    "seed": 42,
                    "verbosity": -1,
                }
                if pos > 0 and neg > 0:
                    occ_params["scale_pos_weight"] = float(neg / pos)

                dtrain_occ = lgb.Dataset(
                    X_occ_train,
                    label=y_occ_train,
                    weight=w_occ_train,
                    categorical_feature=[c for c in cat_cols if c in X_occ_train.columns],
                    free_raw_data=False,
                )
                valid_sets = [dtrain_occ]
                valid_names = ["train"]
                if len(X_occ_val) > 0:
                    dval_occ = lgb.Dataset(
                        X_occ_val,
                        label=y_occ_val,
                        weight=w_occ_val,
                        categorical_feature=[c for c in cat_cols if c in X_occ_val.columns],
                        free_raw_data=False,
                    )
                    valid_sets.append(dval_occ)
                    valid_names.append("val")

                occ_model = lgb.train(
                    occ_params,
                    dtrain_occ,
                    num_boost_round=1000,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=[lgb.early_stopping(50, verbose=False)] if len(X_occ_val) > 0 else None,
                )
                occ_model.save_model(os.path.join(artifact_root, "lgbm_occurrence.txt"))
                metrics_summary["occurrence_model"] = "shared"
                metrics_summary["occurrence_threshold"] = 0.2
                metrics_summary["occurrence_pos_rate_train"] = float(pos / max(1.0, pos + neg))

        # Optuna hyperparameter tuning (using horizon 1 data)
        tuned_params: dict[str, Any] | None = None
        if tune_hyperparameters and _HAS_OPTUNA and all_rows_by_h.get(1):
            if progress_hook:
                progress_hook({"phase": "tuning_hyperparameters", "n_trials": optuna_n_trials})
            tune_df = pd.DataFrame(all_rows_by_h[1])
            tune_df = _add_sample_weights(tune_df, weight_map)
            is_tune_val = pd.to_datetime(tune_df["forecast_ds"]) >= cutoff
            X_tune = tune_df[base_feature_cols].copy()
            X_tune["unique_id"] = X_tune["unique_id"].astype(str)
            X_tune, _ = _encode_categories(X_tune, cat_cols, mappings=mappings)
            X_tune = _add_static_interactions(X_tune, static_interaction_cols)
            X_tune = X_tune.fillna(0)
            y_tune = tune_df["y"].to_numpy(dtype=float)

            X_tune_train = X_tune[~is_tune_val]
            y_tune_train = y_tune[~is_tune_val]
            X_tune_val = X_tune[is_tune_val]
            y_tune_val = y_tune[is_tune_val]

            if len(X_tune_train) >= 50 and len(X_tune_val) >= 10:
                tune_objective = "tweedie" if detrend_method == "none" else "regression"
                tuned_params = _tune_lgbm_hyperparameters(
                    X_tune_train, y_tune_train,
                    X_tune_val, y_tune_val,
                    objective=tune_objective,
                    n_trials=optuna_n_trials,
                    timeout_seconds=optuna_timeout,
                )
                metrics_summary["tuned_hyperparameters"] = tuned_params
                if progress_hook:
                    progress_hook({"phase": "tuning_complete", "best_params": tuned_params})

        for h in range(1, int(horizon) + 1):
            rows = all_rows_by_h[h]
            if not rows:
                continue

            if progress_hook:
                progress_hook(
                    {
                        "phase": "training_horizon",
                        "current_horizon": int(h),
                        "total_horizons": int(horizon),
                    }
                )

            df = pd.DataFrame(rows)
            df = _add_sample_weights(df, weight_map)
            # Keep the label name as y
            y_target = df["y"].to_numpy(dtype=float)

            # Time split by forecast_ds
            is_val = pd.to_datetime(df["forecast_ds"]) >= cutoff

            X = df[base_feature_cols].copy()
            X["unique_id"] = X["unique_id"].astype(str)

            X, _ = _encode_categories(X, cat_cols, mappings=mappings)
            X = _add_static_interactions(X, static_interaction_cols)

            # Ensure numeric
            X = X.fillna(0)

            X_train = X[~is_val]
            y_train = y_target[~is_val]
            X_val = X[is_val]
            y_val = y_target[is_val]
            w_all = df["sample_weight"].to_numpy(dtype=float)
            w_train = w_all[~is_val]
            w_val = w_all[is_val]

            if len(X_train) < 50:
                raise ValueError("Not enough training rows to fit LightGBM. Provide more history or more items.")

            if detrend_method == "none":
                objective = "tweedie"
            else:
                # Detrended targets can be negative; use regression.
                objective = "regression"

            # Use tuned params if available, otherwise defaults
            if tuned_params:
                params = {
                    "objective": objective,
                    "metric": "l2",
                    "learning_rate": tuned_params.get("learning_rate", 0.05),
                    "num_leaves": tuned_params.get("num_leaves", 63),
                    "feature_fraction": tuned_params.get("feature_fraction", 0.8),
                    "bagging_fraction": tuned_params.get("bagging_fraction", 0.8),
                    "bagging_freq": tuned_params.get("bagging_freq", 1),
                    "min_data_in_leaf": int(max(1, tuned_params.get("min_data_in_leaf", lgbm_min_data_in_leaf))),
                    "min_data_in_bin": int(max(1, lgbm_min_data_in_bin)),
                    "lambda_l1": tuned_params.get("lambda_l1", 0.0),
                    "lambda_l2": tuned_params.get("lambda_l2", 0.0),
                    "seed": 42,
                    "verbosity": -1,
                }
            else:
                params = {
                    "objective": objective,
                    "metric": "l2",
                    "learning_rate": 0.05,
                    "num_leaves": 63,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "min_data_in_leaf": int(max(1, lgbm_min_data_in_leaf)),
                    "min_data_in_bin": int(max(1, lgbm_min_data_in_bin)),
                    "seed": 42,
                    "verbosity": -1,
                }
            if objective == "tweedie":
                tweedie_power = tuned_params.get("tweedie_variance_power", 1.3) if tuned_params else 1.3
                params["tweedie_variance_power"] = tweedie_power

            dtrain = lgb.Dataset(
                X_train,
                label=y_train,
                weight=w_train,
                categorical_feature=[c for c in cat_cols if c in X_train.columns],
                free_raw_data=False,
            )
            valid_sets = [dtrain]
            valid_names = ["train"]
            if len(X_val) > 0:
                dval = lgb.Dataset(
                    X_val,
                    label=y_val,
                    weight=w_val,
                    categorical_feature=[c for c in cat_cols if c in X_val.columns],
                    free_raw_data=False,
                )
                valid_sets.append(dval)
                valid_names.append("val")

            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.early_stopping(100, verbose=False)] if len(X_val) > 0 else None,
            )

            models[h] = booster

            # Train quantile model (0.95) for upper bound prediction
            # Use y_orig (original scale) to learn full distribution including trend
            if objective == "tweedie" or objective == "regression":
                # Prepare target: use y_orig if available (original scale), otherwise y
                y_q95 = df["y_orig"].to_numpy(dtype=float) if "y_orig" in df.columns else df["y"].to_numpy(dtype=float)
                y_q95_train = y_q95[~is_val]
                y_q95_val = y_q95[is_val] if len(X_val) > 0 else np.array([])

                # Use same features and weights, but original-scale target
                params_q95 = dict(params)
                params_q95["objective"] = "quantile"
                params_q95["alpha"] = 0.95
                params_q95["metric"] = "quantile"

                dtrain_q95 = lgb.Dataset(
                    X_train,
                    label=y_q95_train,
                    weight=w_train,
                    categorical_feature=[c for c in cat_cols if c in X_train.columns],
                    free_raw_data=False,
                )
                valid_sets_q95 = [dtrain_q95]
                valid_names_q95 = ["train"]
                if len(X_val) > 0:
                    dval_q95 = lgb.Dataset(
                        X_val,
                        label=y_q95_val,
                        weight=w_val,
                        categorical_feature=[c for c in cat_cols if c in X_val.columns],
                        free_raw_data=False,
                    )
                    valid_sets_q95.append(dval_q95)
                    valid_names_q95.append("val")

                booster_q95 = lgb.train(
                    params_q95,
                    dtrain_q95,
                    num_boost_round=2000,
                    valid_sets=valid_sets_q95,
                    valid_names=valid_names_q95,
                    callbacks=[lgb.early_stopping(100, verbose=False)] if len(X_val) > 0 else None,
                )
                # Save to separate file - does NOT overwrite point model
                booster_q95.save_model(os.path.join(artifact_root, f"lgbm_h{h}_q95.txt"))

                # Log validation metrics for quantile model calibration
                if len(X_val) > 0:
                    # Predict quantile on validation set
                    q95_val_pred = booster_q95.predict(X_val)

                    # Get true values on original scale (same as quantile model target)
                    y_val_true_orig = df.loc[is_val, "y_orig"].to_numpy(dtype=float) if "y_orig" in df.columns else y_val

                    # Calculate coverage: fraction of validation samples where true value <= predicted quantile
                    # For a well-calibrated 95th percentile model, coverage should be close to 0.95
                    coverage_95 = float(np.mean(y_val_true_orig <= q95_val_pred))
                    metrics_summary[f"q95_coverage_val_h{h}"] = coverage_95

            # Train magnitude model on nonzero rows only (E[y | y > 0])
            nz_mask = pd.to_numeric(df.get("y_orig", df["y"]), errors="coerce").fillna(0.0) > 0.0
            if bool(nz_mask.any()):
                df_nz = df.loc[nz_mask].copy()
                df_nz = _add_sample_weights(df_nz, weight_map)
                if len(df_nz) >= 50:
                    y_nz = df_nz["y_orig"].to_numpy(dtype=float) if "y_orig" in df_nz.columns else df_nz["y"].to_numpy(dtype=float)
                    is_val_nz = pd.to_datetime(df_nz["forecast_ds"]) >= cutoff
                    X_nz = df_nz[base_feature_cols].copy()
                    X_nz["unique_id"] = X_nz["unique_id"].astype(str)
                    X_nz, _ = _encode_categories(X_nz, cat_cols, mappings=mappings)
                    X_nz = _add_static_interactions(X_nz, static_interaction_cols)
                    X_nz = X_nz.fillna(0)

                    X_nz_train = X_nz[~is_val_nz]
                    y_nz_train = y_nz[~is_val_nz]
                    X_nz_val = X_nz[is_val_nz]
                    y_nz_val = y_nz[is_val_nz]
                    w_nz = df_nz["sample_weight"].to_numpy(dtype=float)
                    w_nz_train = w_nz[~is_val_nz]
                    w_nz_val = w_nz[is_val_nz]

                    all_nonneg_nz = bool(np.all(y_nz_train >= 0) and np.all(y_nz_val >= 0))
                    objective_nz = "tweedie" if all_nonneg_nz else "regression"
                    params_nz = dict(params)
                    params_nz["objective"] = objective_nz
                    params_nz["metric"] = "l2"
                    if objective_nz == "tweedie":
                        params_nz["tweedie_variance_power"] = 1.3

                    dtrain_nz = lgb.Dataset(
                        X_nz_train,
                        label=y_nz_train,
                        weight=w_nz_train,
                        categorical_feature=[c for c in cat_cols if c in X_nz_train.columns],
                        free_raw_data=False,
                    )
                    valid_sets_nz = [dtrain_nz]
                    valid_names_nz = ["train"]
                    if len(X_nz_val) > 0:
                        dval_nz = lgb.Dataset(
                            X_nz_val,
                            label=y_nz_val,
                            weight=w_nz_val,
                            categorical_feature=[c for c in cat_cols if c in X_nz_val.columns],
                            free_raw_data=False,
                        )
                        valid_sets_nz.append(dval_nz)
                        valid_names_nz.append("val")

                    nz_booster = lgb.train(
                        params_nz,
                        dtrain_nz,
                        num_boost_round=2000,
                        valid_sets=valid_sets_nz,
                        valid_names=valid_names_nz,
                        callbacks=[lgb.early_stopping(100, verbose=False)] if len(X_nz_val) > 0 else None,
                    )
                    nz_booster.save_model(os.path.join(artifact_root, f"lgbm_h{h}_nz.txt"))

            # Metrics
            if len(X_val) > 0:
                yhat_model = booster.predict(X_val)
                if occ_model is not None:
                    p_nonzero = occ_model.predict(X_val)
                else:
                    p_nonzero = np.ones_like(yhat_model)
                p_nonzero = np.clip(p_nonzero, 0.0, 1.0)

                # Reconstruct to original scale for metrics.
                if "y_orig" in df.columns:
                    y_true_orig = df.loc[is_val, "y_orig"].to_numpy(dtype=float)
                else:
                    y_true_orig = y_val

                if detrend_method == "none":
                    yhat_orig = np.maximum(0.0, yhat_model)
                elif detrend_method == "linear":
                    trend_val = df.loc[is_val, "trend_model"].to_numpy(dtype=float)
                    yhat_orig = np.maximum(0.0, yhat_model + trend_val)
                else:  # log1p_linear
                    trend_val = df.loc[is_val, "trend_model"].to_numpy(dtype=float)
                    yhat_orig = np.maximum(0.0, np.expm1(yhat_model + trend_val))

                yhat_final = yhat_orig * p_nonzero

                metrics_summary[f"wape_val_h{h}"] = _wape(y_true_orig, yhat_final)

                # Capture residuals for pooled upper-bound calibration (validation-only)
                if "forecast_ds" in df.columns:
                    val_ds = pd.to_datetime(df.loc[is_val, "forecast_ds"], errors="coerce")
                else:
                    val_ds = pd.Series([pd.NaT] * len(y_true_orig))
                val_uids = df.loc[is_val, "unique_id"].astype(str).to_numpy()
                for i in range(len(y_true_orig)):
                    ds_val = val_ds.iloc[i]
                    if pd.isna(ds_val):
                        continue
                    y_true = float(y_true_orig[i])
                    yhat_i = float(yhat_final[i])
                    residual = y_true - yhat_i
                    cv_residual_rows.append(
                        {
                            "model_version": model_version,
                            "unique_id": str(val_uids[i]),
                            "ds": ds_val.date().isoformat(),
                            "horizon": int(h),
                            "y": y_true,
                            "yhat": yhat_i,
                            "residual": float(residual),
                            "positive_excess": float(max(0.0, residual)),
                            "archetype": archetype_by_uid.get(str(val_uids[i]), "unknown"),
                            "fold_id": 0,
                        }
                    )

                if h == 1:
                    # per-item eval for eligibility (h=1)
                    cols = ["unique_id", "y", "lag_1", "y_orig", "lag_1_orig", "trend_model"]
                    keep = [c for c in cols if c in df.columns]
                    df_val = df.loc[is_val, keep].copy()

                    if detrend_method == "none":
                        yhat_tmp = np.maximum(0.0, yhat_model)
                        y_eval = pd.to_numeric(df_val.get("y_orig", df_val["y"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    elif detrend_method == "linear":
                        trend_val = pd.to_numeric(df_val.get("trend_model"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        yhat_tmp = np.maximum(0.0, yhat_model + trend_val)
                        y_eval = pd.to_numeric(df_val.get("y_orig", df_val["y"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    else:
                        trend_val = pd.to_numeric(df_val.get("trend_model"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        yhat_tmp = np.maximum(0.0, np.expm1(yhat_model + trend_val))
                        y_eval = pd.to_numeric(df_val.get("y_orig", df_val["y"]), errors="coerce").fillna(0.0).to_numpy(dtype=float)

                    if occ_model is not None:
                        p_nonzero_h1 = occ_model.predict(X_val)
                        p_nonzero_h1 = np.clip(p_nonzero_h1, 0.0, 1.0)
                    else:
                        p_nonzero_h1 = np.ones_like(yhat_tmp)

                    df_val["yhat_ml"] = yhat_tmp * p_nonzero_h1

                    df_val["y_eval"] = y_eval
                    if "lag_1_orig" in df_val.columns:
                        df_val["yhat_naive"] = pd.to_numeric(df_val["lag_1_orig"], errors="coerce").fillna(0.0)
                    else:
                        df_val["yhat_naive"] = pd.to_numeric(df_val.get("lag_1"), errors="coerce").fillna(0.0)

                    for uid, g in df_val.groupby("unique_id", sort=False):
                        yt = g.get("y_eval", g.get("y_orig", g["y"])).to_numpy(dtype=float)
                        y_ml = g["yhat_ml"].to_numpy(dtype=float)
                        y_nv = g["yhat_naive"].to_numpy(dtype=float)
                        per_item_eval[str(uid)] = {
                            "wape_ml": _wape(yt, y_ml),
                            "wape_naive": _wape(yt, y_nv),
                            "n": float(len(g)),
                        }

            # Save
            booster.save_model(os.path.join(artifact_root, f"lgbm_h{h}.txt"))

        assert feature_cols is not None

        # Build pooled residual quantiles (upper bound calibration)
        residual_quantile_rows: list[dict[str, Any]] = []
        if cv_residual_rows:
            df_res = pd.DataFrame(cv_residual_rows)
            if not df_res.empty:
                def _coerce_h(val: Any) -> int:
                    if isinstance(val, (tuple, list)) and len(val) == 1:
                        val = val[0]
                    try:
                        return int(val)
                    except Exception:
                        return 0

                for (arch, h), g in df_res.groupby(["archetype", "horizon"], dropna=False):
                    vals = g["positive_excess"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) == 0:
                        continue
                    q95 = float(np.quantile(vals, 0.95))
                    residual_quantile_rows.append(
                        {
                            "model_version": model_version,
                            "archetype": str(arch) if arch is not None else "unknown",
                            "horizon": _coerce_h(h),
                            "scale_bucket": None,
                            "q95_excess": q95,
                            "n": int(len(vals)),
                        }
                    )
                # Global fallback per horizon
                for h, g in df_res.groupby(["horizon"], dropna=False):
                    vals = g["positive_excess"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) == 0:
                        continue
                    q95 = float(np.quantile(vals, 0.95))
                    residual_quantile_rows.append(
                        {
                            "model_version": model_version,
                            "archetype": "__global__",
                            "horizon": _coerce_h(h),
                            "scale_bucket": None,
                            "q95_excess": q95,
                            "n": int(len(vals)),
                        }
                    )

        if progress_hook:
            progress_hook({"phase": "writing_artifacts"})

        # Feature spec + category mappings
        feature_spec = {
            "method": "lightgbm",
            "strategy": "direct",
            "freq": "MS",
            "horizon": int(horizon),
            "horizons": list(range(1, int(horizon) + 1)),
            "target": "y",
            "detrend_method": detrend_method,
            "fourier_harmonics": 3,  # k=3 Fourier pairs for month seasonality
            "occurrence_model": "shared" if occ_model is not None else None,
            "occurrence_threshold": 0.2 if occ_model is not None else None,
            "lags": lags,
            "rolling_windows": roll_windows,
            "static_columns": static_cols,
            "static_interaction_columns": static_interaction_cols,
            "exogenous_columns": exogenous_columns,
            "categorical_columns": cat_cols,
            "category_mappings": mappings,
            "feature_columns": feature_cols,
            "stat_feature_columns": ["sf_croston_optimized", "sf_adida", "sf_auto_ets", "sf_theta", "sf_available"],
            "amount_objective": "tweedie" if detrend_method == "none" else "regression",
            "tweedie_variance_power": 1.3,
            "sample_weighting": "inverse_demand",
            "target_encoding_columns": te_cols,
            "target_encoding_maps": te_maps,
            "target_encoding_prior": 10.0,
            "name_cluster_map": name_cluster_map,
            "name_cluster_k": name_cluster_k,
            "residual_quantiles": residual_quantile_rows,
            "quantile_models": True,
            "quantile_alpha": 0.95,
            "hyperparameter_tuning": tune_hyperparameters,
            "tuned_hyperparameters": tuned_params if tuned_params else None,
            "attribute_routing": {
                "calendar": calendar_cols,
                "trend": trend_cols,
                "segment": segment_cols,
                "generic_interaction": generic_interaction_cols,
            },
            "global_yoy": global_yoy,
        }
        spec_path = os.path.join(artifact_root, "feature_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(feature_spec, f, ensure_ascii=False, indent=2)

        spec_hash = self.store.set_feature_spec(model_version, feature_spec)

        if detrend_method != "none":
            trend_path = os.path.join(artifact_root, "trend_params.json")
            with open(trend_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "detrend_method": detrend_method,
                        "per_unique_id": trend_params_by_uid,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        if progress_hook:
            progress_hook({"phase": "writing_duckdb"})

        # Register version
        train_end_ds = pd.to_datetime(base["ds"], errors="coerce").max()
        self.store.create_model_version(
            ModelVersionRow(
                customer_id=self.store.customer_id,
                model_version=model_version,
                created_at=datetime.now(UTC),
                created_by=None,
                status=status,
                freq="MS",
                horizon=int(horizon),
                target="y",
                train_end_ds=train_end_ds.date().isoformat() if pd.notna(train_end_ds) else None,
                artifact_root=artifact_root,
                notes=notes,
                git_sha=None,
            )
        )

        # Persist CV residuals + pooled quantiles
        try:
            if cv_residual_rows:
                self.store.insert_cv_residuals(cv_residual_rows)
            if residual_quantile_rows:
                self.store.upsert_residual_quantiles(residual_quantile_rows)
        except Exception:
            # Don't fail training if residual persistence fails.
            pass

        # Write backtests + eligibility
        backtest_rows: list[dict[str, Any]] = []
        eligibility_rows: list[dict[str, Any]] = []
        explain_rows: list[dict[str, Any]] = []

        # Global feature importance from horizon 1
        if 1 in models:
            names = models[1].feature_name()
            gains = models[1].feature_importance(importance_type="gain")
            top = sorted(zip(names, gains), key=lambda t: t[1], reverse=True)[:10]
            top_features = [{"feature": n, "gain": float(g)} for n, g in top]
        else:
            top_features = []

        # Eligibility based on per-item horizon-1 validation
        # DEBUG: Log base DataFrame info before eligibility loop
        unique_ids_list = base["unique_id"].astype(str).unique().tolist()
        print(f"[DEBUG train_and_register] Starting eligibility loop: base.shape={base.shape}, unique_ids={len(unique_ids_list)}, model_version={model_version}")
        for uid in unique_ids_list:
            series_len = int((base[base["unique_id"].astype(str) == str(uid)]).shape[0])

            evalr = per_item_eval.get(str(uid))
            if series_len < int(min_history_points):
                eligibility_rows.append(
                    {
                        "model_version": model_version,
                        "unique_id": str(uid),
                        "winner_model": "naive",
                        "ml_allowed": False,
                        "ml_preferred": False,
                        "fallback_model": "naive",
                        "reason_code": "SHORT_HISTORY",
                        "confidence": 0.0,
                        "min_history_points": int(min_history_points),
                        "requires_features": [],
                    }
                )
                # Still add to explain_rows for diagnostics API even with short history
                explain_rows.append(
                    {
                        "model_version": model_version,
                        "unique_id": str(uid),
                        "top_features": top_features,
                        "group_contrib": {"reason": "SHORT_HISTORY"},
                        "support_share": None,
                    }
                )
                continue

            if not evalr or evalr.get("n", 0) < 2:
                eligibility_rows.append(
                    {
                        "model_version": model_version,
                        "unique_id": str(uid),
                        "winner_model": "naive",
                        "ml_allowed": False,
                        "ml_preferred": False,
                        "fallback_model": "naive",
                        "reason_code": "NO_EVAL_DATA",
                        "confidence": 0.0,
                        "min_history_points": int(min_history_points),
                        "requires_features": [],
                    }
                )
                # Still add to explain_rows for diagnostics API even with no eval data
                explain_rows.append(
                    {
                        "model_version": model_version,
                        "unique_id": str(uid),
                        "top_features": top_features,
                        "group_contrib": {"reason": "NO_EVAL_DATA"},
                        "support_share": None,
                    }
                )
                continue

            w_ml = float(evalr["wape_ml"])
            w_nv = float(evalr["wape_naive"])

            backtest_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    "model_name": "lgbm_direct_h1",
                    "metric_name": "wape",
                    "metric_value": w_ml,
                    "n_folds": 1,
                    "eval_start_ds": None,
                    "eval_end_ds": None,
                }
            )
            backtest_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    "model_name": "naive",
                    "metric_name": "wape",
                    "metric_value": w_nv,
                    "n_folds": 1,
                    "eval_start_ds": None,
                    "eval_end_ds": None,
                }
            )

            ml_better = w_ml <= (w_nv * (1.0 - float(min_improvement)))
            if ml_better:
                winner = "lgbm_direct"
                allowed = True
                reason = "OK"
                confidence = float(max(0.0, min(1.0, 1.0 - (w_ml / (w_nv + 1e-9)))))
            else:
                winner = "naive"
                allowed = False
                reason = "ML_WORSE_THAN_TS"
                confidence = float(max(0.0, min(1.0, 1.0 - (w_nv / (w_ml + 1e-9)))))

            eligibility_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    "winner_model": winner,
                    "ml_allowed": bool(allowed),
                    "ml_preferred": bool(allowed),
                    "fallback_model": "naive",
                    "reason_code": reason,
                    "confidence": confidence,
                    "min_history_points": int(min_history_points),
                    "requires_features": [],
                }
            )

            explain_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    "top_features": top_features,
                    "group_contrib": {},
                    "support_share": None,
                }
            )

        # Add diagnostic counts to metrics_summary for debugging
        total_items = len(base["unique_id"].astype(str).unique())
        items_with_eval = len(per_item_eval)
        items_short_history = sum(1 for r in eligibility_rows if r.get("reason_code") == "SHORT_HISTORY")
        items_no_eval = sum(1 for r in eligibility_rows if r.get("reason_code") == "NO_EVAL_DATA")
        items_ml_allowed = sum(1 for r in eligibility_rows if r.get("ml_allowed"))
        metrics_summary["_debug_total_items"] = total_items
        metrics_summary["_debug_items_with_eval"] = items_with_eval
        metrics_summary["_debug_items_short_history"] = items_short_history
        metrics_summary["_debug_items_no_eval_data"] = items_no_eval
        metrics_summary["_debug_items_ml_allowed"] = items_ml_allowed
        metrics_summary["_debug_explain_rows_count"] = len(explain_rows)
        metrics_summary["_debug_store_customer_id"] = self.store.customer_id
        metrics_summary["_debug_store_db_path"] = self.store.db_path()
        metrics_summary["_debug_model_version"] = model_version

        # DEBUG: Print explain_rows count after eligibility loop
        print(f"[DEBUG train_and_register] After eligibility loop: explain_rows={len(explain_rows)}, eligibility_rows={len(eligibility_rows)}, model_version={model_version}")

        # Per-item diagnostics (stored as JSON under explain_item_summary.group_contrib_json)
        # - SHAP-like attributions: LightGBM's pred_contrib output (TreeSHAP)
        # - Per-item importance: abs(contribution) ranking
        # - Used/dropped static/exogenous columns (based on last available row)
        # - Basic training diagnostics
        # Calculate SHAP for ALL horizons, not just horizon 1
        if models and feature_cols is not None:
            per_uid_diag: dict[str, dict[str, Any]] = {}
            base_by_uid = base.sort_values(["unique_id", "ds"], kind="mergesort").groupby("unique_id", sort=False)

            for uid, grp in base_by_uid:
                uid = str(uid)
                grp = grp.sort_values("ds", kind="mergesort")
                ds_arr = grp["ds"].to_numpy()
                y_orig = grp["y"].to_numpy(dtype=float)

                diag: dict[str, Any] = {
                    "series_len": int(len(y_orig)),
                    "train_end_ds": pd.Timestamp(ds_arr[-1]).strftime("%Y-%m-%d") if len(ds_arr) else None,
                    "detrend_method": detrend_method,
                }

                # Attach evaluation info (eligibility backtest)
                evalr = per_item_eval.get(uid) or {}
                if evalr:
                    diag.update(
                        {
                            "wape_ml": float(evalr.get("wape_ml")) if evalr.get("wape_ml") is not None else None,
                            "wape_naive": float(evalr.get("wape_naive")) if evalr.get("wape_naive") is not None else None,
                            "eval_n": float(evalr.get("n")) if evalr.get("n") is not None else None,
                        }
                    )

                params = trend_params_by_uid.get(uid)
                if detrend_method == "none" or not params:
                    y_model = y_orig
                elif detrend_method == "linear":
                    trend_hist = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds_arr], dtype=float)
                    y_model = y_orig - trend_hist
                    diag["trend_slope"] = float(params.get("slope", 0.0))
                    diag["trend_intercept"] = float(params.get("intercept", 0.0))
                else:  # log1p_linear
                    yt = np.log1p(np.maximum(0.0, y_orig))
                    trend_hist = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds_arr], dtype=float)
                    y_model = yt - trend_hist
                    diag["trend_slope"] = float(params.get("slope", 0.0))
                    diag["trend_intercept"] = float(params.get("intercept", 0.0))

                # Used/dropped columns (static + exogenous)
                used_static: list[str] = []
                dropped_static: list[str] = []
                for c in static_cols:
                    if c in grp.columns and pd.notna(grp[c].iloc[-1]):
                        used_static.append(c)
                    else:
                        dropped_static.append(c)

                used_exo: list[str] = []
                dropped_exo: list[str] = []
                for c in exogenous_columns or []:
                    if c in grp.columns and pd.notna(grp[c].iloc[-1]):
                        used_exo.append(c)
                    else:
                        dropped_exo.append(c)

                # Build a single feature row for horizon=1 at the last decision point
                lags_local = lags
                roll_local = roll_windows
                t = len(y_model) - 1
                max_needed = max(max(lags_local) - 1, max(roll_local) - 1, 12)
                if t < max_needed:
                    diag["shap"] = {"available": False, "reason": "INSUFFICIENT_LAGS"}
                    per_uid_diag[uid] = {
                        "training_diagnostics": diag,
                        "used_columns": {"static": used_static, "exogenous": used_exo},
                        "dropped_columns": {"static": dropped_static, "exogenous": dropped_exo},
                    }
                    continue

                last_ds = pd.Timestamp(ds_arr[-1])
                forecast_ds = last_ds + pd.offsets.MonthBegin(1)

                r: dict[str, Any] = {
                    "unique_id": uid,
                    "decision_ds": last_ds,
                    "forecast_ds": forecast_ds,
                }
                m = int(forecast_ds.month)
                r["month"] = m
                r["quarter"] = int(((m - 1) // 3) + 1)
                r["year"] = int(forecast_ds.year)
                # Extended Fourier features (k=3 harmonics)
                fourier_feats = _fourier_features(m, k=3)
                r.update(fourier_feats)
                # Backward compat aliases
                r["month_sin"] = fourier_feats["month_sin_1"]
                r["month_cos"] = fourier_feats["month_cos_1"]
                qs, qc = _quarter_sin_cos(int(r["quarter"]))
                r["quarter_sin"] = qs
                r["quarter_cos"] = qc
                r["year_idx"] = float(forecast_ds.year - pd.Timestamp(ds_arr[0]).year)

                # Easter features for paskavara (Easter items)
                easter_feats = _easter_features(forecast_ds)
                r.update(easter_feats)

                # Note: static interaction columns are computed by _add_static_interactions()

                for lag in lags_local:
                    idx = t - (lag - 1)
                    r[f"lag_{lag}"] = float(y_model[idx])

                for w in roll_local:
                    start = t - (w - 1)
                    window = y_model[start : t + 1]
                    r[f"roll_mean_{w}"] = float(np.mean(window))
                    if w >= 2:
                        r[f"roll_std_{w}"] = float(np.std(window, ddof=0))

                r["diff1"] = float(y_model[t] - y_model[t - 1])
                r["diff12"] = float(y_model[t] - y_model[t - 12])

                last12 = y_model[t - 11 : t + 1]
                r["zero_ratio_12"] = float(np.mean(last12 == 0.0))
                r["nonzero_run_length"] = float(_nonzero_run_length(y_model[: t + 1]))

                # Additional features for SHAP accuracy (wrapped in try/except to prevent crashes)
                try:
                    # Conditional-on-nonzero level features (use original scale)
                    window_nz = y_orig[t - 11 : t + 1] if t >= 11 else y_orig[:t + 1]
                    nz_vals = window_nz[window_nz > 0.0]
                    mean_nonzero_12 = float(np.mean(nz_vals)) if len(nz_vals) else 0.0
                    median_nonzero_12 = float(np.median(nz_vals)) if len(nz_vals) else 0.0
                    nz_full_vals = y_orig[y_orig > 0.0]
                    last_nz_val = float(nz_full_vals[-1]) if len(nz_full_vals) else 0.0
                    r["mean_nonzero_12"] = mean_nonzero_12
                    r["median_nonzero_12"] = median_nonzero_12
                    r["last_nonzero_value"] = last_nz_val

                    # Same-month historical features (per item) - last 36 months
                    months = np.array([pd.Timestamp(d).month for d in ds_arr], dtype=int)
                    vals_3y = y_orig[-36:] if len(y_orig) > 36 else y_orig
                    months_3y = months[-36:] if len(months) > 36 else months
                    same_month_stats_local: dict[int, dict[str, float]] = {}
                    for month_i in range(1, 13):
                        hist_vals = vals_3y[months_3y == month_i]
                        if len(hist_vals) == 0:
                            same_month_stats_local[month_i] = {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0}
                        else:
                            same_month_stats_local[month_i] = {
                                "mean": float(np.mean(hist_vals)),
                                "max": float(np.max(hist_vals)),
                                "nonzero_rate": float(np.mean(hist_vals > 0.0)),
                            }
                    m_stats = same_month_stats_local.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
                    r["same_month_mean_3y"] = float(m_stats["mean"])
                    r["same_month_max_3y"] = float(m_stats["max"])
                    r["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
                    r["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

                    mean_last12 = float(np.mean(y_orig[t - 11 : t + 1])) if t >= 11 else float(np.mean(y_orig[:t + 1]))
                    r["seasonal_amplitude_ratio"] = float(m_stats["max"]) / max(mean_last12, 1.0)

                    # Global seasonal components (cross-item learning)
                    if global_monthly_effects is not None:
                        global_month_level = global_monthly_effects.get(m, 0.0)
                        r["global_month_level"] = global_month_level
                        item_month_mean = float(m_stats.get("mean", 0.0))
                        if global_month_level > 0.0:
                            r["item_vs_global_ratio"] = float(item_month_mean / global_month_level)
                        else:
                            r["item_vs_global_ratio"] = 1.0 if item_month_mean == 0.0 else float(item_month_mean)
                    else:
                        r["global_month_level"] = 0.0
                        r["item_vs_global_ratio"] = 1.0

                    # Simple decay signal: slope of last 12 months
                    r["rolling_12_slope"] = float(_rolling_slope(y_model[: t + 1]))
                except Exception:
                    # Fallback to safe defaults if feature calculation fails
                    r["mean_nonzero_12"] = 0.0
                    r["median_nonzero_12"] = 0.0
                    r["last_nonzero_value"] = 0.0
                    r["same_month_mean_3y"] = 0.0
                    r["same_month_max_3y"] = 0.0
                    r["same_month_nonzero_rate_3y"] = 0.0
                    r["item_month_nonzero_rate"] = 0.0
                    r["seasonal_amplitude_ratio"] = 0.0
                    r["global_month_level"] = 0.0
                    r["item_vs_global_ratio"] = 1.0
                    r["rolling_12_slope"] = 0.0

                # Exogenous values at time t
                for k in exogenous_columns or []:
                    if k in grp.columns:
                        v = pd.to_numeric(grp[k].iloc[-1], errors="coerce") if pd.notna(grp[k].iloc[-1]) else 0.0
                        r[k] = float(0.0 if pd.isna(v) else v)

                # Static values
                for k in static_cols:
                    if k in grp.columns:
                        r[k] = grp[k].iloc[-1]

                X = pd.DataFrame([r])
                X["unique_id"] = X["unique_id"].astype(str)
                X, _ = _encode_categories(X, cat_cols, mappings=mappings)
                X = _add_static_interactions(X, static_interaction_cols)
                for col in feature_cols:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_cols].fillna(0)

                # Calculate SHAP for all horizons
                shap_per_horizon = {}
                imp_top = []
                static_contrib = []
                static_values = {}

                try:
                    for h in range(1, int(horizon) + 1):
                        booster = models.get(h)
                        if booster is None:
                            continue

                        try:
                            contrib = booster.predict(X, pred_contrib=True)
                            contrib_row = np.asarray(contrib[0], dtype=float)
                            # LightGBM returns n_features + 1 (bias)
                            bias = float(contrib_row[-1])
                            feat_names = list(booster.feature_name())
                            feat_contrib = contrib_row[:-1]
                            pairs = list(zip(feat_names, feat_contrib))
                            pairs_sorted = sorted(pairs, key=lambda kv: abs(float(kv[1])), reverse=True)
                            contrib_map = {str(n): float(v) for n, v in pairs}

                            top_k = 20
                            shap_top = [{"feature": n, "contribution": float(v)} for n, v in pairs_sorted[:top_k]]

                            # Calculate static feature contributions for this horizon
                            horizon_static_contrib = [
                                {"feature": f, "contribution": float(contrib_map.get(f, 0.0))}
                                for f in used_static
                                if f in contrib_map
                            ]
                            horizon_static_contrib = sorted(horizon_static_contrib, key=lambda kv: abs(float(kv["contribution"])), reverse=True)

                            shap_per_horizon[h] = {
                                "available": True,
                                "method": "pred_contrib",
                                "horizon": h,
                                "bias": bias,
                                "top": shap_top,
                                "static_contrib": horizon_static_contrib,  # NEW: per-horizon static contributions
                            }

                            # For backward compatibility, store horizon 1 data in the old fields
                            if h == 1:
                                imp_top = [{"feature": n, "importance": float(abs(v))} for n, v in pairs_sorted[:top_k]]
                                static_contrib = horizon_static_contrib
                                static_values = {
                                    f: float(X[f].iloc[0]) for f in used_static if f in X.columns
                                }
                        except Exception as e:
                            shap_per_horizon[h] = {"available": False, "reason": f"PRED_CONTRIB_FAILED: {e}", "horizon": h}

                    # For backward compatibility, keep the old 'shap' field with horizon 1 data
                    shap_info = shap_per_horizon.get(1, {"available": False, "reason": "NO_HORIZON_1_MODEL"})
                except Exception as e:
                    shap_info = {"available": False, "reason": f"SHAP_CALCULATION_FAILED: {e}"}
                    shap_per_horizon = {}

                per_uid_diag[uid] = {
                    "shap": shap_info,  # Horizon 1 data for backward compatibility
                    "shap_per_horizon": shap_per_horizon,  # NEW: All horizons
                    "per_item_importance_top": imp_top,
                    "used_columns": {"static": used_static, "exogenous": used_exo},
                    "dropped_columns": {"static": dropped_static, "exogenous": dropped_exo},
                    "static_contrib": static_contrib,
                    "static_values": static_values,
                    "training_diagnostics": diag,
                }

            # Merge diagnostics into explain_rows
            for r in explain_rows:
                uid = str(r.get("unique_id"))
                extra = per_uid_diag.get(uid)
                if not extra:
                    continue
                # Keep existing structure but attach rich per-item payload
                r["group_contrib"] = extra

        self.store.upsert_backtest_metrics(backtest_rows)
        self.store.upsert_eligibility(eligibility_rows)
        # DEBUG: Log before upsert
        print(f"[DEBUG train_and_register] About to upsert_explain_summary: explain_rows={len(explain_rows)}, customer_id={self.store.customer_id}, model_version={model_version}")
        self.store.upsert_explain_summary(explain_rows)

        if progress_hook:
            progress_hook(
                {
                    "phase": "done",
                    "model_version": model_version,
                    "items_trained": int(base["unique_id"].nunique()),
                }
            )

        # Aggregate metrics summary
        if any(k.startswith("wape_val_h") for k in metrics_summary.keys()):
            wapes = [v for k, v in metrics_summary.items() if k.startswith("wape_val_h")]
            metrics_summary["wape_val_mean"] = float(np.mean(wapes))

        return TrainResult(
            customer_id=self.store.customer_id,
            model_version=model_version,
            status=status,
            artifact_root=artifact_root,
            feature_spec_hash=spec_hash,
            items_trained=int(base["unique_id"].nunique()),
            rows_trained=int(sum(len(v) for v in all_rows_by_h.values())),
            metrics_summary=metrics_summary,
        )

    def _load_spec(self, model_version: str) -> tuple[str, dict[str, Any]]:
        artifact_root = self.store.get_model_artifact_root(model_version)
        if not artifact_root:
            raise ValueError(f"Unknown model_version: {model_version}")
        spec_path = os.path.join(artifact_root, "feature_spec.json")
        if not os.path.exists(spec_path):
            raise ValueError(f"Missing feature_spec.json for {model_version}")
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return artifact_root, spec

    def batch_forecast(
        self,
        hist: pd.DataFrame,
        *,
        forecast_periods: int,
        freq: str = "M",
        item_attributes: pd.DataFrame | None = None,
        drivers: pd.DataFrame | None = None,
        exogenous_columns: list[str] | None = None,
        model_version: str | None = None,
        status: str = "prod",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        _ = freq

        model_version = model_version or self.store.get_active_model_version(status=status)
        if not model_version:
            raise ValueError(f"No active model version found for status='{status}'")

        artifact_root, spec = self._load_spec(model_version)
        horizons: list[int] = list(spec.get("horizons") or [])
        feature_cols: list[str] = list(spec.get("feature_columns") or [])
        exo_cols: list[str] = list(spec.get("exogenous_columns") or [])
        static_cols: list[str] = list(spec.get("static_columns") or [])
        stat_feature_cols: list[str] = list(spec.get("stat_feature_columns") or [])
        static_interaction_cols: list[str] = list(spec.get("static_interaction_columns") or [])
        te_cols: list[str] = list(spec.get("target_encoding_columns") or [])
        te_maps: dict[str, Any] = dict(spec.get("target_encoding_maps") or {})
        # Name-token fields may be present in older specs but are no longer used.
        name_token_cols: list[str] = list(spec.get("name_token_columns") or [])
        name_token_buckets = int(spec.get("name_token_buckets") or len(name_token_cols) or 16)
        name_cluster_map: dict[str, Any] = dict(spec.get("name_cluster_map") or {})
        cat_cols: list[str] = list(spec.get("categorical_columns") or [])
        mappings: dict[str, dict[str, int]] = dict(spec.get("category_mappings") or {})
        lags: list[int] = list(spec.get("lags") or [1, 2, 3, 6, 12, 24])
        roll_windows: list[int] = list(spec.get("rolling_windows") or [3, 6, 12])
        occurrence_threshold = float(spec.get("occurrence_threshold") or 0.2)

        detrend_method = str(spec.get("detrend_method") or "none").strip().lower()
        trend_params_by_uid: dict[str, dict[str, Any]] = {}
        trend_path = os.path.join(artifact_root, "trend_params.json")
        if os.path.exists(trend_path):
            try:
                with open(trend_path, "r", encoding="utf-8") as f:
                    trend_blob = json.load(f)
                detrend_method = str(trend_blob.get("detrend_method") or detrend_method).strip().lower()
                trend_params_by_uid = dict(trend_blob.get("per_unique_id") or {})
            except Exception:
                trend_params_by_uid = {}

        H = int(forecast_periods)
        horizons = list(range(1, H + 1))

        base = _normalize_monthly_history(hist)

        if item_attributes is not None and not item_attributes.empty and "item_id" in item_attributes.columns:
            attrs = item_attributes.copy()
            attrs["item_id"] = attrs["item_id"].astype(str)
            base["item_id"] = base["item_id"].astype(str)
            base = base.merge(attrs, on="item_id", how="left", suffixes=("", "_attr"))

        if drivers is not None and not drivers.empty:
            wide = _pivot_long_drivers(drivers)
            if not wide.empty:
                if "item_id" in wide.columns:
                    wide["item_id"] = wide["item_id"].astype(str)
                    base = base.merge(wide, on=["item_id", "ds"], how="left")
                else:
                    base = base.merge(wide, on=["ds"], how="left")

        # Apply target encodings and name features for inference
        if te_cols and te_maps:
            base = _apply_target_encodings(base, te_cols, te_maps)
        if name_cluster_map:
            base = _apply_name_cluster(base, name_cluster_map, name_col="name")
        # name_tok_* features are deprecated and not recomputed at inference.

        # Resolve routing configuration for attribute mechanisms (may be missing for old models)
        routing = spec.get("attribute_routing", {}) or {}
        calendar_cols = list(routing.get("calendar") or [])
        trend_cols = list(routing.get("trend") or [])
        segment_cols = list(routing.get("segment") or [])
        generic_interaction_cols = list(routing.get("generic_interaction") or [])
        global_yoy = dict(spec.get("global_yoy") or {})

        # Precompute segment stats for inference when available
        segment_stats = _precompute_segment_monthly_stats(
            base,
            segment_cols,
            lookback_months=36,
        ) if segment_cols else {}

        if exogenous_columns is None:
            exogenous_columns = [c for c in exo_cols if c in base.columns]
        else:
            exogenous_columns = [c for c in exogenous_columns if c in base.columns]

        base = base.sort_values(["unique_id", "ds"], kind="mergesort")

        # Compute global monthly effects for inference (same as training)
        # This enables cross-item learning and helps static features
        global_monthly_effects = _compute_global_monthly_effects(
            base,
            lookback_months=36,
        )

        month_caps = _build_item_month_caps(base, lookback_years=4)
        cap_nonzero_threshold = 0.25
        cap_multiplier = 2.0
        cap_small_floor = 0.0

        sf_future = _build_statsforecast_future_features(base, max_h=int(max(horizons or [1])), season_length=12)
        sf_future = sf_future.rename(columns={"ds": "forecast_ds"})

        # Load models
        boosters: dict[int, Any] = {}
        nz_boosters: dict[int, Any] = {}
        for h in horizons:
            path = os.path.join(artifact_root, f"lgbm_h{h}.txt")
            if os.path.exists(path):
                boosters[h] = lgb.Booster(model_file=path)
            nz_path = os.path.join(artifact_root, f"lgbm_h{h}_nz.txt")
            if os.path.exists(nz_path):
                nz_boosters[h] = lgb.Booster(model_file=nz_path)

        occ_model: Any | None = None
        occ_path = os.path.join(artifact_root, "lgbm_occurrence.txt")
        if os.path.exists(occ_path):
            occ_model = lgb.Booster(model_file=occ_path)

        # Load quantile models (0.95) for upper bound prediction
        q95_boosters: dict[int, Any] = {}
        for h in horizons:
            q95_path = os.path.join(artifact_root, f"lgbm_h{h}_q95.txt")
            if os.path.exists(q95_path):
                q95_boosters[h] = lgb.Booster(model_file=q95_path)

        residual_quantiles = self.store.get_residual_quantiles(model_version)
        if not residual_quantiles:
            spec_q = spec.get("residual_quantiles") or []
            for r in spec_q:
                arch = r.get("archetype")
                h = r.get("horizon")
                q = r.get("q95_excess")
                if arch is None or h is None or q is None:
                     continue
                residual_quantiles[(str(arch), int(h))] = float(q)

        if not boosters:
            raise ValueError(f"No horizon models found under {artifact_root}")

        forecasts: list[dict[str, Any]] = []

        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            y_orig = grp["y"].to_numpy(dtype=float)
            ds_arr = grp["ds"].to_numpy()
            if len(y_orig) < 6:
                continue

            last_ds = pd.Timestamp(ds_arr[-1])

            params = trend_params_by_uid.get(str(uid))
            if detrend_method != "none" and params:
                if detrend_method == "linear":
                    trend_hist = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds_arr], dtype=float)
                    y = y_orig - trend_hist
                else:  # log1p_linear
                    yt = np.log1p(np.maximum(0.0, y_orig))
                    trend_hist = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds_arr], dtype=float)
                    y = yt - trend_hist
            else:
                y = y_orig

            exo = {c: pd.to_numeric(grp[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in exogenous_columns}
            static: dict[str, Any] = {}
            for c in static_cols:
                static[c] = grp[c].iloc[-1] if c in grp.columns else None

            # Precompute same-month stats for inference (last 36 months) on original scale.
            months = np.array([pd.Timestamp(d).month for d in ds_arr], dtype=int)
            base_y = y_orig
            vals_3y = base_y[-36:] if len(base_y) > 36 else base_y
            months_3y = months[-36:] if len(months) > 36 else months
            same_month_stats: dict[int, dict[str, float]] = {}
            for m in range(1, 13):
                hist_vals = vals_3y[months_3y == m]
                if len(hist_vals) == 0:
                    same_month_stats[m] = {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0}
                else:
                    same_month_stats[m] = {
                        "mean": float(np.mean(hist_vals)),
                        "max": float(np.max(hist_vals)),
                        "nonzero_rate": float(np.mean(hist_vals > 0.0)),
                    }

            overall_nonzero_rate = float(np.mean((vals_3y > 0.0))) if len(vals_3y) else 0.0
            seasonal_strength = float(max([v["nonzero_rate"] for v in same_month_stats.values()] or [0.0]))
            cv_val = _cv(vals_3y if len(vals_3y) else y)
            archetype = _archetype(overall_nonzero_rate, cv_val, seasonal_strength)
            croston_mean = simple_croston_mean(y_orig)
            adida_mean = simple_adida_mean(y_orig, agg=3)
            recent_level = _recent_level(y_orig)
            if recent_level <= 0.0 and croston_mean > 0.0:
                recent_level = 0.4 * float(croston_mean)
            if archetype == "stable":
                stable_window = y_orig[-12:] if len(y_orig) >= 12 else y_orig
                if len(stable_window):
                    recent_level = float(np.mean(stable_window))
            nonzero_count_last_12 = int(np.sum((y_orig[-12:] if len(y_orig) >= 12 else y_orig) > 0.0))
            last_nonzero_age = float(_nonzero_run_length(y_orig))
            recently_active = (last_nonzero_age <= 6) or (nonzero_count_last_12 >= 2)
            recent_slope = _rolling_slope(y_orig)
            mean_last12_item = float(np.mean(base_y[-12:])) if len(base_y) else 0.0

            # Build one row per horizon and predict with the corresponding model
            for h in horizons:
                if h not in boosters:
                    continue
                t = len(y) - 1
                max_needed = max(max(lags) - 1, max(roll_windows) - 1, 12)
                if t < max_needed:
                    continue

                forecast_ds = last_ds + pd.offsets.MonthBegin(h)

                r: dict[str, Any] = {
                    "unique_id": str(uid),
                    "decision_ds": last_ds,
                    "forecast_ds": forecast_ds,
                }
                m = int(forecast_ds.month)
                r["month"] = m
                r["quarter"] = int(((m - 1) // 3) + 1)
                r["year"] = int(forecast_ds.year)
                # Extended Fourier features (k=3 harmonics)
                fourier_feats = _fourier_features(m, k=3)
                r.update(fourier_feats)
                # Backward compat aliases
                r["month_sin"] = fourier_feats["month_sin_1"]
                r["month_cos"] = fourier_feats["month_cos_1"]
                # Quarter Fourier features
                qs, qc = _quarter_sin_cos(int(r["quarter"]))
                r["quarter_sin"] = qs
                r["quarter_cos"] = qc
                # Year index from start of history
                r["year_idx"] = float(forecast_ds.year - pd.Timestamp(ds_arr[0]).year) if len(ds_arr) else 0.0

                # Easter features for paskavara (Easter items)
                easter_feats = _easter_features(forecast_ds)
                r.update(easter_feats)

                for lag in lags:
                    idx = t - (lag - 1)
                    r[f"lag_{lag}"] = float(y[idx]) if idx >= 0 else 0.0

                for w in roll_windows:
                    start = t - (w - 1)
                    window = y[start : t + 1]
                    r[f"roll_mean_{w}"] = float(np.mean(window))
                    if w >= 2:
                        r[f"roll_std_{w}"] = float(np.std(window, ddof=0))

                r["diff1"] = float(y[t] - y[t - 1])
                r["diff12"] = float(y[t] - y[t - 12]) if t >= 12 else 0.0

                last12 = y[t - 11 : t + 1]
                r["zero_ratio_12"] = float(np.mean(last12 == 0.0))
                r["nonzero_run_length"] = float(_nonzero_run_length(y[: t + 1]))

                # Conditional-on-nonzero level features
                window_nz = y_orig[t - 11 : t + 1] if len(y_orig) >= 12 else y_orig
                nz_vals = window_nz[window_nz > 0.0]
                mean_nonzero_12 = float(np.mean(nz_vals)) if len(nz_vals) else 0.0
                median_nonzero_12 = float(np.median(nz_vals)) if len(nz_vals) else 0.0
                nz_full_vals = y_orig[y_orig > 0.0]
                last_nz_val = float(nz_full_vals[-1]) if len(nz_full_vals) else 0.0
                r["mean_nonzero_12"] = mean_nonzero_12
                r["median_nonzero_12"] = median_nonzero_12
                r["last_nonzero_value"] = last_nz_val

                # Same-month historical features (per item)
                m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
                r["same_month_mean_3y"] = float(m_stats["mean"])
                r["same_month_max_3y"] = float(m_stats["max"])
                r["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
                # Alias: per-item month nonzero rate computed server-side.
                r["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])
                r["seasonal_amplitude_ratio"] = float(m_stats["max"]) / max(mean_last12_item, 1.0)

                # Global seasonal components (cross-item learning)
                # Add global month-of-year effect and item vs global ratio
                if global_monthly_effects is not None:
                    global_month_level = global_monthly_effects.get(m, 0.0)
                    r["global_month_level"] = global_month_level
                    # Ratio: how does this item's month pattern compare to global?
                    item_month_mean = float(m_stats.get("mean", 0.0))
                    if global_month_level > 0.0:
                        r["item_vs_global_ratio"] = float(item_month_mean / global_month_level)
                    else:
                        # If global is zero, use item's mean as ratio (or 1.0 if item also zero)
                        r["item_vs_global_ratio"] = 1.0 if item_month_mean == 0.0 else float(item_month_mean)
                else:
                    # Fallback if global_monthly_effects not provided
                    r["global_month_level"] = 0.0
                    r["item_vs_global_ratio"] = 1.0

                # Simple decay signal: slope of last 12 months
                r["rolling_12_slope"] = float(_rolling_slope(y[: t + 1]))

                for k, arr in exo.items():
                    if len(arr) == len(y):
                        r[k] = float(arr[t])

                for k, v in static.items():
                    r[k] = v

                if not sf_future.empty:
                    match = sf_future[(sf_future["unique_id"] == str(uid)) & (sf_future["forecast_ds"] == forecast_ds)]
                    if not match.empty:
                        row_vals = match.iloc[0].to_dict()
                        for k in stat_feature_cols:
                            if k in row_vals:
                                r[k] = row_vals[k]
                        r["sf_available"] = float(row_vals.get("sf_available", 0.0))
                    else:
                        for k in stat_feature_cols:
                            r[k] = 0.0
                        r["sf_available"] = 0.0

                X = pd.DataFrame([r])
                X["unique_id"] = X["unique_id"].astype(str)
                X, _ = _encode_categories(X, cat_cols, mappings=mappings)
                X = _add_static_interactions(X, static_interaction_cols)

                # Track columns before and after filling for diagnostics
                cols_before_fill = set(X.columns)
                missing_cols = [c for c in feature_cols if c not in X.columns]

                for c in feature_cols:
                    if c not in X.columns:
                        X[c] = 0
                X = X[feature_cols].fillna(0)

                # Debug: verify feature count matches expectation
                if X.shape[1] != len(feature_cols):
                    import warnings
                    warnings.warn(
                        f"[lightgbm_forecasts v{_LIGHTGBM_FORECASTS_VERSION}] "
                        f"Feature mismatch: X has {X.shape[1]} cols, expected {len(feature_cols)}. "
                        f"Missing cols filled: {len(missing_cols)}"
                    )

                yhat_model = float(boosters[h].predict(X)[0])
                yhat_model_nz = None
                if h in nz_boosters:
                    yhat_model_nz = float(nz_boosters[h].predict(X)[0])
                if detrend_method == "none" or not params:
                    yhat_amount = float(np.maximum(0.0, yhat_model))
                elif detrend_method == "linear":
                    trend_future = _trend_model_value(forecast_ds, params)
                    yhat_amount = float(np.maximum(0.0, yhat_model + trend_future))
                else:  # log1p_linear
                    trend_future = _trend_model_value(forecast_ds, params)
                    yhat_amount = float(np.maximum(0.0, np.expm1(yhat_model + trend_future)))
                if yhat_model_nz is not None:
                    yhat_amount_nz = float(np.maximum(0.0, yhat_model_nz))
                else:
                    yhat_amount_nz = None

                if occ_model is not None:
                    p_nonzero = float(occ_model.predict(X)[0])
                    p_nonzero = float(np.clip(p_nonzero, 0.0, 1.0))
                else:
                    p_nonzero = 1.0

                month_rate_same = float(m_stats.get("nonzero_rate", 0.0))

                adjustments: dict[str, Any] = {"archetype": archetype}
                anchor_priority = 0

                # Couple occurrence to month history + recent run-length.
                cap_stats = month_caps.get((str(uid), int(forecast_ds.month)))
                month_rate_caps = float(cap_stats.get("nonzero_rate", 1.0)) if cap_stats else 1.0
                run_length = float(_nonzero_run_length(y_orig))
                run_penalty = float(np.exp(-run_length / 6.0))
                if archetype == "seasonal" and month_rate_caps >= 0.3:
                    run_penalty = max(run_penalty, 0.7)

                month_rate_blend = 0.6 * month_rate_same + 0.4 * month_rate_caps
                p_effective = float(np.clip(p_nonzero * max(0.05, month_rate_blend) * run_penalty, 0.0, 1.0))
                p_min = max(0.05, 0.5 * month_rate_blend)

                if nonzero_count_last_12 >= 3:
                    floor_raw = 0.5 * float(mean_nonzero_12) + 0.5 * float(croston_mean)
                else:
                    floor_raw = float(croston_mean)
                beta = 0.7 if p_effective > 0.7 else 0.6
                floor = beta * floor_raw
                amount_used = yhat_amount_nz if yhat_amount_nz is not None else yhat_amount
                yhat = float(p_effective * amount_used + (1.0 - p_effective) * floor)
                if p_effective < max(occurrence_threshold, p_min):
                    if archetype == "seasonal" and month_rate_caps >= 0.3:
                        yhat = max(yhat, 0.3 * floor)
                        adjustments["soft_zero_floor"] = float(0.3 * floor)
                    else:
                        yhat = 0.0
                adjustments["amount_model"] = "nonzero" if yhat_amount_nz is not None else "full"
                adjustments["occurrence_floor"] = floor

                # Conditional-on-nonzero level anchor
                if p_effective > 0.5 and mean_nonzero_12 > 0.0:
                    anchor = 0.6 * float(mean_nonzero_12)
                    if yhat < anchor:
                        yhat = float(anchor)
                        adjustments["nonzero_level_anchor"] = anchor
                        anchor_priority = max(anchor_priority, 1)
                adjustments["mean_nonzero_12"] = mean_nonzero_12
                adjustments["median_nonzero_12"] = median_nonzero_12
                adjustments["last_nonzero_value"] = last_nz_val

                # Improvement #1: recent-level anchor (skip in seasonal-off months)
                if (
                    recent_level > 0.0
                    and (archetype != "seasonal" or month_rate_same >= 0.25)
                    and (last_nonzero_age <= 9)
                    and not (
                        archetype == "seasonal"
                        and month_rate_same >= 0.25
                        and float(m_stats.get("max", 0.0)) > 1.5 * float(recent_level)
                    )
                    and anchor_priority < 3
                ):
                    ramp_up = False
                    nz_full_vals = y_orig[y_orig > 0.0]
                    if len(nz_full_vals) >= 3:
                        last3 = nz_full_vals[-3:]
                        ramp_up = bool(last3[0] < last3[1] < last3[2])
                    slope_thresh = 0.05 * max(recent_level, 1.0)
                    regime_active = bool(recent_slope > slope_thresh and ramp_up and int(h) <= 6)
                    if archetype == "seasonal" and month_rate_same >= 0.25:
                        alpha = 0.95
                    elif archetype == "seasonal":
                        alpha = 0.9
                    elif archetype == "noisy":
                        alpha = 0.5
                    else:
                        alpha = 0.7
                    if regime_active:
                        alpha = min(0.98, alpha + 0.05)
                        adjustments["regime_ramp_up"] = True
                    yhat = float(alpha * yhat + (1.0 - alpha) * recent_level)
                    adjustments["recent_level"] = recent_level
                    adjustments["alpha"] = alpha
                    anchor_priority = max(anchor_priority, 1)

                # Seasonal peak anchor: lift in-season peaks toward historical max
                if archetype == "seasonal" and month_rate_same >= 0.25:
                    peak_alpha = 0.5
                    peak_target = float(m_stats.get("max", 0.0))
                    peak_mean = float(m_stats.get("mean", 0.0))
                    if peak_mean > 0.0 and peak_target > 3.0 * peak_mean:
                        peak_target = 3.0 * peak_mean
                    yhat = float((1.0 - peak_alpha) * yhat + peak_alpha * peak_target)
                    if peak_mean > 0.0 and yhat < 0.8 * peak_mean:
                        yhat = float(0.8 * peak_mean)
                        adjustments["seasonal_mean_floor"] = float(0.8 * peak_mean)
                    adjustments["peak_alpha"] = peak_alpha
                    adjustments["peak_target"] = peak_target
                    anchor_priority = max(anchor_priority, 3)

                # Seasonal amplitude memory: lift in high-amplitude months
                seasonal_amp_ratio = float(m_stats.get("max", 0.0)) / max(mean_last12_item, 1.0)
                if month_rate_same >= 0.3 and seasonal_amp_ratio > 1.5:
                    amp_target = 0.6 * float(m_stats.get("max", 0.0))
                    if yhat < amp_target:
                        yhat = float(amp_target)
                        adjustments["seasonal_amp_anchor"] = amp_target
                        anchor_priority = max(anchor_priority, 3)
                adjustments["seasonal_amplitude_ratio"] = seasonal_amp_ratio

                # Improvement #2: Croston-style floor for intermittent alive items
                if (
                    0.1 < overall_nonzero_rate < 0.5
                    and last_nonzero_age <= 9
                    and nonzero_count_last_12 >= 2
                    and anchor_priority < 3
                ):
                    floor = 0.4 * float(croston_mean)
                    if yhat < floor:
                        yhat = float(floor)
                        adjustments["croston_floor"] = floor
                        anchor_priority = max(anchor_priority, 2)

                # Improvement #3: horizon shrinkage (stable/noisy only)
                if archetype in {"stable", "noisy"} and not adjustments.get("regime_ramp_up"):
                    if archetype == "noisy":
                        beta = 0.05
                    else:
                        beta = 0.02
                    shrink = max(0.0, 1.0 - beta * math.log1p(float(h)))
                    if yhat > recent_level:
                        yhat = float(yhat * shrink)
                        adjustments["shrink"] = shrink

                # Trend memory for stable/nonseasonal items (small nudge)
                if archetype == "stable":
                    trend_weight = 0.2
                    trend_adjust = float(trend_weight * recent_slope * min(float(h), 6.0))
                    if trend_adjust != 0.0:
                        yhat = float(yhat + trend_adjust)
                        adjustments["trend_adjust"] = trend_adjust

                # Level floor for stable items with low CV and flat trend
                if archetype == "stable" and cv_val <= 0.6:
                    flat_thresh = 0.05 * max(recent_level, 1.0)
                    if abs(recent_slope) <= flat_thresh:
                        level_floor = 0.7 * float(recent_level)
                        if yhat < level_floor:
                            if anchor_priority < 2:
                                yhat = float(level_floor)
                                adjustments["level_floor"] = level_floor
                                anchor_priority = max(anchor_priority, 1)
                adjustments["anchor_priority"] = anchor_priority

                # Improvement #4: classical override in narrow cases
                classical_pred = float(np.mean([croston_mean, adida_mean]))
                near_zero = yhat <= max(1.0, 0.1 * max(recent_level, 1.0))
                classical_floor = 0.4 * float(croston_mean)
                classical_cap = max(5.0, 2.0 * max(recent_level, 1.0))
                cv_threshold = 0.8 if archetype == "intermittent" else 0.5
                if (
                    near_zero
                    and cv_val <= cv_threshold
                    and classical_floor <= classical_pred <= classical_cap
                    and last_nonzero_age <= 9
                    and archetype != "seasonal"
                ):
                    yhat = float(classical_pred)
                    adjustments["classical_override"] = classical_pred

                cap_stats = month_caps.get((str(uid), int(forecast_ds.month)))
                if cap_stats is not None:
                    max_y = float(cap_stats.get("max_y", 0.0))
                    nonzero_rate = float(cap_stats.get("nonzero_rate", 0.0))
                    cap_low = max(max_y, cap_small_floor)
                    if nonzero_rate < cap_nonzero_threshold:
                        yhat = min(yhat, cap_low)
                    cap_mult = cap_multiplier
                    if archetype == "seasonal" and month_rate_same >= 0.25:
                        cap_mult = 4.0
                    if adjustments.get("regime_ramp_up"):
                        cap_mult = cap_mult * 1.5
                    cap = max(cap_mult * max_y, cap_small_floor)
                    yhat = min(yhat, cap)
                    adjustments["cap"] = cap
                    adjustments["cap_mult"] = cap_mult

                # Final non-negative clamp
                if yhat < 0.0:
                    yhat = 0.0
                    adjustments["clamped_nonneg"] = True

                # Upper 95% bound: use quantile model if available, otherwise fallback to residual-based
                if h in q95_boosters:
                    # Predict quantile directly using same X as point forecast
                    # Quantile model was trained on y_orig, so prediction is already on original scale
                    q95_model_pred = float(q95_boosters[h].predict(X)[0])
                    q95_pred = float(np.maximum(0.0, q95_model_pred))  # Non-negative clamp

                    # MEDIUM PRIORITY: Add trend adjustment for consistency with point forecast
                    # Even though quantile model trains on y_orig, we add trend in inference
                    # to match the point forecast's trend adjustment logic
                    if detrend_method == "linear" and params:
                        trend_future = _trend_model_value(forecast_ds, params)
                        q95_pred = float(np.maximum(0.0, q95_pred + trend_future))
                    elif detrend_method == "log1p_linear" and params:
                        trend_future = _trend_model_value(forecast_ds, params)
                        q95_pred = float(np.maximum(0.0, np.expm1(q95_pred + trend_future)))
                    # If detrend_method == "none", no trend adjustment needed

                    # HIGH PRIORITY: Apply occurrence model adjustment to upper bound
                    # Point forecast gets zeroed when p_effective < threshold, but quantile model won't.
                    # We need to apply the same occurrence logic for consistency.
                    # Use p_effective from point forecast calculation (already computed above)
                    # Apply conservative floor (0.2) to ensure upper bound reflects some uncertainty even for sparse items
                    p_effective_upper = float(np.clip(p_effective, 0.2, 1.0))  # Conservative floor at 0.2
                    
                    # Scale quantile prediction to be more reasonable relative to point forecast
                    # Quantile model learns full distribution (including zeros), but point forecast is occurrence-adjusted
                    # The quantile model predicts the 95th percentile of the raw distribution, which can be much higher
                    # than the occurrence-adjusted point forecast. We need to scale it down to match the point forecast scale.
                    if yhat > 0 and q95_pred > 0:
                        # Calculate how much the occurrence adjustment reduced the forecast
                        # Point forecast: yhat = p_effective * amount_used + (1-p_effective) * floor
                        # Raw amount (before occurrence): amount_used
                        # Scale factor: yhat / amount_used (how much smaller is yhat vs raw amount)
                        amount_used = yhat_amount_nz if yhat_amount_nz is not None else yhat_amount
                        if amount_used > 0:
                            # How much smaller is the final forecast vs the raw amount?
                            occurrence_scale = yhat / amount_used
                        else:
                            occurrence_scale = 1.0
                        
                        # Apply similar scaling to quantile prediction
                        # But be conservative - don't scale below 60% to maintain uncertainty
                        quantile_scale = max(occurrence_scale, 0.6)  # Don't scale below 60% of original
                        q95_pred_scaled = q95_pred * quantile_scale
                        
                        # Apply occurrence adjustment to scaled quantile
                        upper_95_raw = p_effective_upper * q95_pred_scaled + (1.0 - p_effective_upper) * floor
                    else:
                        # If yhat is zero or very small, use conservative floor-based upper bound
                        upper_95_raw = max(floor, q95_pred * 0.4)  # More conservative scaling for zero forecasts

                    # MEDIUM PRIORITY: Fix cap order - apply monotonicity BEFORE cap
                    # First ensure upper_95 >= yhat (monotonicity constraint)
                    # Then apply cap (this ensures cap doesn't violate monotonicity)
                    # Also ensure upper_95 isn't unreasonably high (max 2.5x yhat for better calibration)
                    # This prevents the upper bound from being 3x+ the point forecast
                    if yhat > 0:
                        max_reasonable = max(yhat * 2.5, floor * 2.0)  # Max 2.5x yhat (more conservative)
                        upper_95_raw = min(upper_95_raw, max_reasonable)
                    
                    upper_95 = float(max(yhat, upper_95_raw, floor))  # Monotonicity: upper_95 >= yhat
                    if "cap" in adjustments:
                        upper_95 = float(min(upper_95, float(adjustments["cap"])))  # Then apply cap

                    # Final non-negative clamp
                    if upper_95 < 0.0:
                        upper_95 = 0.0
                    # Ensure upper_95 is always set (not None) - use yhat as minimum with reasonable multiplier
                    if upper_95 is None or (upper_95 == 0.0 and yhat > 0.0):
                        upper_95 = float(max(yhat * 1.5, floor * 1.5, 1.0))  # Conservative fallback
                    adjustments["upper_95_method"] = "quantile_model"
                    adjustments["p_effective_upper"] = p_effective_upper
                else:
                    # Fallback to existing residual-based calculation
                    q = residual_quantiles.get((archetype, int(h)))
                    if q is None:
                        q = residual_quantiles.get(("__global__", int(h)))
                    if q is None:
                        # fallback: small uplift when calibration missing
                        fallback_base = max(1.0, 0.3 * max(recent_level, 0.0), 0.2 * float(croston_mean))
                        q95_excess = float(fallback_base)
                        adjustments["q95_fallback"] = True
                    else:
                        q95_excess = float(q)

                    upper_95 = float(max(yhat, yhat + q95_excess))

                    if "cap" in adjustments:
                        upper_95 = float(min(upper_95, float(adjustments["cap"])))
                    if upper_95 < 0.0:
                        upper_95 = 0.0
                    # Ensure upper_95 is always set (not None) - use yhat as minimum with reasonable multiplier
                    if upper_95 is None or (upper_95 == 0.0 and yhat > 0.0):
                        upper_95 = float(max(yhat * 1.5, floor * 1.5, 1.0))  # Conservative fallback
                    adjustments["upper_95_method"] = "residual_calibration"

                forecasts.append(
                    {
                        "unique_id": str(uid),
                        "ds": forecast_ds,
                        "yhat": yhat,
                        # Provide a small set of aligned upper-quantiles for downstream inventory logic.
                        # upper_70/upper_90 are derived deterministically from upper_95 to avoid retraining
                        # additional quantile models. (Inventory layer can refine later.)
                        "upper_70": float(max(yhat, yhat + 0.4 * max(0.0, float(upper_95) - yhat), 0.0)),
                        "upper_90": float(max(yhat, yhat + 0.8 * max(0.0, float(upper_95) - yhat), 0.0)),
                        "upper_95": upper_95,
                        "adjustments": adjustments,
                    }
                )

        fcst = pd.DataFrame(forecasts)
        if not fcst.empty:
            fcst["item_id"] = fcst["unique_id"]
            # Month-start semantics: 2026-01-01 represents the total for January 2026.
            fcst["ds"] = pd.to_datetime(fcst["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
            fcst["day"] = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d")
            cols = ["item_id", "day", "yhat"]
            if "upper_70" in fcst.columns:
                cols.append("upper_70")
            if "upper_90" in fcst.columns:
                cols.append("upper_90")
            if "upper_95" in fcst.columns:
                cols.append("upper_95")
            fcst = fcst[cols]

        meta = {
            "model_version": model_version,
            "freq": "MS",
            "strategy": "direct",
            "detrend_method": detrend_method,
            "date_semantics": "month_start_represents_month",
            "occurrence_model": "shared" if occ_model is not None else None,
            "occurrence_threshold": occurrence_threshold if occ_model is not None else None,
            "month_cap": {
                "lookback_years": 4,
                "cap_multiplier": cap_multiplier,
                "nonzero_threshold": cap_nonzero_threshold,
                "small_floor": cap_small_floor,
            },
            "_code_version": _LIGHTGBM_FORECASTS_VERSION,  # Debug marker
        }
        return fcst, meta
