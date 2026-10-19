from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable
import math

import numpy as np
import pandas as pd

from services.duckdb_model_store import DuckDBModelStore, ModelVersionRow
from inventory_algorithm.classical_forecasts import simple_croston_mean, simple_adida_mean
from inventory_algorithm.lgbm_origin import (
    CONFIDENCE_DEFINITION,
    EVALUATION_PROTOCOL,
    FEATURE_SPEC_SCHEMA_VERSION,
    apply_expanding_te_to_rows,
    apply_te_maps_to_frame,
    assign_split_role,
    context_version,
    decide_item_eligibility,
    demand_through,
    label_is_observable,
    month_start,
    origin_split_masks,
    split_rolling_origins,
    wape_with_undefined,
    signed_bias,
)
from inventory_algorithm.lgbm_grid import (
    INTERIOR_GAP_POLICY_OBSERVED_SALES,
    TARGET_OBSERVED_SALES,
    assert_finite_outputs,
    complete_monthly_grid,
    months_since_last_sale,
    seasonal_naive_value,
    validate_monthly_request,
)
from inventory_algorithm.lgbm_sf_join import join_sf_features, normalize_sf_cv_frame, normalize_sf_forecast_frame
from inventory_algorithm.lgbm_predict import (
    AUDIT_POLICY,
    AdjustmentPolicy,
    MEAN_HEAD_HURDLE,
    MEAN_HEAD_TWEEDIE,
    PLANNING_POLICY,
    SEASONAL_SERVE_POLICY,
    apply_adjustment_policy,
    compose_mean,
    coverage_audit,
    cumulative_quantile_not_sum,
    enforce_quantile_order,
    predict_from_heads,
    quantile_to_absolute,
    select_mean_head,
)
from inventory_algorithm.lgbm_drivers import MissingRequiredDriver, attach_target_month_drivers, normalize_drivers
from inventory_algorithm.lgbm_explain import explain_keys, pred_contrib_payload


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
_LIGHTGBM_FORECASTS_VERSION = "2026-09-11-schema-v2-seasonal-quality"

# Independent LGBM quantile heads for inventory upper bounds (safety stock).
UPPER_QUANTILE_ALPHAS: tuple[float, ...] = (0.7, 0.9, 0.95)
NUM_BOOST_ROUND_DEFAULT = 2000
NUM_BOOST_ROUND_NO_VAL = 200
MIN_FULL_TRAIN_ROWS = 50
MIN_REDUCED_TRAIN_ROWS = 10
UNKNOWN_NAME_CLUSTER = -1
PEER_CONTEXT_RECORDS_FORMAT = "records_v2"

# Min validation rows per archetype to use archetype-specific conformal margin (else pool __global__)
CONFORMAL_MARGIN_MIN_N: int = 5

# Target encoding: always encode name/flavour; also encode other high-card categoricals.
TE_MIN_UNIQUES: int = 10
TE_MAX_UNIQUES: int = 5000

# Auto-segment: pool generic categoricals in this cardinality band for cross-item stats.
AUTO_SEGMENT_MIN_UNIQUES: int = 3
AUTO_SEGMENT_MAX_UNIQUES: int = 200

# Identity fields must not be model features: unique_id/name let trees memorize
# the SKU and block pooling by flavour/group/brand (and break new items).
NON_MODEL_FEATURE_COLS = frozenset({
    "y",
    "decision_ds",
    "forecast_ds",
    "y_orig",
    "trend_model",
    "lag_1_orig",
    "yhat_snaive",
    "split_role",
    "fit_cutoff_ds",
    "unique_id",
    "item_id",
    "name",
    "sample_weight",
    "y_observed",
})
IDENTITY_CAT_COLS = frozenset({"unique_id", "item_id", "name"})

# Batch/SHAP need a decision month. Own lags may be zeros; pooling
# (flavour/group/segment) supplies the signal for new items.
MIN_INFERENCE_HISTORY_POINTS: int = 1


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
    weight_train: np.ndarray | None = None,
    weight_val: np.ndarray | None = None,
    categorical_feature: Any = "auto",
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

        cat_kw: dict[str, Any] = {}
        if categorical_feature not in (None, "auto"):
            cat_kw["categorical_feature"] = categorical_feature
        dtrain = lgb.Dataset(X_train, label=y_train, weight=weight_train, free_raw_data=False, **cat_kw)
        dval = lgb.Dataset(X_val, label=y_val, weight=weight_val, reference=dtrain, free_raw_data=False, **cat_kw)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        if LightGBMPruningCallback is not None:
            callbacks.append(LightGBMPruningCallback(trial, "l2", valid_name="val"))

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
            if not np.isfinite(mse):
                raise ValueError("nonfinite mse")
            return mse
        except Exception as exc:
            if _HAS_OPTUNA and optuna is not None and isinstance(exc, optuna.exceptions.TrialPruned):
                raise
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

    finite_trials = [
        t
        for t in study.trials
        if getattr(t, "value", None) is not None and np.isfinite(float(t.value))
    ]
    if not finite_trials:
        raise ValueError("Optuna search failed: all trials nonfinite")
    try:
        best_params = study.best_params
    except ValueError as exc:
        raise ValueError("Optuna search failed: all trials nonfinite") from exc
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


def _apply_target_encodings_to_rows(
    rows: list[dict[str, Any]],
    te_cols: list[str],
    te_maps: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach ``*_te`` columns onto already-built training/explain rows."""
    if not rows or not te_cols or not te_maps:
        return rows
    df = pd.DataFrame(rows)
    df = _apply_target_encodings(df, te_cols, te_maps)
    return df.to_dict(orient="records")


def _model_feature_columns(columns) -> list[str]:
    return [c for c in list(columns) if c not in NON_MODEL_FEATURE_COLS]


def _categorical_feature_columns(static_cat_cols: list[str]) -> list[str]:
    """Categoricals the booster may split on: shared product attrs, not SKU id."""
    return [
        c
        for c in static_cat_cols
        if c not in IDENTITY_CAT_COLS and not str(c).endswith("_te")
    ]


def _prepare_lgbm_X(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cat_cols: list[str],
    mappings: dict[str, dict[str, int]],
    static_interaction_cols: list[str] | None,
) -> pd.DataFrame:
    """Encode categoricals and align to ``feature_cols`` without identity columns.

    Interaction columns created by ``_add_static_interactions`` are appended when
    they are not already listed in ``feature_cols`` (training). At inference the
    spec already includes them, so they stay in the trained order.
    """
    present = [c for c in feature_cols if c in df.columns]
    X = df.loc[:, present].copy()
    X, _ = _encode_categories(X, [c for c in cat_cols if c in X.columns], mappings=mappings)
    X = _add_static_interactions(X, static_interaction_cols)
    extra = [c for c in X.columns if c not in feature_cols]
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    return X[list(feature_cols) + extra].fillna(0)


def _quantile_artifact_suffix(alpha: float) -> str:
    return f"q{int(round(float(alpha) * 100))}"


def _target_encoding_column_selection(static_cols: list[str], base: pd.DataFrame) -> list[str]:
    """Choose static columns for target encoding: name/flavour always; other high-card strings."""
    te_always = {"name", "flavour"}
    out: list[str] = []
    skip = {"name_cluster_id", "item_id", "unique_id"}
    for c in static_cols:
        if c in skip or c not in base.columns:
            continue
        if c in te_always:
            out.append(c)
            continue
        s = base[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)):
            continue
        nu = int(s.nunique(dropna=True))
        if nu < TE_MIN_UNIQUES or nu > TE_MAX_UNIQUES:
            continue
        out.append(c)
    return list(dict.fromkeys(out))


def _auto_segment_columns_from_generic(
    generic_cols: list[str],
    base: pd.DataFrame,
) -> list[str]:
    """Promote medium-cardinality generic categoricals to segment pooling."""
    auto: list[str] = []
    for c in generic_cols:
        if c not in base.columns:
            continue
        s = base[c]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)):
            continue
        nu = int(s.nunique(dropna=True))
        if AUTO_SEGMENT_MIN_UNIQUES <= nu <= AUTO_SEGMENT_MAX_UNIQUES:
            auto.append(c)
    return list(dict.fromkeys(auto))


def _pooled_excess_quantiles_time_cv(
    g: pd.DataFrame,
    alphas: tuple[float, ...],
    *,
    time_cv_folds: int,
) -> dict[float, float]:
    """Pooled quantiles of positive_excess with optional time-block CV (mean across folds)."""
    if g is None or g.empty or "positive_excess" not in g.columns:
        return {a: 0.0 for a in alphas}
    vals_all = pd.to_numeric(g["positive_excess"], errors="coerce").to_numpy(dtype=float)
    vals_all = vals_all[np.isfinite(vals_all)]
    if len(vals_all) == 0:
        return {a: 0.0 for a in alphas}
    if time_cv_folds <= 1 or len(g) < 40 or "ds" not in g.columns:
        return {a: float(np.quantile(vals_all, a)) for a in alphas}

    g2 = g.copy()
    g2["_dsn"] = pd.to_datetime(g2["ds"], errors="coerce")
    g2 = g2.dropna(subset=["_dsn"])
    if g2.empty:
        return {a: float(np.quantile(vals_all, a)) for a in alphas}
    uniq_ds = sorted(g2["_dsn"].unique())
    k = len(uniq_ds)
    if k < time_cv_folds * 2:
        return {a: float(np.quantile(vals_all, a)) for a in alphas}

    out: dict[float, float] = {}
    for a in alphas:
        fold_qs: list[float] = []
        for f in range(time_cv_folds):
            lo = f * k // time_cv_folds
            hi = (f + 1) * k // time_cv_folds
            if lo >= hi:
                continue
            subset = set(uniq_ds[lo:hi])
            mask = g2["_dsn"].isin(subset)
            vals = pd.to_numeric(g2.loc[mask, "positive_excess"], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                fold_qs.append(float(np.quantile(vals, a)))
        out[a] = float(np.mean(fold_qs)) if fold_qs else float(np.quantile(vals_all, a))
    return out


def _finalize_conformal_margins_lookup(
    score_lists: dict[tuple[str, int, float], list[float]],
    alphas: tuple[float, ...],
) -> dict[tuple[str, int], dict[float, float]]:
    """Build (archetype, horizon) -> alpha -> margin to add to LGBM excess (split-conformal style).

    Scores on calibration: max(0, y_true - (yhat_base + q_pred)). Margin per group is the
    alpha-quantile of scores; sparse archetypes fall back to __global__ for same horizon.
    """
    if not score_lists:
        return {}
    q_margin: dict[tuple[str, int, float], float] = {}
    for key, scores in score_lists.items():
        arr = np.asarray(scores, dtype=float)
        if len(arr) == 0:
            q_margin[key] = 0.0
        else:
            _arch, _h, alpha = key
            q_margin[key] = float(np.quantile(arr, float(alpha)))

    out: dict[tuple[str, int], dict[float, float]] = {}
    horizons = sorted({k[1] for k in score_lists})
    for h in horizons:
        for alpha in alphas:
            gk = ("__global__", h, alpha)
            g_margin = float(q_margin.get(gk, 0.0))
            arches = {k[0] for k in score_lists if k[1] == h and k[0] != "__global__"}
            for arch in arches:
                ak = (arch, h, alpha)
                sc = score_lists.get(ak, [])
                if len(sc) >= CONFORMAL_MARGIN_MIN_N:
                    out.setdefault((arch, h), {})[alpha] = float(q_margin.get(ak, g_margin))
                else:
                    out.setdefault((arch, h), {})[alpha] = g_margin
            out.setdefault(("__global__", h), {})[alpha] = g_margin
    return out


def _postprocess_upper_from_quantile_model(
    q_model_pred: float,
    *,
    detrend_method: str,
    params: dict[str, Any] | None,
    forecast_ds: pd.Timestamp,
    yhat: float,
    p_effective: float,
    floor: float,
    yhat_amount_nz: float | None,
    yhat_amount: float,
    cap: float | None,
) -> float:
    """Upper bound = yhat + predicted positive excess (quantile heads trained on residual target).

    Quantile models predict max(0, y_orig - yhat_point_reconstructed) in original scale; the
    served yhat already includes occurrence blending and post-hoc adjustments. Unused kwargs
    are kept for stable call sites.
    """
    _ = (detrend_method, params, forecast_ds, p_effective, yhat_amount_nz, yhat_amount)
    excess = float(np.maximum(0.0, float(q_model_pred)))
    upper = float(yhat + excess)
    upper = max(upper, float(yhat), float(floor))
    if cap is not None:
        upper = min(upper, float(cap))
    if upper < 0.0:
        upper = 0.0
    return upper


def _enforce_upper_quantile_monotonicity(
    yhat: float,
    upper_70: float,
    upper_90: float,
    upper_95: float,
    floor: float,
) -> tuple[float, float, float]:
    u70 = float(max(yhat, floor, upper_70))
    u90 = float(max(yhat, floor, upper_90))
    u95 = float(max(yhat, floor, upper_95))
    u90 = max(u70, u90)
    u95 = max(u90, u95)
    return u70, u90, u95


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
    return {
        "__default__": UNKNOWN_NAME_CLUSTER,
        "mapping": mapping,
        "k": int(k),
        "unseen": "unknown_cluster",
        "unseen_note": (
            "Unseen names map to a dedicated unknown cluster; they are not hashed into "
            "trained KMeans ids. Hash buckets, if used, must be trained as hash features."
        ),
    }, k


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
    default = int((cluster_map or {}).get("__default__", UNKNOWN_NAME_CLUSTER))

    def _assign(name: str) -> int:
        key = str(name)
        if key in mapping:
            return int(mapping[key])
        return default

    df["name_cluster_id"] = df[name_col].astype(str).fillna("").map(_assign).astype(int)
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
        "item_group2",
        "brand",
        "category",
        "sub_category",
        "supplier_group",
        "coverage_group",
        "flavour",
        "color",
        "colour",
        "size",
        "item_type",
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
    skip_exact = {"name", "name_cluster_id"}

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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _segment_key_kind(val: Any) -> str:
    if isinstance(val, (bool, np.bool_)):
        return "bool"
    if isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
        return "int"
    if isinstance(val, (float, np.floating)):
        return "float"
    return "str"


def _restore_segment_key(raw: Any, kind: str) -> Any:
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return bool(raw)
    return str(raw)


def serialize_segment_stats(segment_stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for col, by_seg in dict(segment_stats or {}).items():
        for seg_val, by_month in dict(by_seg or {}).items():
            kind = _segment_key_kind(seg_val)
            stored = seg_val if kind in {"int", "float", "str", "bool"} else str(seg_val)
            for month, payload in dict(by_month or {}).items():
                rec = {
                    "column": str(col),
                    "segment_id": stored,
                    "segment_id_kind": kind,
                    "month": int(month),
                }
                rec.update({k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in dict(payload or {}).items()})
                records.append(rec)
    return records


def deserialize_segment_stats(blob: Any) -> dict[str, dict[Any, dict[int, dict[str, float]]]]:
    """Restore segment_stats without collapsing distinct '5' vs 5 keys or month 1 vs '1'."""
    out: dict[str, dict[Any, dict[int, dict[str, float]]]] = {}
    if isinstance(blob, list):
        rows = blob
    elif isinstance(blob, dict) and blob.get("segment_stats_format") == PEER_CONTEXT_RECORDS_FORMAT:
        rows = list(blob.get("segment_stats_records") or [])
    elif isinstance(blob, dict) and "segment_stats_records" in blob:
        rows = list(blob.get("segment_stats_records") or [])
    else:
        rows = []
        nested = blob if isinstance(blob, dict) else {}
        for col, by_seg in nested.items():
            if not isinstance(by_seg, dict):
                continue
            for seg_raw, by_month in by_seg.items():
                if not isinstance(by_month, dict):
                    continue
                kind = _segment_key_kind(seg_raw)
                try:
                    key = _restore_segment_key(seg_raw, kind)
                except Exception:
                    key = str(seg_raw)
                for m_raw, payload in by_month.items():
                    out.setdefault(str(col), {}).setdefault(key, {})[int(m_raw)] = {
                        str(k): float(v) for k, v in dict(payload or {}).items() if v is not None and str(v) != ""
                    }
        return out
    for r in rows:
        col = str(r.get("column"))
        kind = str(r.get("segment_id_kind") or "str")
        key = _restore_segment_key(r.get("segment_id"), kind)
        month = int(r.get("month"))
        payload = {
            k: float(v)
            for k, v in r.items()
            if k not in {"column", "segment_id", "segment_id_kind", "month"} and v is not None
        }
        out.setdefault(col, {}).setdefault(key, {})[month] = payload
    return out


def serialize_peer_context(
    *,
    customer_id: str,
    context_version: str,
    origin: pd.Timestamp,
    global_monthly: dict[Any, float],
    global_yoy: dict[str, Any],
    segment_stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "context_version": context_version,
        "origin": pd.Timestamp(origin).strftime("%Y-%m-%d"),
        "global_monthly": {str(int(k)): float(v) for k, v in dict(global_monthly or {}).items()},
        "global_monthly_key_kind": "int",
        "global_yoy": global_yoy,
        "segment_stats_format": PEER_CONTEXT_RECORDS_FORMAT,
        "segment_stats_records": serialize_segment_stats(segment_stats),
        "grid_policy": "MS_complete_interior_zeros",
        "interior_gap_policy": INTERIOR_GAP_POLICY_OBSERVED_SALES,
        "target": TARGET_OBSERVED_SALES,
    }


def deserialize_peer_context(blob: dict[str, Any]) -> dict[str, Any]:
    gm_raw = dict(blob.get("global_monthly") or {})
    global_monthly = {int(k): float(v) for k, v in gm_raw.items()}
    if blob.get("segment_stats_format") == PEER_CONTEXT_RECORDS_FORMAT or "segment_stats_records" in blob:
        segment_stats = deserialize_segment_stats(blob)
    else:
        segment_stats = deserialize_segment_stats(blob.get("segment_stats") or {})
    return {
        "global_monthly": global_monthly,
        "global_yoy": dict(blob.get("global_yoy") or {}),
        "segment_stats": segment_stats,
        "context_version": blob.get("context_version"),
        "origin": blob.get("origin"),
    }


def load_peer_context_or_raise(artifact_root: str, spec: dict[str, Any]) -> dict[str, Any]:
    schema = int(spec.get("schema_version") or 1)
    peer_path = os.path.join(artifact_root, str(spec.get("peer_context_file") or "peer_context.json"))
    if not os.path.exists(peer_path):
        if schema >= 2:
            raise ValueError(
                f"Missing required peer_context.json for schema v{schema} at {peer_path}"
            )
        return {}
    with open(peer_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    if not isinstance(blob, dict):
        if schema >= 2:
            raise ValueError("peer_context.json is incompatible (not an object)")
        return {}
    try:
        return deserialize_peer_context(blob)
    except Exception as exc:
        if schema >= 2:
            raise ValueError(f"Incompatible peer_context.json: {exc}") from exc
        return {}


def _peer_stats_as_of(
    base: pd.DataFrame,
    origin: pd.Timestamp,
    segment_cols: list[str],
    trend_cols: list[str],
    *,
    lookback_months: int = 36,
) -> dict[str, Any]:
    hist = demand_through(base, origin)
    return {
        "global_monthly": _compute_global_monthly_effects(hist, lookback_months=lookback_months),
        "segment_stats": _precompute_segment_monthly_stats(
            hist, segment_cols, lookback_months=lookback_months
        )
        if segment_cols
        else {},
        "global_yoy": _compute_global_trend_by_attribute(hist, trend_cols) if trend_cols else {},
    }


def _make_peer_lookup(
    base: pd.DataFrame,
    segment_cols: list[str],
    trend_cols: list[str],
) -> Callable[[pd.Timestamp], dict[str, Any]]:
    cache: dict[pd.Timestamp, dict[str, Any]] = {}

    def _lookup(origin: pd.Timestamp) -> dict[str, Any]:
        key = month_start(origin)
        if key not in cache:
            cache[key] = _peer_stats_as_of(base, key, segment_cols, trend_cols)
        return cache[key]

    return _lookup


def _frozen_peer_lookup(peer_context: dict[str, Any]) -> Callable[[pd.Timestamp], dict[str, Any]]:
    if peer_context.get("segment_stats_format") == PEER_CONTEXT_RECORDS_FORMAT or "segment_stats_records" in peer_context:
        frozen = deserialize_peer_context(peer_context)
    elif "global_monthly" in peer_context and any(isinstance(k, str) for k in dict(peer_context.get("global_monthly") or {})):
        frozen = deserialize_peer_context(peer_context)
    else:
        gm_raw = dict(peer_context.get("global_monthly") or {})
        global_monthly = {}
        for k, v in gm_raw.items():
            try:
                global_monthly[int(k)] = float(v)
            except Exception:
                continue
        frozen = {
            "global_monthly": global_monthly,
            "global_yoy": dict(peer_context.get("global_yoy") or {}),
            "segment_stats": deserialize_segment_stats(peer_context.get("segment_stats") or {}),
        }

    def _lookup(_origin: pd.Timestamp) -> dict[str, Any]:
        return frozen

    return _lookup


def _detrend_with_params(
    ds: np.ndarray,
    y_orig: np.ndarray,
    params: dict[str, Any] | None,
    method: str,
) -> np.ndarray:
    method = (method or "none").strip().lower()
    if method == "none" or not params:
        return np.asarray(y_orig, dtype=float)
    if method == "linear":
        trend = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds], dtype=float)
        return np.asarray(y_orig, dtype=float) - trend
    yt = np.log1p(np.maximum(0.0, np.asarray(y_orig, dtype=float)))
    trend = np.array([_trend_model_value(pd.Timestamp(d), params) for d in ds], dtype=float)
    return yt - trend


def _label_model_scale(y_orig_h: float, forecast_ds: pd.Timestamp, params: dict[str, Any] | None, method: str) -> float:
    method = (method or "none").strip().lower()
    y = float(y_orig_h)
    if method == "none" or not params:
        return y
    trend = float(_trend_model_value(forecast_ds, params))
    if method == "linear":
        return y - trend
    return float(np.log1p(max(0.0, y)) - trend)


def _policy_from_spec(spec: dict[str, Any]) -> AdjustmentPolicy:
    raw = spec.get("adjustment_policy")
    schema = int(spec.get("schema_version") or 1)
    if not isinstance(raw, dict):
        return SEASONAL_SERVE_POLICY if schema >= 2 else PLANNING_POLICY
    policy = AdjustmentPolicy(
        enabled=bool(raw.get("enabled", False)),
        occurrence_blend=bool(raw.get("occurrence_blend", False)),
        nonzero_level_anchor=bool(raw.get("nonzero_level_anchor", False)),
        recent_level_blend=bool(raw.get("recent_level_blend", False)),
        seasonal_floor=bool(raw.get("seasonal_floor", False)),
        croston_floor=bool(raw.get("croston_floor", False)),
        month_cap=bool(raw.get("month_cap", False)),
        non_negative=bool(raw.get("non_negative", True)),
    )
    # Schema v2 defaulted to rules-off. That left December-only items at 0.
    if schema >= 2 and policy.as_dict() == AUDIT_POLICY.as_dict():
        return SEASONAL_SERVE_POLICY
    return policy


def _mean_head_for_horizon(spec_or_map: dict[str, Any], h: int, default: str | None = None) -> str:
    by_h = spec_or_map.get("mean_head_by_horizon") or {}
    key = str(int(h))
    if key in by_h:
        return str(by_h[key])
    if int(h) in by_h:
        return str(by_h[int(h)])
    return str(spec_or_map.get("mean_head") or default or MEAN_HEAD_TWEEDIE)


def _quantile_direct_demand(spec: dict[str, Any]) -> bool:
    if "quantile_direct_demand" in spec:
        return bool(spec.get("quantile_direct_demand"))
    return int(spec.get("schema_version") or 1) >= 2


def _to_original_scale(yhat_model: np.ndarray | float, detrend_method: str, trend_val: np.ndarray | float | None) -> np.ndarray:
    y = np.asarray(yhat_model, dtype=float).ravel()
    method = (detrend_method or "none").strip().lower()
    if method == "none" or trend_val is None:
        return np.maximum(0.0, y)
    trend = np.asarray(trend_val, dtype=float).ravel()
    if trend.size == 1 and y.size > 1:
        trend = np.full_like(y, float(trend[0]))
    if method == "linear":
        return np.maximum(0.0, y + trend)
    return np.maximum(0.0, np.expm1(y + trend))


def _n_threads() -> int | None:
    try:
        if lgb is None:
            return None
        getter = getattr(lgb, "get_num_threads", None)
        if callable(getter):
            return int(getter())
    except Exception:
        return None
    return None


def _emit_phase(progress_hook: Callable[[dict[str, Any]], None] | None, phase: str, **fields: Any) -> float:
    now = time.perf_counter()
    payload = {"phase": phase, **fields}
    if progress_hook:
        progress_hook(payload)
    return now


def _lgb_fit(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    w_val: np.ndarray | None,
    cat_cols: list[str],
    *,
    num_boost_round: int = NUM_BOOST_ROUND_DEFAULT,
    early_stop: int = 100,
    no_val_rounds: int = NUM_BOOST_ROUND_NO_VAL,
    best_iteration_hint: int | None = None,
) -> Any:
    cats = [c for c in cat_cols if c in X_train.columns]
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=w_train,
        categorical_feature=cats or "auto",
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        dval = lgb.Dataset(
            X_val,
            label=y_val,
            weight=w_val,
            categorical_feature=[c for c in cats if c in X_val.columns] or "auto",
            free_raw_data=False,
        )
        valid_sets.append(dval)
        valid_names.append("val")
        rounds = int(num_boost_round)
        callbacks = [lgb.early_stopping(int(early_stop), verbose=False)]
    else:
        hint = int(best_iteration_hint) if best_iteration_hint else int(no_val_rounds)
        rounds = max(1, hint)
        callbacks = None
    return lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def _predictions_for_mask(
    *,
    X: pd.DataFrame,
    df: pd.DataFrame,
    mask: np.ndarray,
    booster: Any,
    nz_booster: Any | None,
    occ_model: Any | None,
    q_boosters: dict[float, Any] | None,
    detrend_method: str,
    head: str,
    policy: AdjustmentPolicy,
    q_margins: dict[float, float] | None = None,
    quantile_direct: bool = True,
) -> list:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    if len(idx) == 0:
        return []
    X_m = X.iloc[idx] if hasattr(X, "iloc") else X[mask]
    yhat_model = np.asarray(booster.predict(X_m), dtype=float)
    trend = (
        df.iloc[idx]["trend_model"].to_numpy(dtype=float)
        if "trend_model" in df.columns
        else np.zeros(len(idx), dtype=float)
    )
    yhat_tweedie = _to_original_scale(yhat_model, detrend_method, trend)
    p_occ = (
        np.clip(np.asarray(occ_model.predict(X_m), dtype=float), 0.0, 1.0)
        if occ_model is not None
        else np.full(len(idx), np.nan)
    )
    yhat_pos = (
        np.maximum(0.0, np.asarray(nz_booster.predict(X_m), dtype=float))
        if nz_booster is not None
        else None
    )
    q_raw: dict[float, np.ndarray] = {}
    if q_boosters:
        for a, qb in q_boosters.items():
            q_raw[float(a)] = np.asarray(qb.predict(X_m), dtype=float)
    out = []
    margins = q_margins or {}
    month_means = (
        pd.to_numeric(df.iloc[idx]["same_month_mean_3y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "same_month_mean_3y" in df.columns
        else np.zeros(len(idx), dtype=float)
    )
    for i in range(len(idx)):
        yt = float(yhat_tweedie[i])
        yp = None if yhat_pos is None else float(yhat_pos[i])
        po = None if not np.isfinite(p_occ[i]) else float(p_occ[i])
        raw, _ = compose_mean(head=head, yhat_tweedie=yt, yhat_positive=yp, p_occ=po)
        q70 = q90 = q95 = None
        if 0.7 in q_raw and 0.9 in q_raw and 0.95 in q_raw:
            q70 = quantile_to_absolute(float(q_raw[0.7][i]), raw, direct_demand=quantile_direct)
            q90 = quantile_to_absolute(float(q_raw[0.9][i]), raw, direct_demand=quantile_direct)
            q95 = quantile_to_absolute(float(q_raw[0.95][i]), raw, direct_demand=quantile_direct)
            if q70 is not None:
                q70 = float(q70) + float(margins.get(0.7, 0.0))
            if q90 is not None:
                q90 = float(q90) + float(margins.get(0.9, 0.0))
            if q95 is not None:
                q95 = float(q95) + float(margins.get(0.95, 0.0))
        out.append(
            predict_from_heads(
                head=head,
                yhat_tweedie=yt,
                yhat_positive=yp,
                p_occ=po,
                policy=policy,
                q70=q70,
                q90=q90,
                q95=q95,
                month_mean=float(month_means[i]),
            )
        )
    return out


def _human_static_value(col: str, raw: Any, mappings: dict[str, dict[str, int]]) -> Any:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    mapping = mappings.get(col) or {}
    if mapping:
        reverse = {int(v): str(k) for k, v in mapping.items()}
        try:
            coded = int(raw)
            if coded in reverse:
                return reverse[coded]
        except Exception:
            pass
        if str(raw) in mapping:
            return str(raw)
    if isinstance(raw, (np.floating, float)) and float(raw).is_integer():
        return str(int(raw)) if mapping else raw
    return raw if isinstance(raw, (str, int, float, bool)) else str(raw)


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
        month_map = None
        if val in col_data:
            month_map = col_data.get(val)
        elif isinstance(val, (np.integer,)) and int(val) in col_data:
            month_map = col_data.get(int(val))
        elif isinstance(val, (np.str_,)) and str(val) in col_data:
            month_map = col_data.get(str(val))
        month_data = None if month_map is None else month_map.get(int(m))
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


def _trailing_same_month_stats(
    y_orig: np.ndarray,
    months: np.ndarray,
    t: int,
    *,
    lookback: int = 36,
) -> dict[int, dict[str, float]]:
    """Per-calendar-month stats from the trailing `lookback` observations ending at index t.

    Only uses history up to and including index t, so it is leak-safe for
    training rows and identical to what inference sees at the series end.
    """
    start = max(0, t + 1 - lookback)
    vals = y_orig[start : t + 1]
    mns = months[start : t + 1]
    out: dict[int, dict[str, float]] = {}
    for m in range(1, 13):
        hist_vals = vals[mns == m]
        if len(hist_vals) == 0:
            out[m] = {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0}
        else:
            out[m] = {
                "mean": float(np.mean(hist_vals)),
                "max": float(np.max(hist_vals)),
                "nonzero_rate": float(np.mean(hist_vals > 0.0)),
            }
    return out


def _assemble_feature_row(
    *,
    unique_id: str,
    t: int,
    h: int,
    ds: np.ndarray,
    y_model: np.ndarray,
    y_orig: np.ndarray,
    lags: list[int],
    roll_windows: list[int],
    static: dict[str, Any],
    exo: dict[str, np.ndarray],
    same_month_stats: dict[int, dict[str, float]],
    lifecycle_feats: dict[str, float],
    calendar_cols: list[str],
    trend_cols: list[str],
    segment_cols: list[str],
    global_monthly: dict[int, float] | None,
    global_yoy: dict[str, float] | None,
    segment_stats: dict[str, dict[Any, dict[int, dict[str, float]]]] | None,
) -> dict[str, Any]:
    """Build the feature row for a forecast made at decision index t for horizon h.

    Single source of truth shared by training-row generation, batch inference
    and SHAP explain so the three paths cannot drift apart. All trailing
    statistics must be computed from history up to and including index t
    (callers pass `same_month_stats` / `lifecycle_feats` computed that way).
    All slices use safe indexing so short series produce shorter windows
    instead of wrapping around.
    """
    decision_ds = pd.Timestamp(ds[t])
    forecast_ds = decision_ds + pd.offsets.MonthBegin(int(h))

    r: dict[str, Any] = {
        "unique_id": str(unique_id),
        "decision_ds": decision_ds,
        "forecast_ds": forecast_ds,
        "horizon": int(h),
    }

    m = int(forecast_ds.month)
    r["month"] = m
    r["quarter"] = int(((m - 1) // 3) + 1)
    r["year"] = int(forecast_ds.year)

    fourier_feats = _fourier_features(m, k=3)
    r.update(fourier_feats)
    r["month_sin"] = fourier_feats["month_sin_1"]
    r["month_cos"] = fourier_feats["month_cos_1"]
    qs, qc = _quarter_sin_cos(int(r["quarter"]))
    r["quarter_sin"] = qs
    r["quarter_cos"] = qc
    r["year_idx"] = float(forecast_ds.year - pd.Timestamp(ds[0]).year)

    r.update(_easter_features(forecast_ds))

    m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
    roll_slope_y = float(_rolling_slope(y_model[: t + 1]))

    if calendar_cols:
        r.update(
            _calendar_proximity_features(
                m,
                static,
                calendar_cols,
                forecast_ds=forecast_ds,
            )
        )
    if trend_cols:
        r.update(
            _trend_attribute_features(
                year_idx=r["year_idx"],
                rolling_12_slope=roll_slope_y,
                static=static,
                trend_cols=trend_cols,
                global_yoy=global_yoy,
            )
        )
    if segment_cols and segment_stats:
        r.update(
            _segment_features_for_row(
                m,
                static,
                segment_cols,
                segment_stats,
                float(m_stats.get("mean", 0.0)),
            )
        )
    r.update(lifecycle_feats)

    for lag in lags:
        idx = t - (lag - 1)
        r[f"lag_{lag}"] = float(y_model[idx]) if idx >= 0 else 0.0
    r["lag_1_orig"] = float(y_orig[t])
    r["yhat_snaive"] = seasonal_naive_value(y_orig, t, int(h))

    for w in roll_windows:
        start = max(0, t - (w - 1))
        window = y_model[start : t + 1]
        r[f"roll_mean_{w}"] = float(np.mean(window))
        if w >= 2:
            r[f"roll_std_{w}"] = float(np.std(window, ddof=0))

    r["diff1"] = float(y_model[t] - y_model[t - 1]) if t >= 1 else 0.0
    r["diff12"] = float(y_model[t] - y_model[t - 12]) if t >= 12 else 0.0

    last12 = y_orig[max(0, t - 11) : t + 1]
    r["zero_ratio_12"] = float(np.mean(last12 == 0.0))
    r["nonzero_run_length"] = float(_nonzero_run_length(y_orig[: t + 1]))
    r["months_since_last_sale"] = float(months_since_last_sale(y_orig, t))

    window_nz = y_orig[max(0, t - 11) : t + 1]
    nz_vals = window_nz[window_nz > 0.0]
    r["mean_nonzero_12"] = float(np.mean(nz_vals)) if len(nz_vals) else 0.0
    r["median_nonzero_12"] = float(np.median(nz_vals)) if len(nz_vals) else 0.0
    nz_hist = y_orig[: t + 1]
    nz_hist_vals = nz_hist[nz_hist > 0.0]
    r["last_nonzero_value"] = float(nz_hist_vals[-1]) if len(nz_hist_vals) else 0.0

    r["same_month_mean_3y"] = float(m_stats["mean"])
    r["same_month_max_3y"] = float(m_stats["max"])
    r["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
    r["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

    mean_last12 = float(np.mean(y_orig[max(0, t - 11) : t + 1]))
    r["seasonal_amplitude_ratio"] = float(m_stats["max"]) / max(mean_last12, 1.0)

    if global_monthly is not None:
        global_month_level = float(global_monthly.get(m, 0.0))
        r["global_month_level"] = global_month_level
        item_month_mean = float(m_stats.get("mean", 0.0))
        if global_month_level > 0.0:
            r["item_vs_global_ratio"] = float(item_month_mean / global_month_level)
        else:
            r["item_vs_global_ratio"] = 1.0 if item_month_mean == 0.0 else float(item_month_mean)
    else:
        r["global_month_level"] = 0.0
        r["item_vs_global_ratio"] = 1.0

    r["rolling_12_slope"] = roll_slope_y

    for k, arr in exo.items():
        # Origin-time value only. Target-month known-future drivers are attached
        # separately via vintage-aware attach_target_month_drivers.
        if len(arr) <= 0 or not (0 <= t < len(arr)):
            r[k] = 0.0
        else:
            r[k] = float(arr[t])

    for k, v in static.items():
        r[k] = v

    return r


def _build_direct_rows_for_item_all_h(
    *,
    unique_id: str,
    ds: np.ndarray,
    y: np.ndarray,
    y_orig: np.ndarray,
    trend_params: dict[str, Any] | None,
    exo: dict[str, np.ndarray],
    static: dict[str, Any],
    horizon: int,
    lags: list[int],
    roll_windows: list[int],
    calendar_cols: list[str] | None = None,
    trend_cols: list[str] | None = None,
    segment_cols: list[str] | None = None,
    global_monthly: dict[int, float] | None = None,
    global_yoy: dict[str, float] | None = None,
    segment_stats: dict[str, dict[Any, dict[int, dict[str, float]]]] | None = None,
    peer_lookup: Callable[[pd.Timestamp], dict[str, Any]] | None = None,
    y_observed: np.ndarray | None = None,
    detrend_method: str | None = None,
    drivers_wide: pd.DataFrame | None = None,
    driver_columns: list[str] | None = None,
    required_drivers: list[str] | None = None,
    history_end: pd.Timestamp | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Build supervised rows for all horizons in one pass over decision points.

    Trailing same-month and lifecycle statistics are recomputed per decision
    point t (history up to t only) so training rows match exactly what
    inference computes at the series end and contain no look-ahead leakage.
    """
    calendar_cols = calendar_cols or []
    trend_cols = trend_cols or []
    segment_cols = segment_cols or []
    driver_columns = driver_columns or []
    required_drivers = required_drivers or []
    method = (detrend_method or "none").strip().lower()

    rows_by_h: dict[int, list[dict[str, Any]]] = {hh: [] for hh in range(1, int(horizon) + 1)}
    n = len(y_orig)
    max_needed = max(max(roll_windows) - 1, 1) if roll_windows else 1
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    hist_end = month_start(history_end) if history_end is not None else month_start(ds[-1])

    for t in range(max_needed, n - 1):
        origin = month_start(ds[t])
        gm = global_monthly
        gy = global_yoy
        ss = segment_stats
        if peer_lookup is not None:
            peers = peer_lookup(origin)
            gm = peers.get("global_monthly")
            gy = peers.get("global_yoy")
            ss = peers.get("segment_stats")

        tp = trend_params
        y_model = y
        if method != "none":
            tp = _fit_trend_params(ds[: t + 1], y_orig[: t + 1], method)
            y_model = _detrend_with_params(ds, y_orig, tp, method)

        same_month_stats = _trailing_same_month_stats(y_orig, months, t)
        lifecycle_feats = _detect_lifecycle_phase(y_orig[: t + 1])

        for hh in range(1, int(horizon) + 1):
            if t + hh > n - 1:
                break
            if not label_is_observable(origin, hh, hist_end):
                break
            if y_observed is not None and int(y_observed[t + hh]) == 0:
                continue
            row = _assemble_feature_row(
                unique_id=unique_id,
                t=t,
                h=hh,
                ds=ds,
                y_model=y_model,
                y_orig=y_orig,
                lags=lags,
                roll_windows=roll_windows,
                static=static,
                exo=exo,
                same_month_stats=same_month_stats,
                lifecycle_feats=lifecycle_feats,
                calendar_cols=calendar_cols,
                trend_cols=trend_cols,
                segment_cols=segment_cols,
                global_monthly=gm,
                global_yoy=gy,
                segment_stats=ss,
            )
            if driver_columns:
                try:
                    attach_target_month_drivers(
                        row,
                        drivers_wide,
                        columns=driver_columns,
                        required=required_drivers,
                    )
                except MissingRequiredDriver:
                    continue

            trend_model_at_forecast = 0.0
            if tp is not None:
                trend_model_at_forecast = _trend_model_value(row["forecast_ds"], tp)
            row["y"] = _label_model_scale(float(y_orig[t + hh]), row["forecast_ds"], tp, method)
            row["y_orig"] = float(y_orig[t + hh])
            row["trend_model"] = float(trend_model_at_forecast)
            if y_observed is not None:
                row["y_observed"] = int(y_observed[t + hh])

            rows_by_h[hh].append(row)

    return rows_by_h


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
    if "cutoff" in cv.columns:
        keep.append("cutoff")
    out = cv.loc[:, [c for c in keep if c in cv.columns]].copy()
    out["sf_available"] = 1.0
    out = out.rename(columns=rename)
    out = out.dropna(subset=["unique_id", "ds"]).reset_index(drop=True)
    try:
        return normalize_sf_cv_frame(out)
    except ValueError:
        return pd.DataFrame()


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
    origin = pd.to_datetime(base["ds"], errors="coerce").max()
    if pd.isna(origin):
        return pd.DataFrame()
    try:
        return normalize_sf_forecast_frame(out, decision_ds=origin)
    except ValueError:
        return pd.DataFrame()


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
        residual_time_cv_folds: int = 2,
        tune_hyperparameters: bool = False,
        optuna_n_trials: int = 30,
        optuna_timeout: int = 300,
        progress_hook: Callable[[dict[str, Any]], None] | None = None,
        required_driver_columns: list[str] | None = None,
    ) -> TrainResult:
        # Monthly-only for v1
        _ = freq  # kept for signature compatibility
        required_driver_columns = list(required_driver_columns or [])

        model_version = model_version or datetime.now(UTC).strftime("v%Y_%m_%d_%H%M%S")
        train_t0 = time.perf_counter()

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
        t_agg = time.perf_counter()
        base = _normalize_monthly_history(hist)
        validate_monthly_request(freq=freq, horizon=int(horizon), item_attributes=item_attributes)
        life_cols = {"active_from", "closed_at"}
        lifecycle = None
        if item_attributes is not None and not item_attributes.empty and life_cols.intersection(item_attributes.columns):
            lifecycle = item_attributes
        base = complete_monthly_grid(
            base, lifecycle=lifecycle, interior_gap_policy=INTERIOR_GAP_POLICY_OBSERVED_SALES
        )

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


        # Merge long drivers (optional) → wide monthly. Known-future values are
        # joined onto the target month in the row builder, not the decision month.
        drivers_wide = pd.DataFrame()
        if drivers is not None and not drivers.empty:
            drivers_wide = normalize_drivers(drivers)
            wide = _pivot_long_drivers(drivers) if "driver_name" in drivers.columns else drivers_wide
            if not wide.empty:
                if "item_id" in wide.columns:
                    wide["item_id"] = wide["item_id"].astype(str)
                if drivers_wide.empty:
                    drivers_wide = wide.copy()
                merge_keys = ["item_id", "ds"] if "item_id" in wide.columns else ["ds"]
                hist_drivers = wide.merge(
                    base.loc[:, [c for c in merge_keys if c in base.columns]].drop_duplicates(),
                    on=merge_keys,
                    how="inner",
                ) if all(k in wide.columns for k in merge_keys) else wide
                if not hist_drivers.empty:
                    drop_meta = [c for c in ("vintage", "vintage_ds", "driver_name", "driver_value") if c in hist_drivers.columns]
                    hist_drivers = hist_drivers.drop(columns=drop_meta, errors="ignore")
                    if "item_id" in hist_drivers.columns:
                        base = base.merge(hist_drivers, on=["item_id", "ds"], how="left")
                    else:
                        base = base.merge(hist_drivers, on=["ds"], how="left")

        # Determine exogenous columns from wide data
        base_cols = {"item_id", "unique_id", "ds", "y", "y_observed"}
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
        segment_cols = list(routed["segment"])
        generic_interaction_cols = list(routed["generic"])
        auto_segment_cols = _auto_segment_columns_from_generic(generic_interaction_cols, base)
        segment_cols = list(dict.fromkeys([*segment_cols, *auto_segment_cols]))
        generic_interaction_cols = [c for c in generic_interaction_cols if c not in set(auto_segment_cols)]

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
            progress_hook({"phase": "feature_gen_start", "elapsed_s": round(time.perf_counter() - train_t0, 3)})
        t_feat = time.perf_counter()

        # Build supervised rows across all items for each horizon
        all_rows_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in range(1, int(horizon) + 1)}

        detrend_method = (detrend_method or "none").strip().lower()
        if detrend_method not in {"none", "linear", "log1p_linear"}:
            raise ValueError("detrend_method must be one of: none, linear, log1p_linear")

        trend_params_by_uid: dict[str, dict[str, Any]] = {}

        base = base.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
        history_end_ds = month_start(pd.to_datetime(base["ds"], errors="coerce").max())
        splits = split_rolling_origins(
            history_end_ds,
            tune_months=max(1, int(val_months) // 2 or 1),
            calibration_months=max(1, int(val_months) // 2 or 1),
            audit_months=max(1, min(3, int(val_months))),
        )
        fit_cutoff_ds = splits.fit_cutoff
        weight_base = demand_through(base, fit_cutoff_ds)
        demand_by_uid = (
            weight_base.groupby("unique_id")["y"].sum().astype(float).to_dict()
            if weight_base is not None and not weight_base.empty else {}
        )
        weight_map = {str(uid): 1.0 / (float(total) + 1.0) for uid, total in demand_by_uid.items()}
        if weight_map:
            weights = np.array(list(weight_map.values()), dtype=float)
            w_med = float(np.median(weights)) if len(weights) else 1.0
            if w_med <= 0:
                w_med = 1.0
            for uid in list(weight_map.keys()):
                weight_map[uid] = float(np.clip(weight_map[uid] / w_med, 0.5, 3.0))

        peer_lookup = _make_peer_lookup(base, segment_cols, trend_cols)
        peer_at_end = _peer_stats_as_of(base, history_end_ds, segment_cols, trend_cols)
        global_monthly_effects = peer_at_end["global_monthly"]
        global_yoy = peer_at_end["global_yoy"]
        segment_stats = peer_at_end["segment_stats"]
        ctx_version = context_version(
            {
                "grid": "MS_complete_interior_zeros",
                "history_end_ds": str(history_end_ds.date()),
                "fit_cutoff_ds": str(fit_cutoff_ds.date()),
                "segment_cols": list(segment_cols),
                "trend_cols": list(trend_cols),
                "peer": "origin_safe_as_of",
            }
        )

        # Archetype assignment for residual pooling (as-of fit cutoff, not full series)
        archetype_by_uid: dict[str, str] = {}
        arch_base = demand_through(base, fit_cutoff_ds)
        for uid, grp in arch_base.groupby("unique_id", sort=False):
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
            y_obs = grp["y_observed"].to_numpy(dtype=int) if "y_observed" in grp.columns else np.ones(len(y_orig), dtype=int)

            trend_params = _fit_trend_params(ds[pd.to_datetime(ds) <= fit_cutoff_ds], y_orig[pd.to_datetime(ds) <= fit_cutoff_ds], detrend_method) if np.any(pd.to_datetime(ds) <= fit_cutoff_ds) else _fit_trend_params(ds, y_orig, detrend_method)
            trend_params_by_uid[str(uid)] = trend_params
            y_model = _detrend_with_params(ds, y_orig, trend_params, detrend_method)

            exo = {c: pd.to_numeric(grp[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in exogenous_columns}

            static: dict[str, Any] = {}
            for c in static_cols:
                v = grp[c].iloc[-1] if c in grp.columns else None
                static[c] = v

            rows_by_h = _build_direct_rows_for_item_all_h(
                unique_id=str(uid),
                ds=ds,
                y=y_model,
                y_orig=y_orig,
                trend_params=trend_params,
                exo=exo,
                static=static,
                horizon=int(horizon),
                lags=lags,
                roll_windows=roll_windows,
                calendar_cols=calendar_cols,
                trend_cols=trend_cols,
                segment_cols=segment_cols,
                global_monthly=global_monthly_effects,
                global_yoy=global_yoy,
                segment_stats=segment_stats,
                peer_lookup=peer_lookup,
                y_observed=y_obs,
                detrend_method=detrend_method,
                drivers_wide=drivers_wide if not drivers_wide.empty else None,
                driver_columns=list(exogenous_columns or []),
                required_drivers=list(required_driver_columns),
                history_end=history_end_ds,
            )
            for h, rows in rows_by_h.items():
                all_rows_by_h[h].extend(rows)

        _emit_phase(
            progress_hook,
            "feature_gen_complete",
            elapsed_s=round(time.perf_counter() - t_feat, 3),
            n_rows=int(sum(len(v) for v in all_rows_by_h.values())),
            fit_cutoff_ds=pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
        )

        # Add per-item StatsForecast features keyed by (item, decision_ds, forecast_ds)
        t_sf = time.perf_counter()
        _emit_phase(progress_hook, "sf_start")
        sf_feats = _build_statsforecast_features(base, max_h=int(horizon), season_length=12)
        sf_unavailable_reason = None
        if sf_feats.empty:
            sf_unavailable_reason = "statsforecast_cv_unavailable"
        for h in range(1, int(horizon) + 1):
            rows = all_rows_by_h.get(h) or []
            if not rows:
                continue
            df_h = pd.DataFrame(rows)
            df_h = join_sf_features(df_h, sf_feats)
            all_rows_by_h[h] = df_h.to_dict(orient="records")
        _emit_phase(
            progress_hook,
            "sf_complete",
            elapsed_s=round(time.perf_counter() - t_sf, 3),
            sf_unavailable_reason=sf_unavailable_reason,
        )

        # Train one model per horizon. Write to a staging directory, then rename.
        existing_root = self.store.get_model_artifact_root(model_version)
        if existing_root:
            with self.store.connect() as con:
                existing = con.execute(
                    "SELECT status FROM model_versions WHERE customer_id=? AND model_version=? LIMIT 1",
                    [self.store.customer_id, model_version],
                ).fetchone()
            if existing and str(existing[0]) == "prod":
                raise ValueError(f"Refusing to overwrite existing prod model_version={model_version}")
        final_artifact_root = os.path.join(self.store.customer_dir(), "artifacts", model_version)
        artifact_root = final_artifact_root + ".partial"
        if os.path.exists(artifact_root):
            shutil.rmtree(artifact_root)
        os.makedirs(artifact_root, exist_ok=True)

        metrics_summary: dict[str, Any] = {
            "strategy": "direct",
            "horizon": int(horizon),
            "val_months": int(val_months),
            "detrend_method": detrend_method,
            "residual_time_cv_folds": int(max(1, residual_time_cv_folds)),
            "upper_quantile_alphas": list(UPPER_QUANTILE_ALPHAS),
        }
        if auto_stats:
            metrics_summary.update(auto_stats)

        models: dict[int, Any] = {}
        feature_cols: list[str] | None = None

        # Only treat truly categorical static columns as categorical — never SKU id.
        cat_cols = _categorical_feature_columns(static_cat_cols)

        # Build stable category mappings once (reused across horizons)
        mappings: dict[str, dict[str, int]] = {}
        for col in cat_cols:
            values = base[col].astype(str).fillna("").unique().tolist()
            values = sorted(set(values))
            mappings[col] = {v: i for i, v in enumerate(values)}

        if not any(all_rows_by_h.values()):
            raise ValueError("No training rows after feature generation (check history length).")

        # Expanding/chronological target encoding: each train row sees maps fit only
        # through its own decision_ds. Persist maps as-of fit_cutoff for serving.
        te_cols = _target_encoding_column_selection(static_cols, base)
        te_maps, _ = _compute_target_encodings(demand_through(base, fit_cutoff_ds), te_cols, prior=10.0)
        base = _apply_target_encodings(base, te_cols, te_maps)
        te_feature_cols = [f"{c}_te" for c in te_cols]
        static_cols = static_cols + [c for c in te_feature_cols if c not in static_cols]
        if te_cols:
            for h, rows in list(all_rows_by_h.items()):
                all_rows_by_h[h] = apply_expanding_te_to_rows(rows, base, te_cols, prior=10.0)

        def _origin_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return origin_split_masks(frame, splits)

        # Eligibility uses calibration origins. Audit is frozen scoring only.
        per_item_eval: dict[str, dict[str, list]] = {}
        cv_residual_rows: list[dict[str, Any]] = []
        conformal_score_lists: dict[tuple[str, int, float], list[float]] = {}
        mean_head_by_horizon: dict[str, str] = {}
        nz_models: dict[int, Any] = {}
        q_models: dict[int, dict[float, Any]] = {}
        last_best_iteration: int | None = None

        # Resolve feature columns once from the first non-empty horizon
        # base_feature_cols: columns from raw rows (used for selecting from raw DataFrames)
        # feature_cols: full list including interaction columns (used for training/spec)
        base_feature_cols: list[str] | None = None
        if feature_cols is None:
            for rows in all_rows_by_h.values():
                if rows:
                    df_spec = pd.DataFrame(rows)
                    base_feature_cols = _model_feature_columns(df_spec.columns)
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
        t_occ = time.perf_counter()
        _emit_phase(progress_hook, "occurrence_start")
        if occ_rows:
            occ_df = pd.DataFrame(occ_rows)
            occ_df = _add_sample_weights(occ_df, weight_map)
            y_occ_src = occ_df["y_orig"] if "y_orig" in occ_df.columns else occ_df["y"]
            y_occ = (pd.to_numeric(y_occ_src, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0).astype(int)
            is_train_occ, is_tune_occ, _is_cal_occ, _is_audit_occ = _origin_masks(occ_df)

            X_occ = _prepare_lgbm_X(
                occ_df,
                base_feature_cols,
                cat_cols=cat_cols,
                mappings=mappings,
                static_interaction_cols=static_interaction_cols,
            )
            feature_cols = list(X_occ.columns)

            X_occ_train = X_occ[is_train_occ]
            y_occ_train = y_occ[is_train_occ]
            X_occ_val = X_occ[is_tune_occ]
            y_occ_val = y_occ[is_tune_occ]
            w_occ = occ_df["sample_weight"].to_numpy(dtype=float)
            w_occ_train = w_occ[is_train_occ]
            w_occ_val = w_occ[is_tune_occ]

            n_occ_train = int(len(X_occ_train))
            if n_occ_train >= MIN_REDUCED_TRAIN_ROWS and y_occ_train.sum() > 0 and (len(y_occ_train) - y_occ_train.sum()) > 0:
                pos = float(y_occ_train.sum())
                neg = float(len(y_occ_train) - pos)
                occ_params = {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "learning_rate": 0.05,
                    "num_leaves": 8 if n_occ_train < MIN_FULL_TRAIN_ROWS else 63,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "min_data_in_leaf": 1 if n_occ_train < MIN_FULL_TRAIN_ROWS else int(max(1, lgbm_min_data_in_leaf)),
                    "min_data_in_bin": int(max(1, lgbm_min_data_in_bin)),
                    "seed": 42,
                    "verbosity": -1,
                }
                occ_model = _lgb_fit(
                    occ_params,
                    X_occ_train,
                    y_occ_train,
                    w_occ_train,
                    X_occ_val,
                    y_occ_val,
                    w_occ_val,
                    cat_cols,
                    num_boost_round=1000,
                    early_stop=50,
                    no_val_rounds=NUM_BOOST_ROUND_NO_VAL,
                )
                occ_model.save_model(os.path.join(artifact_root, "lgbm_occurrence.txt"))
                metrics_summary["occurrence_model"] = "shared"
                metrics_summary["occurrence_threshold"] = 0.2
                metrics_summary["occurrence_pos_rate_train"] = float(pos / max(1.0, pos + neg))
                metrics_summary["occurrence_n_train"] = n_occ_train
                metrics_summary["occurrence_n_tune"] = int(len(X_occ_val))
        _emit_phase(
            progress_hook,
            "occurrence_complete",
            elapsed_s=round(time.perf_counter() - t_occ, 3),
            trained=bool(occ_model is not None),
        )

        # Optuna hyperparameter tuning (fit vs tune origins)
        tuned_params: dict[str, Any] | None = None
        if tune_hyperparameters and _HAS_OPTUNA and all_rows_by_h.get(1):
            t_tune = time.perf_counter()
            _emit_phase(progress_hook, "tuning_start", n_trials=optuna_n_trials)
            tune_df = pd.DataFrame(all_rows_by_h[1])
            tune_df = _add_sample_weights(tune_df, weight_map)
            is_tune_train, is_tune_val, _is_tune_cal, _is_tune_audit = _origin_masks(tune_df)
            X_tune = _prepare_lgbm_X(
                tune_df,
                feature_cols or base_feature_cols,
                cat_cols=cat_cols,
                mappings=mappings,
                static_interaction_cols=static_interaction_cols,
            )
            y_tune = tune_df["y"].to_numpy(dtype=float)
            w_tune = tune_df["sample_weight"].to_numpy(dtype=float)

            X_tune_train = X_tune[is_tune_train]
            y_tune_train = y_tune[is_tune_train]
            X_tune_val = X_tune[is_tune_val]
            y_tune_val = y_tune[is_tune_val]

            if len(X_tune_train) >= MIN_FULL_TRAIN_ROWS and len(X_tune_val) >= 10:
                tune_objective = "tweedie" if detrend_method == "none" else "regression"
                cat_for_tune = [c for c in cat_cols if c in X_tune_train.columns]
                tuned_params = _tune_lgbm_hyperparameters(
                    X_tune_train, y_tune_train,
                    X_tune_val, y_tune_val,
                    objective=tune_objective,
                    n_trials=optuna_n_trials,
                    timeout_seconds=optuna_timeout,
                    weight_train=w_tune[is_tune_train],
                    weight_val=w_tune[is_tune_val],
                    categorical_feature=cat_for_tune or "auto",
                )
                metrics_summary["tuned_hyperparameters"] = tuned_params
            _emit_phase(
                progress_hook,
                "tuning_complete",
                elapsed_s=round(time.perf_counter() - t_tune, 3),
                best_params=tuned_params,
                n_train=int(len(X_tune_train)),
                n_tune=int(len(X_tune_val)),
            )

        for h in range(1, int(horizon) + 1):
            rows = all_rows_by_h[h]
            if not rows:
                metrics_summary[f"support_insufficient_h{h}"] = True
                metrics_summary[f"wape_val_h{h}"] = None
                continue

            t_h = time.perf_counter()
            _emit_phase(
                progress_hook,
                "training_horizon_start",
                current_horizon=int(h),
                total_horizons=int(horizon),
                fit_cutoff_ds=pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
            )

            df = pd.DataFrame(rows)
            df = _add_sample_weights(df, weight_map)
            df["split_role"] = [assign_split_role(d, splits) for d in df["decision_ds"]]
            df["fit_cutoff_ds"] = pd.Timestamp(fit_cutoff_ds)
            y_target = df["y"].to_numpy(dtype=float)

            is_train, is_tune, is_cal, is_audit = _origin_masks(df)
            is_val = is_tune

            X = _prepare_lgbm_X(
                df,
                feature_cols or base_feature_cols,
                cat_cols=cat_cols,
                mappings=mappings,
                static_interaction_cols=static_interaction_cols,
            )
            feature_cols = list(X.columns)

            X_train = X[is_train]
            y_train = y_target[is_train]
            X_val = X[is_val]
            y_val = y_target[is_val]
            w_all = df["sample_weight"].to_numpy(dtype=float)
            w_train = w_all[is_train]
            w_val = w_all[is_val]
            n_train = int(len(X_train))
            n_tune = int(len(X_val))
            n_cal = int(np.sum(is_cal))
            n_audit = int(np.sum(is_audit))
            n_origins = int(pd.to_datetime(df.loc[is_train, "decision_ds"]).nunique()) if n_train else 0

            if n_train < MIN_REDUCED_TRAIN_ROWS:
                metrics_summary[f"support_insufficient_h{h}"] = True
                metrics_summary[f"wape_val_h{h}"] = None
                _emit_phase(
                    progress_hook,
                    "training_horizon_complete",
                    current_horizon=int(h),
                    skipped=True,
                    n_train=n_train,
                    elapsed_s=round(time.perf_counter() - t_h, 3),
                )
                continue

            if detrend_method == "none":
                objective = "tweedie"
            else:
                objective = "regression"

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
            reduced = n_train < MIN_FULL_TRAIN_ROWS
            if reduced:
                params["num_leaves"] = min(int(params.get("num_leaves", 63)), 8)
                params["min_data_in_leaf"] = 1
                metrics_summary[f"reduced_complexity_h{h}"] = True
            if objective == "tweedie":
                tweedie_power = tuned_params.get("tweedie_variance_power", 1.3) if tuned_params else 1.3
                params["tweedie_variance_power"] = tweedie_power

            booster = _lgb_fit(
                params,
                X_train,
                y_train,
                w_train,
                X_val,
                y_val,
                w_val,
                cat_cols,
                num_boost_round=NUM_BOOST_ROUND_DEFAULT,
                early_stop=100,
                no_val_rounds=NUM_BOOST_ROUND_NO_VAL,
                best_iteration_hint=last_best_iteration,
            )
            models[h] = booster
            best_it = getattr(booster, "best_iteration", None)
            if best_it:
                last_best_iteration = int(best_it)
            metrics_summary[f"best_iteration_h{h}"] = int(best_it) if best_it else None
            metrics_summary[f"early_stop_h{h}"] = bool(n_tune > 0)
            metrics_summary[f"n_train_h{h}"] = n_train
            metrics_summary[f"n_tune_h{h}"] = n_tune
            metrics_summary[f"n_cal_h{h}"] = n_cal
            metrics_summary[f"n_audit_h{h}"] = n_audit

            q_boosters_h: dict[float, Any] = {}
            y_q = np.maximum(
                0.0,
                df["y_orig"].to_numpy(dtype=float) if "y_orig" in df.columns else df["y"].to_numpy(dtype=float),
            )
            y_q_train = y_q[is_train]
            y_q_val = y_q[is_val] if n_tune > 0 else np.array([])
            for alpha in UPPER_QUANTILE_ALPHAS:
                params_q = dict(params)
                params_q["objective"] = "quantile"
                params_q["alpha"] = float(alpha)
                params_q["metric"] = "quantile"
                booster_q = _lgb_fit(
                    params_q,
                    X_train,
                    y_q_train,
                    w_train,
                    X_val if n_tune > 0 else None,
                    y_q_val if n_tune > 0 else None,
                    w_val if n_tune > 0 else None,
                    cat_cols,
                    num_boost_round=NUM_BOOST_ROUND_DEFAULT,
                    early_stop=100,
                    no_val_rounds=NUM_BOOST_ROUND_NO_VAL,
                    best_iteration_hint=last_best_iteration,
                )
                suf = _quantile_artifact_suffix(alpha)
                booster_q.save_model(os.path.join(artifact_root, f"lgbm_h{h}_{suf}.txt"))
                q_boosters_h[float(alpha)] = booster_q
            q_models[h] = q_boosters_h

            nz_booster = None
            nz_mask = pd.to_numeric(df.get("y_orig", df["y"]), errors="coerce").fillna(0.0) > 0.0
            if bool(nz_mask.any()):
                df_nz = df.loc[nz_mask].copy()
                df_nz = _add_sample_weights(df_nz, weight_map)
                is_train_nz, is_val_nz, _is_cal_nz, _is_audit_nz = _origin_masks(df_nz)
                n_nz_train = int(np.sum(is_train_nz))
                if n_nz_train >= MIN_REDUCED_TRAIN_ROWS:
                    y_nz = df_nz["y_orig"].to_numpy(dtype=float) if "y_orig" in df_nz.columns else df_nz["y"].to_numpy(dtype=float)
                    X_nz = _prepare_lgbm_X(
                        df_nz,
                        feature_cols or base_feature_cols,
                        cat_cols=cat_cols,
                        mappings=mappings,
                        static_interaction_cols=static_interaction_cols,
                    )
                    X_nz_train = X_nz[is_train_nz]
                    y_nz_train = y_nz[is_train_nz]
                    X_nz_val = X_nz[is_val_nz]
                    y_nz_val = y_nz[is_val_nz]
                    w_nz = df_nz["sample_weight"].to_numpy(dtype=float)
                    w_nz_train = w_nz[is_train_nz]
                    w_nz_val = w_nz[is_val_nz]
                    all_nonneg_nz = bool(np.all(y_nz_train >= 0) and (len(y_nz_val) == 0 or np.all(y_nz_val >= 0)))
                    objective_nz = "tweedie" if all_nonneg_nz else "regression"
                    params_nz = dict(params)
                    params_nz["objective"] = objective_nz
                    params_nz["metric"] = "l2"
                    if objective_nz == "tweedie":
                        params_nz["tweedie_variance_power"] = 1.3
                    if n_nz_train < MIN_FULL_TRAIN_ROWS:
                        params_nz["num_leaves"] = min(int(params_nz.get("num_leaves", 63)), 8)
                        params_nz["min_data_in_leaf"] = 1
                    nz_booster = _lgb_fit(
                        params_nz,
                        X_nz_train,
                        y_nz_train,
                        w_nz_train,
                        X_nz_val,
                        y_nz_val,
                        w_nz_val,
                        cat_cols,
                        num_boost_round=NUM_BOOST_ROUND_DEFAULT,
                        early_stop=100,
                        no_val_rounds=NUM_BOOST_ROUND_NO_VAL,
                        best_iteration_hint=last_best_iteration,
                    )
                    nz_booster.save_model(os.path.join(artifact_root, f"lgbm_h{h}_nz.txt"))
                    nz_models[h] = nz_booster

            # Calibration: choose mean_head on raw heads, score eligibility on the served policy.
            selected_head = MEAN_HEAD_TWEEDIE
            if n_cal > 0:
                cal_tweedie = _predictions_for_mask(
                    X=X, df=df, mask=is_cal, booster=booster, nz_booster=nz_booster,
                    occ_model=occ_model, q_boosters=q_boosters_h, detrend_method=detrend_method,
                    head=MEAN_HEAD_TWEEDIE, policy=AUDIT_POLICY, quantile_direct=True,
                )
                cal_hurdle = _predictions_for_mask(
                    X=X, df=df, mask=is_cal, booster=booster, nz_booster=nz_booster,
                    occ_model=occ_model, q_boosters=q_boosters_h, detrend_method=detrend_method,
                    head=MEAN_HEAD_HURDLE, policy=AUDIT_POLICY, quantile_direct=True,
                )
                y_true_cal = df.loc[is_cal, "y_orig"].to_numpy(dtype=float) if "y_orig" in df.columns else y_target[is_cal]
                yhat_tweedie_cal = np.array([p.yhat for p in cal_tweedie], dtype=float)
                yhat_hurdle_cal = np.array([p.yhat for p in cal_hurdle], dtype=float)
                wape_tweedie_cal = wape_with_undefined(y_true_cal, yhat_tweedie_cal)
                wape_hurdle_cal = wape_with_undefined(y_true_cal, yhat_hurdle_cal)
                metrics_summary[f"wape_calibration_tweedie_h{h}"] = wape_tweedie_cal.get("wape")
                metrics_summary[f"wape_calibration_hurdle_h{h}"] = wape_hurdle_cal.get("wape")
                selected_head = select_mean_head(y_true_cal, yhat_tweedie_cal, yhat_hurdle_cal)
                cal_preds = _predictions_for_mask(
                    X=X, df=df, mask=is_cal, booster=booster, nz_booster=nz_booster,
                    occ_model=occ_model, q_boosters=q_boosters_h, detrend_method=detrend_method,
                    head=selected_head, policy=SEASONAL_SERVE_POLICY, quantile_direct=True,
                )
                yhat_cal = np.array([p.yhat for p in cal_preds], dtype=float)
                metrics_summary[f"wape_calibration_h{h}"] = wape_with_undefined(y_true_cal, yhat_cal).get("wape")

                uids_cal = df.loc[is_cal, "unique_id"].astype(str).to_numpy()
                naive_cal = (
                    pd.to_numeric(df.loc[is_cal, "lag_1_orig"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if "lag_1_orig" in df.columns
                    else np.zeros(n_cal)
                )
                snaive_cal = (
                    pd.to_numeric(df.loc[is_cal, "yhat_snaive"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    if "yhat_snaive" in df.columns
                    else naive_cal
                )
                for i in range(n_cal):
                    uid = str(uids_cal[i])
                    bucket = per_item_eval.setdefault(
                        uid, {"y": [], "ml": [], "naive": [], "snaive": []}
                    )
                    bucket["y"].append(float(y_true_cal[i]))
                    bucket["ml"].append(float(yhat_cal[i]))
                    bucket["naive"].append(float(naive_cal[i]))
                    bucket["snaive"].append(float(snaive_cal[i]))

                for alpha in UPPER_QUANTILE_ALPHAS:
                    ups = []
                    for p in cal_preds:
                        if alpha == 0.7:
                            ups.append(p.upper_70)
                        elif alpha == 0.9:
                            ups.append(p.upper_90)
                        else:
                            ups.append(p.upper_95)
                    if any(u is None for u in ups):
                        continue
                    s_np = np.maximum(0.0, y_true_cal - np.asarray(ups, dtype=float))
                    for ii in range(len(s_np)):
                        au = archetype_by_uid.get(str(uids_cal[ii]), "unknown")
                        conformal_score_lists.setdefault((au, int(h), float(alpha)), []).append(float(s_np[ii]))
                        conformal_score_lists.setdefault(("__global__", int(h), float(alpha)), []).append(float(s_np[ii]))

            mean_head_by_horizon[str(int(h))] = selected_head
            metrics_summary[f"mean_head_h{h}"] = selected_head

            q_margins_h: dict[float, float] = {}
            calibrated_h = False
            for alpha in UPPER_QUANTILE_ALPHAS:
                scores = conformal_score_lists.get(("__global__", int(h), float(alpha))) or []
                if len(scores) >= 5:
                    q_margins_h[float(alpha)] = float(np.quantile(np.asarray(scores, dtype=float), float(alpha)))
                    calibrated_h = True
            metrics_summary[f"quantiles_calibrated_h{h}"] = calibrated_h

            # Audit: frozen head + frozen conformal margins. Never fall back to calibration.
            if n_audit > 0:
                audit_preds = _predictions_for_mask(
                    X=X, df=df, mask=is_audit, booster=booster, nz_booster=nz_booster,
                    occ_model=occ_model, q_boosters=q_boosters_h, detrend_method=detrend_method,
                    head=selected_head, policy=SEASONAL_SERVE_POLICY, q_margins=q_margins_h if calibrated_h else None,
                    quantile_direct=True,
                )
                y_true_audit = df.loc[is_audit, "y_orig"].to_numpy(dtype=float) if "y_orig" in df.columns else y_target[is_audit]
                yhat_audit = np.array([p.yhat for p in audit_preds], dtype=float)
                wape_audit = wape_with_undefined(y_true_audit, yhat_audit)
                metrics_summary[f"wape_val_h{h}"] = wape_audit.get("wape")
                metrics_summary[f"wape_audit_h{h}"] = wape_audit.get("wape")
                metrics_summary[f"wape_undefined_h{h}"] = bool(wape_audit.get("wape_undefined"))
                metrics_summary[f"bias_audit_h{h}"] = signed_bias(y_true_audit, yhat_audit)
                metrics_summary[f"support_insufficient_h{h}"] = False
                for alpha, getter in ((0.7, "upper_70"), (0.9, "upper_90"), (0.95, "upper_95")):
                    ups = np.array([getattr(p, getter) if getattr(p, getter) is not None else np.nan for p in audit_preds], dtype=float)
                    if np.all(np.isfinite(ups)):
                        cov = coverage_audit(y_true_audit, ups, after_caps=False)
                        suf = _quantile_artifact_suffix(alpha)
                        metrics_summary[f"{suf}_coverage_audit_h{h}"] = cov.get("coverage")
                        metrics_summary[f"{suf}_coverage_n_h{h}"] = cov.get("n")

                if "forecast_ds" in df.columns:
                    val_ds = pd.to_datetime(df.loc[is_audit, "forecast_ds"], errors="coerce")
                else:
                    val_ds = pd.Series([pd.NaT] * len(y_true_audit))
                val_uids = df.loc[is_audit, "unique_id"].astype(str).to_numpy()
                dec_ds = pd.to_datetime(df.loc[is_audit, "decision_ds"], errors="coerce")
                roles = df.loc[is_audit, "split_role"].astype(str).to_numpy()
                for i in range(len(y_true_audit)):
                    ds_val = val_ds.iloc[i]
                    if pd.isna(ds_val):
                        continue
                    y_true = float(y_true_audit[i])
                    yhat_i = float(yhat_audit[i])
                    residual = y_true - yhat_i
                    cv_residual_rows.append(
                        {
                            "model_version": model_version,
                            "unique_id": str(val_uids[i]),
                            "ds": ds_val.date().isoformat(),
                            "decision_ds": pd.Timestamp(dec_ds.iloc[i]).strftime("%Y-%m-%d") if pd.notna(dec_ds.iloc[i]) else None,
                            "forecast_ds": ds_val.date().isoformat(),
                            "fit_cutoff_ds": pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
                            "split_role": str(roles[i]),
                            "horizon": int(h),
                            "y": y_true,
                            "yhat": yhat_i,
                            "residual": float(residual),
                            "positive_excess": float(max(0.0, residual)),
                            "archetype": archetype_by_uid.get(str(val_uids[i]), "unknown"),
                            "fold_id": 0,
                        }
                    )
            else:
                metrics_summary[f"wape_val_h{h}"] = None
                metrics_summary[f"wape_audit_h{h}"] = None
                metrics_summary[f"support_insufficient_h{h}"] = True

            booster.save_model(os.path.join(artifact_root, f"lgbm_h{h}.txt"))
            _emit_phase(
                progress_hook,
                "training_horizon_complete",
                current_horizon=int(h),
                elapsed_s=round(time.perf_counter() - t_h, 3),
                n_train=n_train,
                n_tune=n_tune,
                n_cal=n_cal,
                n_audit=n_audit,
                n_origins=n_origins,
                fit_cutoff_ds=pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
                best_iteration=metrics_summary.get(f"best_iteration_h{h}"),
                early_stop=bool(n_tune > 0),
                threads=_n_threads(),
                feature_width=int(X.shape[1]),
                reduced_complexity=bool(reduced),
            )

        if not models:
            raise ValueError("Not enough training rows to fit LightGBM. Provide more history or more items.")

        collapsed_eval: dict[str, dict[str, Any]] = {}
        for uid, bucket in per_item_eval.items():
            yt = np.asarray(bucket.get("y") or [], dtype=float)
            y_ml = np.asarray(bucket.get("ml") or [], dtype=float)
            y_nv = np.asarray(bucket.get("naive") or [], dtype=float)
            y_sn = np.asarray(bucket.get("snaive") or [], dtype=float)
            ml_m = wape_with_undefined(yt, y_ml)
            nv_m = wape_with_undefined(yt, y_nv)
            sn_m = wape_with_undefined(yt, y_sn)
            collapsed_eval[uid] = {
                "wape_ml": ml_m.get("wape"),
                "wape_naive": nv_m.get("wape"),
                "wape_snaive": sn_m.get("wape"),
                "wape_undefined": bool(ml_m.get("wape_undefined")),
                "mae_ml": ml_m.get("mae"),
                "bias_ml": signed_bias(yt, y_ml),
                "n": float(ml_m.get("n") or 0),
            }
        per_item_eval = collapsed_eval
        head_votes = [mean_head_by_horizon.get(str(h)) for h in range(1, int(horizon) + 1)]
        selected_mean_head = (
            MEAN_HEAD_HURDLE
            if head_votes.count(MEAN_HEAD_HURDLE) > head_votes.count(MEAN_HEAD_TWEEDIE)
            else MEAN_HEAD_TWEEDIE
        )
        metrics_summary["mean_head"] = selected_mean_head
        metrics_summary["mean_head_by_horizon"] = mean_head_by_horizon
        metrics_summary["origin_splits"] = splits.as_dict()
        metrics_summary["evaluation_protocol"] = EVALUATION_PROTOCOL
        metrics_summary["confidence_definition"] = CONFIDENCE_DEFINITION
        metrics_summary["schema_version"] = FEATURE_SPEC_SCHEMA_VERSION
        if sf_unavailable_reason:
            metrics_summary["sf_unavailable_reason"] = sf_unavailable_reason

        assert feature_cols is not None

        conformal_m_lookup = _finalize_conformal_margins_lookup(
            conformal_score_lists,
            UPPER_QUANTILE_ALPHAS,
        )
        any_calibrated = bool(
            any(metrics_summary.get(f"quantiles_calibrated_h{h}") for h in range(1, int(horizon) + 1))
        )
        if conformal_m_lookup and any_calibrated:
            metrics_summary["conformal_upper_margins"] = True
            metrics_summary["quantiles_calibrated"] = True
            metrics_summary["uncertainty_status"] = "direct_quantile_calibrated"
        elif q_models:
            metrics_summary["conformal_upper_margins"] = False
            metrics_summary["quantiles_calibrated"] = False
            metrics_summary["uncertainty_status"] = "direct_quantile"
        else:
            metrics_summary["conformal_upper_margins"] = False
            metrics_summary["quantiles_calibrated"] = False
            metrics_summary["uncertainty_status"] = "unavailable"

        def _margins_for_residual_row(arch_s: str, hh: int) -> dict[str, float]:
            m = conformal_m_lookup.get((arch_s, int(hh))) or conformal_m_lookup.get(
                ("__global__", int(hh))
            ) or {}
            return {
                "margin_q70": float(m.get(0.7, 0.0)),
                "margin_q90": float(m.get(0.9, 0.0)),
                "margin_q95": float(m.get(0.95, 0.0)),
            }

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

                cv_folds = int(max(1, residual_time_cv_folds))
                for (arch, h), g in df_res.groupby(["archetype", "horizon"], dropna=False):
                    qs = _pooled_excess_quantiles_time_cv(
                        g,
                        UPPER_QUANTILE_ALPHAS,
                        time_cv_folds=cv_folds,
                    )
                    n = int(len(g))
                    if n == 0:
                        continue
                    hh = _coerce_h(h)
                    akey = str(arch) if arch is not None else "unknown"
                    row_rq = {
                        "model_version": model_version,
                        "archetype": akey,
                        "horizon": hh,
                        "scale_bucket": None,
                        "q70_excess": float(qs[0.7]),
                        "q90_excess": float(qs[0.9]),
                        "q95_excess": float(qs[0.95]),
                        "n": n,
                    }
                    row_rq.update(_margins_for_residual_row(akey, hh))
                    residual_quantile_rows.append(row_rq)
                # Global fallback per horizon
                for h, g in df_res.groupby(["horizon"], dropna=False):
                    qs = _pooled_excess_quantiles_time_cv(
                        g,
                        UPPER_QUANTILE_ALPHAS,
                        time_cv_folds=cv_folds,
                    )
                    n = int(len(g))
                    if n == 0:
                        continue
                    hh = _coerce_h(h)
                    row_gq = {
                        "model_version": model_version,
                        "archetype": "__global__",
                        "horizon": hh,
                        "scale_bucket": None,
                        "q70_excess": float(qs[0.7]),
                        "q90_excess": float(qs[0.9]),
                        "q95_excess": float(qs[0.95]),
                        "n": n,
                    }
                    row_gq.update(_margins_for_residual_row("__global__", hh))
                    residual_quantile_rows.append(row_gq)

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
            "quantile_alphas": list(UPPER_QUANTILE_ALPHAS),
            "quantile_alpha": 0.95,
            "hyperparameter_tuning": tune_hyperparameters,
            "tuned_hyperparameters": tuned_params if tuned_params else None,
            "attribute_routing": {
                "calendar": calendar_cols,
                "trend": trend_cols,
                "segment": segment_cols,
                "generic_interaction": generic_interaction_cols,
                "auto_segment_columns": auto_segment_cols,
            },
            "global_yoy": global_yoy,
            "schema_version": FEATURE_SPEC_SCHEMA_VERSION,
            "fit_cutoff_ds": pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
            "history_end_ds": pd.Timestamp(history_end_ds).strftime("%Y-%m-%d"),
            "context_version": ctx_version,
            "mean_head": metrics_summary.get("mean_head", MEAN_HEAD_TWEEDIE),
            "mean_head_by_horizon": mean_head_by_horizon,
            "adjustment_policy": SEASONAL_SERVE_POLICY.as_dict(),
            "quantile_direct_demand": True,
            "quantiles_calibrated": bool(metrics_summary.get("quantiles_calibrated")),
            "uncertainty_status": metrics_summary.get("uncertainty_status") or "direct_quantile",
            "confidence_definition": CONFIDENCE_DEFINITION,
            "origin_splits": splits.as_dict(),
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "artifact_complete": False,
            "code_version": _LIGHTGBM_FORECASTS_VERSION,
            "sf_unavailable_reason": sf_unavailable_reason,
            "peer_context_file": "peer_context.json",
            "required_driver_columns": list(required_driver_columns),
            "interior_gap_policy": INTERIOR_GAP_POLICY_OBSERVED_SALES,
            "target_definition": TARGET_OBSERVED_SALES,
        }
        spec_path = os.path.join(artifact_root, "feature_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(feature_spec, f, ensure_ascii=False, indent=2, default=_json_default)

        peer_path = os.path.join(artifact_root, "peer_context.json")
        with open(peer_path, "w", encoding="utf-8") as f:
            json.dump(
                serialize_peer_context(
                    customer_id=self.store.customer_id,
                    context_version=ctx_version,
                    origin=history_end_ds,
                    global_monthly=global_monthly_effects,
                    global_yoy=global_yoy,
                    segment_stats=segment_stats,
                ),
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

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

        if os.path.isdir(final_artifact_root):
            shutil.rmtree(final_artifact_root)
        os.replace(artifact_root, final_artifact_root)
        artifact_root = final_artifact_root
        feature_spec["artifact_complete"] = True
        spec_path = os.path.join(artifact_root, "feature_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(feature_spec, f, ensure_ascii=False, indent=2, default=_json_default)
        spec_hash = self.store.set_feature_spec(model_version, feature_spec)

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

        # Eligibility: ML vs last-value naive only. Seasonal naive is fallback, not the gate.
        unique_ids_list = base["unique_id"].astype(str).unique().tolist()
        for uid in unique_ids_list:
            series_len = int((base[base["unique_id"].astype(str) == str(uid)]).shape[0])
            evalr = per_item_eval.get(str(uid)) or {}
            elig = decide_item_eligibility(
                series_len=series_len,
                min_history_points=int(min_history_points),
                n_eval=evalr.get("n", 0),
                wape_ml=evalr.get("wape_ml"),
                wape_naive=evalr.get("wape_naive"),
                wape_snaive=evalr.get("wape_snaive"),
                min_improvement=float(min_improvement),
            )
            eligibility_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    **elig,
                    "min_history_points": int(min_history_points),
                    "requires_features": [],
                }
            )
            if evalr.get("wape_ml") is not None and evalr.get("wape_naive") is not None:
                backtest_rows.append(
                    {
                        "model_version": model_version,
                        "unique_id": str(uid),
                        "model_name": "lgbm_direct_h1",
                        "metric_name": "wape",
                        "metric_value": float(evalr["wape_ml"]),
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
                        "metric_value": float(evalr["wape_naive"]),
                        "n_folds": 1,
                        "eval_start_ds": None,
                        "eval_end_ds": None,
                    }
                )
            explain_rows.append(
                {
                    "model_version": model_version,
                    "unique_id": str(uid),
                    "top_features": top_features,
                    "group_contrib": {"reason": elig.get("reason_code")},
                    "support_share": None,
                }
            )

        # Add diagnostic counts to metrics_summary for debugging
        total_items = len(base["unique_id"].astype(str).unique())
        items_with_eval = len(per_item_eval)
        items_short_history = sum(1 for r in eligibility_rows if r.get("reason_code") == "SHORT_HISTORY")
        items_sparse_holdout = sum(1 for r in eligibility_rows if r.get("reason_code") == "OK_SPARSE_HOLDOUT")
        items_ml_allowed = sum(1 for r in eligibility_rows if r.get("ml_allowed"))
        metrics_summary["_debug_total_items"] = total_items
        metrics_summary["_debug_items_with_eval"] = items_with_eval
        metrics_summary["_debug_items_short_history"] = items_short_history
        metrics_summary["_debug_items_sparse_holdout"] = items_sparse_holdout
        metrics_summary["_debug_items_ml_allowed"] = items_ml_allowed
        metrics_summary["_debug_explain_rows_count"] = len(explain_rows)
        metrics_summary["_debug_store_customer_id"] = self.store.customer_id
        metrics_summary["_debug_store_db_path"] = self.store.db_path()
        metrics_summary["_debug_model_version"] = model_version

        # Per-item diagnostics (stored as JSON under explain_item_summary.group_contrib_json)
        # - SHAP-like attributions: LightGBM's pred_contrib output (TreeSHAP)
        # - Per-item importance: abs(contribution) ranking
        # - Used/dropped static/exogenous columns (based on last available row)
        # - Basic training diagnostics
        # Calculate SHAP for ALL horizons, not just horizon 1
        t_explain = time.perf_counter()
        _emit_phase(progress_hook, "explain_start")
        if models and feature_cols is not None:
            per_uid_diag: dict[str, dict[str, Any]] = {}
            base_by_uid = base.sort_values(["unique_id", "ds"], kind="mergesort").groupby("unique_id", sort=False)

            # Future statsforecast features so explain rows match batch inference inputs.
            try:
                sf_explain = _build_statsforecast_future_features(base, max_h=int(horizon), season_length=12)
                if not sf_explain.empty:
                    sf_explain["unique_id"] = sf_explain["unique_id"].astype(str)
            except Exception:
                sf_explain = pd.DataFrame()

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
                if len(y_orig) < MIN_INFERENCE_HISTORY_POINTS:
                    diag["shap"] = {"available": False, "reason": "EMPTY_HISTORY"}
                    per_uid_diag[uid] = {
                        "training_diagnostics": diag,
                        "used_columns": {"static": [], "exogenous": []},
                        "dropped_columns": {
                            "static": list(static_cols),
                            "exogenous": list(exogenous_columns or []),
                        },
                    }
                    continue

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

                # Build per-horizon feature rows at the last decision point using the
                # shared row builder, so explanations describe the exact feature
                # vectors that batch inference feeds each horizon's model.
                t = len(y_model) - 1
                last_ds = pd.Timestamp(ds_arr[-1])
                months_arr = np.array([pd.Timestamp(d).month for d in ds_arr], dtype=int)
                same_month_stats_t = _trailing_same_month_stats(y_orig, months_arr, t)
                lifecycle_feats_t = _detect_lifecycle_phase(y_orig)
                exo_arrays = {
                    c: pd.to_numeric(grp[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    for c in (exogenous_columns or [])
                    if c in grp.columns
                }
                static_map: dict[str, Any] = {
                    c: (grp[c].iloc[-1] if c in grp.columns else None) for c in static_cols
                }

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
                            r = _assemble_feature_row(
                                unique_id=uid,
                                t=t,
                                h=h,
                                ds=ds_arr,
                                y_model=y_model,
                                y_orig=y_orig,
                                lags=lags,
                                roll_windows=roll_windows,
                                static=static_map,
                                exo=exo_arrays,
                                same_month_stats=same_month_stats_t,
                                lifecycle_feats=lifecycle_feats_t,
                                calendar_cols=calendar_cols,
                                trend_cols=trend_cols,
                                segment_cols=segment_cols,
                                global_monthly=global_monthly_effects,
                                global_yoy=global_yoy,
                                segment_stats=segment_stats,
                            )
                            forecast_ds_h = pd.Timestamp(r["forecast_ds"])
                            if required_driver_columns or (exogenous_columns):
                                try:
                                    attach_target_month_drivers(
                                        r,
                                        drivers_wide if not drivers_wide.empty else None,
                                        columns=list(exogenous_columns or []),
                                        required=list(required_driver_columns),
                                    )
                                except MissingRequiredDriver:
                                    shap_per_horizon[h] = {
                                        "available": False,
                                        "reason": "MISSING_REQUIRED_DRIVER",
                                        "horizon": h,
                                    }
                                    continue
                            if te_cols and te_maps:
                                framed = apply_te_maps_to_frame(pd.DataFrame([r]), te_cols, te_maps)
                                r = framed.iloc[0].to_dict()

                            # StatsForecast features for the future month (matches batch inference)
                            if sf_explain is not None and not sf_explain.empty:
                                match = sf_explain[
                                    (sf_explain["unique_id"] == uid)
                                    & (sf_explain["forecast_ds"] == forecast_ds_h)
                                ]
                                if "decision_ds" in sf_explain.columns:
                                    match = match[match["decision_ds"] == last_ds]
                                if not match.empty:
                                    row_vals = match.iloc[0].to_dict()
                                    for k in ["sf_croston_optimized", "sf_adida", "sf_auto_ets", "sf_theta"]:
                                        if k in row_vals:
                                            r[k] = row_vals[k]
                                    r["sf_available"] = float(row_vals.get("sf_available", 0.0))

                            X = _prepare_lgbm_X(
                                pd.DataFrame([r]),
                                feature_cols,
                                cat_cols=cat_cols,
                                mappings=mappings,
                                static_interaction_cols=static_interaction_cols,
                            )

                            served_head = _mean_head_for_horizon(
                                {"mean_head_by_horizon": mean_head_by_horizon, "mean_head": selected_mean_head},
                                h,
                            )
                            explain_booster = booster
                            occurrence_payload = None
                            amount_payload = None
                            composition = "unconditional_mean"
                            if served_head == MEAN_HEAD_HURDLE and occ_model is not None and h in nz_models:
                                occurrence_payload = pred_contrib_payload(occ_model, X)
                                amount_payload = pred_contrib_payload(nz_models[h], X)
                                explain_booster = nz_models[h]
                                composition = "hurdle: p_occ * E[y | y > 0]; occurrence and positive-amount SHAP are separate"
                            payload = pred_contrib_payload(explain_booster, X)
                            bias = float(payload["bias"])
                            feat_contrib = payload["contributions"]
                            pairs_sorted = payload["pairs_sorted"]
                            contrib_map = payload["contrib_map"]
                            recon = payload["reconcile"]
                            raw_score = float(payload["raw_score"])
                            margin_sum = float(payload["margin_sum"])

                            top_k = 20
                            shap_top = [{"feature": n, "contribution": float(v)} for n, v in pairs_sorted[:top_k]]
                            rest_contribution = float(sum(float(v) for _, v in pairs_sorted[top_k:]))

                            yhat_point = float(explain_booster.predict(X)[0])
                            if detrend_method == "none" or not params:
                                yhat_amount = float(max(0.0, yhat_point))
                                base_amount = float(max(0.0, math.exp(min(bias, 50.0))))
                            elif detrend_method == "linear":
                                trend_future = float(_trend_model_value(forecast_ds_h, params))
                                yhat_amount = float(max(0.0, yhat_point + trend_future))
                                base_amount = float(max(0.0, bias + trend_future))
                            else:
                                trend_future = float(_trend_model_value(forecast_ds_h, params))
                                yhat_amount = float(max(0.0, math.expm1(min(yhat_point + trend_future, 50.0))))
                                base_amount = float(max(0.0, math.expm1(min(bias + trend_future, 50.0))))

                            yhat_tweedie_pt = float(
                                _to_original_scale(booster.predict(X)[0], detrend_method, _trend_model_value(forecast_ds_h, params) if params else 0.0)[0]
                            )
                            yhat_pos_pt = None
                            p_occ_pt = None
                            if occ_model is not None:
                                p_occ_pt = float(np.clip(occ_model.predict(X)[0], 0.0, 1.0))
                            if h in nz_models:
                                yhat_pos_pt = float(max(0.0, nz_models[h].predict(X)[0]))
                            served = predict_from_heads(
                                head=served_head,
                                yhat_tweedie=yhat_tweedie_pt,
                                yhat_positive=yhat_pos_pt,
                                p_occ=p_occ_pt,
                                policy=AUDIT_POLICY,
                            )

                            horizon_static_contrib = [
                                {"feature": f, "contribution": float(contrib_map.get(f, 0.0))}
                                for f in used_static
                                if f in contrib_map
                            ]
                            horizon_static_contrib = sorted(horizon_static_contrib, key=lambda kv: abs(float(kv["contribution"])), reverse=True)

                            output_space = "log_mean" if detrend_method == "none" else "units"
                            link = "log" if detrend_method == "none" else "identity"

                            shap_per_horizon[h] = {
                                "available": True,
                                "method": "pred_contrib",
                                "head": served_head,
                                "composition": composition,
                                "horizon": h,
                                "decision_ds": last_ds.strftime("%Y-%m-%d"),
                                "forecast_ds": forecast_ds_h.strftime("%Y-%m-%d"),
                                "bias": bias,
                                "margin_sum": margin_sum,
                                "raw_score": raw_score,
                                "rest_contribution": rest_contribution,
                                "yhat_amount": yhat_amount,
                                "base_amount": base_amount,
                                "output_space": output_space,
                                "link": link,
                                "reconcile": recon,
                                "occurrence": (
                                    {
                                        "bias": occurrence_payload["bias"],
                                        "raw_score": occurrence_payload["raw_score"],
                                        "reconcile": occurrence_payload["reconcile"],
                                        "top": [
                                            {"feature": n, "contribution": float(v)}
                                            for n, v in occurrence_payload["pairs_sorted"][:top_k]
                                        ],
                                    }
                                    if occurrence_payload is not None
                                    else None
                                ),
                                "positive_amount": (
                                    {
                                        "bias": amount_payload["bias"],
                                        "raw_score": amount_payload["raw_score"],
                                        "reconcile": amount_payload["reconcile"],
                                        "top": [
                                            {"feature": n, "contribution": float(v)}
                                            for n, v in amount_payload["pairs_sorted"][:top_k]
                                        ],
                                    }
                                    if amount_payload is not None
                                    else None
                                ),
                                "keys": explain_keys(
                                    model_version=model_version,
                                    spec_hash=spec_hash,
                                    context_version=ctx_version,
                                    item_id=uid,
                                    origin=last_ds.strftime("%Y-%m-%d"),
                                    forecast_ds=forecast_ds_h.strftime("%Y-%m-%d"),
                                    horizon=int(h),
                                    scenario_id="baseline",
                                ),
                                "top": shap_top,
                                "static_contrib": horizon_static_contrib,
                                "unit_allocation_is_display_only": True,
                                "adjustment_trace": served.adjustments,
                                "yhat_served": float(served.yhat),
                                "snapshot": True,
                                "snapshot_note": (
                                    "Train-time explain payload; changing origin or drivers invalidates it."
                                ),
                            }

                            if h == 1:
                                imp_top = [{"feature": n, "importance": float(abs(v))} for n, v in pairs_sorted[:top_k]]
                                static_contrib = horizon_static_contrib
                                static_values = {
                                    f: _human_static_value(f, static_map.get(f), mappings)
                                    for f in used_static
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

        _emit_phase(progress_hook, "explain_complete", elapsed_s=round(time.perf_counter() - t_explain, 3))

        _emit_phase(progress_hook, "persist_start")
        self.store.upsert_backtest_metrics(backtest_rows)
        self.store.upsert_eligibility(eligibility_rows)
        self.store.upsert_explain_summary(explain_rows)
        _emit_phase(progress_hook, "persist_complete")

        if progress_hook:
            progress_hook(
                {
                    "phase": "done",
                    "model_version": model_version,
                    "items_trained": int(base["unique_id"].nunique()),
                    "elapsed_s": round(time.perf_counter() - train_t0, 3),
                    "fit_cutoff_ds": pd.Timestamp(fit_cutoff_ds).strftime("%Y-%m-%d"),
                    "threads": _n_threads(),
                }
            )

        # Aggregate metrics summary
        if any(k.startswith("wape_val_h") for k in metrics_summary.keys()):
            wapes = [
                v
                for k, v in metrics_summary.items()
                if k.startswith("wape_val_h") and v is not None and np.isfinite(float(v))
            ]
            metrics_summary["wape_val_mean"] = float(np.mean(wapes)) if wapes else None

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
        validate_monthly_request(freq=freq, horizon=H, item_attributes=item_attributes)

        base = _normalize_monthly_history(hist)
        life_cols = {"active_from", "closed_at"}
        lifecycle = None
        if item_attributes is not None and not item_attributes.empty and life_cols.intersection(item_attributes.columns):
            lifecycle = item_attributes
        base = complete_monthly_grid(
            base,
            lifecycle=lifecycle,
            interior_gap_policy=str(spec.get("interior_gap_policy") or INTERIOR_GAP_POLICY_OBSERVED_SALES),
        )

        if item_attributes is not None and not item_attributes.empty and "item_id" in item_attributes.columns:
            attrs = item_attributes.copy()
            attrs["item_id"] = attrs["item_id"].astype(str)
            base["item_id"] = base["item_id"].astype(str)
            base = base.merge(attrs, on="item_id", how="left", suffixes=("", "_attr"))

        drivers_wide = pd.DataFrame()
        if drivers is not None and not drivers.empty:
            drivers_wide = normalize_drivers(drivers)
            wide = _pivot_long_drivers(drivers) if "driver_name" in drivers.columns else drivers_wide
            if not wide.empty:
                if "item_id" in wide.columns:
                    wide["item_id"] = wide["item_id"].astype(str)
                merge_keys = ["item_id", "ds"] if "item_id" in wide.columns else ["ds"]
                if all(k in wide.columns for k in merge_keys) and all(k in base.columns for k in merge_keys):
                    hist_drivers = wide.merge(base.loc[:, merge_keys].drop_duplicates(), on=merge_keys, how="inner")
                    if not hist_drivers.empty:
                        drop_meta = [c for c in ("vintage", "vintage_ds", "driver_name", "driver_value") if c in hist_drivers.columns]
                        hist_drivers = hist_drivers.drop(columns=drop_meta, errors="ignore")
                        base = base.merge(hist_drivers, on=merge_keys, how="left")

        # Apply target encodings and name features for inference
        if te_cols and te_maps:
            base = _apply_target_encodings(base, te_cols, te_maps)
        if name_cluster_map:
            base = _apply_name_cluster(base, name_cluster_map, name_col="name")

        routing = spec.get("attribute_routing", {}) or {}
        calendar_cols = list(routing.get("calendar") or [])
        trend_cols = list(routing.get("trend") or [])
        segment_cols = list(routing.get("segment") or [])
        generic_interaction_cols = list(routing.get("generic_interaction") or [])
        global_yoy = dict(spec.get("global_yoy") or {})

        schema_version = int(spec.get("schema_version") or 1)
        frozen = load_peer_context_or_raise(artifact_root, spec)
        if frozen:
            global_monthly_effects = frozen["global_monthly"]
            global_yoy = frozen.get("global_yoy") or global_yoy
            segment_stats = frozen.get("segment_stats") or {}
        else:
            segment_stats = _precompute_segment_monthly_stats(
                base, segment_cols, lookback_months=36
            ) if segment_cols else {}
            global_monthly_effects = _compute_global_monthly_effects(base, lookback_months=36)

        if exogenous_columns is None:
            exogenous_columns = [c for c in exo_cols if c in base.columns]
        else:
            exogenous_columns = [c for c in exogenous_columns if c in base.columns]

        base = base.sort_values(["unique_id", "ds"], kind="mergesort")

        month_caps = _build_item_month_caps(base, lookback_years=4)
        cap_nonzero_threshold = 0.25
        cap_multiplier = 2.0
        cap_small_floor = 0.0
        policy = _policy_from_spec(spec)
        mean_head = str(spec.get("mean_head") or MEAN_HEAD_TWEEDIE)
        quantile_direct = _quantile_direct_demand(spec)
        quantiles_calibrated = bool(spec.get("quantiles_calibrated"))
        uncertainty_status = str(spec.get("uncertainty_status") or ("direct_quantile" if quantile_direct else "v1_excess"))
        required_drivers = list(spec.get("required_driver_columns") or [])
        driver_columns_serve = list(exogenous_columns or [])

        sf_future = _build_statsforecast_future_features(base, max_h=int(max(horizons or [1])), season_length=12)

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

        # Load independent quantile models (70/90/95) when present; legacy bundles may only ship q95.
        q_boosters_by_h: dict[int, dict[float, Any]] = {}
        for h in horizons:
            qh: dict[float, Any] = {}
            for alpha in UPPER_QUANTILE_ALPHAS:
                suf = _quantile_artifact_suffix(alpha)
                qpath = os.path.join(artifact_root, f"lgbm_h{h}_{suf}.txt")
                if os.path.exists(qpath):
                    qh[alpha] = lgb.Booster(model_file=qpath)
            if qh:
                q_boosters_by_h[h] = qh

        residual_quantiles: dict[tuple[str, int], dict[str, float]] = self.store.get_residual_quantiles(
            model_version
        )
        if not residual_quantiles:
            spec_q = spec.get("residual_quantiles") or []
            for r in spec_q:
                arch = r.get("archetype")
                h = r.get("horizon")
                q95 = r.get("q95_excess")
                if arch is None or h is None or q95 is None:
                    continue
                q95f = float(q95)
                q70v = r.get("q70_excess")
                q90v = r.get("q90_excess")
                residual_quantiles[(str(arch), int(h))] = {
                    "q70": float(q70v) if q70v is not None else float(0.4 * q95f),
                    "q90": float(q90v) if q90v is not None else float(0.8 * q95f),
                    "q95": q95f,
                    "margin_q70": float(r["margin_q70"]) if r.get("margin_q70") is not None else 0.0,
                    "margin_q90": float(r["margin_q90"]) if r.get("margin_q90") is not None else 0.0,
                    "margin_q95": float(r["margin_q95"]) if r.get("margin_q95") is not None else 0.0,
                }

        if not boosters:
            raise ValueError(f"No horizon models found under {artifact_root}")

        n_input_series = int(base["unique_id"].astype(str).nunique())
        n_skipped_empty_history = 0
        n_horizons_skipped_missing_model = 0

        forecasts: list[dict[str, Any]] = []

        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            y_orig = grp["y"].to_numpy(dtype=float)
            ds_arr = grp["ds"].to_numpy()
            if len(y_orig) < MIN_INFERENCE_HISTORY_POINTS:
                n_skipped_empty_history += 1
                for h in horizons:
                    forecasts.append(
                        {
                            "unique_id": str(uid),
                            "ds": pd.Timestamp(ds_arr[-1]) + pd.offsets.MonthBegin(int(h)) if len(ds_arr) else pd.NaT,
                            "yhat": 0.0,
                            "upper_70": 0.0,
                            "upper_90": 0.0,
                            "upper_95": 0.0,
                            "adjustments": [],
                            "failed": True,
                            "reason_code": "EMPTY_HISTORY",
                        }
                    )
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
            same_month_stats = _trailing_same_month_stats(base_y, months, len(base_y) - 1)

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
            lifecycle_feats = _detect_lifecycle_phase(y_orig)

            # Build one row per horizon and predict with the corresponding model
            t = len(y) - 1
            # Do not skip short series: lags/rolls zero-fill, and pooling features
            # (flavour/group/segment) let new items borrow peer demand.

            for h in horizons:
                if h not in boosters:
                    n_horizons_skipped_missing_model += 1
                    continue

                # Shared row builder: identical feature computation to training rows.
                r = _assemble_feature_row(
                    unique_id=str(uid),
                    t=t,
                    h=int(h),
                    ds=ds_arr,
                    y_model=y,
                    y_orig=y_orig,
                    lags=lags,
                    roll_windows=roll_windows,
                    static=static,
                    exo=exo,
                    same_month_stats=same_month_stats,
                    lifecycle_feats=lifecycle_feats,
                    calendar_cols=calendar_cols,
                    trend_cols=trend_cols,
                    segment_cols=segment_cols,
                    global_monthly=global_monthly_effects,
                    global_yoy=global_yoy,
                    segment_stats=segment_stats,
                )
                forecast_ds = pd.Timestamp(r["forecast_ds"])
                m = int(forecast_ds.month)
                m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
                mean_nonzero_12 = float(r["mean_nonzero_12"])
                median_nonzero_12 = float(r["median_nonzero_12"])
                last_nz_val = float(r["last_nonzero_value"])

                try:
                    if driver_columns_serve:
                        attach_target_month_drivers(
                            r,
                            drivers_wide if not drivers_wide.empty else None,
                            columns=driver_columns_serve,
                            required=required_drivers,
                        )
                    if not sf_future.empty:
                        match = sf_future[
                            (sf_future["unique_id"] == str(uid)) & (sf_future["forecast_ds"] == forecast_ds)
                        ]
                        if "decision_ds" in sf_future.columns:
                            match = match[match["decision_ds"] == last_ds]
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

                    X = _prepare_lgbm_X(
                        pd.DataFrame([r]),
                        feature_cols,
                        cat_cols=cat_cols,
                        mappings=mappings,
                        static_interaction_cols=static_interaction_cols,
                    )
                    yhat_model = float(boosters[h].predict(X)[0])
                    yhat_model_nz = float(nz_boosters[h].predict(X)[0]) if h in nz_boosters else None
                    if detrend_method == "none" or not params:
                        yhat_tweedie = float(np.maximum(0.0, yhat_model))
                    elif detrend_method == "linear":
                        trend_future = _trend_model_value(forecast_ds, params)
                        yhat_tweedie = float(np.maximum(0.0, yhat_model + trend_future))
                    else:
                        trend_future = _trend_model_value(forecast_ds, params)
                        yhat_tweedie = float(np.maximum(0.0, np.expm1(yhat_model + trend_future)))
                    yhat_positive = None if yhat_model_nz is None else float(np.maximum(0.0, yhat_model_nz))
                    p_occ = float(np.clip(occ_model.predict(X)[0], 0.0, 1.0)) if occ_model is not None else None
                    head_h = _mean_head_for_horizon(spec, int(h), default=mean_head)
                    raw_mean, _ = compose_mean(
                        head=head_h,
                        yhat_tweedie=yhat_tweedie,
                        yhat_positive=yhat_positive,
                        p_occ=p_occ,
                    )

                    cap_stats = month_caps.get((str(uid), int(forecast_ds.month)))
                    month_cap = None
                    if cap_stats is not None:
                        month_cap = max(cap_multiplier * float(cap_stats.get("max_y", 0.0)), cap_small_floor)

                    q70 = q90 = q95 = None
                    qh = q_boosters_by_h.get(h, {})
                    if 0.7 in qh and 0.9 in qh and 0.95 in qh:
                        q70 = quantile_to_absolute(float(qh[0.7].predict(X)[0]), raw_mean, direct_demand=quantile_direct)
                        q90 = quantile_to_absolute(float(qh[0.9].predict(X)[0]), raw_mean, direct_demand=quantile_direct)
                        q95 = quantile_to_absolute(float(qh[0.95].predict(X)[0]), raw_mean, direct_demand=quantile_direct)
                    elif residual_quantiles and not quantile_direct:
                        trip = residual_quantiles.get((archetype, int(h))) or residual_quantiles.get(("__global__", int(h)))
                        if trip:
                            q70 = float(raw_mean + float(trip.get("q70", 0.0)))
                            q90 = float(raw_mean + float(trip.get("q90", 0.0)))
                            q95 = float(raw_mean + float(trip.get("q95", 0.0)))
                    if quantiles_calibrated and residual_quantiles:
                        trip = residual_quantiles.get((archetype, int(h))) or residual_quantiles.get(("__global__", int(h)))
                        if trip and q70 is not None:
                            q70 = float(q70) + float(trip.get("margin_q70", 0.0))
                            q90 = float(q90) + float(trip.get("margin_q90", 0.0))
                            q95 = float(q95) + float(trip.get("margin_q95", 0.0))

                    pred = predict_from_heads(
                        head=head_h,
                        yhat_tweedie=yhat_tweedie,
                        yhat_positive=yhat_positive,
                        p_occ=p_occ,
                        policy=policy,
                        q70=q70,
                        q90=q90,
                        q95=q95,
                        month_mean=float(m_stats.get("mean", 0.0)),
                        month_max=float(m_stats.get("max", 0.0)),
                        recent_level=float(recent_level),
                        croston=float(croston_mean),
                        month_cap=month_cap,
                    )
                    u70 = pred.upper_70
                    u90 = pred.upper_90
                    u95 = pred.upper_95
                    if u70 is None and u90 is None and u95 is None:
                        row_status = "unavailable"
                    else:
                        row_status = uncertainty_status
                    forecasts.append(
                        {
                            "unique_id": str(uid),
                            "ds": forecast_ds,
                            "yhat": float(pred.yhat),
                            "yhat_raw": float(pred.yhat_raw),
                            "upper_70": None if u70 is None else float(u70),
                            "upper_90": None if u90 is None else float(u90),
                            "upper_95": None if u95 is None else float(u95),
                            "planning_ceiling": pred.planning_ceiling,
                            "adjustments": pred.adjustments,
                            "mean_head": pred.head,
                            "failed": False,
                            "uncertainty_status": row_status,
                        }
                    )
                except MissingRequiredDriver as exc:
                    forecasts.append(
                        {
                            "unique_id": str(uid),
                            "ds": forecast_ds,
                            "yhat": 0.0,
                            "upper_70": 0.0,
                            "upper_90": 0.0,
                            "upper_95": 0.0,
                            "adjustments": [{"name": "missing_required_driver", "before": 0.0, "after": 0.0, "delta": 0.0}],
                            "failed": True,
                            "reason_code": "MISSING_REQUIRED_DRIVER",
                            "error": str(exc),
                        }
                    )
                except Exception as exc:
                    forecasts.append(
                        {
                            "unique_id": str(uid),
                            "ds": forecast_ds,
                            "yhat": 0.0,
                            "upper_70": 0.0,
                            "upper_90": 0.0,
                            "upper_95": 0.0,
                            "adjustments": [{"name": "predict_failed", "before": 0.0, "after": 0.0, "delta": 0.0}],
                            "failed": True,
                            "reason_code": "PREDICT_FAILED",
                            "error": str(exc),
                        }
                    )

        fcst = pd.DataFrame(forecasts)
        if not fcst.empty:
            fcst["item_id"] = fcst["unique_id"]
            n_output_series = int(fcst["item_id"].astype(str).nunique())
            # Month-start semantics: 2026-01-01 represents the total for January 2026.
            fcst["ds"] = pd.to_datetime(fcst["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
            fcst["day"] = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d")
            cols = ["item_id", "day", "yhat", "upper_70", "upper_90", "upper_95", "adjustments"]
            extra = [c for c in ["failed", "reason_code", "planning_ceiling", "yhat_raw", "mean_head", "uncertainty_status"] if c in fcst.columns]
            fcst = fcst[cols + extra]
            assert_finite_outputs(fcst["yhat"].to_numpy(), name="yhat")
            for qcol in ("upper_70", "upper_90", "upper_95"):
                if qcol in fcst.columns:
                    finite = pd.to_numeric(fcst[qcol], errors="coerce").dropna()
                    if len(finite):
                        assert_finite_outputs(finite.to_numpy(), name=qcol)
        else:
            n_output_series = 0

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
            # Phase 2: coverage diagnostics (input vs output series)
            "batch_coverage": {
                "n_input_series": int(n_input_series),
                "n_output_series": int(n_output_series),
                "n_skipped_empty_history": int(n_skipped_empty_history),
                "n_skipped_short_history_lt6": int(n_skipped_empty_history),
                "n_horizon_skips_missing_booster": int(n_horizons_skipped_missing_model),
                "forecast_horizons_requested": int(H),
                "horizons_with_models": sorted(int(h) for h in boosters.keys()),
            },
            "_code_version": _LIGHTGBM_FORECASTS_VERSION,
            "upper_quantile_alphas": list(UPPER_QUANTILE_ALPHAS),
            "schema_version": int(spec.get("schema_version") or 1),
            "confidence_definition": spec.get("confidence_definition") or CONFIDENCE_DEFINITION,
            "mean_head": mean_head,
            "mean_head_by_horizon": spec.get("mean_head_by_horizon"),
            "context_version": spec.get("context_version"),
            "fit_cutoff_ds": spec.get("fit_cutoff_ds"),
            "history_end_ds": spec.get("history_end_ds"),
            "adjustment_policy": policy.as_dict(),
            "quantile_direct_demand": quantile_direct,
            "quantiles_calibrated": quantiles_calibrated,
            "uncertainty_status": uncertainty_status,
        }
        return fcst, meta
