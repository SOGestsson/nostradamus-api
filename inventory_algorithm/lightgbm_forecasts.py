from __future__ import annotations

import json
import os
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
# Feature engineering (direct strategy)
# -------------------------

def _month_sin_cos(month: int) -> tuple[float, float]:
    # Use 0-based month index for cyclic encoding (Jan=0,...,Dec=11)
    ang = 2.0 * np.pi * ((float(month) - 1.0) / 12.0)
    return float(np.sin(ang)), float(np.cos(ang))


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    n = len(y)
    max_needed = max(max(roll_windows) - 1, 1) if roll_windows else 1

    # Precompute per-month historical stats for same-month features (last 36 months).
    same_month_stats: dict[int, dict[str, float]] = {}
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    vals_3y = y[-36:] if len(y) > 36 else y
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
        s, c = _month_sin_cos(m)
        row["month_sin"] = s
        row["month_cos"] = c

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

        # Same-month historical features (per item)
        m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
        row["same_month_mean_3y"] = float(m_stats["mean"])
        row["same_month_max_3y"] = float(m_stats["max"])
        row["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
        # Alias: per-item month nonzero rate computed server-side.
        row["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

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

        if progress_hook:
            progress_hook({"phase": "building_features"})

        # Build supervised rows across all items for each horizon
        all_rows_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in range(1, int(horizon) + 1)}

        detrend_method = (detrend_method or "none").strip().lower()
        if detrend_method not in {"none", "linear", "log1p_linear"}:
            raise ValueError("detrend_method must be one of: none, linear, log1p_linear")

        trend_params_by_uid: dict[str, dict[str, Any]] = {}

        base = base.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
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
                rows = _build_direct_rows_for_item(
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
                )
                all_rows_by_h[h].extend(rows)

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
        static_cat_cols = [
            c
            for c in static_cols
            if c in base.columns and (pd.api.types.is_object_dtype(base[c]) or pd.api.types.is_categorical_dtype(base[c]))
        ]
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

        # For eligibility: evaluate horizon 1 per-item
        per_item_eval: dict[str, dict[str, float]] = {}

        # Resolve feature columns once from the first non-empty horizon
        if feature_cols is None:
            for rows in all_rows_by_h.values():
                if rows:
                    df_spec = pd.DataFrame(rows)
                    excluded = {"y", "decision_ds", "forecast_ds", "y_orig", "trend_model", "lag_1_orig"}
                    feature_cols = [c for c in df_spec.columns if c not in excluded]
                    break
        if not feature_cols:
            raise ValueError("Could not determine feature columns (no training rows).")

        # Train shared occurrence model (y > 0) across all horizons
        occ_rows: list[dict[str, Any]] = []
        for rows in all_rows_by_h.values():
            if rows:
                occ_rows.extend(rows)
        occ_model: Any | None = None
        if occ_rows:
            occ_df = pd.DataFrame(occ_rows)
            y_occ_src = occ_df["y_orig"] if "y_orig" in occ_df.columns else occ_df["y"]
            y_occ = (pd.to_numeric(y_occ_src, errors="coerce").fillna(0.0).to_numpy(dtype=float) > 0.0).astype(int)
            is_val_occ = pd.to_datetime(occ_df["forecast_ds"]) >= cutoff

            X_occ = occ_df[feature_cols].copy()
            X_occ["unique_id"] = X_occ["unique_id"].astype(str)
            X_occ, _ = _encode_categories(X_occ, cat_cols, mappings=mappings)
            X_occ = X_occ.fillna(0)

            X_occ_train = X_occ[~is_val_occ]
            y_occ_train = y_occ[~is_val_occ]
            X_occ_val = X_occ[is_val_occ]
            y_occ_val = y_occ[is_val_occ]

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
                    categorical_feature=[c for c in cat_cols if c in X_occ_train.columns],
                    free_raw_data=False,
                )
                valid_sets = [dtrain_occ]
                valid_names = ["train"]
                if len(X_occ_val) > 0:
                    dval_occ = lgb.Dataset(
                        X_occ_val,
                        label=y_occ_val,
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
            # Keep the label name as y
            y_target = df["y"].to_numpy(dtype=float)

            # Time split by forecast_ds
            is_val = pd.to_datetime(df["forecast_ds"]) >= cutoff

            X = df[feature_cols].copy()
            X["unique_id"] = X["unique_id"].astype(str)

            X, _ = _encode_categories(X, cat_cols, mappings=mappings)

            # Ensure numeric
            X = X.fillna(0)

            X_train = X[~is_val]
            y_train = y_target[~is_val]
            X_val = X[is_val]
            y_val = y_target[is_val]

            if len(X_train) < 50:
                raise ValueError("Not enough training rows to fit LightGBM. Provide more history or more items.")

            if detrend_method == "none":
                all_nonneg = bool(np.all(y_train >= 0) and np.all(y_val >= 0))
                objective = "poisson" if all_nonneg else "regression"
            else:
                # Detrended targets can be negative; use regression.
                objective = "regression"

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

            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=[c for c in cat_cols if c in X_train.columns], free_raw_data=False)
            valid_sets = [dtrain]
            valid_names = ["train"]
            if len(X_val) > 0:
                dval = lgb.Dataset(X_val, label=y_val, categorical_feature=[c for c in cat_cols if c in X_val.columns], free_raw_data=False)
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

                yhat_orig = yhat_orig * p_nonzero

                metrics_summary[f"wape_val_h{h}"] = _wape(y_true_orig, yhat_orig)

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
            "occurrence_model": "shared" if occ_model is not None else None,
            "occurrence_threshold": 0.2 if occ_model is not None else None,
            "lags": lags,
            "rolling_windows": roll_windows,
            "static_columns": static_cols,
            "exogenous_columns": exogenous_columns,
            "categorical_columns": cat_cols,
            "category_mappings": mappings,
            "feature_columns": feature_cols,
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
        for uid in base["unique_id"].astype(str).unique().tolist():
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

        # Per-item diagnostics (stored as JSON under explain_item_summary.group_contrib_json)
        # - SHAP-like attributions: LightGBM's pred_contrib output (TreeSHAP)
        # - Per-item importance: abs(contribution) ranking
        # - Used/dropped static/exogenous columns (based on last available row)
        # - Basic training diagnostics
        booster_h1 = models.get(1)
        if booster_h1 is not None and feature_cols is not None:
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
                s, c = _month_sin_cos(m)
                r["month_sin"] = s
                r["month_cos"] = c

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
                for col in feature_cols:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_cols].fillna(0)

                try:
                    contrib = booster_h1.predict(X, pred_contrib=True)
                    contrib_row = np.asarray(contrib[0], dtype=float)
                    # LightGBM returns n_features + 1 (bias)
                    bias = float(contrib_row[-1])
                    feat_names = list(booster_h1.feature_name())
                    feat_contrib = contrib_row[:-1]
                    pairs = list(zip(feat_names, feat_contrib))
                    pairs_sorted = sorted(pairs, key=lambda kv: abs(float(kv[1])), reverse=True)

                    top_k = 20
                    shap_top = [{"feature": n, "contribution": float(v)} for n, v in pairs_sorted[:top_k]]
                    imp_top = [{"feature": n, "importance": float(abs(v))} for n, v in pairs_sorted[:top_k]]

                    shap_info = {
                        "available": True,
                        "method": "pred_contrib",
                        "horizon": 1,
                        "bias": bias,
                        "top": shap_top,
                    }
                except Exception as e:
                    shap_info = {"available": False, "reason": f"PRED_CONTRIB_FAILED: {e}"}
                    imp_top = []

                per_uid_diag[uid] = {
                    "shap": shap_info,
                    "per_item_importance_top": imp_top,
                    "used_columns": {"static": used_static, "exogenous": used_exo},
                    "dropped_columns": {"static": dropped_static, "exogenous": dropped_exo},
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

        if exogenous_columns is None:
            exogenous_columns = [c for c in exo_cols if c in base.columns]
        else:
            exogenous_columns = [c for c in exogenous_columns if c in base.columns]

        base = base.sort_values(["unique_id", "ds"], kind="mergesort")

        month_caps = _build_item_month_caps(base, lookback_years=4)
        cap_nonzero_threshold = 0.25
        cap_multiplier = 2.0
        cap_small_floor = 0.0

        # Load models
        boosters: dict[int, Any] = {}
        for h in horizons:
            path = os.path.join(artifact_root, f"lgbm_h{h}.txt")
            if os.path.exists(path):
                boosters[h] = lgb.Booster(model_file=path)

        occ_model: Any | None = None
        occ_path = os.path.join(artifact_root, "lgbm_occurrence.txt")
        if os.path.exists(occ_path):
            occ_model = lgb.Booster(model_file=occ_path)

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

            # Precompute same-month stats for inference (last 36 months).
            months = np.array([pd.Timestamp(d).month for d in ds_arr], dtype=int)
            vals_3y = y[-36:] if len(y) > 36 else y
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
                s, c = _month_sin_cos(m)
                r["month_sin"] = s
                r["month_cos"] = c

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

                # Same-month historical features (per item)
                m_stats = same_month_stats.get(m, {"mean": 0.0, "max": 0.0, "nonzero_rate": 0.0})
                r["same_month_mean_3y"] = float(m_stats["mean"])
                r["same_month_max_3y"] = float(m_stats["max"])
                r["same_month_nonzero_rate_3y"] = float(m_stats["nonzero_rate"])
                # Alias: per-item month nonzero rate computed server-side.
                r["item_month_nonzero_rate"] = float(m_stats["nonzero_rate"])

                # Simple decay signal: slope of last 12 months
                r["rolling_12_slope"] = float(_rolling_slope(y[: t + 1]))

                for k, arr in exo.items():
                    if len(arr) == len(y):
                        r[k] = float(arr[t])

                for k, v in static.items():
                    r[k] = v

                X = pd.DataFrame([r])
                X["unique_id"] = X["unique_id"].astype(str)
                X, _ = _encode_categories(X, cat_cols, mappings=mappings)

                for c in feature_cols:
                    if c not in X.columns:
                        X[c] = 0
                X = X[feature_cols].fillna(0)

                yhat_model = float(boosters[h].predict(X)[0])
                if detrend_method == "none" or not params:
                    yhat_amount = float(np.maximum(0.0, yhat_model))
                elif detrend_method == "linear":
                    trend_future = _trend_model_value(forecast_ds, params)
                    yhat_amount = float(np.maximum(0.0, yhat_model + trend_future))
                else:  # log1p_linear
                    trend_future = _trend_model_value(forecast_ds, params)
                    yhat_amount = float(np.maximum(0.0, np.expm1(yhat_model + trend_future)))

                if occ_model is not None:
                    p_nonzero = float(occ_model.predict(X)[0])
                    p_nonzero = float(np.clip(p_nonzero, 0.0, 1.0))
                else:
                    p_nonzero = 1.0

                # Couple occurrence to month history + recent run-length.
                cap_stats = month_caps.get((str(uid), int(forecast_ds.month)))
                month_rate = float(cap_stats.get("nonzero_rate", 1.0)) if cap_stats else 1.0
                run_length = float(_nonzero_run_length(y_orig))
                run_penalty = float(np.exp(-run_length / 6.0))

                p_effective = float(np.clip(p_nonzero * max(0.05, month_rate) * run_penalty, 0.0, 1.0))
                p_min = max(0.05, 0.5 * month_rate)

                yhat = float(yhat_amount * p_effective)
                if p_effective < max(occurrence_threshold, p_min):
                    yhat = 0.0

                adjustments: dict[str, Any] = {"archetype": archetype}

                # Improvement #1: recent-level anchor (skip in seasonal-off months)
                month_rate = float(m_stats.get("nonzero_rate", 0.0))
                if (
                    recent_level > 0.0
                    and (archetype != "seasonal" or month_rate >= 0.25)
                    and (last_nonzero_age <= 9)
                ):
                    if archetype == "seasonal":
                        alpha = 0.9
                    elif archetype == "noisy":
                        alpha = 0.5
                    else:
                        alpha = 0.7
                    yhat = float(alpha * yhat + (1.0 - alpha) * recent_level)
                    adjustments["recent_level"] = recent_level
                    adjustments["alpha"] = alpha

                # Seasonal peak anchor: lift in-season peaks toward historical max
                if archetype == "seasonal" and month_rate >= 0.25:
                    peak_alpha = 0.5
                    peak_target = float(m_stats.get("max", 0.0))
                    peak_mean = float(m_stats.get("mean", 0.0))
                    if peak_mean > 0.0 and peak_target > 3.0 * peak_mean:
                        peak_target = 3.0 * peak_mean
                    yhat = float((1.0 - peak_alpha) * yhat + peak_alpha * peak_target)
                    adjustments["peak_alpha"] = peak_alpha
                    adjustments["peak_target"] = peak_target

                # Improvement #2: Croston-style floor for intermittent alive items
                if (
                    0.1 < overall_nonzero_rate < 0.5
                    and last_nonzero_age <= 9
                    and nonzero_count_last_12 >= 2
                ):
                    floor = 0.4 * float(croston_mean)
                    if yhat < floor:
                        yhat = float(floor)
                        adjustments["croston_floor"] = floor

                # Improvement #3: horizon shrinkage (apply only when above recent level)
                if not (archetype == "seasonal" and month_rate >= 0.25):
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
                            yhat = float(level_floor)
                            adjustments["level_floor"] = level_floor

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
                    if archetype == "seasonal" and month_rate >= 0.25:
                        cap_mult = 4.0
                    cap = max(cap_mult * max_y, cap_small_floor)
                    yhat = min(yhat, cap)
                    adjustments["cap"] = cap
                    adjustments["cap_mult"] = cap_mult

                forecasts.append({"unique_id": str(uid), "ds": forecast_ds, "yhat": yhat, "adjustments": adjustments})

        fcst = pd.DataFrame(forecasts)
        if not fcst.empty:
            fcst["item_id"] = fcst["unique_id"]
            # Month-start semantics: 2026-01-01 represents the total for January 2026.
            fcst["ds"] = pd.to_datetime(fcst["ds"], errors="coerce").dt.to_period("M").dt.to_timestamp(how="start")
            fcst["day"] = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d")
            fcst = fcst[["item_id", "day", "yhat"]]

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
        }
        return fcst, meta
