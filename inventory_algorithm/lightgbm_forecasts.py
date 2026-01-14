from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from services.duckdb_model_store import DuckDBModelStore, ModelVersionRow


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

    df["ds"] = _to_month_start(df["day"])
    df["y"] = pd.to_numeric(df["actual_sale"], errors="coerce").fillna(0.0).astype(float)

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


# -------------------------
# Feature engineering (direct strategy)
# -------------------------

def _month_sin_cos(month: int) -> tuple[float, float]:
    ang = 2.0 * np.pi * (float(month) / 12.0)
    return float(np.sin(ang)), float(np.cos(ang))


def _nonzero_run_length(values: np.ndarray) -> int:
    # months since last non-zero ending at current index
    for i in range(len(values) - 1, -1, -1):
        if float(values[i]) != 0.0:
            return int(len(values) - 1 - i)
    return int(len(values))


def _build_direct_rows_for_item(
    *,
    unique_id: str,
    ds: np.ndarray,
    y: np.ndarray,
    exo: dict[str, np.ndarray],
    static: dict[str, Any],
    horizon: int,
    lags: list[int],
    roll_windows: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    n = len(y)
    max_needed = max(max(lags) - 1, max(roll_windows) - 1, 12) if (lags or roll_windows) else 12

    for t in range(0, n - horizon):
        if t < max_needed:
            continue

        forecast_ds = pd.Timestamp(ds[t]) + pd.offsets.MonthBegin(horizon)
        target = float(y[t + horizon])

        row: dict[str, Any] = {
            "unique_id": unique_id,
            "decision_ds": pd.Timestamp(ds[t]),
            "forecast_ds": forecast_ds,
            "y": target,
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
            row[f"lag_{lag}"] = float(y[idx])

        for w in roll_windows:
            start = t - (w - 1)
            window = y[start : t + 1]
            row[f"roll_mean_{w}"] = float(np.mean(window))
            if w >= 2:
                row[f"roll_std_{w}"] = float(np.std(window, ddof=0))

        # Trend-ish
        row["diff1"] = float(y[t] - y[t - 1])
        row["diff12"] = float(y[t] - y[t - 12])

        # Intermittency
        last12 = y[t - 11 : t + 1]
        row["zero_ratio_12"] = float(np.mean(last12 == 0.0))
        row["nonzero_run_length"] = float(_nonzero_run_length(y[: t + 1]))

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
        model_version: str | None = None,
        status: str = "staging",
        notes: str | None = None,
        min_history_points: int = 24,
        min_improvement: float = 0.02,
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

        # Feature recipe (monthly)
        lags = [1, 2, 3, 6, 12, 24]
        roll_windows = [3, 6, 12]

        if progress_hook:
            progress_hook({"phase": "building_features"})

        # Build supervised rows across all items for each horizon
        all_rows_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in range(1, int(horizon) + 1)}

        base = base.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            ds = grp["ds"].to_numpy()
            y = grp["y"].to_numpy(dtype=float)

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
                    y=y,
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
        }

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

            # Define feature columns once
            if feature_cols is None:
                excluded = {"y", "decision_ds", "forecast_ds"}
                feature_cols = [c for c in df.columns if c not in excluded]

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

            all_nonneg = bool(np.all(y_train >= 0) and np.all(y_val >= 0))
            objective = "poisson" if all_nonneg else "regression"

            params = {
                "objective": objective,
                "metric": "l2",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "min_data_in_leaf": 50,
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
                yhat = np.maximum(0.0, booster.predict(X_val))
                metrics_summary[f"wape_val_h{h}"] = _wape(y_val, yhat)

                if h == 1:
                    # per-item eval for eligibility (h=1)
                    df_val = df.loc[is_val, ["unique_id", "y", "lag_1"]].copy()
                    df_val["yhat_ml"] = yhat
                    df_val["yhat_naive"] = pd.to_numeric(df_val["lag_1"], errors="coerce").fillna(0.0)

                    for uid, g in df_val.groupby("unique_id", sort=False):
                        yt = g["y"].to_numpy(dtype=float)
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

        # Load models
        boosters: dict[int, Any] = {}
        for h in horizons:
            path = os.path.join(artifact_root, f"lgbm_h{h}.txt")
            if os.path.exists(path):
                boosters[h] = lgb.Booster(model_file=path)

        if not boosters:
            raise ValueError(f"No horizon models found under {artifact_root}")

        forecasts: list[dict[str, Any]] = []

        for uid, grp in base.groupby("unique_id", sort=False):
            grp = grp.sort_values("ds", kind="mergesort")
            y = grp["y"].to_numpy(dtype=float)
            ds_arr = grp["ds"].to_numpy()
            if len(y) < 25:
                continue

            last_ds = pd.Timestamp(ds_arr[-1])

            exo = {c: pd.to_numeric(grp[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in exogenous_columns}
            static: dict[str, Any] = {}
            for c in static_cols:
                static[c] = grp[c].iloc[-1] if c in grp.columns else None

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
                    r[f"lag_{lag}"] = float(y[idx])

                for w in roll_windows:
                    start = t - (w - 1)
                    window = y[start : t + 1]
                    r[f"roll_mean_{w}"] = float(np.mean(window))
                    if w >= 2:
                        r[f"roll_std_{w}"] = float(np.std(window, ddof=0))

                r["diff1"] = float(y[t] - y[t - 1])
                r["diff12"] = float(y[t] - y[t - 12])

                last12 = y[t - 11 : t + 1]
                r["zero_ratio_12"] = float(np.mean(last12 == 0.0))
                r["nonzero_run_length"] = float(_nonzero_run_length(y[: t + 1]))

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

                yhat = float(np.maximum(0.0, boosters[h].predict(X)[0]))
                forecasts.append({"unique_id": str(uid), "ds": forecast_ds, "yhat": yhat})

        fcst = pd.DataFrame(forecasts)
        if not fcst.empty:
            fcst["item_id"] = fcst["unique_id"]
            fcst["day"] = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d")
            fcst = fcst[["item_id", "day", "yhat"]]

        meta = {"model_version": model_version, "freq": "MS", "strategy": "direct"}
        return fcst, meta
