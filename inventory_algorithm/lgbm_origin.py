"""Origin-safe evaluation splits, metrics, and expanding target encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEATURE_SPEC_SCHEMA_VERSION = 2
CONFIDENCE_DEFINITION = "wape_improvement_ratio_vs_naive"

SPLIT_FIT = "fit"
SPLIT_TUNE = "tune"
SPLIT_CALIBRATION = "calibration"
SPLIT_AUDIT = "audit"

EVALUATION_PROTOCOL = {
    "type": "fixed_model_temporal_holdout",
    "not": "rolling_origin_refit",
    "roles": {
        "fit": (
            "decision_ds <= fit_cutoff_ds and forecast_ds <= fit_cutoff_ds; these rows train the booster"
        ),
        "tune": "Optuna and early stopping only",
        "calibration": (
            "choose mean_head / mean_head_by_horizon, routing vs last-value naive "
            "(seasonal naive is fallback only), and quantile calibration; never used as audit"
        ),
        "audit": (
            "score the frozen procedure only; no head vote, routing threshold, conformal fit, "
            "or silent fallback to calibration rows"
        ),
    },
    "label_observability": (
        "train labels must be observable at fit_cutoff_ds, not only at history_end_ds"
    ),
    "small_n": (
        "never mix holdout into train; use reduced-complexity fit or mark the horizon unavailable"
    ),
}


@dataclass(frozen=True)
class OriginSplits:
    """Fixed-model temporal holdout. One fitted model per horizon; audit never used for selection."""

    history_end: pd.Timestamp
    fit_cutoff: pd.Timestamp
    tune_origins: list[pd.Timestamp] = field(default_factory=list)
    calibration_origins: list[pd.Timestamp] = field(default_factory=list)
    audit_origins: list[pd.Timestamp] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def _fmt(xs: list[pd.Timestamp]) -> list[str]:
            return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in xs]

        return {
            "history_end_ds": pd.Timestamp(self.history_end).strftime("%Y-%m-%d"),
            "fit_cutoff_ds": pd.Timestamp(self.fit_cutoff).strftime("%Y-%m-%d"),
            "tune_origins": _fmt(self.tune_origins),
            "calibration_origins": _fmt(self.calibration_origins),
            "audit_origins": _fmt(self.audit_origins),
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "residual_time_cv_folds_note": (
                "residual_time_cv_folds pools held-out residuals; it is not model cross-validation"
            ),
        }


def month_start(ts) -> pd.Timestamp:
    return pd.Timestamp(pd.Timestamp(ts).to_period("M").to_timestamp(how="start"))


def split_rolling_origins(
    history_end: pd.Timestamp,
    *,
    tune_months: int = 3,
    calibration_months: int = 3,
    audit_months: int = 3,
    min_origin: pd.Timestamp | None = None,
) -> OriginSplits:
    """Walk back from history_end: audit, then calibration, then tune. Remainder is fit data.

    This is a fixed-model temporal holdout (one booster per horizon), not rolling-origin
    refitting. Origins are decision months. Train labels at origin T for horizon h are
    legal only when T+h <= fit_cutoff (observable at the fit cutoff). Evaluation rows
    may use labels through history_end.
    """
    end = month_start(history_end)
    audit_months = int(max(1, audit_months))
    calibration_months = int(max(1, calibration_months))
    tune_months = int(max(1, tune_months))

    audit = [end - pd.offsets.MonthBegin(i) for i in range(audit_months, 0, -1)]
    cal_end = audit[0] - pd.offsets.MonthBegin(1)
    calibration = [cal_end - pd.offsets.MonthBegin(i) for i in range(calibration_months - 1, -1, -1)]
    tune_end = calibration[0] - pd.offsets.MonthBegin(1)
    tune = [tune_end - pd.offsets.MonthBegin(i) for i in range(tune_months - 1, -1, -1)]
    fit_cutoff = tune[0] - pd.offsets.MonthBegin(1)

    if min_origin is not None:
        min_o = month_start(min_origin)

        def _keep(xs: list[pd.Timestamp]) -> list[pd.Timestamp]:
            return [x for x in xs if pd.Timestamp(x) >= min_o]

        audit = _keep(audit)
        calibration = _keep(calibration)
        tune = _keep(tune)

    if not audit:
        audit = [end]
    if not calibration:
        calibration = [audit[0] - pd.offsets.MonthBegin(1)]
    if not tune:
        tune = [calibration[0] - pd.offsets.MonthBegin(1)]
    fit_cutoff = pd.Timestamp(tune[0]) - pd.offsets.MonthBegin(1)
    return OriginSplits(
        history_end=end,
        fit_cutoff=month_start(fit_cutoff),
        tune_origins=[month_start(x) for x in tune],
        calibration_origins=[month_start(x) for x in calibration],
        audit_origins=[month_start(x) for x in audit],
    )


def target_month(decision_ds: pd.Timestamp, horizon: int) -> pd.Timestamp:
    return month_start(decision_ds) + pd.offsets.MonthBegin(int(horizon))


def label_is_observable(decision_ds: pd.Timestamp, horizon: int, available_through: pd.Timestamp) -> bool:
    """True when y at T+h is already realized by `available_through` (fit cutoff or panel end)."""
    return pd.Timestamp(target_month(decision_ds, horizon)) <= month_start(available_through)


def origin_membership(decision_ds: pd.Series, origins: list[pd.Timestamp]) -> np.ndarray:
    """Boolean mask: decision month is one of the listed origins."""
    members = {month_start(o) for o in origins}
    return pd.to_datetime(decision_ds, errors="coerce").map(month_start).isin(members).to_numpy()


def assign_split_role(decision_ds, splits: OriginSplits) -> str:
    d = month_start(decision_ds)
    audit = {month_start(x) for x in splits.audit_origins}
    cal = {month_start(x) for x in splits.calibration_origins}
    tune = {month_start(x) for x in splits.tune_origins}
    if d in audit:
        return SPLIT_AUDIT
    if d in cal:
        return SPLIT_CALIBRATION
    if d in tune:
        return SPLIT_TUNE
    return SPLIT_FIT


def origin_split_masks(
    frame: pd.DataFrame,
    splits: OriginSplits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (is_train, is_tune, is_cal, is_audit). Roles are disjoint even when n is small.

    Train requires decision_ds <= fit_cutoff AND forecast_ds <= fit_cutoff so labels
    are observable at the fit cutoff. Never expands train into tune/cal/audit.
    """
    if frame is None or frame.empty:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty, empty
    decision = pd.to_datetime(frame["decision_ds"], errors="coerce")
    if "forecast_ds" in frame.columns:
        forecast = pd.to_datetime(frame["forecast_ds"], errors="coerce")
    else:
        forecast = decision
    fit_cutoff = month_start(splits.fit_cutoff)
    dec_m = decision.map(month_start)
    fcst_m = forecast.map(month_start)
    is_tune = origin_membership(frame["decision_ds"], splits.tune_origins)
    is_cal = origin_membership(frame["decision_ds"], splits.calibration_origins)
    is_audit = origin_membership(frame["decision_ds"], splits.audit_origins)
    is_train = (dec_m <= fit_cutoff) & (fcst_m <= fit_cutoff) & (~is_tune) & (~is_cal) & (~is_audit)
    return (
        np.asarray(is_train, dtype=bool),
        np.asarray(is_tune, dtype=bool),
        np.asarray(is_cal, dtype=bool),
        np.asarray(is_audit, dtype=bool),
    )


def demand_through(base: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    ds = pd.to_datetime(base["ds"], errors="coerce")
    return base.loc[ds <= month_start(origin)].copy()


def wape_with_undefined(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-12:
        return {"wape": None, "wape_undefined": True, "mae": mae, "n": int(len(y_true))}
    return {
        "wape": float(np.sum(np.abs(y_true - y_pred)) / denom),
        "wape_undefined": False,
        "mae": mae,
        "n": int(len(y_true)),
    }


def _fallback_model(wape_naive: float | None, wape_snaive: float | None) -> str:
    """When ML is not served, pick the better of last-value vs seasonal naive."""
    if wape_snaive is None and wape_naive is None:
        return "seasonal_naive"
    if wape_snaive is None:
        return "naive"
    if wape_naive is None:
        return "seasonal_naive"
    if float(wape_snaive) < float(wape_naive):
        return "seasonal_naive"
    return "naive"


def decide_item_eligibility(
    *,
    series_len: int,
    min_history_points: int,
    n_eval: float | int | None,
    wape_ml: float | None,
    wape_naive: float | None,
    wape_snaive: float | None = None,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    """Allow ML unless it loses to last-value naive. Seasonal naive is fallback, not the gate.

    A short off-season calibration window (all zeros) does not force last-value 0. That path
    zeros December-only items. Gating against seasonal naive copies last year for most SKUs.
    """
    fallback = _fallback_model(wape_naive, wape_snaive)
    if int(series_len) < int(min_history_points):
        return {
            "winner_model": "naive",
            "ml_allowed": False,
            "ml_preferred": False,
            "fallback_model": "naive",
            "reason_code": "SHORT_HISTORY",
            "confidence": 0.0,
        }
    n = int(n_eval or 0)
    if n < 1 or wape_ml is None or wape_naive is None:
        return {
            "winner_model": "lgbm_direct",
            "ml_allowed": True,
            "ml_preferred": True,
            "fallback_model": fallback,
            "reason_code": "OK_SPARSE_HOLDOUT",
            "confidence": 0.0,
        }
    w_ml = float(wape_ml)
    w_nv = float(wape_naive)
    ml_better = w_ml <= (w_nv * (1.0 - float(min_improvement)))
    if ml_better:
        return {
            "winner_model": "lgbm_direct",
            "ml_allowed": True,
            "ml_preferred": True,
            "fallback_model": fallback,
            "reason_code": "OK",
            "confidence": float(max(0.0, min(1.0, 1.0 - (w_ml / (w_nv + 1e-9))))),
        }
    return {
        "winner_model": fallback,
        "ml_allowed": False,
        "ml_preferred": False,
        "fallback_model": fallback,
        "reason_code": "ML_WORSE_THAN_TS",
        "confidence": float(max(0.0, min(1.0, 1.0 - (w_nv / (w_ml + 1e-9))))),
    }


def signed_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_pred - y_true))


def expanding_target_encoding_maps(
    history: pd.DataFrame,
    origin: pd.Timestamp,
    cols: list[str],
    *,
    prior: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """TE maps using demand with ds <= origin (target month itself is excluded because origin is decision_ds)."""
    df = demand_through(history, origin)
    if df is None or df.empty or not cols or "y" not in df.columns:
        return {}
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
        smooth = (stats["mean"] * stats["count"] + global_mean * float(prior)) / (
            stats["count"] + float(prior)
        )
        mapping = {str(k): float(v) for k, v in smooth.to_dict().items()}
        enc_maps[col] = {"__global__": global_mean, "mapping": mapping}
    return enc_maps


def apply_te_maps_to_frame(
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


def apply_expanding_te_to_rows(
    rows: list[dict[str, Any]],
    history: pd.DataFrame,
    te_cols: list[str],
    *,
    prior: float = 10.0,
) -> list[dict[str, Any]]:
    """Attach *_te using maps fitted only on demand through each row's decision_ds."""
    if not rows or not te_cols:
        return rows
    cache: dict[pd.Timestamp, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for r in rows:
        origin = month_start(r["decision_ds"])
        if origin not in cache:
            cache[origin] = expanding_target_encoding_maps(history, origin, te_cols, prior=prior)
        framed = apply_te_maps_to_frame(pd.DataFrame([r]), te_cols, cache[origin])
        out.append(framed.iloc[0].to_dict())
    return out


def context_version(parts: dict[str, Any]) -> str:
    import hashlib
    import json

    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
