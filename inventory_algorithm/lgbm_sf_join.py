"""StatsForecast feature joins keyed by origin, item, and target month."""

from __future__ import annotations

from typing import Any

import pandas as pd

from inventory_algorithm.lgbm_origin import month_start

SF_FEATURE_COLS = [
    "sf_croston_optimized",
    "sf_adida",
    "sf_auto_ets",
    "sf_theta",
    "sf_available",
]


def _rename_sf_models(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "CrostonOptimized": "sf_croston_optimized",
        "ADIDA": "sf_adida",
        "AutoETS": "sf_auto_ets",
        "Theta": "sf_theta",
    }
    return df.rename(columns=rename)


def normalize_sf_cv_frame(cv: pd.DataFrame) -> pd.DataFrame:
    """Treat StatsForecast cutoff as decision_ds and ds as forecast_ds."""
    if cv is None or cv.empty:
        return pd.DataFrame(columns=["unique_id", "decision_ds", "forecast_ds", "horizon", *SF_FEATURE_COLS])
    out = _rename_sf_models(cv.copy())
    if "cutoff" not in out.columns:
        raise ValueError("StatsForecast CV frame is missing cutoff (decision_ds).")
    out["unique_id"] = out["unique_id"].astype(str)
    out["decision_ds"] = pd.to_datetime(out["cutoff"], errors="coerce").map(month_start)
    out["forecast_ds"] = pd.to_datetime(out["ds"], errors="coerce").map(month_start)
    out = out.dropna(subset=["unique_id", "decision_ds", "forecast_ds"])
    out["horizon"] = (
        (out["forecast_ds"].dt.year * 12 + out["forecast_ds"].dt.month)
        - (out["decision_ds"].dt.year * 12 + out["decision_ds"].dt.month)
    ).astype(int)
    if "sf_available" not in out.columns:
        out["sf_available"] = 1.0
    keep = ["unique_id", "decision_ds", "forecast_ds", "horizon"] + [
        c for c in SF_FEATURE_COLS if c in out.columns
    ]
    out = out.loc[:, keep]
    _assert_unique_sf_keys(out)
    return out.reset_index(drop=True)


def normalize_sf_forecast_frame(fcst: pd.DataFrame, *, decision_ds: pd.Timestamp) -> pd.DataFrame:
    if fcst is None or fcst.empty:
        return pd.DataFrame(columns=["unique_id", "decision_ds", "forecast_ds", "horizon", *SF_FEATURE_COLS])
    out = _rename_sf_models(fcst.copy())
    origin = month_start(decision_ds)
    out["unique_id"] = out["unique_id"].astype(str)
    out["decision_ds"] = origin
    out["forecast_ds"] = pd.to_datetime(out["ds"], errors="coerce").map(month_start)
    out = out.dropna(subset=["unique_id", "forecast_ds"])
    out["horizon"] = (
        (out["forecast_ds"].dt.year * 12 + out["forecast_ds"].dt.month)
        - (origin.year * 12 + origin.month)
    ).astype(int)
    if "sf_available" not in out.columns:
        out["sf_available"] = 1.0
    keep = ["unique_id", "decision_ds", "forecast_ds", "horizon"] + [
        c for c in SF_FEATURE_COLS if c in out.columns
    ]
    out = out.loc[:, keep]
    _assert_unique_sf_keys(out)
    return out.reset_index(drop=True)


def _assert_unique_sf_keys(df: pd.DataFrame) -> None:
    keys = ["unique_id", "decision_ds", "forecast_ds"]
    dup = int(df.duplicated(keys).sum())
    if dup:
        raise ValueError(
            f"StatsForecast features have {dup} duplicate (unique_id, decision_ds, forecast_ds) keys."
        )


def join_sf_features(
    rows: pd.DataFrame,
    sf: pd.DataFrame,
    *,
    stat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Left-join SF features. Row count must not change. Duplicates are errors, not first/last."""
    stat_cols = stat_cols or [c for c in SF_FEATURE_COLS if c != "sf_available"]
    n_before = int(len(rows))
    if rows is None or rows.empty:
        return rows
    if sf is None or sf.empty:
        out = rows.copy()
        for c in [*stat_cols, "sf_available"]:
            if c not in out.columns:
                out[c] = 0.0
        return out
    _assert_unique_sf_keys(sf)
    left_keys = ["unique_id", "decision_ds", "forecast_ds"]
    for k in left_keys:
        if k not in rows.columns:
            raise ValueError(f"supervised rows missing join key {k}")
    work = rows.copy()
    work["unique_id"] = work["unique_id"].astype(str)
    work["decision_ds"] = pd.to_datetime(work["decision_ds"]).map(month_start)
    work["forecast_ds"] = pd.to_datetime(work["forecast_ds"]).map(month_start)
    sf_j = sf.copy()
    sf_j["unique_id"] = sf_j["unique_id"].astype(str)
    keep = left_keys + [c for c in [*stat_cols, "sf_available"] if c in sf_j.columns]
    merged = work.merge(sf_j[keep], on=left_keys, how="left", validate="many_to_one")
    if int(len(merged)) != n_before:
        raise ValueError(
            f"StatsForecast join changed row count: {n_before} -> {len(merged)}"
        )
    if "sf_available" not in merged.columns:
        merged["sf_available"] = 0.0
    merged["sf_available"] = merged["sf_available"].fillna(0.0)
    for c in stat_cols:
        if c in merged.columns:
            merged[c] = merged[c].where(merged["sf_available"] > 0.0, 0.0).fillna(0.0)
        else:
            merged[c] = 0.0
    return merged
