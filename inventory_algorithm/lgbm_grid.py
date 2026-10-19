"""Monthly calendar grid, missing vs observed zeros, input validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MONTHLY_FREQS = {"M", "MS", "ME", "MONTH", "MONTHLY"}
INTERIOR_GAP_POLICY_OBSERVED_SALES = "observed_sales_zeros"
TARGET_OBSERVED_SALES = "observed_sales"


def is_monthly_freq(freq: str | None) -> bool:
    f = str(freq or "M").strip().upper()
    return f in MONTHLY_FREQS or f.startswith("M")


def validate_monthly_request(
    *,
    freq: str | None,
    horizon: int,
    item_attributes: pd.DataFrame | None = None,
) -> None:
    if not is_monthly_freq(freq):
        raise ValueError("LightGBM forecasts are monthly-only; freq must be M/MS.")
    if int(horizon) < 1 or int(horizon) > 24:
        raise ValueError("horizon/forecast_periods must be between 1 and 24.")
    if item_attributes is not None and not item_attributes.empty and "item_id" in item_attributes.columns:
        ids = item_attributes["item_id"].astype(str)
        if int(ids.duplicated().sum()) > 0:
            raise ValueError("item_attributes must contain unique item_id rows.")


def complete_monthly_grid(
    hist: pd.DataFrame,
    *,
    lifecycle: pd.DataFrame | None = None,
    interior_gap_policy: str = INTERIOR_GAP_POLICY_OBSERVED_SALES,
) -> pd.DataFrame:
    """Complete each item's calendar MS grid from first to last observed month.

    Interior gaps become observed zeros (y_observed=1) under interior_gap_policy=
    observed_sales_zeros. The modeled target is observed sales, not unconstrained
    demand: we do not invent stockouts or availability. Months outside an optional
    lifecycle window are not imputed as demand (y_observed=0) and should not be labels.
    """
    _ = interior_gap_policy
    if hist is None or hist.empty:
        return hist
    df = hist.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds", "unique_id"] if "unique_id" in df.columns else ["ds"])
    if df.empty:
        return df
    if "unique_id" not in df.columns:
        df["unique_id"] = df["item_id"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df["y_observed"] = 1

    life: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
    if lifecycle is not None and not lifecycle.empty:
        lf = lifecycle.copy()
        id_col = "item_id" if "item_id" in lf.columns else "unique_id"
        lf[id_col] = lf[id_col].astype(str)
        for _, row in lf.iterrows():
            uid = str(row[id_col])
            start = pd.to_datetime(row.get("active_from"), errors="coerce")
            end = pd.to_datetime(row.get("closed_at"), errors="coerce")
            life[uid] = (
                None if pd.isna(start) else pd.Timestamp(start).to_period("M").to_timestamp("start"),
                None if pd.isna(end) else pd.Timestamp(end).to_period("M").to_timestamp("start"),
            )

    extra = [c for c in df.columns if c not in {"unique_id", "item_id", "ds", "y", "y_observed"}]
    pieces: list[pd.DataFrame] = []
    for uid, grp in df.groupby("unique_id", sort=False):
        grp = grp.sort_values("ds")
        start = grp["ds"].min()
        end = grp["ds"].max()
        win = life.get(str(uid))
        if win:
            if win[0] is not None:
                start = min(start, win[0])
            if win[1] is not None:
                end = max(end, win[1])
        full = pd.date_range(start=start, end=end, freq="MS")
        idx = grp.drop_duplicates("ds", keep="last").set_index("ds")
        aligned = idx.reindex(full)
        aligned.index.name = "ds"
        aligned = aligned.reset_index()
        aligned["unique_id"] = str(uid)
        if "item_id" in grp.columns:
            aligned["item_id"] = str(grp["item_id"].iloc[-1])
        else:
            aligned["item_id"] = str(uid)
        aligned["y"] = pd.to_numeric(aligned["y"], errors="coerce").fillna(0.0)
        # Interior gaps on the first→last observed span are observed sales zeros
        # (not unconstrained demand).
        aligned["y_observed"] = 1
        if win:
            ds = pd.to_datetime(aligned["ds"])
            if win[0] is not None:
                aligned.loc[ds < win[0], "y_observed"] = 0
            if win[1] is not None:
                aligned.loc[ds > win[1], "y_observed"] = 0
        for c in extra:
            if c in aligned.columns:
                aligned[c] = aligned[c].ffill().bfill()
        pieces.append(aligned)
    if not pieces:
        return df
    out = pd.concat(pieces, ignore_index=True)
    out["ds"] = pd.to_datetime(out["ds"])
    return out.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def seasonal_naive_value(y_orig: np.ndarray, t: int, h: int) -> float:
    """Calendar-aligned seasonal naive: y_orig[t + h - 12] on a completed MS grid.

    Last-value naive is y_orig[t]. lag_12 = y[t-11] is the 12-month lag of the
    origin, not the seasonal pair of the target month, and must not be used here.
    """
    idx = int(t) + int(h) - 12
    y = np.asarray(y_orig, dtype=float)
    if idx < 0 or idx >= len(y):
        return float(y[t]) if 0 <= int(t) < len(y) else 0.0
    return float(y[idx])


def calendar_lag_value(y: np.ndarray, ds: np.ndarray, t: int, lag_months: int) -> float:
    """Lag by calendar months on a completed MS grid (positional == calendar)."""
    idx = t - int(lag_months)
    if idx < 0:
        return 0.0
    return float(y[idx])


def months_since_last_sale(y_orig: np.ndarray, t: int) -> float:
    hist = np.asarray(y_orig[: t + 1], dtype=float)
    nz = np.where(hist > 0.0)[0]
    if len(nz) == 0:
        return float(t + 1)
    return float(t - int(nz[-1]))


def assert_finite_outputs(values: np.ndarray, *, name: str) -> None:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
