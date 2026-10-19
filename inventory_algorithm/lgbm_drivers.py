"""Known-future drivers: vintage, target-month join, missing required columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from inventory_algorithm.lgbm_origin import month_start

VINTAGE_OBSERVED = "observed"
VINTAGE_KNOWN_FUTURE = "known_future"
VINTAGE_SCENARIO = "scenario"


class MissingRequiredDriver(ValueError):
    pass


def normalize_drivers(drivers: pd.DataFrame | None) -> pd.DataFrame:
    if drivers is None or drivers.empty:
        return pd.DataFrame()
    df = drivers.copy()
    if "day" in df.columns and "ds" not in df.columns:
        df["ds"] = pd.to_datetime(df["day"], errors="coerce")
    if "ds" not in df.columns:
        raise ValueError("drivers require a day or ds column")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").map(month_start)
    if "vintage" not in df.columns:
        df["vintage"] = VINTAGE_OBSERVED
    df["vintage"] = df["vintage"].astype(str).str.lower().fillna(VINTAGE_OBSERVED)
    if "vintage_ds" in df.columns:
        df["vintage_ds"] = pd.to_datetime(df["vintage_ds"], errors="coerce").map(
            lambda x: month_start(x) if pd.notna(x) else pd.NaT
        )
    return df.dropna(subset=["ds"])


def _vintage_allows(*, vintage: str, target: pd.Timestamp, origin: pd.Timestamp, vintage_ds) -> bool:
    origin = month_start(origin)
    target = month_start(target)
    v = str(vintage or VINTAGE_OBSERVED).strip().lower()
    if v in {VINTAGE_KNOWN_FUTURE, VINTAGE_SCENARIO}:
        if vintage_ds is None or (isinstance(vintage_ds, float) and pd.isna(vintage_ds)) or pd.isna(vintage_ds):
            return True
        return month_start(vintage_ds) <= origin
    # observed (default): the target-month realization is only known once target <= origin
    return target <= origin


def driver_value_at_target(
    drivers_wide: pd.DataFrame,
    *,
    item_id: str,
    forecast_ds: pd.Timestamp,
    column: str,
    origin: pd.Timestamp,
    required: bool = False,
) -> float | None:
    """Return driver at the target month only if vintage says it is known at origin.

    observed: allowed only when forecast_ds <= origin (already realized).
    known_future / scenario: allowed at forecast_ds in live forecasts; in backtests
    allowed only if vintage_ds is missing or vintage_ds <= origin.
    """
    if drivers_wide is None or drivers_wide.empty:
        if required:
            raise MissingRequiredDriver(f"required driver {column} missing for item {item_id}")
        return None
    work = normalize_drivers(drivers_wide)
    target = month_start(forecast_ds)
    origin_m = month_start(origin)
    if "item_id" in work.columns:
        scoped = work[work["item_id"].astype(str) == str(item_id)]
        if scoped.empty:
            scoped = work
    else:
        scoped = work

    value_col = "driver_value" if "driver_name" in scoped.columns else column
    if "driver_name" in scoped.columns:
        hit = scoped[(scoped["driver_name"].astype(str) == str(column)) & (scoped["ds"] == target)]
    elif column not in scoped.columns:
        hit = pd.DataFrame()
    else:
        hit = scoped[scoped["ds"] == target]

    if hit.empty:
        if required:
            raise MissingRequiredDriver(f"required driver {column} missing at {target.date()} for {item_id}")
        return None

    row = hit.iloc[0]
    vintage = str(row["vintage"]) if "vintage" in hit.columns else VINTAGE_OBSERVED
    vintage_ds = row["vintage_ds"] if "vintage_ds" in hit.columns else None
    if not _vintage_allows(vintage=vintage, target=target, origin=origin_m, vintage_ds=vintage_ds):
        if required:
            raise MissingRequiredDriver(
                f"required driver {column} at {target.date()} is not known at origin {origin_m.date()} "
                f"(vintage={vintage})"
            )
        return None

    val = pd.to_numeric(row[value_col], errors="coerce")
    if pd.isna(val):
        if required:
            raise MissingRequiredDriver(f"required driver {column} is null at {target.date()} for {item_id}")
        return None
    return float(val)


def attach_target_month_drivers(
    row: dict[str, Any],
    drivers_wide: pd.DataFrame | None,
    *,
    columns: list[str],
    required: list[str] | None = None,
) -> dict[str, Any]:
    required = required or []
    if not columns:
        return row
    if drivers_wide is None or drivers_wide.empty:
        for c in required:
            raise MissingRequiredDriver(f"required driver {c} missing")
        return row
    origin = month_start(row["decision_ds"])
    forecast_ds = month_start(row["forecast_ds"])
    item_id = str(row.get("unique_id") or row.get("item_id"))
    for c in columns:
        val = driver_value_at_target(
            drivers_wide,
            item_id=item_id,
            forecast_ds=forecast_ds,
            column=c,
            origin=origin,
            required=c in required,
        )
        if val is None:
            continue
        row[c] = val
        row[f"{c}_at_target"] = 1.0
    return row
