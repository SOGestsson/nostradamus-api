"""Paskavara-gated Easter placement for classical / AutoModel forecasts.

Easter Sunday moves between 22 March and 25 April, so a monthly SeasonalNaive
copies last year's calendar month and misses the spike when the feast changes
month. AutoModel is univariate and most of its candidates ignore regressors, so
the helper strips Easter out of history, lets the model forecast the rest, then
places the event onto this year's March/April using a known shopping window.

Only item ids passed in by the caller are touched. Missing / empty list is a
no-op so other clients keep current AutoModel behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

# Consumer sales sit before Easter Sunday: the week before Holy Week plus Holy
# Week itself. Ends the day before Easter so Sunday itself carries no weight.
WINDOW_BEFORE_EASTER = 21
WINDOW_THROUGH_BEFORE_EASTER = 1
EVENT_MONTHS = (3, 4)
HALFLIFE_YEARS = 2.0
UPPER_COLS = ("forecast_upper_70", "forecast_upper_90", "forecast_upper_95")
RESULT_UPPER_KEYS = ("upper_70", "upper_90", "upper_95")


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm. Easter falls 22 March through 25 April."""

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
    return date(int(year), int(month), int(day))


def monthly_easter_weights(year: int) -> dict[int, float]:
    """Share of the shopping window that falls in each calendar month.

    Window is Easter−21 through Easter−1. With 21 days this is March and/or
    April; the two weights sum to 1.
    """

    sunday = easter_sunday(year)
    start = sunday - timedelta(days=WINDOW_BEFORE_EASTER)
    end = sunday - timedelta(days=WINDOW_THROUGH_BEFORE_EASTER)
    counts: dict[int, int] = {}
    day = start
    while day <= end:
        counts[day.month] = counts.get(day.month, 0) + 1
        day += timedelta(days=1)
    total = float(sum(counts.values())) or 1.0
    return {month: n / total for month, n in counts.items()}


def _clean_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


@dataclass(frozen=True)
class EasterPlan:
    """Per-item event totals to place on the forecast after the model returns."""

    event_forecast: dict[str, float] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.event_forecast)

    @property
    def is_empty(self) -> bool:
        return not self.event_forecast


def _event_totals_by_year(
    hist_df: pd.DataFrame,
    easter_ids: set[str],
    *,
    item_col: str,
    date_col: str,
    sales_col: str,
) -> dict[str, dict[int, float]]:
    totals: dict[str, dict[int, float]] = {}
    items = hist_df[item_col].map(_clean_id)
    dates = pd.to_datetime(hist_df[date_col], errors="coerce")
    sales = pd.to_numeric(hist_df[sales_col], errors="coerce").fillna(0.0)
    mask = items.isin(easter_ids) & dates.notna() & dates.dt.month.isin(EVENT_MONTHS)
    if not bool(mask.any()):
        return totals
    years = dates.dt.year
    for item, year, sale in zip(items[mask], years[mask], sales[mask]):
        totals.setdefault(str(item), {})
        totals[str(item)][int(year)] = totals[str(item)].get(int(year), 0.0) + float(sale)
    return totals


def _recency_weighted_mean(by_year: dict[int, float], *, halflife: float = HALFLIFE_YEARS) -> float:
    if not by_year:
        return 0.0
    latest = max(by_year)
    num = 0.0
    den = 0.0
    for year, total in by_year.items():
        age = float(latest - year)
        weight = 0.5 ** (age / halflife) if halflife > 0 else 1.0
        num += weight * float(total)
        den += weight
    return num / den if den else 0.0


def prepare_easter_history(
    hist_df: pd.DataFrame,
    easter_ids: Iterable[object],
    *,
    item_col: str = "item_id",
    date_col: str = "day",
    sales_col: str = "actual_sale",
) -> tuple[pd.DataFrame, EasterPlan]:
    """Strip Easter from history for flagged items; leave everyone else untouched.

    Returns a copy only when at least one flagged item is present. Event
    forecasts are computed from the original March+April totals.
    """

    ids = {_clean_id(i) for i in easter_ids if _clean_id(i)}
    if hist_df is None or hist_df.empty or not ids:
        return hist_df, EasterPlan()
    if item_col not in hist_df.columns or date_col not in hist_df.columns:
        return hist_df, EasterPlan()
    if sales_col not in hist_df.columns:
        return hist_df, EasterPlan()

    present = set(hist_df[item_col].map(_clean_id)) & ids
    if not present:
        return hist_df, EasterPlan()

    totals = _event_totals_by_year(
        hist_df, present, item_col=item_col, date_col=date_col, sales_col=sales_col
    )
    event_forecast = {
        item: _recency_weighted_mean(by_year)
        for item, by_year in totals.items()
        if by_year
    }
    plan = EasterPlan(event_forecast)

    out = hist_df.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")
    items = out[item_col].map(_clean_id)
    sales = pd.to_numeric(out[sales_col], errors="coerce")
    adjusted = sales.copy()
    for idx in out.index:
        item = items.at[idx]
        if item not in present:
            continue
        ts = dates.at[idx]
        if ts is pd.NaT or pd.isna(ts):
            continue
        month = int(ts.month)
        if month not in EVENT_MONTHS:
            continue
        year = int(ts.year)
        event = totals.get(item, {}).get(year, 0.0)
        if event <= 0:
            continue
        weight = monthly_easter_weights(year).get(month, 0.0)
        raw = sales.at[idx]
        if pd.isna(raw):
            continue
        adjusted.at[idx] = max(0.0, float(raw) - event * weight)
    out[sales_col] = adjusted
    return out, plan


def apply_easter_to_forecasts(
    fcst_df: pd.DataFrame,
    plan: EasterPlan,
    *,
    item_col: str = "item_id",
    date_col: str = "forecast_date",
    forecast_col: str = "forecast",
) -> pd.DataFrame:
    """Add the Easter event onto March/April forecast points. Band width kept."""

    if fcst_df is None or fcst_df.empty or plan.is_empty:
        return fcst_df
    if item_col not in fcst_df.columns or date_col not in fcst_df.columns:
        return fcst_df
    if forecast_col not in fcst_df.columns:
        return fcst_df

    out = fcst_df.copy()
    items = out[item_col].map(_clean_id)
    dates = pd.to_datetime(out[date_col], errors="coerce")
    lifts = pd.Series(0.0, index=out.index)
    for idx in out.index:
        item = items.at[idx]
        event = plan.event_forecast.get(item)
        if event is None or event <= 0:
            continue
        ts = dates.at[idx]
        if ts is pd.NaT or pd.isna(ts):
            continue
        month = int(ts.month)
        if month not in EVENT_MONTHS:
            continue
        weight = monthly_easter_weights(int(ts.year)).get(month, 0.0)
        if weight <= 0:
            continue
        lifts.at[idx] = float(event) * weight

    if not bool((lifts > 0).any()):
        return out

    out[forecast_col] = pd.to_numeric(out[forecast_col], errors="coerce").fillna(0.0) + lifts
    for col in UPPER_COLS:
        if col not in out.columns:
            continue
        upper = pd.to_numeric(out[col], errors="coerce")
        out[col] = upper + lifts
    return out


def apply_easter_to_item_results(results: list[dict], plan: EasterPlan) -> list[dict]:
    """Place Easter onto generate_forecast item payloads (lists of dates/values)."""

    if not results or plan.is_empty:
        return results

    out: list[dict] = []
    for raw in results:
        if not isinstance(raw, dict):
            out.append(raw)
            continue
        item = dict(raw)
        event = plan.event_forecast.get(_clean_id(item.get("item_id")))
        dates = item.get("forecast_dates") or []
        vals = item.get("forecast") or []
        if event is None or event <= 0 or not isinstance(dates, list) or not isinstance(vals, list):
            out.append(item)
            continue
        lifts: list[float] = []
        n = min(len(dates), len(vals))
        for i in range(n):
            ts = pd.to_datetime(dates[i], errors="coerce")
            if ts is pd.NaT or pd.isna(ts):
                lifts.append(0.0)
                continue
            month = int(ts.month)
            if month not in EVENT_MONTHS:
                lifts.append(0.0)
                continue
            lifts.append(float(event) * monthly_easter_weights(int(ts.year)).get(month, 0.0))
        item["forecast"] = [float(vals[i]) + lifts[i] for i in range(n)]
        for key in RESULT_UPPER_KEYS:
            upper = item.get(key)
            if isinstance(upper, list) and len(upper) >= n:
                item[key] = [float(upper[i]) + lifts[i] for i in range(n)]
        out.append(item)
    return out
