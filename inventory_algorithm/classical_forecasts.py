# classical_forecasts.py (or drop into your existing module)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Callable, Optional

try:
    # NOTE: Do not import nixtla at module import time.
    # Auto-model (StatsForecast) must not require or touch Nixtla.
    pass
except Exception:
    pass


def _lazy_import_timegpt_client():
    """Lazy import of Nixtla client.

    This guarantees StatsForecast-only paths (including auto_model selection)
    never import or instantiate the Nixtla SDK.
    """
    try:
        from nixtla import NixtlaClient  # type: ignore
    except Exception as e:
        raise RuntimeError("Nixtla 'nixtla' package not available. Install with `pip install nixtla`.") from e
    return NixtlaClient

# Optional local fallback models via StatsForecast
def _lazy_import_nixtla_models():
    from statsforecast.models import (
        AutoARIMA,
        AutoETS,
        CrostonOptimized,
        Naive,
        SeasonalNaive,
        ADIDA,
        OptimizedTheta,
        Theta,
        AutoCES,
        HistoricAverage,
        WindowAverage,
        SeasonalWindowAverage,
    )
    return {
        'auto_arima': AutoARIMA,
        'auto_ets': AutoETS,
        'croston_optimized': CrostonOptimized,
        'naive': Naive,
        'seasonal_naive': SeasonalNaive,
        'adida': ADIDA,
        'optimized_theta': OptimizedTheta,
        'theta': Theta,
        'auto_ces': AutoCES,
        'historic_average': HistoricAverage,
        'window_average': WindowAverage,
        'seasonal_window_average': SeasonalWindowAverage,
    }


def simple_croston_mean(y: np.ndarray) -> float:
    """Return a simple Croston-style mean for intermittent demand."""
    y = np.asarray(y, dtype=float)
    idx = np.where(y > 0.0)[0]
    if len(idx) == 0:
        return 0.0
    sizes = y[idx]
    if len(idx) == 1:
        return float(sizes[0])
    intervals = np.diff(idx)
    mean_size = float(np.mean(sizes))
    mean_interval = float(np.mean(intervals)) if float(np.mean(intervals)) > 0 else 1.0
    return float(mean_size / mean_interval)


def simple_adida_mean(y: np.ndarray, agg: int = 3) -> float:
    """Return a simple ADIDA-style mean (aggregate then average back to monthly scale)."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return 0.0
    if agg <= 1:
        return float(np.mean(y))
    n = len(y) - (len(y) % agg)
    if n <= 0:
        return float(np.mean(y))
    y2 = y[:n].reshape(-1, agg).sum(axis=1)
    return float(np.mean(y2) / agg)


def _metric_func_from_name(name: str) -> tuple[str, Optional[Callable]]:
    """Map a user-friendly metric string to utilsforecast loss function.

    Special values:
    - 'robust' / 'rank' / 'rmse+mae': rank-aggregate RMSE and MAE per series.
    - 'wape': weighted absolute percentage error (lower is better)
    - 'wape_bias': WAPE primary; prefer models within bias threshold when possible
    """
    from utilsforecast.losses import rmse, mae

    metric = (name or '').strip().lower()
    if metric in ('robust', 'rank', 'rmse+mae', 'rmse_mae'):
        return 'robust', None
    if metric in ('wape',):
        return 'wape', None
    if metric in ('wape_bias', 'wape+bias', 'wape_bias_pct', 'wape_biaspct'):
        return 'wape_bias', None
    if metric in ('rmse',):
        return 'rmse', rmse
    if metric in ('mae',):
        return 'mae', mae
    raise ValueError("Unsupported metric. Use 'rmse', 'mae', 'robust', 'wape', or 'wape_bias'.")


def _safe_wape_and_bias(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    """Return (wape, bias_pct) using safe denominators.

    WAPE = sum(|y - yhat|) / sum(|y|)
    bias_pct = 100 * sum(yhat - y) / sum(|y|)

    Uses sum(|y|) to avoid division by zero on sparse/zero series.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float('inf'), float('inf')
    y = y[mask]
    yhat = yhat[mask]

    denom = float(np.sum(np.abs(y)))
    if denom <= 1e-12:
        # All-zero or near-zero: WAPE is ill-defined; fall back to absolute error scale.
        # This still provides a consistent ordering across models.
        wape = float(np.mean(np.abs(y - yhat)))
        bias_pct = float(np.mean(yhat - y))
        return wape, bias_pct

    err = yhat - y
    wape = float(np.sum(np.abs(err)) / denom)
    bias_pct = float(100.0 * np.sum(err) / denom)
    return wape, bias_pct


def _pick_model_wape_bias_penalty(
    wape_vals: pd.Series,
    bias_pct_vals: pd.Series,
    *,
    rel_eps: float = 0.02,
    abs_eps: float = 0.005,
    seasonal_naive_min_wape_advantage: float = 0.06,
    bias_ok_pct: float = 10.0,
    bias_scale_pct: float = 20.0,
    weight: float = 0.25,
    prefer_seasonal_naive: bool = False,
) -> str:
    """Pick a model using WAPE primary with a symmetric |bias| penalty.

    - WAPE is the primary objective.
    - Bias is symmetric: over/under forecasts are equally bad.
    - Bias only affects choice among models whose WAPE is within a small band
      of the best WAPE (to keep WAPE as the main focus).

    Score definition (in percent points):
      W = 100 * wape
      B = |bias_pct|
      penalty = weight * max(0, B - bias_ok)^2 / bias_scale
      score = W + penalty

    Edge case: if wape values are on an absolute-error scale (all-zero denom
    fallback), then bias_pct is not a percentage; disable the bias penalty.
    """
    if wape_vals.empty:
        return 'Naive'

    wape_vals = wape_vals.astype(float)
    bias_abs = bias_pct_vals.astype(float).abs()

    best_wape = float(wape_vals.min())
    # Use both relative and absolute epsilon to be stable across scales.
    band = max(float(abs_eps), abs(best_wape) * float(rel_eps))
    close = wape_vals <= (best_wape + band)
    if bool(close.any()):
        w_close = wape_vals[close]
        b_close = bias_abs[close]
    else:
        w_close = wape_vals
        b_close = bias_abs

    # If WAPE is on absolute-error scale (near-zero denom fallback), bias_pct is
    # not actually a percent. In that case, disable the penalty and just pick
    # the lowest WAPE (tie-break by |bias| for determinism).
    if best_wape > 5.0:
        picked = (
            pd.DataFrame({'wape': w_close, 'abs_bias': b_close})
            .sort_values(['wape', 'abs_bias'], ascending=True)
            .index[0]
        )
        return str(picked)

    # Penalize excessive |bias| above an "ok" zone.
    excess = (b_close - float(bias_ok_pct)).clip(lower=0.0)
    denom = float(bias_scale_pct) if float(bias_scale_pct) > 0 else 1.0
    penalty = float(weight) * (excess * excess) / denom
    w_pct = 100.0 * w_close
    score = w_pct + penalty

    # Policy: avoid picking SeasonalNaive when an adaptive model is essentially tied.
    # Rationale: SeasonalNaive is a strong baseline but can be brittle under level shifts.
    # Keep it only when it is materially better on WAPE.
    # Exception: event-seasonal items (prefer_seasonal_naive=True) — SN is the natural
    # model and should not be penalized.
    seasonal_name = 'SeasonalNaive'
    if seasonal_name in w_close.index and not prefer_seasonal_naive:
        alternatives = [
            m for m in [
                'AutoETS', 'Theta', 'OptimizedTheta', 'AutoARIMA',
                'HistoricAverage', 'SeasonalWindowAverage',
            ]
            if m in w_close.index
        ]
        if alternatives:
            best_alt = float(w_close.loc[alternatives].min())
            seasonal_wape = float(w_close.loc[seasonal_name])
            advantage = best_alt - seasonal_wape
            if advantage < float(seasonal_naive_min_wape_advantage):
                score_alt = score.loc[alternatives]
                picked_alt = (
                    pd.DataFrame({'score': score_alt, 'wape': w_close.loc[alternatives], 'abs_bias': b_close.loc[alternatives]})
                    .sort_values(['score', 'wape', 'abs_bias'], ascending=True)
                    .index[0]
                )
                return str(picked_alt)

    picked = (
        pd.DataFrame({'score': score, 'wape': w_close, 'abs_bias': b_close})
        .sort_values(['score', 'wape', 'abs_bias'], ascending=True)
        .index[0]
    )
    return str(picked)


def _build_candidate_model_factories(season_length: int) -> list[tuple[str, Callable[[], object]]]:
    """Candidate StatsForecast model factories (explicitly excludes TimeGPT/LightGPT)."""
    models_dict = _lazy_import_nixtla_models()
    # NOTE: AutoCES is intentionally excluded from the auto-model candidate set
    # because it can fail to fit for some series and would otherwise abort the
    # entire cross-validation run.
    seasonal = {
        'auto_arima',
        'auto_ets',
        'seasonal_naive',
        'theta',
        'optimized_theta',
        'seasonal_window_average',
    }
    keys = [
        'naive',
        'historic_average',
        'seasonal_naive',
        'seasonal_window_average',
        'auto_arima',
        'auto_ets',
        'theta',
        'optimized_theta',
        'croston_optimized',
        'adida',
    ]
    specs: list[tuple[str, Callable[[], object]]] = []
    for key in keys:
        ModelClass = models_dict[key]
        if key in seasonal:
            if key == 'seasonal_window_average':
                # window_size=1 means "use last season"; alias must match class name
                # because downstream expects the forecast column to be that name.
                specs.append((
                    ModelClass.__name__,
                    lambda cls=ModelClass: cls(season_length=season_length, window_size=1, alias=cls.__name__),
                ))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
        else:
            if key == 'window_average':
                # Not used as default fallback (we use HistoricAverage for variable windows),
                # but keep a small window available for selection.
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(window_size=3, alias=cls.__name__)))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls()))
    return specs


def _build_model_factories_for_keys(keys: list[str], season_length: int) -> list[tuple[str, Callable[[], object]]]:
    models_dict = _lazy_import_nixtla_models()
    seasonal = {
        'auto_arima',
        'auto_ets',
        'seasonal_naive',
        'theta',
        'optimized_theta',
        'seasonal_window_average',
    }
    specs: list[tuple[str, Callable[[], object]]] = []
    for key in keys:
        if key not in models_dict:
            continue
        ModelClass = models_dict[key]
        if key in seasonal:
            if key == 'seasonal_window_average':
                specs.append((
                    ModelClass.__name__,
                    lambda cls=ModelClass: cls(season_length=season_length, window_size=1, alias=cls.__name__),
                ))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
        else:
            if key == 'window_average':
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(window_size=3, alias=cls.__name__)))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls()))
    return specs


def _series_profile(y: np.ndarray) -> dict[str, float]:
    """Cheap per-series stats used to select candidate model sets."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n == 0:
        return {'n': 0.0, 'zero_frac': 1.0, 'adi': float('inf'), 'cv2': float('inf'), 'trend_corr': 0.0}

    zero_frac = float(np.mean(y <= 0))
    nz_idx = np.flatnonzero(y > 0)
    if nz_idx.size >= 2:
        adi = float(np.diff(nz_idx).mean())
    elif nz_idx.size == 1:
        adi = float(n)
    else:
        adi = float('inf')

    y_pos = y[y > 0]
    if y_pos.size >= 2 and float(y_pos.mean()) > 0:
        cv2 = float((y_pos.std(ddof=1) / y_pos.mean()) ** 2)
    else:
        cv2 = float('inf') if y_pos.size == 0 else 0.0

    # Trend proxy: correlation with time index (bounded [-1,1]).
    t = np.arange(n, dtype=float)
    if n >= 3 and float(np.std(y)) > 0:
        trend_corr = float(np.corrcoef(t, y)[0, 1])
        if not np.isfinite(trend_corr):
            trend_corr = 0.0
    else:
        trend_corr = 0.0

    return {'n': float(n), 'zero_frac': zero_frac, 'adi': adi, 'cv2': cv2, 'trend_corr': trend_corr}


def canonical_forecaster_freq(freq: str | None) -> str:
    """Single source of truth for monthly ``freq`` in ClassicalForecasts and the forecast API.

    Convention (matches ``api/v1/forecast.py``):
    - ``M``, ``MS``, and ``ME`` (any case) → ``MS`` (first day of month, ``YYYY-MM-01``).
      There is **no** month-end period index: ``ME`` is accepted only as a legacy alias
      and is normalized the same as ``M`` / ``MS``.
    - ``None`` / empty → ``D``.
    - Other offsets (``D``, ``W-SUN``, …) returned stripped with original casing.
    """
    if freq is None:
        return 'D'
    s = str(freq).strip()
    if not s:
        return 'D'
    u = s.upper()
    if u in ('M', 'MS', 'ME'):
        return 'MS'
    return s


def _normalize_monthly_ds_to_period_anchor(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Snap ``ds`` to the first day of each calendar month, then sum ``y`` per month.

    ``pd.date_range(start=..., freq='MS')`` is anchored to the *first* month boundary
    on or after ``start``. If ``start`` is month-end (e.g. 2024-01-31) or mid-month,
    that grid skips the month that the row actually belongs to and shifts every
    observation one period — seasonal models then peak in the wrong calendar month.

    Call this before any monthly ``date_range`` / reindex used with StatsForecast.
    """
    if canonical_forecaster_freq(freq) != 'MS':
        return df
    out = df.copy()
    d = pd.to_datetime(out['ds'])
    out['ds'] = d.dt.to_period('M').dt.to_timestamp(how='start')
    out = out.groupby(['unique_id', 'ds'], as_index=False)['y'].sum()
    return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def _monthly_anchor_now() -> pd.Timestamp:
    """First day of the current calendar month (for extending sparse monthly history)."""
    return pd.Timestamp.now(tz=None).to_period('M').to_timestamp(how='start')


def _regularize_panel_extend_end(freq: str, data_max: pd.Timestamp) -> pd.Timestamp:
    """Extend regularized monthly panels through max(data, current month).

    Imports often omit months with no sales, so a dead SKU's last row can be its
    last *sale* month — there is no long explicit zero tail for heuristics.
    Padding through the current month makes trailing zeros real for auto_model.
    """
    cf = canonical_forecaster_freq(freq)
    if cf == 'MS':
        anchor = _monthly_anchor_now()
        dm = pd.to_datetime(data_max)
        return max(dm, anchor)
    return pd.to_datetime(data_max)


def _regularize_panel_time_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return a panel with a complete time index per unique_id.

    Many StatsForecast models expect a regular time grid. Real inventory history
    often has missing timestamps (no transactions recorded). For demand series,
    missing periods should typically be treated as 0.

    The panel ends at each series' last observation — it does NOT extend to the
    current month. Callers control the history cutoff; adding future zeros would
    shift the forecast start date.

    Input df must have columns: ['unique_id','ds','y']. For monthly freq, ``ds``
    must already be month-start anchors (see :func:`_normalize_monthly_ds_to_period_anchor`
    and :meth:`ClassicalForecasts._to_statsforecast_df`).
    """
    if df.empty:
        return df
    range_freq = canonical_forecaster_freq(freq)
    out_parts: list[pd.DataFrame] = []
    for uid, g in df.groupby('unique_id', sort=False):
        g = g.sort_values('ds')
        start = pd.to_datetime(g['ds'].iloc[0])
        end = pd.to_datetime(g['ds'].iloc[-1])
        full = pd.date_range(start=start, end=end, freq=range_freq)
        # If already regular, skip reindex work.
        if len(full) == len(g) and pd.Index(g['ds']).is_monotonic_increasing:
            out_parts.append(g)
            continue
        gg = g.set_index('ds')
        gg = gg.reindex(full)
        gg.index.name = 'ds'
        gg = gg.reset_index()
        gg['unique_id'] = str(uid)
        gg['y'] = gg['y'].fillna(0.0)
        out_parts.append(gg.loc[:, ['unique_id', 'ds', 'y']])
    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)


def _recent_tail_stable_level(
    y: np.ndarray,
    *,
    tail_n: int = 8,
    max_cv: float = 0.55,
    min_mean: float = 0.5,
) -> tuple[bool, float]:
    """Detect a flat-ish positive tail (new stable demand level vs dying SKU).

    Returns (is_stable, mean_of_last_tail_n_observations).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    tn = int(min(tail_n, y.size)) if y.size else 0
    if tn < max(4, tail_n // 2):
        return False, 0.0
    t = y[-tn:]
    m = float(np.mean(t))
    if m < float(min_mean):
        return False, m
    if t.size < 2:
        return True, m
    s = float(np.std(t, ddof=1))
    cv = s / m if m > 1e-12 else float("inf")
    if cv > float(max_cv):
        return False, m
    return True, m


def _seasonal_naive_lag_regime_mismatch(y: np.ndarray, season_length: int, *, tail_n: int = 8) -> bool:
    """SeasonalNaive uses y[t-s]; if the recent stable level differs sharply from the lag window, SN is misleading.

    Bidirectional: fires when recent >> lag (SN would under-forecast) **and** when
    lag >> recent (SN would over-forecast after a level drop).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    s = int(season_length)
    if y.size < s + tail_n:
        return False
    recent = y[-tail_n:]
    lag = y[-tail_n - s : -s]
    mr = float(np.mean(recent))
    ml = float(np.mean(lag))
    ref = max(mr, ml)
    if ref < 1.0:
        return False
    if recent.size < 2:
        cv_r = 0.0
    else:
        cv_r = float(np.std(recent, ddof=1)) / max(mr, 1e-12)
    if cv_r > 0.62:
        return False
    ratio = min(mr, ml) / max(mr, ml) if ref > 1e-12 else 1.0
    return bool(ratio < 0.42)


def _strong_yearly_seasonality(y: np.ndarray, season_length: int) -> bool:
    """True when y aligns with y lagged by one season (e.g. summer peaks), despite zeros.

    Intermittent-demand heuristics often mis-classify strong annual patterns with long
    off-seasons; this is a cheap lag-``season_length`` correlation check on trimmed y.
    """
    y = np.asarray(y, dtype=float)
    s = int(season_length)
    if s < 2 or y.size < 2 * s:
        return False
    a = y[s:]
    b = y[:-s]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < s:
        return False
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) <= 1e-12 or float(np.std(bb)) <= 1e-12:
        return False
    r = float(np.corrcoef(aa, bb)[0, 1])
    return bool(np.isfinite(r) and r >= 0.38)


def _bucket_series(profile: dict[str, float], season_length: int, min_arima_len: int) -> str:
    n = int(profile['n'])
    zero_frac = float(profile['zero_frac'])
    adi = float(profile['adi'])
    cv2 = float(profile['cv2'])
    trend_corr = float(profile['trend_corr'])

    # Very short histories: skip CV.
    # IMPORTANT: history length drives this decision; season_length is a model
    # configuration and should not decide whether a series is "too short".
    if n < 20:
        return 'short'

    # Intermittent demand heuristic (Syntetos-Boylan style): ADI + CV^2.
    if (zero_frac >= 0.30) or (adi > 1.32 and cv2 > 0.49):
        return 'intermittent'

    # Seasonal if enough history for at least ~2 seasons.
    if season_length >= 2 and n >= 2 * season_length:
        return 'seasonal'

    # Trend if correlation is strong and history is long-ish.
    if n >= min_arima_len and abs(trend_corr) >= 0.5:
        return 'trend'

    return 'smooth'


def _auto_model_force_naive_trailing_zeros(y_full: np.ndarray, *, min_months: int = 24) -> bool:
    """True when the last ``min_months`` observations are all (near) zero.

    Long inactive tails should not be extrapolated with ETS/seasonal models that
    keep a small positive smoothed level (common user complaint: 'dead' SKUs).
    Naive then repeats the last value, typically 0 on a regular monthly grid.
    """
    y = np.asarray(y_full, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < int(min_months):
        return False
    tail = y[-int(min_months) :]
    return bool(np.all(tail <= 0.0))


def _auto_model_force_naive_long_silence_after_last_sale(
    y_full: np.ndarray,
    ds: pd.Series,
    *,
    min_silent_months: int = 18,
) -> bool:
    """True when the last *positive* demand was many calendar months ago.

    Compares the last sale date to the current calendar month rather than to the
    series end so that sparse imports (which stop at the last sale) are still
    flagged as dead.
    """
    y = np.asarray(y_full, dtype=float)
    ds = pd.to_datetime(ds, errors='coerce').reset_index(drop=True)
    if y.size == 0 or len(ds) != len(y):
        return False
    nz = np.flatnonzero(y > 1e-12)
    if nz.size == 0:
        return True
    last_i = int(nz[-1])
    last_sale = ds.iloc[last_i]
    if pd.isna(last_sale):
        return False
    now = _monthly_anchor_now()
    silent_m = (now.year - last_sale.year) * 12 + (now.month - last_sale.month)
    return int(silent_m) >= int(min_silent_months)


def _auto_model_exclude_seasonal_naive(
    y_full: np.ndarray,
    *,
    season_length: int,
    event_seasonal: bool = False,
    stable_recent_level: bool = False,
) -> bool:
    """True when SeasonalNaive (y[t]=y[t-s]) is likely misleading.

    - Recent all-zero tail: same month last year may show a peak while the item
      is effectively discontinued for the last several months.
    - Year-over-year collapse: prior seasonal year had real volume, last year
      near zero — repeating the old seasonal profile is usually wrong.
      **Skipped** when ``stable_recent_level``: demand stepped down but the last
      months sit at a stable positive level (not discontinuation).

    Event-seasonal SKUs (e.g. Christmas-heavy) often have long off-season zero
    runs and volatile recent years; those patterns are *not* discontinuation.
    For them we skip these exclusions and let CV pick (``long_trailing_zero_run``
    / ``long_silence_after_last_sale`` still force Naive when appropriate).
    """
    if event_seasonal:
        return False
    y = np.asarray(y_full, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < max(6, min(season_length, 12)):
        return False
    tail_n = min(6, y.size)
    if bool(np.all(y[-tail_n:] <= 0.0)):
        return True
    if y.size < 2 * int(season_length):
        return False
    prev = float(np.mean(y[-2 * int(season_length) : -int(season_length)]))
    curr = float(np.mean(y[-int(season_length) :]))
    if prev > 1.0 and curr < 0.35 * prev:
        if stable_recent_level:
            return False
        return True
    return False


def _auto_model_maybe_prefer_level_under_stable_tail(
    *,
    best_by_uid: dict[str, str],
    stable_tail_uid: dict[str, bool],
    bucket_by_uid: dict[str, str],
    wape_scores_map: dict[str, dict[str, float]],
    rmse_scores_map: dict[str, dict[str, float]],
    mae_scores_map: dict[str, dict[str, float]],
    metric_name: str,
) -> None:
    """When the recent tail is a stable positive level, prefer HA/SWA if CV scores are competitive.

    Reduces AutoETS/Theta/ARIMA extrapolating a past decline down to ~0 while demand
    has already settled at a lower but stable run rate.
    """
    adaptive = {'AutoETS', 'Theta', 'OptimizedTheta', 'AutoARIMA'}
    level_models = ('HistoricAverage', 'SeasonalWindowAverage')
    # Never override adaptive picks for trending series — level models can't follow trends.
    buckets_ok = {'seasonal', 'smooth'}
    for uid, pick in list(best_by_uid.items()):
        if not stable_tail_uid.get(uid, False):
            continue
        if bucket_by_uid.get(uid) not in buckets_ok:
            continue
        if pick not in adaptive:
            continue

        if metric_name in ('wape', 'wape_bias'):
            pw = wape_scores_map.get(uid, {}).get(pick)
            if pw is None or not np.isfinite(float(pw)):
                continue
            pool: list[str] = [pick]
            for alt in level_models:
                aw = wape_scores_map.get(uid, {}).get(alt)
                if aw is None or not np.isfinite(float(aw)):
                    continue
                # Level model must actually beat the adaptive model on WAPE.
                if float(aw) <= float(pw):
                    pool.append(alt)
            if len(pool) > 1:
                best_by_uid[uid] = min(pool, key=lambda m: float(wape_scores_map[uid][m]))
        elif metric_name == 'robust':
            pr = rmse_scores_map.get(uid, {}).get(pick)
            pm = mae_scores_map.get(uid, {}).get(pick)
            if pr is None or pm is None:
                continue
            if not np.isfinite(float(pr)) or not np.isfinite(float(pm)):
                continue
            pool_r: list[str] = [pick]
            for alt in level_models:
                ar = rmse_scores_map.get(uid, {}).get(alt)
                am = mae_scores_map.get(uid, {}).get(alt)
                if ar is None or am is None:
                    continue
                if not np.isfinite(float(ar)) or not np.isfinite(float(am)):
                    continue
                # Level model must beat or match on both RMSE and MAE.
                if float(ar) <= float(pr) and float(am) <= float(pm):
                    pool_r.append(alt)
            if len(pool_r) > 1:
                best_by_uid[uid] = min(
                    pool_r,
                    key=lambda m: float(rmse_scores_map[uid].get(m, np.inf)) + float(mae_scores_map[uid].get(m, np.inf)),
                )


def _rerank_pick_excluding_seasonal_naive(
    *,
    uid: str,
    metric_name: str,
    best_by_uid: dict[str, str],
    rmse_scores_map: dict[str, dict[str, float]],
    mae_scores_map: dict[str, dict[str, float]],
    wape_scores_map: dict[str, dict[str, float]],
    bias_scores_map: dict[str, dict[str, float]],
    metric_scores: dict[str, dict[str, float]],
) -> None:
    """If selection is SeasonalNaive but that model is disallowed, pick the next best in-place."""
    if best_by_uid.get(uid) != 'SeasonalNaive':
        return
    if metric_name == 'robust':
        rm = {m: v for m, v in rmse_scores_map.get(uid, {}).items() if m != 'SeasonalNaive'}
        ma = {m: v for m, v in mae_scores_map.get(uid, {}).items() if m != 'SeasonalNaive'}
        models = sorted(set(rm.keys()) & set(ma.keys()))
        if not models:
            best_by_uid[uid] = 'Naive'
            return
        rmse_vals = pd.Series({m: rm.get(m, np.inf) for m in models})
        mae_vals = pd.Series({m: ma.get(m, np.inf) for m in models})
        total_rank = rmse_vals.rank(method='min').add(mae_vals.rank(method='min'), fill_value=0)
        picked = str(total_rank.idxmin())
        if np.isfinite(rmse_vals.get(picked, np.inf)):
            best_by_uid[uid] = picked
        else:
            best_by_uid[uid] = 'Naive'
        return
    if metric_name in ('wape', 'wape_bias'):
        wape_vals = pd.Series({m: v for m, v in wape_scores_map.get(uid, {}).items() if m != 'SeasonalNaive'})
        if wape_vals.empty:
            best_by_uid[uid] = 'Naive'
            return
        bias_vals = pd.Series({m: float(bias_scores_map.get(uid, {}).get(m, np.inf)) for m in wape_vals.index})
        if metric_name == 'wape_bias':
            picked = _pick_model_wape_bias_penalty(
                wape_vals,
                bias_vals,
                rel_eps=0.02,
                abs_eps=0.005,
                bias_ok_pct=10.0,
                bias_scale_pct=20.0,
                weight=0.25,
            )
        else:
            picked = (
                pd.DataFrame({'wape': wape_vals, 'abs_bias_pct': bias_vals.abs()})
                .sort_values(['wape', 'abs_bias_pct'], ascending=True)
                .index[0]
            )
        if np.isfinite(float(wape_scores_map[uid].get(str(picked), np.inf))):
            best_by_uid[uid] = str(picked)
        else:
            best_by_uid[uid] = 'Naive'
        return
    ms = {m: v for m, v in metric_scores.get(uid, {}).items() if m != 'SeasonalNaive'}
    if not ms:
        best_by_uid[uid] = 'Naive'
        return
    best_by_uid[uid] = str(min(ms.items(), key=lambda kv: kv[1])[0])


class ClassicalForecasts:
    """
    Plug-in forecaster with two modes:
      - 'timegpt'   -> Nixtla TimeGPT (cloud API)
      - 'local'     -> StatsForecast classical models (AutoARIMA/ETS/Croston/ADIDA etc.)
    Returns monthly forecast arrays compatible with your simulator.

    Conventions:
      - Input history is a DataFrame with columns: ['day', 'actual_sale', 'item_id']
      - Monthly ``freq``: ``M``, ``MS``, and legacy ``ME`` are **month-start** (stored as ``MS``).
        See :func:`canonical_forecaster_freq`.
    """

    def __init__(self,
                 mode: str = 'timegpt',
                 api_key: str | None = None,
                 model: str | None = None,     # e.g., 'timegpt-1', 'timegpt-1-long-horizon'
                 quantiles: list[float] | None = None,   # e.g., [0.1,0.5,0.8,0.95]
                 local_model: str = 'auto_arima',    # 'auto_arima'|'auto_ets'|'croston_optimized'|'adida'|'theta'
                 season_length: int = 12,  # Seasonality period (12=yearly cycle in monthly data)
                 freq: str = 'M',  # 'M'/'MS'/'ME' → month-start (canonical MS); 'D'/'W-*' daily/weekly
                 ):
        self.mode = mode
        self.quantiles = quantiles or []
        self.model_name = model
        self.freq = canonical_forecaster_freq(freq)
        self._client = None
        self.local_model = local_model
        self.season_length = season_length

        if self.mode == 'timegpt':
            NixtlaClient = _lazy_import_timegpt_client()
            self._client = NixtlaClient(api_key=api_key or os.environ.get("NIXTLA_API_KEY"))

    # ---------- TimeGPT path ----------
    def _timegpt_forecast_path(self, hist: pd.DataFrame, h: int) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          'ds' (timestamp), 'yhat' (point), and possibly quantile columns when requested.
        """
        df = hist.rename(columns={'day':'ds','actual_sale':'y'}).loc[:, ['ds','y']].copy()
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Ensure continuous timestamps with no gaps (TimeGPT requirement)
        df['ds'] = pd.to_datetime(df['ds'])
        df['unique_id'] = '_'
        df = _normalize_monthly_ds_to_period_anchor(df, str(self.freq))
        df = df.drop(columns=['unique_id'])

        # Reindex to fill any missing dates
        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq=self.freq)
        df = df.set_index('ds').reindex(date_range).reset_index()
        df.columns = ['ds', 'y']
        
        # Fill missing values (pandas 2.x compatible)
        df['y'] = df['y'].ffill().bfill().fillna(0)
        
        # Drop duplicates if any
        df = df.drop_duplicates(subset=['ds'], keep='first')

        kwargs = dict(df=df, h=h, freq=self.freq, time_col='ds', target_col='y')
        if self.model_name:
            kwargs['model'] = self.model_name
        if self.quantiles:
            kwargs['quantiles'] = self.quantiles

        fcst = self._client.forecast(**kwargs)
        out = pd.DataFrame({'ds': pd.to_datetime(fcst['ds'])})
        out['yhat'] = fcst.get('TimeGPT', fcst.get('TimeGPT-q-50', np.nan))
        # Attach quantiles if present
        for q in self.quantiles:
            key = f"TimeGPT-q-{int(q*100)}"
            if key in fcst.columns:
                out[key] = fcst[key]
        return out

    # ---------- Local fallback path ----------
    def _local_forecast_path(self, hist: pd.DataFrame, h: int) -> pd.DataFrame:
        """
        Local fallback using StatsForecast models (AutoARIMA, ETS, Croston, etc.).
        Returns DataFrame with 'ds', 'yhat', and upper quantiles.
        """
        from statsforecast import StatsForecast
        
        models_dict = _lazy_import_nixtla_models()
        
        if self.local_model not in models_dict:
            raise ValueError(f"Unknown local_model '{self.local_model}'. Available: {list(models_dict.keys())}")
        
        ModelClass = models_dict[self.local_model]
        
        # Prepare data for StatsForecast format
        df = hist.rename(columns={'day':'ds','actual_sale':'y'}).copy()
        df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        df['unique_id'] = 'item'
        df['ds'] = pd.to_datetime(df['ds'])
        df = _normalize_monthly_ds_to_period_anchor(df, str(self.freq))

        # Initialize model with appropriate parameters
        seasonal_models = ['auto_arima', 'auto_ets', 'seasonal_naive', 'theta', 'optimized_theta', 'auto_ces']
        
        if self.local_model in seasonal_models:
            model_instance = ModelClass(season_length=self.season_length)
        else:
            model_instance = ModelClass()
        
        # Initialize StatsForecast
        sf = StatsForecast(
            models=[model_instance],
            freq=self.freq,
            n_jobs=1
        )
        
        # Fit and forecast. StatsForecast 2.x requires prediction_intervals when level is passed.
        sf.fit(df)
        try:
            from statsforecast.utils import ConformalIntervals
            n_windows = 2
            min_len = df.groupby('unique_id').size().min() if 'unique_id' in df.columns else len(df)
            if min_len > n_windows * h:
                intervals = ConformalIntervals(h=h, n_windows=n_windows)
                fcst = sf.forecast(h=h, df=df, level=[70, 90, 95], prediction_intervals=intervals)
            else:
                fcst = sf.forecast(h=h, df=df)
        except Exception:
            fcst = sf.forecast(h=h, df=df)
        
        # Build output DataFrame
        last_ds = pd.to_datetime(df['ds'].iloc[-1])
        future_ds = pd.date_range(start=last_ds, periods=h+1, freq=self.freq)[1:]  # Skip first (it's last_ds)
        
        # Get point forecast column (exclude interval columns like *-hi-XX, *-lo-XX)
        forecast_cols = [
            col for col in fcst.columns
            if col not in ['unique_id', 'ds'] and ('-hi-' not in col) and ('-lo-' not in col)
        ]
        if not forecast_cols:
            raise ValueError(f"No forecast column found in output for {self.local_model}")
        forecast_col = forecast_cols[0]
        yhat = fcst[forecast_col].to_numpy(dtype=float)

        def _upper_for(level: int) -> Optional[np.ndarray]:
            col = f"{forecast_col}-hi-{level}"
            if col in fcst.columns:
                return fcst[col].to_numpy(dtype=float)
            cands = [c for c in fcst.columns if c.endswith(f"-hi-{level}")]
            if cands:
                return fcst[cands[0]].to_numpy(dtype=float)
            return None

        # Prefer StatsForecast intervals when available; otherwise derive from a conservative fallback.
        u95 = _upper_for(95)
        if u95 is None:
            u95 = np.maximum(yhat * 1.5, yhat + 1.0)
        upper_95 = np.maximum(np.maximum(u95, yhat), 0.0)

        u90 = _upper_for(90)
        u70 = _upper_for(70)
        gap = np.maximum(0.0, upper_95 - yhat)
        upper_90 = np.maximum(np.maximum(u90 if u90 is not None else (yhat + 0.8 * gap), yhat), 0.0)
        upper_70 = np.maximum(np.maximum(u70 if u70 is not None else (yhat + 0.4 * gap), yhat), 0.0)
        # Enforce nesting: yhat <= upper_70 <= upper_90 <= upper_95
        upper_90 = np.minimum(upper_90, upper_95)
        upper_70 = np.minimum(upper_70, upper_90)
        
        out = pd.DataFrame({
            'ds': future_ds,
            'yhat': yhat,
            'upper_70': upper_70,
            'upper_90': upper_90,
            'upper_95': upper_95
        })
        # Replace any non-finite values with safe fallbacks based on yhat.
        for col in ['yhat', 'upper_70', 'upper_90', 'upper_95']:
            vals = out[col].to_numpy(dtype=float)
            mask = ~np.isfinite(vals)
            if np.any(mask):
                # For yhat, fall back to 0; for quantiles, fall back to yhat.
                if col == 'yhat':
                    vals[mask] = 0.0
                else:
                    vals[mask] = np.maximum(out['yhat'].to_numpy(dtype=float)[mask], 0.0)
                out[col] = vals
        return out

    def _to_statsforecast_df(self, hist: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
        """Normalize input history to StatsForecast format: ['unique_id','ds','y']."""
        if {'unique_id', 'ds', 'y'}.issubset(hist.columns):
            df = hist.loc[:, ['unique_id', 'ds', 'y']].copy()
            id_map = {str(uid): uid for uid in df['unique_id'].unique()}
            df['unique_id'] = df['unique_id'].astype(str)
        elif {'item_id', 'day', 'actual_sale'}.issubset(hist.columns):
            id_map = {str(item_id): item_id for item_id in hist['item_id'].unique()}
            df = hist.rename(columns={'item_id': 'unique_id', 'day': 'ds', 'actual_sale': 'y'}).loc[:, ['unique_id', 'ds', 'y']].copy()
            df['unique_id'] = df['unique_id'].astype(str)
        else:
            raise ValueError("hist must have columns ['item_id','day','actual_sale'] or ['unique_id','ds','y']")

        df['ds'] = pd.to_datetime(df['ds'])
        df = _normalize_monthly_ds_to_period_anchor(df, str(self.freq))
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df, id_map

    def auto_model_forecast_panel(
        self,
        hist: pd.DataFrame,
        h: int,
        metric: str = 'robust',
        cv_h: Optional[int] = None,
        n_windows: int = 2,
        lookback_days: Optional[int] = None,
        lookback_periods: Optional[int] = None,
        bias_threshold_pct: float = float('inf'),
    ) -> pd.DataFrame:
        """Select best StatsForecast model per series and forecast.

        Returns DataFrame with columns: ['unique_id', 'ds', 'yhat', 'model_used', 'upper_70', 'upper_90', 'upper_95'].
        """
        from statsforecast import StatsForecast
        from utilsforecast.evaluation import evaluate

        if self.freq is None:
            raise ValueError("freq is not set")
        if h <= 0:
            raise ValueError('h must be > 0')

        df, _ = self._to_statsforecast_df(hist)
        if df.empty:
            raise ValueError('Empty history')

        if lookback_periods is not None and int(lookback_periods) > 0:
            # Frequency-agnostic: keep only the last N observations per series.
            df = (
                df.sort_values(['unique_id', 'ds'])
                .groupby('unique_id', as_index=False, sort=False)
                .tail(int(lookback_periods))
            )
        elif lookback_days is not None and int(lookback_days) > 0:
            # Time-based: keep only observations within last N days per series.
            max_ds = df.groupby('unique_id', as_index=False)['ds'].transform('max')
            cutoff = max_ds - pd.Timedelta(days=int(lookback_days))
            df = df.loc[df['ds'] >= cutoff]

        if df.empty:
            raise ValueError('Empty history after lookback filter')

        # Ensure each series has a complete time index for the requested frequency.
        # This dramatically reduces model failures in StatsForecast CV and prevents
        # the "no_cv_scores -> default Naive" outcome.
        df = _regularize_panel_time_index(df, freq=str(self.freq))
        if df.empty:
            raise ValueError('Empty history after regularization')

        metric_name, metric_fn = _metric_func_from_name(metric)

        is_monthly = canonical_forecaster_freq(str(self.freq)) == 'MS'
        # For monthly series we always assume yearly seasonality (12) for model configuration.
        season_for_models = 12 if is_monthly else int(self.season_length)

        if cv_h is not None:
            cv_h_eff = int(cv_h)
        elif is_monthly:
            cv_h_eff = int(min(h, 6))
        else:
            cv_h_eff = int(min(h, max(1, season_for_models)))
        cv_h_eff = max(1, cv_h_eff)

        debug = bool(os.getenv('AUTO_MODEL_DEBUG'))
        debug_reason: dict[str, dict[str, object]] = {}

        # Heuristic tuning knobs (kept internal to avoid API churn)
        min_arima_len = 36 if is_monthly else max(50, 3 * max(1, int(self.season_length)))

        counts = df.groupby('unique_id', as_index=True).size()
        min_len = (cv_h_eff * max(1, int(n_windows))) + 2

        # Bucket series to choose smaller candidate sets (speed) and avoid
        # obviously-wrong models (accuracy).
        buckets: dict[str, list[str]] = {}
        best_by_uid: dict[str, str] = {uid: 'Naive' for uid in counts.index.tolist()}
        # Track uids that look like "event-seasonal" (most demand in the same month each year).
        event_seasonal_uid: dict[str, bool] = {}
        # Track uids with a strong monotone trend, where naive forecasts are often too flat.
        strong_trend_uid: dict[str, bool] = {}
        stable_tail_uid: dict[str, bool] = {}
        stable_tail_mean_uid: dict[str, float] = {}
        bucket_by_uid: dict[str, str] = {}
        exclude_seasonal_naive_uid: dict[str, bool] = {}
        dead_sku_uid: set[str] = set()

        def _looks_event_seasonal(g: pd.DataFrame, season_length: int) -> bool:
            """Heuristic: demand volume mostly concentrated in 1-2 calendar months each year.

            Uses demand *volume* share (sum of y per month / total y), not occurrence
            count, so a December spike of 2500 correctly dominates over small Oct/Nov sales.
            """
            if season_length < 2:
                return False
            if len(g) < 2 * season_length:
                return False
            nz = g[g['y'] > 0.0].copy()
            if nz.empty:
                return False
            nz['month'] = pd.to_datetime(nz['ds']).dt.month
            vol_by_month = nz.groupby('month')['y'].sum()
            total = float(vol_by_month.sum())
            if total <= 0.0:
                return False
            share = (vol_by_month / total).sort_values(ascending=False)
            top1 = float(share.iloc[0])
            top2 = float(share.iloc[:2].sum()) if len(share) >= 2 else top1
            return top1 >= 0.50 or top2 >= 0.65

        for uid, n_obs in counts.items():
            g_uid = df.loc[df['unique_id'] == uid, ['unique_id', 'ds', 'y']].copy()
            y_full = g_uid['y'].to_numpy(dtype=float)

            # Long trailing zero run: avoid ETS / seasonal methods that keep a small
            # positive level; Naive repeats last value (0 on regularized months).
            if _auto_model_force_naive_trailing_zeros(y_full, min_months=24):
                best_by_uid[uid] = 'Naive'
                exclude_seasonal_naive_uid[uid] = True
                dead_sku_uid.add(uid)
                if debug:
                    debug_reason[uid] = {
                        'reason': 'long_trailing_zero_run',
                        'picked': 'Naive',
                        'n_obs': int(n_obs),
                    }
                continue

            if _auto_model_force_naive_long_silence_after_last_sale(
                y_full, g_uid['ds'], min_silent_months=18
            ):
                best_by_uid[uid] = 'Naive'
                exclude_seasonal_naive_uid[uid] = True
                dead_sku_uid.add(uid)
                if debug:
                    debug_reason[uid] = {
                        'reason': 'long_silence_after_last_sale',
                        'picked': 'Naive',
                        'n_obs': int(n_obs),
                    }
                continue

            # Event-seasonal must be known before SeasonalNaive exclusions: long zero
            # tails are normal off-season for those SKUs, not discontinuation.
            event_seasonal_uid[uid] = _looks_event_seasonal(g_uid, int(season_for_models))

            # Trim leading zeros for profiling/bucketing so launch-phase zeros don't
            # dominate n, zero_frac, or trend detection. Keep zeros after the first
            # positive month (true stockouts/off-season).
            mask_pos = y_full > 0.0
            if mask_pos.any():
                first_pos = int(np.argmax(mask_pos))
                y_eff = y_full[first_pos:]
            else:
                y_eff = y_full

            # Stable-tail on y_eff (not y_full) so padded trailing zeros from
            # panel extension don't mask a real stable demand level.
            stable_recent, stable_tail_mean = _recent_tail_stable_level(y_eff)
            stable_tail_uid[uid] = bool(stable_recent)
            stable_tail_mean_uid[uid] = float(stable_tail_mean)

            ex_sn = _auto_model_exclude_seasonal_naive(
                y_full,
                season_length=int(season_for_models),
                event_seasonal=bool(event_seasonal_uid.get(uid, False)),
                stable_recent_level=bool(stable_recent),
            )
            if (
                not bool(event_seasonal_uid.get(uid, False))
                and _seasonal_naive_lag_regime_mismatch(y_full, int(season_for_models))
            ):
                ex_sn = True
            exclude_seasonal_naive_uid[uid] = ex_sn
            prof = _series_profile(y_eff)
            bucket = _bucket_series(prof, season_length=season_for_models, min_arima_len=min_arima_len)
            n_eff = int(prof.get('n', float(len(y_eff))))  # effective history length
            # Structural off-season zeros -> intermittent bucket + Croston/ADIDA CV; force
            # seasonal pool for single-month event peaks or strong year-over-year shape.
            if bucket == 'intermittent' and int(n_eff) >= 2 * int(season_for_models):
                if bool(event_seasonal_uid.get(uid, False)) or _strong_yearly_seasonality(
                    y_eff, int(season_for_models)
                ):
                    bucket = 'seasonal'
            # Strong trend: high absolute correlation with time index.
            try:
                trend_corr = float(prof.get('trend_corr', 0.0))  # type: ignore[arg-type]
            except Exception:
                trend_corr = 0.0
            strong_trend_uid[uid] = bool(abs(trend_corr) >= 0.7)

            # Heuristic selections for very short series (skip CV entirely)
            if bucket == 'short' or int(n_eff) < min_len:
                # Intermittent series: if we skip CV due to length, avoid collapsing to Naive
                # (Naive tends to win trivial off-season windows).
                if bucket == 'intermittent':
                    best_by_uid[uid] = 'ADIDA' if int(n_obs) >= 3 else 'Naive'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': best_by_uid[uid],
                            'bucket': bucket,
                            'n_obs': int(n_obs),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                    continue

                # Monthly fallbacks:
                # - Intermittent already handled above.
                # - 12-23 months: use a short-horizon average/smoothing instead of copying
                #   last year. Favor SeasonalWindowAverage for \"swinging\" items and a
                #   simple seasonal ETS for clearly trending ones.
                # - Otherwise: simple moving average over all available months -> HistoricAverage.
                if is_monthly and 12 <= int(n_eff) <= 23:
                    # Reuse trend_corr from prof; strong trend -> AutoETS, otherwise SWA.
                    try:
                        trend_corr = float(prof.get('trend_corr', 0.0))  # type: ignore[arg-type]
                    except Exception:
                        trend_corr = 0.0
                    if abs(trend_corr) >= 0.5:
                        best_by_uid[uid] = 'AutoETS'
                    else:
                        best_by_uid[uid] = 'SeasonalWindowAverage'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': best_by_uid[uid],
                            'bucket': bucket,
                            'n_obs': int(n_eff),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                    continue

                if is_monthly and int(n_eff) < 12:
                    best_by_uid[uid] = 'HistoricAverage'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': 'HistoricAverage',
                            'bucket': bucket,
                            'n_obs': int(n_eff),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                    continue

                if int(season_for_models) >= 2 and int(n_obs) >= int(season_for_models) + 1:
                    if exclude_seasonal_naive_uid.get(uid, False):
                        best_by_uid[uid] = 'HistoricAverage'
                        pick_reason = 'HistoricAverage'
                    else:
                        best_by_uid[uid] = 'SeasonalNaive'
                        pick_reason = 'SeasonalNaive'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': pick_reason,
                            'bucket': bucket,
                            'n_obs': int(n_obs),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                else:
                    best_by_uid[uid] = 'HistoricAverage' if is_monthly and int(n_obs) >= 2 else 'Naive'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': best_by_uid[uid],
                            'bucket': bucket,
                            'n_obs': int(n_obs),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                continue

            buckets.setdefault(bucket, []).append(uid)
            bucket_by_uid[uid] = bucket

        def _candidate_keys_for_bucket(
            bucket: str,
            n_obs: int,
            any_event_seasonal: bool,
            any_strong_trend: bool,
        ) -> list[str]:
            # Keep candidate sets small for speed.
            if bucket == 'intermittent':
                # Intermittent series: prefer intermittent-demand models (+ seasonal naive).
                # NOTE: We intentionally exclude plain Naive here because it tends to win
                # error metrics on mostly-zero holdout windows by predicting 0, which is
                # often not the desired behaviour for replenishment planning.
                # For monthly series with at least ~2 seasons of history, always include
                # at least one seasonal model in the candidate set so strong seasonal
                # patterns (e.g., Christmas-only items) can be captured.
                base_keys = ['croston_optimized', 'adida', 'seasonal_naive']
                if int(season_for_models) >= 2 and int(n_obs) >= 2 * int(season_for_models):
                    if any_event_seasonal:
                        # For clearly event-seasonal series, bias towards seasonal models
                        # but keep one intermittent model as backup.
                        return [
                            'seasonal_naive',
                            'seasonal_window_average',
                            'historic_average',
                            'auto_ets',
                            'croston_optimized',
                        ]
                    # Otherwise, just add a single seasonal model to the pool.
                    base_keys.append('auto_ets')
                    base_keys.extend(['historic_average', 'seasonal_window_average'])
                return base_keys
            if bucket == 'seasonal':
                # Level baselines compete when demand has stepped to a new stable run rate.
                return [
                    'seasonal_naive',
                    'seasonal_window_average',
                    'historic_average',
                    'auto_ets',
                    'theta',
                    'optimized_theta',
                ]
            if bucket == 'trend':
                keys = ['historic_average', 'seasonal_window_average', 'auto_ets', 'theta', 'optimized_theta', 'naive']
                # If we have at least one full season, allow SeasonalNaive even if the
                # seasonal detector didn't put the series into the seasonal bucket yet
                # (common for monthly series with ~13-23 months of history).
                if int(self.season_length) >= 2 and int(n_obs) >= int(self.season_length) + 1:
                    keys.insert(0, 'seasonal_naive')
                # For strongly trending series, bias away from plain Naive (which tends to
                # extrapolate the last value and under-react to clear trends).
                if any_strong_trend:
                    keys = [k for k in keys if k != 'naive']
                if n_obs >= min_arima_len:
                    keys.insert(0, 'auto_arima')
                return keys
            # smooth: weak seasonality; keep level baselines so stable run rates can win CV.
            keys = ['historic_average', 'seasonal_window_average', 'auto_ets', 'theta', 'optimized_theta', 'naive']
            if n_obs >= min_arima_len:
                keys.insert(0, 'auto_arima')
            return keys

        # Per-uid CV scores (initialized before bucket loop so reranking always has defined maps).
        rmse_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        mae_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        wape_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        bias_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        metric_scores: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}

        # Score each bucket with per-model CV (robust to individual model failures).
        for bucket, uids in buckets.items():
            df_bucket = df[df['unique_id'].isin(uids)]
            if df_bucket.empty:
                continue

            # Seasonal monthly items need a full-year CV horizon so SeasonalNaive
            # gets evaluated across a complete cycle (peak + off-season).
            if is_monthly and bucket == 'seasonal' and cv_h is None:
                bucket_cv_h = int(min(h, int(season_for_models)))
            else:
                bucket_cv_h = cv_h_eff
            bucket_cv_h = max(1, bucket_cv_h)

            # Determine max n_obs in bucket to decide if AutoARIMA is allowed.
            max_n = int(counts.loc[uids].max())
            any_event = any(bool(event_seasonal_uid.get(uid, False)) for uid in uids)
            any_strong_trend = any(bool(strong_trend_uid.get(uid, False)) for uid in uids)
            model_specs = _build_model_factories_for_keys(
                _candidate_keys_for_bucket(bucket, max_n, any_event, any_strong_trend),
                season_length=int(season_for_models),
            )
            if not model_specs:
                continue

            for model_name, factory in model_specs:
                try:
                    sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
                    cv = sf_one.cross_validation(
                        df=df_bucket,
                        h=bucket_cv_h,
                        step_size=bucket_cv_h,
                        n_windows=max(1, int(n_windows)),
                    )

                    if metric_name == 'robust':
                        from utilsforecast.losses import rmse, mae
                        scores = evaluate(cv, metrics=[rmse, mae])
                        # Mean across cutoffs
                        rmse_mean = scores[scores['metric'] == 'rmse'].groupby('unique_id', as_index=True)[model_name].mean()
                        mae_mean = scores[scores['metric'] == 'mae'].groupby('unique_id', as_index=True)[model_name].mean()
                        for uid in uids:
                            v1 = float(rmse_mean.get(uid, np.inf))
                            v2 = float(mae_mean.get(uid, np.inf))
                            rmse_scores_map[uid][model_name] = v1
                            mae_scores_map[uid][model_name] = v2
                    elif metric_name in ('wape', 'wape_bias'):
                        # Compute per-series WAPE and bias% directly from CV paths.
                        if model_name not in cv.columns:
                            continue
                        for uid, g in cv.groupby('unique_id', sort=False):
                            y = g['y'].to_numpy(dtype=float)
                            yhat = g[model_name].to_numpy(dtype=float)
                            wape_v, bias_v = _safe_wape_and_bias(y, yhat)
                            wape_scores_map[str(uid)][model_name] = wape_v
                            bias_scores_map[str(uid)][model_name] = bias_v
                    else:
                        scores = evaluate(cv, metrics=[metric_fn])
                        m = scores[scores['metric'] == metric_name].groupby('unique_id', as_index=True)[model_name].mean()
                        for uid in uids:
                            metric_scores[uid][model_name] = float(m.get(uid, np.inf))

                except Exception:
                    # If a model can't be fit for this bucket, skip it.
                    continue

            for uid in uids:
                if metric_name == 'robust':
                    if not rmse_scores_map[uid] or not mae_scores_map[uid]:
                        if debug and uid not in debug_reason and best_by_uid.get(uid) == 'Naive':
                            debug_reason[uid] = {
                                'reason': 'no_cv_scores',
                                'picked': 'Naive',
                                'bucket': bucket,
                            }
                        continue
                    models = sorted(set(rmse_scores_map[uid].keys()) | set(mae_scores_map[uid].keys()))
                    if not models:
                        continue
                    rmse_vals = pd.Series({m: rmse_scores_map[uid].get(m, np.inf) for m in models})
                    mae_vals = pd.Series({m: mae_scores_map[uid].get(m, np.inf) for m in models})
                    total_rank = rmse_vals.rank(method='min').add(mae_vals.rank(method='min'), fill_value=0)
                    picked = str(total_rank.idxmin())
                    if np.isfinite(rmse_scores_map[uid].get(picked, np.inf)):
                        best_by_uid[uid] = picked
                elif metric_name in ('wape', 'wape_bias'):
                    if not wape_scores_map[uid]:
                        continue
                    models = list(wape_scores_map[uid].keys())
                    wape_vals = pd.Series({m: wape_scores_map[uid].get(m, np.inf) for m in models})
                    bias_vals = pd.Series({m: float(bias_scores_map[uid].get(m, np.inf)) for m in models})

                    if metric_name == 'wape_bias':
                        picked = _pick_model_wape_bias_penalty(
                            wape_vals,
                            bias_vals,
                            rel_eps=0.02,
                            abs_eps=0.005,
                            bias_ok_pct=10.0,
                            bias_scale_pct=20.0,
                            weight=0.25,
                            prefer_seasonal_naive=bool(event_seasonal_uid.get(uid, False)),
                        )
                    else:
                        # Plain 'wape': pick best WAPE; deterministic tie-break by |bias|.
                        picked = (
                            pd.DataFrame({'wape': wape_vals, 'abs_bias_pct': bias_vals.abs()})
                            .sort_values(['wape', 'abs_bias_pct'], ascending=True)
                            .index[0]
                        )
                    if np.isfinite(float(wape_scores_map[uid].get(str(picked), np.inf))):
                        best_by_uid[uid] = str(picked)
                else:
                    if not metric_scores[uid]:
                        if debug and uid not in debug_reason and best_by_uid.get(uid) == 'Naive':
                            debug_reason[uid] = {
                                'reason': 'no_cv_scores',
                                'picked': 'Naive',
                                'bucket': bucket,
                            }
                        continue
                    best_by_uid[uid] = str(min(metric_scores[uid].items(), key=lambda kv: kv[1])[0])

        # SeasonalNaive repeats y[t-season]; drop it when recent history shows a dead tail
        # or YoY collapse (handled per-uid; CV runs on full bucket candidate sets).
        for uid in list(best_by_uid.keys()):
            if not exclude_seasonal_naive_uid.get(uid, False):
                continue
            _rerank_pick_excluding_seasonal_naive(
                uid=str(uid),
                metric_name=metric_name,
                best_by_uid=best_by_uid,
                rmse_scores_map=rmse_scores_map,
                mae_scores_map=mae_scores_map,
                wape_scores_map=wape_scores_map,
                bias_scores_map=bias_scores_map,
                metric_scores=metric_scores,
            )

        _auto_model_maybe_prefer_level_under_stable_tail(
            best_by_uid=best_by_uid,
            stable_tail_uid=stable_tail_uid,
            bucket_by_uid=bucket_by_uid,
            wape_scores_map=wape_scores_map,
            rmse_scores_map=rmse_scores_map,
            mae_scores_map=mae_scores_map,
            metric_name=metric_name,
        )

        if debug:
            # Summarise why we ended up with Naive defaults.
            reasons = {}
            samples: dict[str, list[str]] = {}
            for uid, picked in best_by_uid.items():
                if picked != 'Naive':
                    continue
                reason = str(debug_reason.get(uid, {}).get('reason', 'naive_won_or_unknown'))
                reasons[reason] = reasons.get(reason, 0) + 1
                if len(samples.get(reason, [])) < 5:
                    samples.setdefault(reason, []).append(str(uid))
            if reasons:
                print('AUTO_MODEL_DEBUG naive reasons:', reasons)
                print('AUTO_MODEL_DEBUG naive sample_uids:', samples)

        # Forecast per chosen model in batches.
        by_model: dict[str, list[str]] = {}
        for uid, model_name in best_by_uid.items():
            by_model.setdefault(model_name, []).append(uid)

        parts: list[pd.DataFrame] = []
        # Use the full set of factories so we can forecast whatever was selected
        # in any bucket.
        all_model_specs = _build_candidate_model_factories(int(season_for_models))
        for model_name, factory in all_model_specs:
            uids = by_model.get(model_name)
            if not uids:
                continue
            sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
            subset = df[df['unique_id'].isin(uids)]
            try:
                from statsforecast.utils import ConformalIntervals
                min_len = subset.groupby('unique_id').size().min()
                n_windows = 2
                if min_len > n_windows * h:
                    intervals = ConformalIntervals(h=h, n_windows=n_windows)
                    fcst = sf_one.forecast(df=subset, h=h, level=[70, 90, 95], prediction_intervals=intervals)
                else:
                    fcst = sf_one.forecast(df=subset, h=h)
            except Exception:
                fcst = sf_one.forecast(df=subset, h=h)
            if model_name not in fcst.columns:
                raise RuntimeError(f"Expected forecast column '{model_name}' not found")
            part = fcst.loc[:, ['unique_id', 'ds']].copy()
            yhat = fcst[model_name].to_numpy(dtype=float)
            part['yhat'] = yhat
            part['model_used'] = model_name
            def _upper_for(level: int) -> Optional[np.ndarray]:
                col = f"{model_name}-hi-{level}"
                if col in fcst.columns:
                    return fcst[col].to_numpy(dtype=float)
                cands = [c for c in fcst.columns if c.endswith(f"-hi-{level}")]
                if cands:
                    return fcst[cands[0]].to_numpy(dtype=float)
                return None

            u95 = _upper_for(95)
            if u95 is None:
                u95 = np.maximum(yhat * 1.5, yhat + 1.0)
            upper_95 = np.maximum(np.maximum(u95, yhat), 0.0)

            u90 = _upper_for(90)
            u70 = _upper_for(70)
            gap = np.maximum(0.0, upper_95 - yhat)
            upper_90 = np.maximum(np.maximum(u90 if u90 is not None else (yhat + 0.8 * gap), yhat), 0.0)
            upper_70 = np.maximum(np.maximum(u70 if u70 is not None else (yhat + 0.4 * gap), yhat), 0.0)
            upper_90 = np.minimum(upper_90, upper_95)
            upper_70 = np.minimum(upper_70, upper_90)

            part['upper_70'] = upper_70
            part['upper_90'] = upper_90
            part['upper_95'] = upper_95
            parts.append(part)

        if not parts:
            raise RuntimeError('Failed to generate forecasts for any series')

        out = pd.concat(parts, ignore_index=True)
        out['yhat'] = out['yhat'].clip(lower=0.0)

        # Dead SKUs: force forecast to zero regardless of what Naive produced
        # (without panel extension, Naive repeats the last positive sale).
        if dead_sku_uid:
            dead_mask = out['unique_id'].isin(dead_sku_uid)
            for col in ['yhat', 'upper_70', 'upper_90', 'upper_95']:
                if col in out.columns:
                    out.loc[dead_mask, col] = 0.0

        # Replace non-finite quantiles with safe fallbacks and enforce nesting.
        for col in ['upper_70', 'upper_90', 'upper_95']:
            vals = out[col].to_numpy(dtype=float)
            mask = ~np.isfinite(vals)
            if np.any(mask):
                vals[mask] = np.maximum(out['yhat'].to_numpy(dtype=float)[mask], 0.0)
                out[col] = vals
        out['upper_95'] = out['upper_95'].clip(lower=out['yhat'])
        out['upper_90'] = out['upper_90'].clip(lower=out['yhat'], upper=out['upper_95'])
        out['upper_70'] = out['upper_70'].clip(lower=out['yhat'], upper=out['upper_90'])

        # Post-forecast sanity floor: only for UIDs whose recent tail was genuinely
        # stable (low CV). Seasonal items have volatile tails by design — their
        # median forecast *should* be low (off-season zeros), so the floor must
        # never fire on them.
        floor_ratio = 0.15
        floor_min_tail = 5.0
        for uid in out['unique_id'].unique():
            if not stable_tail_uid.get(uid, False):
                continue
            tm = float(stable_tail_mean_uid.get(uid, 0.0))
            if tm < floor_min_tail:
                continue
            mask_uid = out['unique_id'] == uid
            med = float(out.loc[mask_uid, 'yhat'].median())
            if med < floor_ratio * tm:
                out.loc[mask_uid, 'yhat'] = tm
                out.loc[mask_uid, 'model_used'] = out.loc[mask_uid, 'model_used'].astype(str) + ':floor'
                gap = tm * 0.5
                out.loc[mask_uid, 'upper_70'] = tm + 0.4 * gap
                out.loc[mask_uid, 'upper_90'] = tm + 0.8 * gap
                out.loc[mask_uid, 'upper_95'] = tm + gap

        return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    def auto_model_forecast_single(
        self,
        item_hist: pd.DataFrame,
        h: int,
        metric: str = 'robust',
        cv_h: Optional[int] = None,
        n_windows: int = 2,
        lookback_days: Optional[int] = None,
        lookback_periods: Optional[int] = None,
    ) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
        """Auto-select a model for a single series; returns (forecast_path, model_used, upper_70, upper_90, upper_95)."""
        df = item_hist.rename(columns={'day': 'ds', 'actual_sale': 'y'}).loc[:, ['ds', 'y']].copy()
        df['unique_id'] = 'item'
        panel = self.auto_model_forecast_panel(
            df,
            h=h,
            metric=metric,
            cv_h=cv_h,
            n_windows=n_windows,
            lookback_days=lookback_days,
            lookback_periods=lookback_periods,
        )
        model_used = str(panel['model_used'].iloc[0])
        path = panel['yhat'].to_numpy(dtype=float)
        upper_70 = panel['upper_70'].to_numpy(dtype=float)
        upper_90 = panel['upper_90'].to_numpy(dtype=float)
        upper_95 = panel['upper_95'].to_numpy(dtype=float)
        return path, model_used, upper_70, upper_90, upper_95

    def forecast_panel_with_selected_models(
        self,
        hist: pd.DataFrame,
        h: int,
        model_by_uid: dict[str, str],
    ) -> pd.DataFrame:
        """Forecast a panel using a pre-selected model per series.

        This bypasses CV and is intended for cases where model selection was done
        on a transformed representation (e.g., monthly aggregates) but forecasting
        should be performed at the original frequency.

        Returns DataFrame with columns: ['unique_id', 'ds', 'yhat', 'model_used', 'upper_70', 'upper_90', 'upper_95'].
        """
        from statsforecast import StatsForecast

        if h <= 0:
            raise ValueError('h must be > 0')

        df, _ = self._to_statsforecast_df(hist)
        if df.empty:
            raise ValueError('Empty history')
        df = _regularize_panel_time_index(df, freq=str(self.freq))
        if df.empty:
            raise ValueError('Empty history after regularization')

        # Build a mapping from model class name -> factory.
        factories = {name: factory for name, factory in _build_candidate_model_factories(int(self.season_length))}

        by_model: dict[str, list[str]] = {}
        for uid in df['unique_id'].unique().tolist():
            model_name = model_by_uid.get(str(uid), 'Naive')
            by_model.setdefault(model_name, []).append(str(uid))

        parts: list[pd.DataFrame] = []
        for model_name, uids in by_model.items():
            factory = factories.get(model_name)
            if factory is None:
                # Unknown model name -> safe fallback
                factory = factories.get('Naive')
                model_name = 'Naive'
            sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
            subset = df[df['unique_id'].isin(uids)]
            try:
                from statsforecast.utils import ConformalIntervals
                min_len = subset.groupby('unique_id').size().min()
                n_windows = 2
                if min_len > n_windows * h:
                    intervals = ConformalIntervals(h=h, n_windows=n_windows)
                    fcst = sf_one.forecast(df=subset, h=h, level=[70, 90, 95], prediction_intervals=intervals)
                else:
                    fcst = sf_one.forecast(df=subset, h=h)
            except Exception:
                fcst = sf_one.forecast(df=subset, h=h)
            if model_name not in fcst.columns:
                raise RuntimeError(f"Expected forecast column '{model_name}' not found")
            part = fcst.loc[:, ['unique_id', 'ds']].copy()
            yhat = fcst[model_name].to_numpy(dtype=float)
            part['yhat'] = yhat
            part['model_used'] = model_name
            def _upper_for(level: int) -> Optional[np.ndarray]:
                col = f"{model_name}-hi-{level}"
                if col in fcst.columns:
                    return fcst[col].to_numpy(dtype=float)
                cands = [c for c in fcst.columns if c.endswith(f"-hi-{level}")]
                if cands:
                    return fcst[cands[0]].to_numpy(dtype=float)
                return None

            u95 = _upper_for(95)
            if u95 is None:
                u95 = np.maximum(yhat * 1.5, yhat + 1.0)
            upper_95 = np.maximum(np.maximum(u95, yhat), 0.0)

            u90 = _upper_for(90)
            u70 = _upper_for(70)
            gap = np.maximum(0.0, upper_95 - yhat)
            upper_90 = np.maximum(np.maximum(u90 if u90 is not None else (yhat + 0.8 * gap), yhat), 0.0)
            upper_70 = np.maximum(np.maximum(u70 if u70 is not None else (yhat + 0.4 * gap), yhat), 0.0)
            upper_90 = np.minimum(upper_90, upper_95)
            upper_70 = np.minimum(upper_70, upper_90)

            part['upper_70'] = upper_70
            part['upper_90'] = upper_90
            part['upper_95'] = upper_95
            parts.append(part)

        if not parts:
            raise RuntimeError('Failed to generate forecasts for any series')

        out = pd.concat(parts, ignore_index=True)
        out['yhat'] = out['yhat'].clip(lower=0.0)
        # Replace non-finite quantiles with safe fallbacks and enforce nesting.
        for col in ['upper_70', 'upper_90', 'upper_95']:
            vals = out[col].to_numpy(dtype=float)
            mask = ~np.isfinite(vals)
            if np.any(mask):
                vals[mask] = np.maximum(out['yhat'].to_numpy(dtype=float)[mask], 0.0)
                out[col] = vals
        out['upper_95'] = out['upper_95'].clip(lower=out['yhat'])
        out['upper_90'] = out['upper_90'].clip(lower=out['yhat'], upper=out['upper_95'])
        out['upper_70'] = out['upper_70'].clip(lower=out['yhat'], upper=out['upper_90'])
        return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # ---------- Public: daily path ----------
    def daily_path(
        self, item_hist: pd.DataFrame, periods: int
    ) -> tuple[np.ndarray, Optional[dict[str, np.ndarray]]]:
        """
        Returns (path, quantiles) where path is a length=periods np.array of forecasts (float),
        aligned to the next period after the last 'day' in item_hist. quantiles is a dict of
        per-period upper quantiles (e.g. upper_70/upper_90/upper_95) when available, or None.
        """
        if self.mode == 'timegpt':
            fcst = self._timegpt_forecast_path(item_hist, periods)
            path = fcst['yhat'].to_numpy(dtype=float)
            return np.maximum(path, 0.0), None
        if self.local_model in ('auto_model', 'automodel'):
            path, _, upper_70, upper_90, upper_95 = self.auto_model_forecast_single(item_hist, h=periods)
            return np.maximum(path, 0.0), {
                'upper_70': np.maximum(upper_70, 0.0),
                'upper_90': np.maximum(upper_90, 0.0),
                'upper_95': np.maximum(upper_95, 0.0),
            }
        fcst = self._local_forecast_path(item_hist, periods)
        path = fcst['yhat'].to_numpy(dtype=float)
        return np.maximum(path, 0.0), {
            'upper_70': fcst['upper_70'].to_numpy(dtype=float),
            'upper_90': fcst['upper_90'].to_numpy(dtype=float),
            'upper_95': fcst['upper_95'].to_numpy(dtype=float),
        }

    # ---------- Public: lead-time totals for service levels ----------
    def leadtime_total_quantile(self,
                                item_hist: pd.DataFrame,
                                L: int,
                                serv_lev: float = 0.95,
                                trials: int = 2000) -> float:
        """
        If quantiles are available from TimeGPT, use them period-wise + MC to approximate
        the lead-time total. Otherwise, bootstrap residuals around point forecast.
        """
        use_quantiles = (self.mode == 'timegpt' and len(self.quantiles) > 0)
        if use_quantiles and (serv_lev in self.quantiles):
            qkey = f"TimeGPT-q-{int(serv_lev*100)}"
            fcst = self._timegpt_forecast_path(item_hist, L)
            if qkey in fcst.columns:
                return float(np.maximum(fcst[qkey].to_numpy(), 0.0).sum())

        periods = L
        if self.mode == 'timegpt' and self.quantiles:
            fcst = self._timegpt_forecast_path(item_hist, periods)
            qs = sorted(set(self.quantiles + [0.5]))
            qcols = [f"TimeGPT-q-{int(q*100)}" for q in qs if f"TimeGPT-q-{int(q*100)}" in fcst.columns]
            if qcols:
                Q = np.array(qs[:len(qcols)])
                grid = fcst[qcols].to_numpy()
                rng = np.random.default_rng()
                totals = np.zeros(trials)
                for t in range(trials):
                    u = rng.random(size=periods)
                    draws = np.zeros(periods)
                    for day in range(periods):
                        draws[day] = np.interp(u[day], Q, grid[day, :])
                    totals[t] = np.maximum(draws, 0.0).sum()
                return float(np.quantile(totals, serv_lev, method='higher'))

        path, _ = self.daily_path(item_hist, periods)
        hist = item_hist['actual_sale'].to_numpy(dtype=float)
        if len(hist) >= self.season_length + 1:
            resids = hist[self.season_length:] - hist[:-self.season_length]
        else:
            resids = hist - np.median(hist)
        resids = resids[np.isfinite(resids)]
        if len(resids) == 0:
            resids = np.array([0.0])

        rng = np.random.default_rng()
        totals = np.zeros(trials)
        for t in range(trials):
            noise = rng.choice(resids, size=periods, replace=True)
            draws = np.maximum(path + noise, 0.0)
            totals[t] = draws.sum()
        return float(np.quantile(totals, serv_lev, method='higher'))
