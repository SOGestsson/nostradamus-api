# classical_forecasts.py (or drop into your existing module)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Any, Callable, Optional

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


# Lag-family models forecast some weighted reuse of historical same-season values
# (SeasonalNaive: y[t-s]; SeasonalWindowAverage: mean of last k same-season values).
# Both share the failure mode where the historical lag values are no longer
# representative of the current level (e.g. post step-down). Treat them as a unit
# in anti-lag guards.
_LAG_FAMILY_MODELS: tuple[str, ...] = ('SeasonalNaive', 'SeasonalWindowAverage')


def _pick_model_wape_bias_penalty(
    wape_vals: pd.Series,
    bias_pct_vals: pd.Series,
    *,
    rel_eps: float = 0.02,
    abs_eps: float = 0.005,
    seasonal_naive_min_wape_advantage: float = 0.08,
    bias_ok_pct: float = 10.0,
    bias_scale_pct: float = 20.0,
    weight: float = 0.25,
    prefer_seasonal_naive: bool = False,
    wape_std_by_model: Optional[pd.Series] = None,
    unstable_std_threshold: float = 0.15,
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

    Lag-family demotion (``_LAG_FAMILY_MODELS``):
      When a lag-family model would win, require it to beat the best non-lag
      alternative by at least ``seasonal_naive_min_wape_advantage`` (in WAPE
      points). When ``wape_std_by_model`` is supplied and the lag pick's
      per-window WAPE std exceeds ``unstable_std_threshold``, the required
      advantage doubles — addressing the "lag won because one CV window
      happened to align" failure mode.
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

    # Bias-penalty pick within the close band.
    pre_pick = (
        pd.DataFrame({'score': score, 'wape': w_close, 'abs_bias': b_close})
        .sort_values(['score', 'wape', 'abs_bias'], ascending=True)
        .index[0]
    )

    # Lag-family demotion (post bias-penalty). Applies to BOTH SeasonalNaive
    # and SeasonalWindowAverage — they share the "stale historical values"
    # failure mode. Skip for event-seasonal items (the lag IS the signal).
    #
    # Compare against the best non-lag alternative across the FULL candidate
    # set (not the close band) so the threshold actually bites: the close band
    # is typically <1pp wide, but the threshold is intended to fire on
    # 6–10pp leads.
    if pre_pick in _LAG_FAMILY_MODELS and not prefer_seasonal_naive:
        fixed_non_lag = [
            'AutoETS', 'Theta', 'OptimizedTheta', 'AutoARIMA',
            'HistoricAverage', 'WindowAverage',
        ]
        # Only longer MA aliases (≥6) count as non-lag alternatives — MA3 is
        # too reactive / noisy to trust as a level anchor demoting SN/SWA.
        ma_aliases = [
            m for m in wape_vals.index
            if (_ma_window_from_alias(str(m)) or 0) >= 6
        ]
        non_lag_full = [m for m in fixed_non_lag if m in wape_vals.index] + ma_aliases
        if non_lag_full:
            best_lag_wape = float(wape_vals.loc[pre_pick])
            best_alt_wape = float(wape_vals.loc[non_lag_full].min())
            advantage_required = float(seasonal_naive_min_wape_advantage)
            # Stability scaling: if the lag pick's per-window WAPE std is high,
            # require a larger advantage. Directly addresses "lag won because
            # one CV window happened to align" — the SN/SWA over-selection
            # failure mode.
            if wape_std_by_model is not None:
                try:
                    lag_std = float(wape_std_by_model.get(pre_pick, 0.0))
                except Exception:
                    lag_std = 0.0
                if np.isfinite(lag_std) and lag_std > float(unstable_std_threshold):
                    advantage_required *= 2.0
            advantage = best_alt_wape - best_lag_wape
            if advantage < advantage_required:
                # Demote: pick the best non-lag alternative by score (bias-aware).
                score_alt_full = pd.Series(
                    {m: 100.0 * float(wape_vals.loc[m]) for m in non_lag_full}
                )
                bias_alt_full = pd.Series(
                    {m: float(bias_pct_vals.loc[m]) for m in non_lag_full}
                ).abs()
                excess_alt = (bias_alt_full - float(bias_ok_pct)).clip(lower=0.0)
                score_alt_full = score_alt_full + (
                    float(weight) * (excess_alt * excess_alt) / denom
                )
                picked_alt = (
                    pd.DataFrame(
                        {
                            'score': score_alt_full,
                            'wape': wape_vals.loc[non_lag_full],
                            'abs_bias': bias_alt_full,
                        }
                    )
                    .sort_values(['score', 'wape', 'abs_bias'], ascending=True)
                    .index[0]
                )
                return str(picked_alt)

    return str(pre_pick)


# Default window sizes for the lag/level model factories. Centralized so CV-time
# minimum-observation gating and forecast-time factories use the same numbers.
SWA_WINDOW_SIZE: int = 3  # SeasonalWindowAverage: average last N same-season values.
WA_WINDOW_SIZE: int = 12  # WindowAverage: trailing-N level (≈ "recent run rate").

# Moving-average (MA) family: a small set of trailing WindowAverage variants with
# distinct aliases so CV can compare them as separate candidates. MA12 is the
# primary "recent level" model on monthly data — on a non-stationary series
# (level shifts, regime changes) it is categorically different from
# HistoricAverage because it ignores history outside the last year. MA6 is a
# shorter-window level proxy; MA3 is reserved for narrow short-history cases
# only. No upper gate on MA12: the earlier rationale (MA12 ≈ HA on stable
# long histories) is wrong precisely on the items where MA12 matters —
# level-shifted series whose old regime dominates HA's average.
MA_ALL_WINDOWS: tuple[int, ...] = (3, 6, 12)


def _ma_windows_for_history(n_obs: int) -> list[int]:
    """Return the MA windows that make sense given a history length.

    Only gate: the lower bound ``1.5 * w``. Below that, the window "average"
    has too little data to stabilise (and typically matches HA within a few
    percent), so the alias adds no information. No upper gate — MA12 ≠ HA on
    non-stationary series regardless of how much history is available.
    """
    return [w for w in MA_ALL_WINDOWS if int(n_obs) >= int(1.5 * w)]


def _build_ma_factories(windows: list[int]) -> list[tuple[str, Callable[[], object]]]:
    """Build WindowAverage factories with distinct aliases (``MA3``, ``MA6``, ``MA12``).

    StatsForecast dedupes by alias, so two WindowAverage factories with
    different ``window_size`` but the same class-name alias will overwrite each
    other in the CV output frame. Using ``MA{w}`` aliases keeps them distinct.
    """
    from statsforecast.models import WindowAverage
    specs: list[tuple[str, Callable[[], object]]] = []
    for w in windows:
        alias = f"MA{int(w)}"
        specs.append(
            (alias, lambda w=int(w), a=alias: WindowAverage(window_size=w, alias=a))
        )
    return specs


def _ma_window_from_alias(alias: str) -> Optional[int]:
    """Parse ``"MA<k>"`` → ``k``. Returns None if the alias is not an MA alias."""
    if not isinstance(alias, str) or not alias.startswith('MA'):
        return None
    tail = alias[2:]
    if not tail.isdigit():
        return None
    return int(tail)


def _min_obs_for_model_cv(
    model_name: str,
    *,
    season_length: int,
    cv_h: int,
    n_windows: int,
) -> int:
    """Conservative minimum obs needed for ``model_name`` to fit *every* CV training window.

    StatsForecast's ``cross_validation`` produces ``n_windows`` cutoffs; the
    earliest training window has length ``n_obs - cv_h * n_windows``. A model
    that can't fit that window will raise and (because we wrap the whole bucket
    in try/except) wipe out scores for the whole bucket. Filtering uids per
    model up front avoids that.
    """
    base = int(cv_h) * max(1, int(n_windows)) + 2
    if model_name == 'SeasonalWindowAverage':
        # Needs ``window_size`` complete seasonal cycles in the smallest train window.
        return base + SWA_WINDOW_SIZE * int(season_length)
    if model_name == 'WindowAverage':
        return base + WA_WINDOW_SIZE
    ma_w = _ma_window_from_alias(model_name)
    if ma_w is not None:
        return base + ma_w
    if model_name == 'SeasonalNaive':
        return base + int(season_length)
    if model_name == 'AutoARIMA':
        return max(base, 36)
    return base


def _conformal_forecast_n_windows(min_len: int, h: int) -> int:
    """Adaptive conformal window count for forecast-time intervals.

    More windows -> more calibration residuals per horizon step -> better
    calibrated intervals. Capped at 5 to bound cost and keep enough training
    data in the earliest window.
    """
    if min_len <= h + 2:
        return 1
    return max(1, min(5, (min_len - h - 2) // max(h, 1)))


def _statsforecast_forecast_with_conformal(sf, df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forecast with conformal intervals when history length allows."""
    from statsforecast.utils import ConformalIntervals

    min_len = int(df.groupby('unique_id').size().min()) if 'unique_id' in df.columns else len(df)
    n_windows = _conformal_forecast_n_windows(min_len, h)
    if min_len > n_windows * h:
        intervals = ConformalIntervals(h=h, n_windows=n_windows)
        return sf.forecast(h=h, df=df, level=[70, 90, 95], prediction_intervals=intervals)
    return sf.forecast(h=h, df=df)


def _upper_column_from_fcst(fcst: pd.DataFrame, model_name: str, level: int) -> Optional[np.ndarray]:
    col = f"{model_name}-hi-{level}"
    if col in fcst.columns:
        return fcst[col].to_numpy(dtype=float)
    cands = [c for c in fcst.columns if c.endswith(f"-hi-{level}")]
    if cands:
        return fcst[cands[0]].to_numpy(dtype=float)
    return None


def _conformal_corrected_quantile(arr: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)*level)-th smallest score.

    Unlike interpolated ``np.quantile``, this carries the standard split-conformal
    coverage guarantee. With small n the required rank can exceed n; we clip to
    the max score (slightly anti-conservative but the best available bound).
    """
    n = len(arr)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * float(level)))
    srt = np.sort(arr)
    if k >= n:
        return float(srt[-1])
    return float(srt[k - 1])


def _excess_quantiles_from_values(
    values: list[float],
    baselines: Optional[list[float]] = None,
) -> dict[str, float]:
    """Absolute (and, when baselines given, relative) conformal excess quantiles.

    ``baselines`` are the predicted/reference levels each excess was measured
    against. Relative quantiles let the band scale with the forecast level so a
    peak-month error does not inflate off-season months by the same absolute
    amount.
    """
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    if baselines is not None and len(baselines) == len(values):
        base = np.asarray(baselines, dtype=float)
        mask = np.isfinite(arr) & np.isfinite(base)
        arr = arr[mask]
        base = base[mask]
    else:
        base = None
        arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    out = {
        'q70': _conformal_corrected_quantile(arr, 0.70),
        'q90': _conformal_corrected_quantile(arr, 0.90),
        'q95': _conformal_corrected_quantile(arr, 0.95),
    }
    if base is not None and len(base) == len(arr):
        # min(abs, rel) of two level-alpha bounds only guarantees 2*alpha-1
        # coverage, so each side is computed at the Bonferroni-adjusted level
        # 1-(1-alpha)/2 to keep the combined bound at the nominal level.
        # Drop near-zero baselines: excess/1.0 from quiet months produces
        # absurd relative quantiles (e.g. 15x) that defeat off-season capping.
        pos_base = base[base > 0]
        min_base = 1.0
        if len(pos_base) > 0:
            min_base = max(1.0, float(np.quantile(pos_base, 0.25)))
        ok = base >= min_base
        if np.any(ok):
            ratios = arr[ok] / np.maximum(base[ok], 1.0)
            for key, level in (('q70', 0.70), ('q90', 0.90), ('q95', 0.95)):
                adj = 1.0 - (1.0 - level) / 2.0
                out[f'{key}_hi'] = _conformal_corrected_quantile(arr, adj)
                out[f'{key}_rel'] = _conformal_corrected_quantile(ratios, adj)
    return out


def _residual_excess_for_level(q: dict[str, float], level_key: str, yhat_val: float) -> Optional[float]:
    """Per-row excess above yhat from residual quantiles.

    Uses min(absolute, relative*yhat), each at the Bonferroni-adjusted level:
    the absolute quantile is dominated by peak-month errors, the relative one
    by low-month errors, so the min keeps the band proportional to the
    forecast level without losing peak coverage or nominal validity.
    """
    abs_v = q.get(level_key)
    if abs_v is None:
        return None
    rel_v = q.get(f'{level_key}_rel')
    if rel_v is None:
        return float(abs_v)
    abs_adj = q.get(f'{level_key}_hi', abs_v)
    return float(min(abs_adj, rel_v * max(yhat_val, 1.0)))


def _merge_excess_quantile_dicts(
    cv_q: dict[str, float],
    hist_q: dict[str, float],
) -> dict[str, float]:
    """Combine CV and historical residual quantile dicts.

    Absolute quantiles take the elementwise max so a historical/CV spike
    signal is not dropped. Relative quantiles take the tighter (min) ratio
    when *both* sides are informative; an uninformative near-zero CV band
    must not zero-out a useful historical relative cap (and a huge CV ratio
    from peak-month errors on a flat mean must not defeat off-season capping).
    """
    if not cv_q:
        return dict(hist_q)
    if not hist_q:
        return dict(cv_q)
    cv95 = float(cv_q.get('q95', 0.0) or 0.0)
    h95 = float(hist_q.get('q95', 0.0) or 0.0)
    out = dict(cv_q)
    for key in ('q70', 'q90', 'q95', 'q70_hi', 'q90_hi', 'q95_hi'):
        hv = hist_q.get(key)
        if hv is None:
            continue
        cvv = out.get(key)
        out[key] = float(hv) if cvv is None else float(max(float(cvv), float(hv)))
    cv_informative = cv95 >= 0.25 * max(h95, 1.0)
    hist_informative = h95 >= 0.25 * max(cv95, 1.0)
    for key in ('q70_rel', 'q90_rel', 'q95_rel'):
        hv = hist_q.get(key)
        cvv = out.get(key)
        if hist_informative and not cv_informative and hv is not None:
            out[key] = float(hv)
        elif cv_informative and not hist_informative and cvv is not None:
            out[key] = float(cvv)
        elif hv is not None and cvv is not None:
            out[key] = float(min(float(cvv), float(hv)))
        elif hv is not None:
            out[key] = float(hv)
        elif cvv is not None:
            out[key] = float(cvv)
    return out


def _is_spiky_intermittent(ypos: np.ndarray, *, min_zero_frac: float = 0.4) -> bool:
    """True for event/spike demand: mostly-idle months between bursts.

    Measured over the *active* span (first to last positive) so that leading or
    trailing pad zeros from panel regularization don't make a continuously
    selling item look intermittent. Continuous items must keep their plain
    residual/conformal bands: same-month reasoning would clamp them to last
    year's value for the month and understate ordinary month-to-month noise.
    """
    if ypos is None or len(ypos) == 0:
        return False
    nz = ypos > 0.0
    if not bool(nz.any()):
        return False
    first = int(np.argmax(nz))
    last = len(nz) - 1 - int(np.argmax(nz[::-1]))
    span = ypos[first:last + 1]
    if len(span) < 6:
        # Too short to characterise; treat as spiky (wider band is the safer error).
        return True
    return float(np.mean(span <= 0.0)) >= float(min_zero_frac)


# Exponential forgetting factor (per year) for event-recurrence evidence. 0.7
# gives an evidence half-life of ~2 years, so a season that was missed recently
# counts for more than one missed five years ago.
_EVENT_DISCOUNT_LAMBDA = 0.7

_UPPER_LEVELS: tuple[float, ...] = (0.70, 0.90, 0.95)


def _event_probability(occurred_newest_first: np.ndarray, lam: float = _EVENT_DISCOUNT_LAMBDA) -> float:
    """P(event recurs) from its per-year occurrence history, newest first.

    Discounted Beta(1,1) posterior mean: recent years carry more weight, and the
    prior keeps the estimate strictly inside (0, 1) — an item still being
    forecast is never certain to stay silent, nor certain to fire again.
    """
    occ = np.asarray(occurred_newest_first, dtype=float)
    if len(occ) == 0:
        return 0.0
    w = float(lam) ** np.arange(len(occ), dtype=float)
    a = 1.0 + float(np.sum(w * occ))
    b = 1.0 + float(np.sum(w * (1.0 - occ)))
    return a / (a + b)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    total = float(w.sum())
    if total <= 0.0 or len(v) == 0:
        return float(v[-1]) if len(v) else 0.0
    cum = (np.cumsum(w) - 0.5 * w) / total
    return float(np.interp(float(q), cum, v))


def _offpeak_positive_pool(
    ypos: np.ndarray,
    months: np.ndarray,
    *,
    frac_of_max: float = 0.2,
) -> np.ndarray:
    """Positive sales from the item's non-event months.

    A month that has never sold still has some chance of selling, but its scale
    is the item's *quiet* trade, not its event. Months whose own maximum is a
    small fraction of the series maximum supply that scale.
    """
    if len(ypos) == 0 or len(months) != len(ypos):
        return np.asarray([], dtype=float)
    series_max = float(ypos.max())
    if series_max <= 0.0:
        return np.asarray([], dtype=float)
    threshold = float(frac_of_max) * series_max
    keep = np.zeros(len(ypos), dtype=bool)
    for m in np.unique(months):
        m_mask = months == m
        if float(ypos[m_mask].max()) < threshold:
            keep |= m_mask
    pool = ypos[keep]
    return pool[pool > 0.0]


def _mixture_limits(
    p_event: float,
    mags: np.ndarray,
    mag_w: np.ndarray,
    ceiling: float,
    yhat_val: float,
) -> dict[int, float]:
    """Upper limits per level for a Bernoulli(p) x magnitude mixture."""
    resolvable = len(mags) / (len(mags) + 1.0)
    q_at_resolvable = _weighted_quantile(mags, mag_w, resolvable)
    limits: dict[int, float] = {}
    for level in _UPPER_LEVELS:
        key = int(round(level * 100))
        if p_event <= (1.0 - level):
            # The event is too unlikely to appear in this level's tail at all.
            limits[key] = float(yhat_val)
            continue
        q_level = 1.0 - (1.0 - level) / p_event
        if q_level <= resolvable:
            limits[key] = _weighted_quantile(mags, mag_w, q_level)
        else:
            # Interpolate across the tail the samples can't resolve so the levels
            # stay ordered instead of all pinning to the ceiling.
            frac = (q_level - resolvable) / max(1e-9, 1.0 - resolvable)
            limits[key] = q_at_resolvable + frac * (ceiling - q_at_resolvable)
    return limits


def _event_month_limits(
    y_hist: Optional[np.ndarray],
    yhat_val: float,
    *,
    ds_hist: Optional[pd.Series] = None,
    forecast_month: Optional[int] = None,
    horizon_pos: int = 1,
    horizon_len: int = 12,
) -> tuple[Optional[dict[int, float]], float]:
    """Per-level upper limits for one month of spiky/event demand.

    Models the month as a mixture: with probability ``p`` the event happens and
    demand is drawn from the magnitudes that month has produced before, and with
    probability ``1-p`` it is zero. The upper limit at level ``alpha`` is then

        0 (i.e. yhat)                       if p <= 1 - alpha
        quantile(magnitudes, 1-(1-alpha)/p) otherwise

    Consequences worth knowing: while ``p`` is above ``1-alpha`` the limit sits
    near the event's own size and barely moves as ``p`` decays, because the
    tail of the mixture *is* the event. What decays smoothly is the point
    forecast (``p`` times the mean magnitude). As an item goes dormant the
    levels switch off in turn — upper_70 once p < 30%, upper_90 below 10%,
    upper_95 below 5% — so the spread between them is the "how much safety does
    this need" signal rather than any single level.

    ``p`` is a discounted Beta posterior over the month's occurrence history and
    magnitudes are recency-weighted, so both a declining event size and a
    lengthening silence pull the limits down.

    Returns ``(limits_by_level, ramp_floor_excess)``. ``limits_by_level`` is
    None when this month has no recurrence history to reason from (continuous
    item, or fewer than two years of the month producing sales); a sparse series
    then gets ``ramp_floor_excess``, a floor growing with the horizon toward its
    recent max, since we know a spike can come but not when.
    """
    if y_hist is None or len(y_hist) == 0:
        return None, 0.0
    y = np.asarray(y_hist, dtype=float)
    if len(y) == 0:
        return None, 0.0
    ypos = np.maximum(np.where(np.isfinite(y), y, 0.0), 0.0)
    if float(ypos.max()) <= 0.0:
        return None, 0.0
    if not _is_spiky_intermittent(ypos):
        return None, 0.0
    n_pos = int(np.sum(ypos > 0.0))

    def _sparse_ramp_floor() -> float:
        if n_pos == 0 or n_pos > 3:
            return 0.0
        gmax = float(ypos[-36:].max()) if len(ypos) > 36 else float(ypos.max())
        if gmax <= 0.0:
            return 0.0
        ramp = min(1.0, max(1, int(horizon_pos)) / max(1, int(horizon_len)))
        return max(0.0, gmax - float(yhat_val)) * ramp

    if (
        forecast_month is None
        or ds_hist is None
        or len(ds_hist) != len(ypos)
    ):
        return None, _sparse_ramp_floor()

    ds = pd.to_datetime(pd.Series(ds_hist).reset_index(drop=True))
    months = ds.dt.month.to_numpy()
    # Same-month values across years, zeros included: a year where the event did
    # not happen is evidence, and dropping it makes a stale peak look current.
    same_mask = months == int(forecast_month)
    if int(np.sum(same_mask)) < 2:
        return None, _sparse_ramp_floor()
    order = np.argsort(ds[same_mask].to_numpy())
    same_vals = ypos[same_mask][order]

    occurred = same_vals > 0.0
    p_event = _event_probability(occurred[::-1])
    mags = same_vals[occurred]

    if len(mags) >= 2:
        # This month recurs: use its own magnitudes, newest weighted heaviest.
        mag_w = (_EVENT_DISCOUNT_LAMBDA ** np.arange(len(mags), dtype=float))[::-1]
        # Headroom beyond the observed max equal to the month's own year-over-year
        # growth (nothing for a flat or declining month).
        growth = 0.0
        if float(mags[-2]) > 0.0:
            growth = min(0.2, max(0.0, (float(mags[-1]) - float(mags[-2])) / float(mags[-2])))
        ceiling = float(mags.max()) * (1.0 + growth)
        return _mixture_limits(p_event, mags, mag_w, ceiling, yhat_val), 0.0

    if n_pos <= 3:
        # Single-spike series: we know a spike can come but not when, so the band
        # grows with the horizon instead of claiming to know the season.
        return None, _sparse_ramp_floor()

    # Month has never (or only once) sold while the item trades in other months.
    # A zero-width interval here would assert certainty of no sales, so borrow the
    # scale of the item's quiet trade and keep this month's own low probability.
    pool = _offpeak_positive_pool(ypos, months)
    if len(mags) == 1:
        pool = np.concatenate([pool, mags])
    if len(pool) == 0:
        return None, 0.0
    return (
        _mixture_limits(p_event, pool, np.ones(len(pool)), float(pool.max()), yhat_val),
        0.0,
    )


def _historical_excess_quantiles(
    y: np.ndarray,
    ds: Optional[pd.Series] = None,
    *,
    lookback: int = 36,
) -> dict[str, float]:
    """Quantiles of positive forecast excess from trailing history."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return {}
    tail = y[-lookback:] if len(y) > lookback else y
    excesses: list[float] = []
    baselines: list[float] = []
    if ds is not None and len(ds) == len(y):
        ds_tail = pd.to_datetime(ds.iloc[-len(tail):])
        months = ds_tail.dt.month.to_numpy()
        for i, val in enumerate(tail):
            if not np.isfinite(val):
                continue
            same_month = tail[months == months[i]]
            baseline = float(np.mean(same_month)) if len(same_month) > 0 else float(np.mean(tail))
            if baseline <= 0:
                baseline = float(np.median(tail[tail > 0])) if np.any(tail > 0) else 1.0
            excesses.append(max(0.0, float(val) - baseline))
            baselines.append(baseline)
    else:
        baseline = float(np.median(tail[tail > 0])) if np.any(tail > 0) else float(np.mean(tail))
        if baseline <= 0:
            baseline = 1.0
        for v in tail:
            if np.isfinite(v):
                excesses.append(max(0.0, float(v) - baseline))
                baselines.append(baseline)
    return _excess_quantiles_from_values(excesses, baselines)


def _last_resort_upper_excess(yhat_val: float, y_hist: Optional[np.ndarray] = None) -> float:
    """Uncalibrated last-resort excess above yhat when no conformal/CV/history quantiles."""
    if y_hist is not None and len(y_hist) >= 6:
        recent = np.asarray(y_hist[-12:], dtype=float)
        recent = recent[np.isfinite(recent)]
        if len(recent) > 0:
            mean_y = float(np.mean(recent))
            if mean_y > 0:
                cv_y = float(np.std(recent) / mean_y)
                return max(yhat_val * 0.5, 2.0 * cv_y * yhat_val, 1.0)
    return max(yhat_val * 0.5, 1.0)


def _attach_upper_quantiles(
    yhat: np.ndarray,
    fcst: pd.DataFrame,
    model_name: str,
    *,
    uid_series: Optional[pd.Series] = None,
    ds_series: Optional[pd.Series] = None,
    cv_excess_by_uid: Optional[dict[str, dict[str, float]]] = None,
    historical_excess_by_uid: Optional[dict[str, dict[str, float]]] = None,
    y_hist_by_uid: Optional[dict[str, np.ndarray]] = None,
    ds_hist_by_uid: Optional[dict[str, pd.Series]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Attach upper_70/90/95 combining conformal columns and residual/history bands.

    StatsForecast conformal intervals here calibrate on very few residuals per
    horizon step (1-5 windows), which empirically under-covers. The CV holdout /
    historical excess band uses a larger sample, so when both are available we
    take the elementwise max (each is a valid upper bound; max preserves the
    stronger coverage).
    """
    yhat = np.asarray(yhat, dtype=float)
    n = len(yhat)
    u70 = _upper_column_from_fcst(fcst, model_name, 70)
    u90 = _upper_column_from_fcst(fcst, model_name, 90)
    u95 = _upper_column_from_fcst(fcst, model_name, 95)

    # Residual/history band per row: fallback when conformal is missing,
    # backstop (max) when it is present but thinly calibrated.
    r95 = np.full(n, np.nan)
    r90 = np.full(n, np.nan)
    r70 = np.full(n, np.nan)
    lim = {70: np.full(n, np.nan), 90: np.full(n, np.nan), 95: np.full(n, np.nan)}
    horizon_pos_by_uid: dict[str, int] = {}
    horizon_len_by_uid: dict[str, int] = {}
    if uid_series is not None:
        for u, cnt in uid_series.astype(str).value_counts().items():
            horizon_len_by_uid[str(u)] = int(cnt)
    for i in range(n):
        uid = str(uid_series.iloc[i]) if uid_series is not None else None
        cv_q: dict[str, float] = {}
        hist_q: dict[str, float] = {}
        if uid and cv_excess_by_uid:
            cv_q = cv_excess_by_uid.get(uid) or {}
        if uid and historical_excess_by_uid:
            hist_q = historical_excess_by_uid.get(uid) or {}
        q = _merge_excess_quantile_dicts(cv_q, hist_q)
        yh = float(yhat[i])
        y_hist = y_hist_by_uid.get(uid) if uid and y_hist_by_uid else None
        ds_hist = ds_hist_by_uid.get(uid) if uid and ds_hist_by_uid else None
        forecast_month: Optional[int] = None
        if ds_series is not None and i < len(ds_series):
            try:
                forecast_month = int(pd.Timestamp(ds_series.iloc[i]).month)
            except Exception:
                forecast_month = None
        h_pos = horizon_pos_by_uid.get(uid or '', 0) + 1
        horizon_pos_by_uid[uid or ''] = h_pos
        h_len = max(horizon_len_by_uid.get(uid or '', n), 12)
        month_limits, spike_floor = _event_month_limits(
            y_hist,
            yh,
            ds_hist=ds_hist,
            forecast_month=forecast_month,
            horizon_pos=h_pos,
            horizon_len=h_len,
        )
        if month_limits:
            for key, arr in lim.items():
                arr[i] = float(month_limits[key])
        if q:
            e95 = _residual_excess_for_level(q, 'q95', yh)
            e90 = _residual_excess_for_level(q, 'q90', yh)
            e70 = _residual_excess_for_level(q, 'q70', yh)
            if e95 is None:
                e95 = _last_resort_upper_excess(yh, y_hist)
            e95 = max(float(e95), spike_floor)
            if e90 is None:
                e90 = 0.8 * e95
            else:
                e90 = max(float(e90), 0.8 * spike_floor)
            if e70 is None:
                e70 = 0.4 * e95
            else:
                e70 = max(float(e70), 0.4 * spike_floor)
            r95[i] = yh + e95
            r90[i] = yh + e90
            r70[i] = yh + e70
        elif u95 is None or spike_floor > 0.0:
            # Last resort / spike floor: uncalibrated when conformal & residual
            # bands are missing; still apply same-month hist-max floor when present.
            excess = _last_resort_upper_excess(yh, y_hist) if u95 is None else 0.0
            excess = max(float(excess), spike_floor)
            r95[i] = yh + excess
            r90[i] = yh + 0.8 * excess
            r70[i] = yh + 0.4 * excess

    # np.fmax ignores NaN, so rows without residual info keep the conformal value.
    u95 = r95 if u95 is None else np.fmax(u95, r95)
    u90 = r90 if u90 is None else np.fmax(u90, r90)
    u70 = r70 if u70 is None else np.fmax(u70, r70)

    u95 = np.nan_to_num(np.asarray(u95, dtype=float), nan=0.0)
    upper_95 = np.maximum(np.maximum(u95, yhat), 0.0)
    # Event months: the mixture limit derived from that month's own recurrence
    # history replaces the residual/conformal band, which is calibrated across
    # all months and cannot see whether this one fires. Where the point forecast
    # already exceeds the limit the month's history says nothing useful about the
    # error, so the model's own band is kept rather than collapsing to zero width.
    has_lim = np.isfinite(lim[95]) & (lim[95] > yhat)
    if np.any(has_lim):
        upper_95 = np.where(has_lim, lim[95], upper_95)
    gap = np.maximum(0.0, upper_95 - yhat)
    # Rows still NaN at 90/70 (conformal missing those levels, no residual info)
    # fall back to a proportional share of the 95 gap.
    u90 = np.where(np.isnan(u90), yhat + 0.8 * gap, u90)
    u70 = np.where(np.isnan(u70), yhat + 0.4 * gap, u70)
    if np.any(has_lim):
        u90 = np.where(has_lim, np.maximum(lim[90], yhat), u90)
        u70 = np.where(has_lim, np.maximum(lim[70], yhat), u70)
    upper_90 = np.maximum(np.maximum(u90, yhat), 0.0)
    upper_70 = np.maximum(np.maximum(u70, yhat), 0.0)
    upper_90 = np.minimum(upper_90, upper_95)
    upper_70 = np.minimum(upper_70, upper_90)
    return upper_70, upper_90, upper_95


def _build_candidate_model_factories(season_length: int) -> list[tuple[str, Callable[[], object]]]:
    """Candidate StatsForecast model factories (explicitly excludes TimeGPT/LightGPT).

    Used by the forecast step to look up factories by model-name / alias for
    already-selected uids, so every alias that can appear as a CV winner must
    have a factory here. That includes the ``WindowAverage`` class name (for
    back-compat with callers that may pass stored 'WindowAverage' picks) and
    the MA{3,6,12} aliases used by the MA family.
    """
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
        'window_average',
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
                # window_size=SWA_WINDOW_SIZE makes SWA a smoothed seasonal model
                # (average of the last N same-season values), distinct from
                # SeasonalNaive which is window_size=1.
                specs.append((
                    ModelClass.__name__,
                    lambda cls=ModelClass: cls(season_length=season_length, window_size=SWA_WINDOW_SIZE, alias=cls.__name__),
                ))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
        else:
            if key == 'window_average':
                # Back-compat: legacy single WindowAverage(window_size=12) under its
                # class-name alias. New runs select from the MA family below; this
                # entry is only reached if a caller passes 'WindowAverage' as a
                # stored model name.
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(window_size=WA_WINDOW_SIZE, alias=cls.__name__)))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls()))

    # MA family: always registered under all windows so the forecast-time lookup
    # finds a factory for any MA{k} alias, even if CV only selects a subset.
    specs.extend(_build_ma_factories(list(MA_ALL_WINDOWS)))
    return specs


def _build_model_factories_for_keys(keys: list[str], season_length: int) -> list[tuple[str, Callable[[], object]]]:
    """Build factories for the subset of candidate keys active in a given bucket.

    Supports pseudo-keys ``ma3``/``ma6``/``ma12`` that map to WindowAverage
    with distinct aliases, so CV can score them as independent candidates.
    """
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
        if key.startswith('ma') and key[2:].isdigit():
            w = int(key[2:])
            specs.extend(_build_ma_factories([w]))
            continue
        if key not in models_dict:
            continue
        ModelClass = models_dict[key]
        if key in seasonal:
            if key == 'seasonal_window_average':
                specs.append((
                    ModelClass.__name__,
                    lambda cls=ModelClass: cls(season_length=season_length, window_size=SWA_WINDOW_SIZE, alias=cls.__name__),
                ))
            else:
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
        else:
            if key == 'window_average':
                specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(window_size=WA_WINDOW_SIZE, alias=cls.__name__)))
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


def _trim_panel_pre_gap(df: pd.DataFrame, freq: str, *, gap_threshold_months: int = 12) -> pd.DataFrame:
    """Trim history before any large contiguous-zero block (data gap).

    Real production data often has multi-month "holes" — periods where sales
    weren't tracked, the SKU was paused, or imports were skipped — that
    ``_regularize_panel_time_index`` fills with zero. To downstream models
    that's indistinguishable from a real off-season, but it poisons CV
    scoring for seasonal models:

      * SeasonalNaive's lag-12 lookup hits zero-filled rows on one side of
        the gap and real values on the other, producing garbage WAPE.
      * Theta / AutoETS see a non-stationary level and pick degenerate
        configurations.
      * HistoricAverage's mean is dragged down by the zero block.

    The pragmatic fix: when a contiguous run of ``>= gap_threshold_months``
    zero months exists, treat everything before it as pre-gap history and
    drop it. The post-gap block is what reflects current demand. Items
    without large gaps are returned unchanged.

    Concrete example (production item Kjoris_1/106501):
      raw rows span Jan 2022 – May 2026 with all of 2023 missing → 12-month
      zero block after regularization. Trimming leaves Jan 2024 – May 2026
      (29 months), which the dead-zone gate then routes to the seasonal-
      aware mini-CV instead of a flat HistoricAverage that ignores the
      Jul 2024 = 3795 / Jul 2025 = 3920 peaks.

    Threshold rationale: 12 months catches genuine year-long gaps while
    leaving real off-season runs untouched. Even narrow event-seasonal
    items (Christmas-only) have at most ~10 contiguous zero months in any
    given year because the peak is in December; a 12-month zero run is
    diagnostic of missing data rather than seasonality.

    Notes:
      * Operates per ``unique_id``; gaps in one series don't affect others.
      * Only the *most recent* gap is used as the trim cutoff. If a series
        has multiple gaps, all pre-most-recent-gap data is dropped.
      * Series with all-zero history are unchanged (no real data to keep).
    """
    if df.empty or int(gap_threshold_months) <= 0:
        return df
    if canonical_forecaster_freq(freq) != 'MS':
        return df  # gap detection is monthly-specific
    out_parts: list[pd.DataFrame] = []
    for uid, g in df.groupby('unique_id', sort=False):
        g = g.sort_values('ds').reset_index(drop=True)
        y = pd.to_numeric(g['y'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        n = len(y)
        if n == 0 or not np.any(y > 0.0):
            out_parts.append(g)
            continue
        # Find contiguous zero-runs and their (start_idx, end_idx_exclusive).
        # We walk the array once, tracking runs that are flanked by real data
        # on both sides (a leading zero block isn't a "gap" — it's launch
        # phase, which the leading-zero-trim handler downstream addresses).
        first_pos = int(np.argmax(y > 0.0))
        last_pos = n - 1 - int(np.argmax(y[::-1] > 0.0))
        if last_pos <= first_pos:
            out_parts.append(g)
            continue

        # Default: no trim. Leading zeros (before ``first_pos``) are
        # explicitly NOT a gap — that's launch-phase, handled by the
        # leading-zero trim downstream (``y_eff = y_full[first_pos:]``).
        # Only zero runs flanked by real data on both sides count.
        cutoff_idx = 0
        in_run = False
        run_start = 0
        for i in range(first_pos, last_pos + 1):
            if y[i] <= 0.0:
                if not in_run:
                    in_run = True
                    run_start = i
            elif in_run:
                in_run = False
                run_len = i - run_start
                if run_len >= int(gap_threshold_months):
                    # Trim cutoff: end of the gap (this i, the first real
                    # observation after the gap). Use the LATEST gap end
                    # encountered so we keep only the most recent block.
                    cutoff_idx = i

        if cutoff_idx > 0:
            g = g.iloc[cutoff_idx:].reset_index(drop=True)
        out_parts.append(g)

    if not out_parts:
        return df
    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)


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


def _auto_model_detect_sparse_noise(
    y_eff: np.ndarray,
    ds_eff: pd.Series,
    *,
    min_nonzero_months: int = 6,
    max_nonzero_fraction: float = 0.15,
    trend_corr_threshold: float = 0.5,
    recurring_month_min_years: int = 2,
) -> bool:
    """True when history has too few non-zero points to justify a structured model.

    Catches items like "3 random sales in 3 different months across 3 years"
    where CV on classical forecasters can pick seasonal/trend/ETS models that
    invent patterns out of noise. When this fires, the caller should route the
    series to ``HistoricAverage`` (flat mean-of-all level) rather than run CV.

    All four conditions must hold:
      - fewer than ``min_nonzero_months`` non-zero observations in total
      - non-zero fraction < ``max_nonzero_fraction`` (implicitly requires long
        history — e.g. with defaults, series must have >= 40 periods before
        this gate can fire)
      - no strong monotone trend (|trend_corr| < ``trend_corr_threshold``)
      - no recurring calendar month (no single month with non-zero sales in
        ``recurring_month_min_years`` or more distinct years)

    The trend and recurrence checks are safety nets: if a short-history item
    genuinely shows a pattern, let CV run on it. Only flat-ish random sparse
    data gets short-circuited.
    """
    y = np.asarray(y_eff, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return False  # handled by the no-positive-history path upstream

    nz_count = int(np.sum(y > 0))
    if nz_count == 0:
        # All-zero history is silence, not sparse noise. Upstream already
        # routes these to the dead-SKU path via long-silence; returning False
        # here keeps the helper's contract defensive if that upstream gate
        # ever changes.
        return False
    if nz_count >= int(min_nonzero_months):
        return False

    nz_frac = float(nz_count) / float(y.size)
    if nz_frac >= float(max_nonzero_fraction):
        return False

    # Trend safety net: if the few non-zero points themselves line up as a
    # trend against their positions in time, don't short-circuit — CV / trend
    # models may still produce a useful forecast.
    #
    # Correlation is computed on the non-zero *subset* (positions vs values),
    # not the full zero-padded series. On sparse data the zero mass dilutes
    # the full-series Pearson r well below 0.5 even for a cleanly trending
    # tail, so a full-series check would be effectively inert here.
    #
    # Require >= 4 non-zero points: Pearson on only 3 points is unstable (a
    # single outlier can tip |r| above 0.5 spuriously). Items with exactly
    # 3 sparse sales fall through to the short-circuit regardless of shape.
    if nz_count >= 4:
        nz_mask_tmp = y > 0
        nz_positions = np.flatnonzero(nz_mask_tmp).astype(float)
        nz_values = y[nz_mask_tmp]
        if float(np.std(nz_values)) > 0:
            r = float(np.corrcoef(nz_positions, nz_values)[0, 1])
            if np.isfinite(r) and abs(r) >= float(trend_corr_threshold):
                return False

    # Recurrence safety net: if a calendar month shows non-zero sales in
    # multiple distinct years, there IS a seasonal signal — keep CV eligible.
    try:
        ds_pd = pd.to_datetime(ds_eff, errors='coerce').reset_index(drop=True)
    except Exception:
        ds_pd = None
    if ds_pd is not None and len(ds_pd) == y.size:
        nz_mask = y > 0
        if int(np.sum(nz_mask)) >= 2:
            months = ds_pd[nz_mask].dt.month
            years = ds_pd[nz_mask].dt.year
            per_month_years = (
                pd.DataFrame({'m': months.values, 'y': years.values})
                .groupby('m')['y']
                .nunique()
            )
            if int(per_month_years.max()) >= int(recurring_month_min_years):
                return False

    return True


def _auto_model_event_seasonal_dead_zone_select(
    g_uid: pd.DataFrame,
    *,
    h: int,
    season_length: int,
    n_obs: int,
    freq: str,
) -> tuple[Optional[str], dict[str, Any]]:
    """Run a small per-uid CV to pick a model for event-seasonal items in the
    "CV dead zone" (24-37 months on monthly h>=12 data).

    The bucket-level seasonal CV uses ``cv_h = min(h, season_length)`` which
    sets the per-model min-obs gate for ``SeasonalNaive`` to ~38 months.
    Items below that threshold get SN/SWA/AutoARIMA filtered out and the pool
    collapses to AutoETS + HistoricAverage. AutoETS on 1-2 cycles typically
    picks a non-seasonal config and forecasts a flat low line.

    This helper runs a relaxed CV with a shorter horizon so SN qualifies, and
    scores it against a small set of seasonal-aware alternatives:

      - SeasonalNaive: lag-12 baseline.
      - AutoETS: searches (E, T, S) by AICc; on 2 cycles usually drops the
        seasonal component but may pick (A, A, N) for clearly trending items.
      - Theta / OptimizedTheta: classical seasonal decomposition; estimates
        seasonal indices by averaging same-month values, more robust on
        short series than AutoETS's MLE.
      - HistoricAverage: honest level fallback.

    Returns ``(best_model_name, debug_info)``. ``best_model_name`` is the
    StatsForecast class name suitable for ``best_by_uid[uid] = ...``. If CV
    cannot be run (insufficient data even at the relaxed cv_h, or all models
    raise), returns ``(None, ...)`` and the caller should fall back to a
    deterministic SN pick.
    """
    try:
        from statsforecast import StatsForecast  # local import: speed
    except Exception as e:  # pragma: no cover - statsforecast is a hard dep
        return None, {'reason': 'cv_failed', 'error': str(e)}

    # Adaptive CV settings: smaller cv_h to fit SN at min-obs <= n_obs.
    # For h >= 6 use cv_h = 6; otherwise mirror h. Two windows when feasible
    # (n_obs >= 26 with cv_h=6), single window otherwise so n_obs in [24, 25]
    # still gets evaluated.
    dz_cv_h = max(1, int(min(h, 6)))
    sn_min_obs_two = dz_cv_h * 2 + 2 + int(season_length)
    dz_n_windows = 2 if int(n_obs) >= int(sn_min_obs_two) else 1

    # Hard floor: even single-window CV needs cv_h + 2 + season for SN to
    # have any training data. If n_obs is below that, bail out.
    sn_min_obs_one = dz_cv_h + 2 + int(season_length)
    if int(n_obs) < int(sn_min_obs_one):
        return None, {
            'reason': 'cv_failed',
            'detail': 'n_obs_below_sn_min_obs_one_window',
            'n_obs': int(n_obs),
            'sn_min_obs_one': int(sn_min_obs_one),
        }

    models_dict = _lazy_import_nixtla_models()
    candidates: list[tuple[str, Callable[[], object]]] = [
        ('SeasonalNaive', lambda: models_dict['seasonal_naive'](season_length=season_length)),
        ('AutoETS', lambda: models_dict['auto_ets'](season_length=season_length)),
        ('Theta', lambda: models_dict['theta'](season_length=season_length)),
        ('OptimizedTheta', lambda: models_dict['optimized_theta'](season_length=season_length)),
        ('HistoricAverage', lambda: models_dict['historic_average']()),
    ]

    df_uid = g_uid.loc[:, ['unique_id', 'ds', 'y']].copy()
    df_uid['unique_id'] = df_uid['unique_id'].astype(str)

    scores: dict[str, float] = {}
    biases: dict[str, float] = {}
    for name, factory in candidates:
        try:
            sf_one = StatsForecast(models=[factory()], freq=freq, n_jobs=1)
            cv = sf_one.cross_validation(
                df=df_uid,
                h=int(dz_cv_h),
                step_size=int(dz_cv_h),
                n_windows=int(dz_n_windows),
            )
        except Exception:
            continue
        if name not in cv.columns:
            continue
        y_true = pd.to_numeric(cv['y'], errors='coerce').to_numpy(dtype=float)
        y_hat = pd.to_numeric(cv[name], errors='coerce').to_numpy(dtype=float)
        wape, bias = _safe_wape_and_bias(y_true, y_hat)
        if not np.isfinite(wape):
            continue
        scores[name] = float(wape)
        biases[name] = float(bias)

    if not scores:
        return None, {
            'reason': 'cv_failed',
            'detail': 'all_models_raised_or_unscored',
            'dz_cv_h': int(dz_cv_h),
            'dz_n_windows': int(dz_n_windows),
        }

    # Lag-family vs non-lag carve-out.
    #
    # Theta / OptimizedTheta / AutoETS estimate seasonality from the data.
    # On <3 cycles their seasonal decomposition is unreliable: they often
    # win CV on a 6-month holdout (mostly off-season months) but their
    # 12-step forecast collapses to a near-flat extrapolation that
    # completely misses the next peak. Concrete failure observed on real
    # production items 103402/103403 (28 months, leading zeros): Theta's
    # CV WAPE 0.501 beat SN 0.556 by 11%, but Theta's Jul forecast was
    # 19.6 vs SN+peak_ratio's 353 — an order-of-magnitude undershoot.
    #
    # Require a stronger advantage on short series so SN is the default
    # pivot when seasonality is detected but data is sparse. SN combined
    # with the peak_ratio post-correction handles year-over-year
    # growth/decline without depending on trend extrapolation.
    cycles = float(n_obs) / float(season_length) if season_length > 0 else 0.0
    sn_advantage_required = 0.05 if cycles >= 3.0 else 0.20
    sn_wape = scores.get('SeasonalNaive')
    other_best_score = min(
        (v for k, v in scores.items() if k != 'SeasonalNaive'),
        default=None,
    )
    sn_unavailable = sn_wape is None or not np.isfinite(sn_wape)
    if sn_unavailable:
        # SN couldn't be scored (too few observations even at relaxed cv_h);
        # fall back to the best of whatever did score.
        best = min(scores.keys(), key=lambda k: scores[k])
    elif (
        other_best_score is not None
        and other_best_score > 0
        and (sn_wape - other_best_score) / max(other_best_score, 1e-9) > sn_advantage_required
    ):
        # Non-lag candidate beats SN by more than the required advantage —
        # promote the non-lag winner. On <3 cycles the threshold is 20% so
        # we don't fall for short-data Theta/AutoETS that win CV but
        # extrapolate to a flat forecast.
        best = min(
            (k for k in scores.keys() if k != 'SeasonalNaive'),
            key=lambda k: scores[k],
        )
    else:
        # SN is the safe pivot: either it has the lowest WAPE or no non-lag
        # candidate beats it by enough margin to be trusted on this much
        # data. The peak_ratio post-correction will adjust for YoY trend.
        best = 'SeasonalNaive'

    return best, {
        'reason': 'event_seasonal_dead_zone_cv',
        'picked': best,
        'scores': {k: float(v) for k, v in scores.items()},
        'biases': {k: float(v) for k, v in biases.items()},
        'dz_cv_h': int(dz_cv_h),
        'dz_n_windows': int(dz_n_windows),
        'n_obs': int(n_obs),
        'cycles': float(cycles),
        'sn_advantage_required': float(sn_advantage_required),
    }


def _auto_model_compute_peak_ratio(
    g_uid: pd.DataFrame,
    *,
    season_length: int,
    min_clip: float = 0.7,
    max_clip: float = 1.4,
) -> Optional[float]:
    """Year-over-year peak-month volume ratio, used to scale a SeasonalNaive
    forecast for event-seasonal items in the dead zone.

    Identifies the peak month (highest total volume across all years) and
    returns ``current_year_peak / prior_year_peak`` clipped to a conservative
    band so noise on a single peak can't blow up the forecast.

    Returns ``None`` when:
      - history is too short for two full seasonal cycles
      - either the current or prior year peak observation is missing
      - the prior year peak is non-positive (avoid divide-by-near-zero)

    The clip range ``[0.7, 1.4]`` is intentionally conservative: with only
    two cycles the ratio has high sampling variance, so we limit the
    correction to ±30-40%. For items with three or more cycles the ratio
    is averaged across the available pairs (more robust); the mini-CV in
    ``_auto_model_event_seasonal_dead_zone_select`` is the primary
    differentiator and this correction is a small additional nudge.
    """
    try:
        ds_pd = pd.to_datetime(g_uid['ds'], errors='coerce').reset_index(drop=True)
    except Exception:
        return None
    y = pd.to_numeric(g_uid['y'], errors='coerce').reset_index(drop=True)
    if len(y) < 2 * int(season_length):
        return None

    # Find the peak month by total volume across all years.
    nz = y > 0.0
    if not bool(nz.any()):
        return None
    df_local = pd.DataFrame({'ds': ds_pd, 'y': y, 'm': ds_pd.dt.month, 'yr': ds_pd.dt.year})
    vol_by_month = df_local.loc[nz, :].groupby('m')['y'].sum()
    if vol_by_month.empty:
        return None
    peak_month = int(vol_by_month.idxmax())

    peak_obs = (
        df_local.loc[df_local['m'] == peak_month, ['yr', 'y']]
        .groupby('yr', as_index=True)['y']
        .sum()
        .sort_index()
    )
    if len(peak_obs) < 2:
        return None

    # Pairwise ratios across consecutive years; average for robustness.
    years = peak_obs.index.to_list()
    ratios: list[float] = []
    for i in range(1, len(years)):
        prev = float(peak_obs.iloc[i - 1])
        curr = float(peak_obs.iloc[i])
        if prev <= 1e-9:
            continue
        ratios.append(float(curr / prev))
    if not ratios:
        return None

    ratio = float(np.mean(ratios))
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    return float(np.clip(ratio, float(min_clip), float(max_clip)))


def _auto_model_exclude_seasonal_naive(
    y_full: np.ndarray,
    *,
    season_length: int,
    event_seasonal: bool = False,
    stable_recent_level: bool = False,  # noqa: ARG001 (kept for backward compat)
) -> bool:
    """True when SeasonalNaive (y[t]=y[t-s]) is likely misleading.

    - Recent all-zero tail: same month last year may show a peak while the item
      is effectively discontinued for the last several months.
    - Year-over-year collapse: prior seasonal year had real volume, last year
      <35% of that — repeating the old seasonal profile is usually wrong.
      A *stable* recent level after the collapse is the strongest evidence the
      old seasonal values are no longer representative; SN is excluded in that
      case too. The level-prefer reranker then steers selection to a level
      model (HistoricAverage / WindowAverage) for the new run rate.

    Event-seasonal SKUs (e.g. Christmas-heavy) often have long off-season zero
    runs and volatile recent years; those patterns are *not* discontinuation.
    For them we skip these exclusions and let CV pick (``long_trailing_zero_run``
    / ``long_silence_after_last_sale`` still force Naive when appropriate).

    The ``stable_recent_level`` argument is retained for backward compatibility
    but no longer changes the result: previously it short-circuited the YoY
    collapse exclusion, which kept SN eligible in exactly the post-step-down
    cohort where its absolute-lag forecasts are most wrong.
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
    event_seasonal_uid: Optional[dict[str, bool]] = None,
    strong_trend_uid: Optional[dict[str, bool]] = None,
) -> None:
    """When the recent tail is a stable positive level, prefer a level model.

    Reduces two failure modes for items that have settled at a new (often lower)
    stable run rate:

      * AutoETS/Theta/ARIMA extrapolating a past decline down to ~0.
      * SeasonalNaive/SeasonalWindowAverage repeating last year's *absolute*
        seasonal values (which are no longer representative).

    The target pool is ``HistoricAverage``, ``MA6``, ``MA12`` and legacy
    ``WindowAverage`` — *excluding* ``MA3`` which is too reactive to serve as a
    level anchor. On WAPE ties, prefer LONGER windows (more stable) — the
    reranker's job is to replace a jittery adaptive pick, so reactivity is
    anti-goal here.

    In the ``seasonal`` bucket a CV margin of 1 pp WAPE is required before
    demoting — enough to filter CV noise (typically 1–2 pp between candidates)
    without blocking legitimate level-model wins. A 3 pp margin was tried
    initially and was too strict; it blocked ~20 MA12 wins in A/B testing.

    Event-seasonal items: lag-family picks are kept (the lag IS the signal).
    Strong-trend items: skipped entirely — a flat level can't follow a trend.
    """
    demote_from = {
        'AutoETS', 'Theta', 'OptimizedTheta', 'AutoARIMA',
        'SeasonalNaive', 'SeasonalWindowAverage',
    }
    # Ordering hint used on exact WAPE ties / equal-score ranks. The reranker's
    # job is to replace a jittery adaptive pick with a STABLE level estimate,
    # so on ties prefer LONGER windows (more stable). MA3 is intentionally
    # excluded from the target pool entirely — 3 months is too reactive/noisy
    # to function as a level anchor on monthly data (v3 evidence: MA3 bias
    # median −28%, multiple full collapses on confectionery items).
    _level_order_hint: dict[str, int] = {
        'MA12': 0, 'WindowAverage': 1, 'MA6': 2, 'HistoricAverage': 3,
    }

    def _level_candidates_for_uid(uid: str) -> list[str]:
        """Collect level-model aliases that were actually scored for ``uid``.

        Excludes MA3 — too noisy for demote-to use. Includes MA6, MA12, legacy
        WindowAverage, and HistoricAverage.
        """
        if metric_name in ('wape', 'wape_bias'):
            scored = set(wape_scores_map.get(uid, {}).keys())
        else:
            scored = set(rmse_scores_map.get(uid, {}).keys()) & set(mae_scores_map.get(uid, {}).keys())
        candidates: list[str] = []
        for name in ('MA12', 'WindowAverage', 'MA6', 'HistoricAverage'):
            if name in scored:
                candidates.append(name)
        return candidates

    # Open to 'trend' bucket too — a strong-trend guard below prevents us from
    # demoting picks on genuinely trending series, but mild-trend series that
    # have settled into a stable tail should behave like smooth/seasonal.
    buckets_ok = {'seasonal', 'smooth', 'trend'}
    event_uid = event_seasonal_uid or {}
    trend_uid = strong_trend_uid or {}
    for uid, pick in list(best_by_uid.items()):
        if not stable_tail_uid.get(uid, False):
            continue
        bucket = bucket_by_uid.get(uid)
        if bucket not in buckets_ok:
            continue
        if pick not in demote_from:
            continue
        # Never demote on a strong-trend series — a flat level forecast is worse
        # than the trend-aware model the picker chose.
        if bool(trend_uid.get(uid, False)):
            continue
        # Don't demote lag-family picks for event-seasonal items: the lag IS the
        # forecast signal there (Christmas-only, etc.).
        if pick in _LAG_FAMILY_MODELS and bool(event_uid.get(uid, False)):
            continue

        level_models = _level_candidates_for_uid(uid)
        if not level_models:
            continue

        # Seasonal bucket: require a small CV margin before demoting. CV noise
        # between candidates on monthly seasonal series is typically 1–2 pp;
        # 1 pp filters the noise without blocking legitimate MA12 wins. The
        # earlier 3 pp margin was too tight and blocked ~20 MA12 wins over
        # AutoETS where MA12 was genuinely better by 1–3 pp (v4 A/B).
        margin_required = 0.01 if bucket == 'seasonal' else 0.0

        if metric_name in ('wape', 'wape_bias'):
            pw = wape_scores_map.get(uid, {}).get(pick)
            if pw is None or not np.isfinite(float(pw)):
                continue
            pool: list[str] = [pick]
            for alt in level_models:
                aw = wape_scores_map.get(uid, {}).get(alt)
                if aw is None or not np.isfinite(float(aw)):
                    continue
                # Level model must beat the current pick by at least margin_required.
                if float(aw) + margin_required <= float(pw):
                    pool.append(alt)
            if len(pool) > 1:
                # Sort by WAPE, then ordering hint (longer MA wins on ties).
                best_by_uid[uid] = min(
                    pool,
                    key=lambda m: (float(wape_scores_map[uid][m]), _level_order_hint.get(m, 9)),
                )
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
                # Level model must beat or match on both RMSE and MAE. Apply the
                # seasonal margin on RMSE (scale-aware) by requiring strict
                # improvement of at least margin_required * |pr|.
                rmse_margin = margin_required * abs(float(pr))
                if float(ar) + rmse_margin <= float(pr) and float(am) <= float(pm):
                    pool_r.append(alt)
            if len(pool_r) > 1:
                best_by_uid[uid] = min(
                    pool_r,
                    key=lambda m: (
                        float(rmse_scores_map[uid].get(m, np.inf)) + float(mae_scores_map[uid].get(m, np.inf)),
                        _level_order_hint.get(m, 9),
                    ),
                )


def _rerank_pick_excluding_lag_family(
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
    """If selection is in the lag family but disallowed, pick the next best in-place.

    "Lag family" = ``_LAG_FAMILY_MODELS`` (SeasonalNaive + SeasonalWindowAverage).
    Both forecast some weighted reuse of historical same-season values and share
    the failure mode where those historical values no longer represent current
    demand (post step-down, dead tails).
    """
    if best_by_uid.get(uid) not in _LAG_FAMILY_MODELS:
        return
    excluded = set(_LAG_FAMILY_MODELS)
    if metric_name == 'robust':
        rm = {m: v for m, v in rmse_scores_map.get(uid, {}).items() if m not in excluded}
        ma = {m: v for m, v in mae_scores_map.get(uid, {}).items() if m not in excluded}
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
        wape_vals = pd.Series(
            {m: v for m, v in wape_scores_map.get(uid, {}).items() if m not in excluded}
        )
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
    ms = {m: v for m, v in metric_scores.get(uid, {}).items() if m not in excluded}
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
            fcst = _statsforecast_forecast_with_conformal(sf, df, h)
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

        hist_q = _historical_excess_quantiles(df['y'].to_numpy(), df['ds'])
        y_hist = {'item': df['y'].to_numpy(dtype=float)}
        ds_hist = {'item': pd.Series(pd.to_datetime(df['ds']))}
        upper_70, upper_90, upper_95 = _attach_upper_quantiles(
            yhat,
            fcst,
            forecast_col,
            uid_series=pd.Series(['item'] * len(yhat)),
            ds_series=pd.Series(future_ds),
            historical_excess_by_uid={'item': hist_q} if hist_q else None,
            y_hist_by_uid=y_hist,
            ds_hist_by_uid=ds_hist,
        )
        
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

        # Trim history before any large data gap (>=12 contiguous zero months
        # after regularization). On real catalogs, ~46% of items can have
        # year-long calendar gaps in their imports — these poison
        # SeasonalNaive's lag-12 alignment and unfairly favor flat models
        # like HistoricAverage. Keeping only the post-most-recent-gap block
        # restores honest CV scoring for seasonal candidates without
        # affecting items that don't have gaps.
        df = _trim_panel_pre_gap(df, freq=str(self.freq))
        if df.empty:
            raise ValueError('Empty history after pre-gap trim')

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
        # Three stability tiers, increasingly permissive:
        #   stable_tail_uid       max_cv=0.55 — strict; gates short-history Naive fallback.
        #   stable_tail_loose_uid max_cv=0.62 — middling; gates the level-prefer reranker
        #                                       so noisier-but-real level shifts get caught
        #                                       at selection (cleaner uncertainty bands).
        #   floor_eligible_uid    max_cv=0.70 — loose; only used by the post-forecast floor.
        stable_tail_uid: dict[str, bool] = {}
        stable_tail_loose_uid: dict[str, bool] = {}
        stable_tail_mean_uid: dict[str, float] = {}
        floor_eligible_uid: dict[str, bool] = {}
        floor_eligible_mean_uid: dict[str, float] = {}
        bucket_by_uid: dict[str, str] = {}
        exclude_seasonal_naive_uid: dict[str, bool] = {}
        dead_sku_uid: set[str] = set()
        # Peak-ratio correction for event-seasonal items in the dead zone where
        # SeasonalNaive won the mini-CV. Maps uid -> multiplicative scalar
        # (clipped) applied to SN's forecast at output time so that genuinely
        # growing items don't get pinned to last year's value.
        peak_ratio_correction_uid: dict[str, float] = {}

        def _looks_event_seasonal(g: pd.DataFrame, season_length: int) -> bool:
            """Heuristic: demand volume mostly concentrated in 1-2 calendar months each year.

            Uses demand *volume* share (sum of y per month / total y), not occurrence
            count, so a December spike of 2500 correctly dominates over small Oct/Nov sales.

            Requires a recurrence check: the top-volume month must appear in at
            least 2 distinct calendar years. Without this, a series with only
            a handful of sparse sales (e.g. 3 sales in 3 different years) trips
            the concentration gate by accident — the top month's "share" is
            simply 1/3, which beats the 45% threshold on occurrence alone, but
            there's no actual repeating pattern for SeasonalNaive to exploit.
            """
            if season_length < 2:
                return False
            if len(g) < 2 * season_length:
                return False
            nz = g[g['y'] > 0.0].copy()
            if nz.empty:
                return False
            nz_ds = pd.to_datetime(nz['ds'])
            nz['month'] = nz_ds.dt.month
            vol_by_month = nz.groupby('month')['y'].sum()
            total = float(vol_by_month.sum())
            if total <= 0.0:
                return False
            share = (vol_by_month / total).sort_values(ascending=False)
            top1 = float(share.iloc[0])
            top2 = float(share.iloc[:2].sum()) if len(share) >= 2 else top1
            top3 = float(share.iloc[:3].sum()) if len(share) >= 3 else top2
            passes_concentration = (top1 >= 0.45 or top2 >= 0.60 or top3 >= 0.75)
            if not passes_concentration:
                return False

            # Recurrence: the peak month must show non-zero sales in ≥ 2 years.
            # This distinguishes genuine event-seasonality (same month, multiple
            # years) from sparse sales that concentrate by volume only because
            # there are few of them.
            peak_month = int(share.index[0])
            peak_years = int(nz_ds[nz['month'] == peak_month].dt.year.nunique())
            return peak_years >= 2

        for uid, n_obs in counts.items():
            g_uid = df.loc[df['unique_id'] == uid, ['unique_id', 'ds', 'y']].copy()
            y_full = g_uid['y'].to_numpy(dtype=float)

            # Event-seasonal detection runs first so that seasonal items with
            # expected off-season silence aren't misclassified as dead SKUs.
            event_seasonal_uid[uid] = _looks_event_seasonal(g_uid, int(season_for_models))

            # Long trailing zero run: 24+ months of actual zeros in the data is
            # strong evidence of death even for seasonal items.
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

            # Long calendar-time silence since last sale. Skip for event-seasonal
            # items where multi-month off-season gaps are normal, not discontinuation.
            if (
                not event_seasonal_uid.get(uid, False)
                and _auto_model_force_naive_long_silence_after_last_sale(
                    y_full, g_uid['ds'], min_silent_months=24
                )
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

            # Trim leading zeros for profiling/bucketing so launch-phase zeros don't
            # dominate n, zero_frac, or trend detection. Keep zeros after the first
            # positive month (true stockouts/off-season).
            mask_pos = y_full > 0.0
            if mask_pos.any():
                first_pos = int(np.argmax(mask_pos))
                y_eff = y_full[first_pos:]
                ds_eff = g_uid['ds'].iloc[first_pos:]
            else:
                y_eff = y_full
                ds_eff = g_uid['ds']

            # Sparse-noise items: rarely sold, no trend, no recurring calendar
            # month. CV on these tends to pick spurious ETS / seasonal patterns
            # (or Croston-family models that collapse to ~0). Short-circuit to
            # HistoricAverage — mean-of-all is the honest flat level for
            # genuinely random sparse demand. Conservative minimum: no yhat
            # override, no new model label; if HA proves to under-forecast on
            # real data, a mean-of-positives override can be added as a
            # follow-up with before/after numbers.
            if _auto_model_detect_sparse_noise(y_eff, ds_eff):
                best_by_uid[uid] = 'HistoricAverage'
                exclude_seasonal_naive_uid[uid] = True
                if debug:
                    debug_reason[uid] = {
                        'reason': 'sparse_noise',
                        'picked': 'HistoricAverage',
                        'nonzero_count': int(np.sum(y_eff > 0)),
                        'n_obs': int(n_obs),
                    }
                continue

            # Stable-tail on y_eff (not y_full) so padded trailing zeros from
            # panel extension don't mask a real stable demand level.
            stable_recent, stable_tail_mean = _recent_tail_stable_level(y_eff)
            stable_tail_uid[uid] = bool(stable_recent)
            stable_tail_mean_uid[uid] = float(stable_tail_mean)

            # Middling stability tier (max_cv=0.62) for the level-prefer reranker.
            # Wider than the strict 0.55 so noisier-but-real level shifts get caught
            # at *selection* (the chosen model's uncertainty bands are then honest);
            # tighter than the floor's 0.70 so we don't over-demote AutoETS/Theta on
            # genuinely volatile series.
            if stable_recent:
                stable_tail_loose_uid[uid] = True
            else:
                lo_ok, _lo_mean = _recent_tail_stable_level(y_eff, max_cv=0.62)
                stable_tail_loose_uid[uid] = bool(lo_ok)

            # Looser stability check for the post-forecast sanity floor. The floor
            # only needs a rough demand level, not a tight estimate, so allow higher
            # CV (0.70) to catch more collapsing items.
            if stable_recent:
                floor_eligible_uid[uid] = True
                floor_eligible_mean_uid[uid] = float(stable_tail_mean)
            else:
                fl_ok, fl_mean = _recent_tail_stable_level(y_eff, max_cv=0.70)
                floor_eligible_uid[uid] = bool(fl_ok)
                floor_eligible_mean_uid[uid] = float(fl_mean)

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
            # Same lag-family failure mode applies to SeasonalWindowAverage:
            # both reuse historical same-season values. Track jointly under the
            # legacy ``exclude_seasonal_naive_uid`` flag (the rerank function
            # strips the whole _LAG_FAMILY_MODELS set when this is True).
            exclude_seasonal_naive_uid[uid] = ex_sn

            # Event-seasonal items in the CV "dead zone".
            #
            # On monthly data the seasonal CV bucket runs with bucket_cv_h =
            # min(h, season_for_models) (line ~1851). For h>=12 that's a
            # 12-step CV horizon, which pushes ``_min_obs_for_model_cv`` for
            # SeasonalNaive (``base + season``) to ~38 months. Items with
            # 24-37 months of *panel* history therefore have SN/SWA/AutoARIMA
            # filtered out before CV; the seasonal pool collapses to AutoETS
            # + HistoricAverage, AutoETS picks a non-seasonal config on 1-2
            # cycles of training data, and the forecast is a flat low line
            # that misses the next peak.
            #
            # We also short-circuit items that would otherwise land in the
            # ``'smooth'`` bucket (n_eff < 24 due to leading zeros). That
            # bucket's pool doesn't include SeasonalNaive at all, so an
            # event-seasonal SKU with first sale ~6 months in (e.g. summer-
            # only Kjörís) would silently get AutoETS-flat. This gate uses
            # ``n_obs`` (full regularized panel) rather than ``n_eff`` so
            # leading zeros don't disqualify it — SN forecasts the requested
            # horizon h<=12 by lagging into real data, not the leading zeros.
            #
            # Conditions:
            #   - is_monthly (gate is monthly-specific by construction)
            #   - yearly seasonality detected by EITHER detector:
            #       * event_seasonal_uid: strict — single-month dominant
            #         peak (top1 share >= 0.45) with recurrence in >=2 years.
            #         Catches narrow-event items (Christmas, ramp-events).
            #       * _strong_yearly_seasonality(y_full): broad — lag-12
            #         correlation >= 0.38 on the full panel. Catches broad
            #         seasonal humps (e.g. ice-cream summer season Apr-Sep
            #         peaking in Jul) where no single month exceeds 45% of
            #         volume share but the year-over-year shape repeats.
            #         Verified on production items 103401-103403, 104318:
            #         top1=0.25-0.34 (fail strict) but lag-12 corr passes.
            #     Using OR rather than AND: both detectors are conservative
            #     in their own way (concentration threshold vs correlation
            #     threshold) and miss legitimate seasonal items each.
            #   - not exclude_seasonal_naive_uid (dead tail / YoY collapse /
            #     lag-vs-recent regime mismatch all bypassed for event-
            #     seasonal upstream, but checked defensively)
            #   - 2*season <= n_obs < sn_gate: long enough for either
            #     detector to even run, short enough that the seasonal CV
            #     path filters SN out.
            yearly_seasonal_for_dz = bool(
                event_seasonal_uid.get(uid, False)
            ) or bool(
                _strong_yearly_seasonality(y_full, int(season_for_models))
            )
            if (
                is_monthly
                and yearly_seasonal_for_dz
                and not exclude_seasonal_naive_uid.get(uid, False)
            ):
                # Mirror the bucket_cv_h that the seasonal CV path uses
                # (line ~1851: min(h, season_for_models)) so this gate fires
                # iff the CV path would otherwise filter SN out.
                _bucket_cv_h_seasonal = max(1, int(min(h, season_for_models)))
                _sn_filter_gate = (
                    _bucket_cv_h_seasonal * max(1, int(n_windows))
                    + 2
                    + int(season_for_models)
                )
                # The gate must fire for two distinct failure modes:
                #   (1) SN would be filtered out by min-obs (covered by
                #       _sn_filter_gate above; depends on n_windows).
                #   (2) Standard CV is unreliable on the well-known monthly
                #       dead zone [2*season, 3*season+h] — particularly when
                #       n_windows=1 (which the API uses by default for
                #       season_length=forecast_periods=12). With a single
                #       12-month holdout, leading-zero alignment can poison
                #       SN's lag-12 baseline and Naive wins by predicting a
                #       flat last value. Verified on production items
                #       103401-103403, 104318: n_obs=28, n_windows=1 →
                #       _sn_filter_gate=26 misses them; without the dead-zone
                #       upper bound, the gate skips and SN never gets a fair
                #       comparison. The mini-CV uses smaller cv_h + multiple
                #       windows internally so it is robust to this regime.
                _dead_zone_upper = 3 * int(season_for_models) + int(h)
                _sn_gate = max(_sn_filter_gate, _dead_zone_upper)
                if 2 * int(season_for_models) <= int(n_obs) < int(_sn_gate):
                    # Run a relaxed mini-CV (smaller cv_h so SN is eligible)
                    # over a small set of seasonal-aware candidates. Pick the
                    # winner by WAPE with a small lag-family advantage
                    # requirement; SeasonalNaive wins for items where last
                    # year is the best estimate of next year, Theta /
                    # OptimizedTheta / AutoETS for items with a clear trend.
                    dz_pick, dz_info = _auto_model_event_seasonal_dead_zone_select(
                        g_uid,
                        h=int(h),
                        season_length=int(season_for_models),
                        n_obs=int(n_obs),
                        freq=str(self.freq),
                    )
                    if dz_pick is None:
                        # Fallback: deterministic SN. The CV path further
                        # downstream would have left this item with AutoETS-
                        # flat or HistoricAverage; SN is at least
                        # seasonally-aware.
                        best_by_uid[uid] = 'SeasonalNaive'
                        dz_info = dict(dz_info or {})
                        dz_info.update(
                            {
                                'reason': dz_info.get('reason', 'cv_failed'),
                                'picked': 'SeasonalNaive',
                                'fallback': True,
                            }
                        )
                    else:
                        best_by_uid[uid] = dz_pick

                    # Peak-ratio correction is a post-forecast nudge for the
                    # SN winners only — Theta/AutoETS already model trend
                    # internally, so applying a multiplicative ratio on top
                    # would double-count. The correction is conservative
                    # (clipped to [0.7, 1.4]) so it can't blow up the
                    # forecast on noisy single-cycle ratios.
                    if best_by_uid[uid] == 'SeasonalNaive':
                        ratio = _auto_model_compute_peak_ratio(
                            g_uid, season_length=int(season_for_models)
                        )
                        if ratio is not None and abs(float(ratio) - 1.0) > 1e-3:
                            peak_ratio_correction_uid[uid] = float(ratio)

                    if debug:
                        debug_entry: dict[str, Any] = {
                            'reason': 'event_seasonal_cv_dead_zone',
                            'picked': best_by_uid[uid],
                            'n_obs': int(n_obs),
                            'sn_gate': int(_sn_gate),
                            'season_for_models': int(season_for_models),
                            'bucket_cv_h_seasonal': int(_bucket_cv_h_seasonal),
                            'n_windows': int(n_windows),
                        }
                        debug_entry.update(dz_info or {})
                        if uid in peak_ratio_correction_uid:
                            debug_entry['peak_ratio'] = float(
                                peak_ratio_correction_uid[uid]
                            )
                        debug_reason[uid] = debug_entry
                    continue

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
            # 0.55 (was 0.7) aligns with the bucket classifier (line 557, abs>=0.5)
            # and the 12-23m fallback (line 1221, abs>=0.5). 0.7 was so strict that
            # genuinely trending items with seasonal noise (|r|≈0.55-0.65) bypassed
            # the trend-bucket lag-family demotion entirely.
            try:
                trend_corr = float(prof.get('trend_corr', 0.0))  # type: ignore[arg-type]
            except Exception:
                trend_corr = 0.0
            strong_trend_uid[uid] = bool(abs(trend_corr) >= 0.55)

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
                    # Reuse trend_corr from prof.
                    # - Strong trend → AutoETS (level + slope on a single year of data).
                    # - Stable recent tail → Naive (repeats the current run rate).
                    # - Otherwise → HistoricAverage. MA6 was tried here but
                    #   regressed on items with latent monthly seasonality (too
                    #   short to detect at <24m), where a forecast origin in
                    #   off-season projected low into the next peak.
                    # NOTE: We deliberately do NOT pick SeasonalWindowAverage here.
                    # With ``SWA_WINDOW_SIZE`` ≥ 2, SWA needs at least two complete
                    # seasonal cycles, so it can't fit on 12–23 monthly observations.
                    try:
                        trend_corr = float(prof.get('trend_corr', 0.0))  # type: ignore[arg-type]
                    except Exception:
                        trend_corr = 0.0
                    if abs(trend_corr) >= 0.5:
                        best_by_uid[uid] = 'AutoETS'
                    elif stable_tail_uid.get(uid, False):
                        best_by_uid[uid] = 'Naive'
                    else:
                        best_by_uid[uid] = 'HistoricAverage'
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
                    # HistoricAverage operates on the full regularized panel
                    # (including zero-filled months before the first sale),
                    # which dilutes the forecast. Naive repeats the last value
                    # and is better when the recent tail is at a stable level.
                    if stable_tail_uid.get(uid, False):
                        best_by_uid[uid] = 'Naive'
                    else:
                        best_by_uid[uid] = 'HistoricAverage'
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
                            'window_average',
                            'auto_ets',
                            'croston_optimized',
                        ]
                    # Otherwise, just add a single seasonal model to the pool.
                    base_keys.append('auto_ets')
                    base_keys.extend(['historic_average', 'window_average', 'seasonal_window_average'])
                return base_keys
            # MA family available for this history length. MA3 is excluded from
            # CV candidate pools — on monthly data 3 months is too short to
            # function as a stable level estimate, and the v3 A/B showed MA3
            # winning CV on noise patterns (bias median −28%, multiple full
            # collapses). MA3 is retained only as a registered factory for
            # back-compat; it is never selected fresh.
            ma_windows = _ma_windows_for_history(int(n_obs))
            ma_keys_long = [f'ma{w}' for w in ma_windows if w in (6, 12)]
            if bucket == 'seasonal':
                keys = [
                    'seasonal_naive',
                    'seasonal_window_average',
                    'historic_average',
                    'auto_ets',
                    'theta',
                    'optimized_theta',
                ]
                # Event-seasonal items (e.g., Christmas-only): MA-of-zero through
                # the off-season would forecast 0 post-peak. Keep the bucket
                # lag-heavy and drop the MA level family and both Thetas.
                if any_event_seasonal:
                    keys = [k for k in keys if k not in ('theta', 'optimized_theta')]
                else:
                    # Seasonal bucket: MA6 is half a cycle and mechanically
                    # broken when the forecast origin sits in the off-season
                    # (averages low months, projects low into a peak). Only MA12
                    # belongs here — it averages a full cycle so the forecast is
                    # independent of which season the origin falls in.
                    keys.extend([k for k in ma_keys_long if k == 'ma12'])
                return keys
            if bucket == 'trend':
                keys = [
                    'historic_average',
                    'seasonal_window_average',
                    'auto_ets',
                    'theta',
                    'optimized_theta',
                    'naive',
                ]
                # MA6/MA12 serve as level baselines for mild trends / level shifts.
                keys.extend(ma_keys_long)
                # If we have at least one full season, allow SeasonalNaive even if the
                # seasonal detector didn't put the series into the seasonal bucket yet
                # (common for monthly series with ~13–23 months of history).
                # IMPORTANT: gate on ``season_for_models`` (12 for monthly) rather than
                # ``self.season_length`` so non-monthly panels with a different stored
                # season_length don't accidentally diverge from the actual frequency.
                if int(season_for_models) >= 2 and int(n_obs) >= int(season_for_models) + 1:
                    keys.insert(0, 'seasonal_naive')
                # Strongly trending series: lag-family models can't follow trend, and
                # plain Naive systematically under-reacts. Drop all three — and also
                # drop MA6/MA12 (flat level is a poor substitute for a true trend
                # model), leaving AutoETS / Theta / AutoARIMA to compete.
                if any_strong_trend:
                    keys = [
                        k for k in keys
                        if k not in ('naive', 'seasonal_naive', 'seasonal_window_average', 'ma6', 'ma12')
                    ]
                if n_obs >= min_arima_len:
                    keys.insert(0, 'auto_arima')
                return keys
            # smooth: weak seasonality; MA family is the natural level baseline here
            # because "recent stable run rate" is exactly what smooth series look like.
            keys = [
                'historic_average',
                'seasonal_window_average',
                'auto_ets',
                'theta',
                'optimized_theta',
                'naive',
            ]
            keys.extend(ma_keys_long)
            if n_obs >= min_arima_len:
                keys.insert(0, 'auto_arima')
            return keys

        # Per-uid CV scores (initialized before bucket loop so reranking always has defined maps).
        rmse_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        mae_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        wape_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        bias_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        # Per-window WAPE for stability scaling in the lag-family advantage check.
        # {uid: {model_name: [wape_window_0, wape_window_1, ...]}}.
        wape_per_window_map: dict[str, dict[str, list[float]]] = {uid: {} for uid in counts.index}
        metric_scores: dict[str, dict[str, float]] = {uid: {} for uid in counts.index}
        # Per uid/model: (positive excess, predicted level) pairs from CV holdouts.
        cv_excess_by_uid_model: dict[str, dict[str, list[tuple[float, float]]]] = {str(uid): {} for uid in counts.index}

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

            # Adaptive n_windows: with default n_windows=2 and cv_h_eff=6, monthly
            # series with 30+ months can support 3+ windows, which materially
            # de-noises the ranking. Cap at 4 to bound CV cost. Use the bucket's
            # MIN n_obs so CV runs successfully for every uid in the bucket.
            min_n_bucket = int(counts.loc[uids].min())
            head_room = max(0, (min_n_bucket - bucket_cv_h) // max(1, bucket_cv_h))
            bucket_n_windows = int(min(4, max(int(n_windows), head_room)))
            bucket_n_windows = max(1, bucket_n_windows)

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

            uid_n_obs = counts.loc[uids]
            for model_name, factory in model_specs:
                # Per-uid filter: only run this model on series with enough history
                # to fit every CV training window. Avoids one short series taking
                # down the entire model for a bucket via raised exceptions.
                min_obs_needed = _min_obs_for_model_cv(
                    model_name,
                    season_length=int(season_for_models),
                    cv_h=int(bucket_cv_h),
                    n_windows=int(bucket_n_windows),
                )
                qualified_uids = [
                    str(u) for u in uid_n_obs.index
                    if int(uid_n_obs.loc[u]) >= int(min_obs_needed)
                ]
                if not qualified_uids:
                    continue
                df_model = df_bucket[df_bucket['unique_id'].isin(qualified_uids)]
                if df_model.empty:
                    continue

                try:
                    sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
                    cv = sf_one.cross_validation(
                        df=df_model,
                        h=bucket_cv_h,
                        step_size=bucket_cv_h,
                        n_windows=bucket_n_windows,
                    )

                    if model_name in cv.columns:
                        for uid, g in cv.groupby('unique_id', sort=False):
                            y_cv = g['y'].to_numpy(dtype=float)
                            yhat_cv = g[model_name].to_numpy(dtype=float)
                            pairs = [
                                (max(0.0, float(yi - yhi)), float(yhi))
                                for yi, yhi in zip(y_cv, yhat_cv)
                                if np.isfinite(yi) and np.isfinite(yhi)
                            ]
                            if pairs:
                                cv_excess_by_uid_model[str(uid)][model_name] = pairs

                    if metric_name == 'robust':
                        from utilsforecast.losses import rmse, mae
                        scores = evaluate(cv, metrics=[rmse, mae])
                        # Mean across cutoffs
                        rmse_mean = scores[scores['metric'] == 'rmse'].groupby('unique_id', as_index=True)[model_name].mean()
                        mae_mean = scores[scores['metric'] == 'mae'].groupby('unique_id', as_index=True)[model_name].mean()
                        for uid in qualified_uids:
                            v1 = float(rmse_mean.get(uid, np.inf))
                            v2 = float(mae_mean.get(uid, np.inf))
                            rmse_scores_map[uid][model_name] = v1
                            mae_scores_map[uid][model_name] = v2
                    elif metric_name in ('wape', 'wape_bias'):
                        # Compute per-series WAPE and bias% directly from CV paths.
                        # Also track per-cutoff (per-window) WAPE for stability scaling.
                        if model_name not in cv.columns:
                            continue
                        cutoff_col = 'cutoff' if 'cutoff' in cv.columns else None
                        for uid, g in cv.groupby('unique_id', sort=False):
                            y = g['y'].to_numpy(dtype=float)
                            yhat = g[model_name].to_numpy(dtype=float)
                            wape_v, bias_v = _safe_wape_and_bias(y, yhat)
                            wape_scores_map[str(uid)][model_name] = wape_v
                            bias_scores_map[str(uid)][model_name] = bias_v
                            if cutoff_col is not None:
                                per_window: list[float] = []
                                for _co, gw in g.groupby(cutoff_col, sort=False):
                                    yw = gw['y'].to_numpy(dtype=float)
                                    yhw = gw[model_name].to_numpy(dtype=float)
                                    w_w, _ = _safe_wape_and_bias(yw, yhw)
                                    if np.isfinite(w_w):
                                        per_window.append(float(w_w))
                                if per_window:
                                    wape_per_window_map[str(uid)][model_name] = per_window
                    else:
                        scores = evaluate(cv, metrics=[metric_fn])
                        m = scores[scores['metric'] == metric_name].groupby('unique_id', as_index=True)[model_name].mean()
                        for uid in qualified_uids:
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
                        # Per-window WAPE std for stability scaling of the lag-family
                        # advantage requirement.
                        per_window = wape_per_window_map.get(uid, {})
                        wape_std = pd.Series({
                            m: float(np.std(vs, ddof=1)) if len(vs) >= 2 else 0.0
                            for m, vs in per_window.items()
                        })
                        picked = _pick_model_wape_bias_penalty(
                            wape_vals,
                            bias_vals,
                            rel_eps=0.02,
                            abs_eps=0.005,
                            bias_ok_pct=10.0,
                            bias_scale_pct=20.0,
                            weight=0.25,
                            prefer_seasonal_naive=bool(event_seasonal_uid.get(uid, False)),
                            wape_std_by_model=wape_std if not wape_std.empty else None,
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

        # Post-CV sanity: if the selected model has CV WAPE > 200%, it's unstable
        # for this series. Fall back to the best alternative with finite WAPE.
        if metric_name in ('wape', 'wape_bias'):
            for uid in list(best_by_uid.keys()):
                pick = best_by_uid[uid]
                pick_wape = wape_scores_map.get(uid, {}).get(pick)
                if pick_wape is None or not np.isfinite(float(pick_wape)):
                    continue
                if float(pick_wape) <= 2.0:
                    continue
                alts = {
                    m: v for m, v in wape_scores_map.get(uid, {}).items()
                    if m != pick and np.isfinite(v) and v < float(pick_wape)
                }
                if alts:
                    best_by_uid[uid] = min(alts, key=alts.get)

        # Lag-family models (SeasonalNaive / SeasonalWindowAverage) repeat
        # historical same-season values; drop them when recent history shows a
        # dead tail, YoY collapse, or lag-vs-recent regime mismatch.
        for uid in list(best_by_uid.keys()):
            if not exclude_seasonal_naive_uid.get(uid, False):
                continue
            _rerank_pick_excluding_lag_family(
                uid=str(uid),
                metric_name=metric_name,
                best_by_uid=best_by_uid,
                rmse_scores_map=rmse_scores_map,
                mae_scores_map=mae_scores_map,
                wape_scores_map=wape_scores_map,
                bias_scores_map=bias_scores_map,
                metric_scores=metric_scores,
            )

        # Use the *looser* stability tier for the reranker so noisier-but-real
        # level shifts are caught at selection (not only at the post-forecast
        # floor). Skip lag-family demotion for event-seasonal items.
        _auto_model_maybe_prefer_level_under_stable_tail(
            best_by_uid=best_by_uid,
            stable_tail_uid=stable_tail_loose_uid,
            bucket_by_uid=bucket_by_uid,
            wape_scores_map=wape_scores_map,
            rmse_scores_map=rmse_scores_map,
            mae_scores_map=mae_scores_map,
            metric_name=metric_name,
            event_seasonal_uid=event_seasonal_uid,
            strong_trend_uid=strong_trend_uid,
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

        cv_excess_by_uid: dict[str, dict[str, float]] = {}
        for uid, model_name in best_by_uid.items():
            pairs = cv_excess_by_uid_model.get(str(uid), {}).get(str(model_name), [])
            q = _excess_quantiles_from_values(
                [p[0] for p in pairs], [p[1] for p in pairs]
            )
            if q:
                cv_excess_by_uid[str(uid)] = q

        historical_excess_by_uid: dict[str, dict[str, float]] = {}
        y_hist_by_uid: dict[str, np.ndarray] = {}
        ds_hist_by_uid: dict[str, pd.Series] = {}
        for uid, g in df.groupby('unique_id', sort=False):
            uid_s = str(uid)
            y_hist_by_uid[uid_s] = g['y'].to_numpy(dtype=float)
            ds_hist_by_uid[uid_s] = pd.Series(pd.to_datetime(g['ds']))
            hq = _historical_excess_quantiles(g['y'].to_numpy(dtype=float), g['ds'])
            if hq:
                historical_excess_by_uid[uid_s] = hq

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
                fcst = _statsforecast_forecast_with_conformal(sf_one, subset, h)
            except Exception:
                fcst = sf_one.forecast(df=subset, h=h)
            if model_name not in fcst.columns:
                raise RuntimeError(f"Expected forecast column '{model_name}' not found")
            part = fcst.loc[:, ['unique_id', 'ds']].copy()
            yhat = fcst[model_name].to_numpy(dtype=float)
            part['yhat'] = yhat
            part['model_used'] = model_name
            upper_70, upper_90, upper_95 = _attach_upper_quantiles(
                yhat,
                fcst,
                model_name,
                uid_series=part['unique_id'],
                ds_series=part['ds'],
                cv_excess_by_uid=cv_excess_by_uid,
                historical_excess_by_uid=historical_excess_by_uid,
                y_hist_by_uid=y_hist_by_uid,
                ds_hist_by_uid=ds_hist_by_uid,
            )
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

        # Peak-ratio correction: scale SeasonalNaive's lag-12 forecast by the
        # year-over-year peak ratio for event-seasonal items in the dead zone
        # where SN won the mini-CV. SN by construction repeats last year's
        # value; for items with a clear YoY trend on the peak month (e.g.
        # 35% growth), this nudges the forecast in the trend's direction
        # without abandoning the seasonal lag pattern. Skip if the uid was
        # already marked dead.
        if peak_ratio_correction_uid:
            for uid, ratio in peak_ratio_correction_uid.items():
                if uid in dead_sku_uid:
                    continue
                mask_uid = out['unique_id'] == uid
                if not bool(mask_uid.any()):
                    continue
                # Only apply when SeasonalNaive is actually the model used
                # (the mini-CV's pick can be overridden by downstream rerankers
                # in pathological cases — defensive check).
                model_used = out.loc[mask_uid, 'model_used'].astype(str).iloc[0]
                if model_used != 'SeasonalNaive':
                    continue
                for col in ['yhat', 'upper_70', 'upper_90', 'upper_95']:
                    if col in out.columns:
                        out.loc[mask_uid, col] = (
                            out.loc[mask_uid, col].astype(float) * float(ratio)
                        )
                # Tag the model label so the correction is visible in the
                # picked-model output.
                out.loc[mask_uid, 'model_used'] = (
                    out.loc[mask_uid, 'model_used'].astype(str) + ':peak_ratio'
                )

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

        # Post-forecast sanity floor: uses a looser stability criterion
        # (max_cv=0.70) than the level-prefer reranking (0.55). The floor only
        # needs a rough demand level, so we accept noisier tails here.
        floor_ratio = 0.15
        floor_min_tail = 5.0
        for uid in out['unique_id'].unique():
            if not floor_eligible_uid.get(uid, False):
                continue
            tm = float(floor_eligible_mean_uid.get(uid, 0.0))
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
        df = _trim_panel_pre_gap(df, freq=str(self.freq))
        if df.empty:
            raise ValueError('Empty history after pre-gap trim')

        # Build a mapping from model class name -> factory.
        factories = {name: factory for name, factory in _build_candidate_model_factories(int(self.season_length))}

        by_model: dict[str, list[str]] = {}
        for uid in df['unique_id'].unique().tolist():
            model_name = model_by_uid.get(str(uid), 'Naive')
            by_model.setdefault(model_name, []).append(str(uid))

        historical_excess_by_uid: dict[str, dict[str, float]] = {}
        y_hist_by_uid: dict[str, np.ndarray] = {}
        ds_hist_by_uid: dict[str, pd.Series] = {}
        for uid, g in df.groupby('unique_id', sort=False):
            uid_s = str(uid)
            y_hist_by_uid[uid_s] = g['y'].to_numpy(dtype=float)
            ds_hist_by_uid[uid_s] = pd.Series(pd.to_datetime(g['ds']))
            hq = _historical_excess_quantiles(g['y'].to_numpy(dtype=float), g['ds'])
            if hq:
                historical_excess_by_uid[uid_s] = hq

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
                fcst = _statsforecast_forecast_with_conformal(sf_one, subset, h)
            except Exception:
                fcst = sf_one.forecast(df=subset, h=h)
            if model_name not in fcst.columns:
                raise RuntimeError(f"Expected forecast column '{model_name}' not found")
            part = fcst.loc[:, ['unique_id', 'ds']].copy()
            yhat = fcst[model_name].to_numpy(dtype=float)
            part['yhat'] = yhat
            part['model_used'] = model_name
            upper_70, upper_90, upper_95 = _attach_upper_quantiles(
                yhat,
                fcst,
                model_name,
                uid_series=part['unique_id'],
                ds_series=part['ds'],
                historical_excess_by_uid=historical_excess_by_uid,
                y_hist_by_uid=y_hist_by_uid,
                ds_hist_by_uid=ds_hist_by_uid,
            )
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
