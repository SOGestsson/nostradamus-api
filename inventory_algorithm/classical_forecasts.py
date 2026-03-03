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
    seasonal_naive_min_wape_advantage: float = 0.015,
    bias_ok_pct: float = 10.0,
    bias_scale_pct: float = 20.0,
    weight: float = 0.25,
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
    seasonal_name = 'SeasonalNaive'
    if seasonal_name in w_close.index:
        adaptive = [m for m in ['AutoETS', 'Theta', 'OptimizedTheta', 'AutoARIMA'] if m in w_close.index]
        if adaptive:
            best_adaptive = float(w_close.loc[adaptive].min())
            seasonal_wape = float(w_close.loc[seasonal_name])
            # SeasonalNaive advantage is how much lower its WAPE is.
            advantage = best_adaptive - seasonal_wape
            if advantage < float(seasonal_naive_min_wape_advantage):
                # Choose the best adaptive model by the same penalized score.
                score_adaptive = score.loc[adaptive]
                picked_adaptive = (
                    pd.DataFrame({'score': score_adaptive, 'wape': w_close.loc[adaptive], 'abs_bias': b_close.loc[adaptive]})
                    .sort_values(['score', 'wape', 'abs_bias'], ascending=True)
                    .index[0]
                )
                return str(picked_adaptive)

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


def _regularize_panel_time_index(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Return a panel with a complete time index per unique_id.

    Many StatsForecast models expect a regular time grid. Real inventory history
    often has missing timestamps (no transactions recorded). For demand series,
    missing periods should typically be treated as 0.

    Input df must have columns: ['unique_id','ds','y'].
    """
    if df.empty:
        return df
    out_parts: list[pd.DataFrame] = []
    for uid, g in df.groupby('unique_id', sort=False):
        g = g.sort_values('ds')
        start = pd.to_datetime(g['ds'].iloc[0])
        end = pd.to_datetime(g['ds'].iloc[-1])
        full = pd.date_range(start=start, end=end, freq=freq)
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

class ClassicalForecasts:
    """
    Plug-in forecaster with two modes:
      - 'timegpt'   -> Nixtla TimeGPT (cloud API)
      - 'local'     -> StatsForecast classical models (AutoARIMA/ETS/Croston/ADIDA etc.)
    Returns monthly forecast arrays compatible with your simulator.

    Conventions:
      - Input history is a DataFrame with columns: ['day', 'actual_sale', 'item_id']
            - Frequency 'M' for monthly data (month end)
    """

    def __init__(self,
                 mode: str = 'timegpt',
                 api_key: str | None = None,
                 model: str | None = None,     # e.g., 'timegpt-1', 'timegpt-1-long-horizon'
                 quantiles: list[float] | None = None,   # e.g., [0.1,0.5,0.8,0.95]
                 local_model: str = 'auto_arima',    # 'auto_arima'|'auto_ets'|'croston_optimized'|'adida'|'theta'
                 season_length: int = 12,  # Seasonality period (12=yearly cycle in monthly data)
                 freq: str = 'M',  # Pandas freq: 'M'=monthly, 'D'=daily, 'W'=weekly
                 ):
        self.mode = mode
        self.quantiles = quantiles or []
        self.model_name = model
        # Backwards compatibility: accept month-start shorthand.
        # NOTE: Keep month-start as 'MS' (do not convert to month-end).
        self.freq = 'MS' if (freq or '').strip().lower() == 'ms' else freq
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
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df, id_map

    def auto_model_forecast_panel(
        self,
        hist: pd.DataFrame,
        h: int,
        metric: str = 'robust',
        cv_h: Optional[int] = None,
        n_windows: int = 1,
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

        freq_upper = str(self.freq or '').strip().upper()
        is_monthly = freq_upper in ('M', 'ME', 'MS') or freq_upper.startswith('M')
        # For monthly series we always assume yearly seasonality (12) for model configuration.
        season_for_models = 12 if is_monthly else int(self.season_length)

        cv_h_eff = int(cv_h) if cv_h is not None else int(min(h, max(1, season_for_models)))
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

        for uid, n_obs in counts.items():
            y = df.loc[df['unique_id'] == uid, 'y'].to_numpy(dtype=float)
            prof = _series_profile(y)
            bucket = _bucket_series(prof, season_length=season_for_models, min_arima_len=min_arima_len)

            # Heuristic selections for very short series (skip CV entirely)
            if bucket == 'short' or int(n_obs) < min_len:
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
                # - 12-23 months: SeasonalWindowAverage is typically a better baseline than Naive.
                # - Otherwise: simple moving average over all available months -> HistoricAverage.
                if is_monthly and 12 <= int(n_obs) <= 23:
                    best_by_uid[uid] = 'SeasonalWindowAverage'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': 'SeasonalWindowAverage',
                            'bucket': bucket,
                            'n_obs': int(n_obs),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                    continue

                if is_monthly and int(n_obs) < 12:
                    best_by_uid[uid] = 'HistoricAverage'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': 'HistoricAverage',
                            'bucket': bucket,
                            'n_obs': int(n_obs),
                            'min_len': int(min_len),
                            'season_for_models': int(season_for_models),
                            'cv_h_eff': int(cv_h_eff),
                            'n_windows': int(n_windows),
                        }
                    continue

                if int(season_for_models) >= 2 and int(n_obs) >= int(season_for_models) + 1:
                    best_by_uid[uid] = 'SeasonalNaive'
                    if debug:
                        debug_reason[uid] = {
                            'reason': 'short_or_insufficient_len',
                            'picked': 'SeasonalNaive',
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

        def _candidate_keys_for_bucket(bucket: str, n_obs: int) -> list[str]:
            # Keep candidate sets small for speed.
            if bucket == 'intermittent':
                # Intermittent series: prefer intermittent-demand models (+ seasonal naive).
                # NOTE: We intentionally exclude plain Naive here because it tends to win
                # error metrics on mostly-zero holdout windows by predicting 0, which is
                # often not the desired behaviour for replenishment planning.
                return ['croston_optimized', 'adida', 'seasonal_naive']
            if bucket == 'seasonal':
                # Seasonal series: bias away from Naive so we actually test seasonal models.
                return ['seasonal_naive', 'auto_ets', 'theta', 'optimized_theta']
            if bucket == 'trend':
                keys = ['auto_ets', 'theta', 'optimized_theta', 'naive']
                # If we have at least one full season, allow SeasonalNaive even if the
                # seasonal detector didn't put the series into the seasonal bucket yet
                # (common for monthly series with ~13-23 months of history).
                if int(self.season_length) >= 2 and int(n_obs) >= int(self.season_length) + 1:
                    keys.insert(0, 'seasonal_naive')
                if n_obs >= min_arima_len:
                    keys.insert(0, 'auto_arima')
                return keys
            # smooth
            keys = ['auto_ets', 'theta', 'optimized_theta', 'naive']
            if int(self.season_length) >= 2 and int(n_obs) >= int(self.season_length) + 1:
                keys.insert(0, 'seasonal_naive')
            if n_obs >= min_arima_len:
                keys.insert(0, 'auto_arima')
            return keys

        # Score each bucket with per-model CV (robust to individual model failures).
        for bucket, uids in buckets.items():
            df_bucket = df[df['unique_id'].isin(uids)]
            if df_bucket.empty:
                continue

            # Determine max n_obs in bucket to decide if AutoARIMA is allowed.
            max_n = int(counts.loc[uids].max())
            model_specs = _build_model_factories_for_keys(
                _candidate_keys_for_bucket(bucket, max_n),
                season_length=int(season_for_models),
            )
            if not model_specs:
                continue

            # Collect per-uid per-model scores
            metric_scores: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            rmse_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            mae_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            wape_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            bias_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}

            for model_name, factory in model_specs:
                try:
                    sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
                    cv = sf_one.cross_validation(
                        df=df_bucket,
                        h=cv_h_eff,
                        step_size=cv_h_eff,
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
                        # WAPE is the main focus. If WAPE is close, penalize excessive |bias|.
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
        out['upper_95'] = out['upper_95'].clip(lower=out['yhat'])
        out['upper_90'] = out['upper_90'].clip(lower=out['yhat'], upper=out['upper_95'])
        out['upper_70'] = out['upper_70'].clip(lower=out['yhat'], upper=out['upper_90'])
        return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    def auto_model_forecast_single(
        self,
        item_hist: pd.DataFrame,
        h: int,
        metric: str = 'robust',
        cv_h: Optional[int] = None,
        n_windows: int = 1,
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
