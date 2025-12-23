# classical_forecasts.py (or drop into your existing module)
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Callable, Optional

try:
    from nixtla import NixtlaClient
    _HAS_TIMEGPT = True
except Exception:
    _HAS_TIMEGPT = False

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
    }


def _metric_func_from_name(name: str) -> tuple[str, Optional[Callable]]:
    """Map a user-friendly metric string to utilsforecast loss function.

    Special values:
    - 'robust' / 'rank' / 'rmse+mae': rank-aggregate RMSE and MAE per series.
    """
    from utilsforecast.losses import rmse, mae

    metric = (name or '').strip().lower()
    if metric in ('robust', 'rank', 'rmse+mae', 'rmse_mae'):
        return 'robust', None
    if metric in ('rmse',):
        return 'rmse', rmse
    if metric in ('mae',):
        return 'mae', mae
    raise ValueError("Unsupported metric. Use 'rmse', 'mae', or 'robust'.")


def _build_candidate_model_factories(season_length: int) -> list[tuple[str, Callable[[], object]]]:
    """Candidate StatsForecast model factories (explicitly excludes TimeGPT/LightGPT)."""
    models_dict = _lazy_import_nixtla_models()
    # NOTE: AutoCES is intentionally excluded from the auto-model candidate set
    # because it can fail to fit for some series and would otherwise abort the
    # entire cross-validation run.
    seasonal = {'auto_arima', 'auto_ets', 'seasonal_naive', 'theta', 'optimized_theta'}
    keys = ['naive', 'seasonal_naive', 'auto_arima', 'auto_ets', 'theta', 'optimized_theta', 'croston_optimized', 'adida']
    specs: list[tuple[str, Callable[[], object]]] = []
    for key in keys:
        ModelClass = models_dict[key]
        if key in seasonal:
            specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
        else:
            specs.append((ModelClass.__name__, lambda cls=ModelClass: cls()))
    return specs


def _build_model_factories_for_keys(keys: list[str], season_length: int) -> list[tuple[str, Callable[[], object]]]:
    models_dict = _lazy_import_nixtla_models()
    seasonal = {'auto_arima', 'auto_ets', 'seasonal_naive', 'theta', 'optimized_theta'}
    specs: list[tuple[str, Callable[[], object]]] = []
    for key in keys:
        if key not in models_dict:
            continue
        ModelClass = models_dict[key]
        if key in seasonal:
            specs.append((ModelClass.__name__, lambda cls=ModelClass: cls(season_length=season_length)))
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


def _bucket_series(profile: dict[str, float], season_length: int, min_arima_len: int) -> str:
    n = int(profile['n'])
    zero_frac = float(profile['zero_frac'])
    adi = float(profile['adi'])
    cv2 = float(profile['cv2'])
    trend_corr = float(profile['trend_corr'])

    # Very short histories: skip CV and skip AutoARIMA.
    if n < max(20, season_length + 5):
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
      - Frequency 'MS' for monthly data (month start)
    """

    def __init__(self,
                 mode: str = 'timegpt',
                 api_key: str | None = None,
                 model: str | None = None,     # e.g., 'timegpt-1', 'timegpt-1-long-horizon'
                 quantiles: list[float] | None = None,   # e.g., [0.1,0.5,0.8,0.95]
                 local_model: str = 'auto_arima',    # 'auto_arima'|'auto_ets'|'croston_optimized'|'adida'|'theta'
                 season_length: int = 12,  # Seasonality period (12=yearly cycle in monthly data)
                 freq: str = 'MS',  # Pandas freq: 'MS'=monthly, 'D'=daily, 'W'=weekly
                 ):
        self.mode = mode
        self.quantiles = quantiles or []
        self.model_name = model
        self.freq = freq
        self._client = None
        self.local_model = local_model
        self.season_length = season_length

        if self.mode == 'timegpt':
            if not _HAS_TIMEGPT:
                raise RuntimeError("Nixtla 'nixtla' package not available. Install with `pip install nixtla`.")
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
        Returns DataFrame with 'ds' and 'yhat'.
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
        
        # Fit and forecast
        sf.fit(df)
        fcst = sf.forecast(h=h, df=df)
        
        # Build output DataFrame
        last_ds = pd.to_datetime(df['ds'].iloc[-1])
        future_ds = pd.date_range(start=last_ds, periods=h+1, freq=self.freq)[1:]  # Skip first (it's last_ds)
        
        # Get forecast column
        forecast_cols = [col for col in fcst.columns if col not in ['unique_id', 'ds']]
        if not forecast_cols:
            raise ValueError(f"No forecast column found in output for {self.local_model}")
        
        forecast_col = forecast_cols[0]
        
        out = pd.DataFrame({
            'ds': future_ds,
            'yhat': fcst[forecast_col].to_numpy(dtype=float)
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
    ) -> pd.DataFrame:
        """Select best StatsForecast model per series and forecast.

        Returns DataFrame with columns: ['unique_id', 'ds', 'yhat', 'model_used'].
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

        metric_name, metric_fn = _metric_func_from_name(metric)
        cv_h_eff = int(cv_h) if cv_h is not None else int(min(h, max(1, self.season_length)))
        cv_h_eff = max(1, cv_h_eff)

        # Heuristic tuning knobs (kept internal to avoid API churn)
        min_arima_len = max(50, 3 * max(1, int(self.season_length)))

        counts = df.groupby('unique_id', as_index=True).size()
        min_len = (cv_h_eff * max(1, int(n_windows))) + 2

        # Bucket series to choose smaller candidate sets (speed) and avoid
        # obviously-wrong models (accuracy).
        buckets: dict[str, list[str]] = {}
        best_by_uid: dict[str, str] = {uid: 'Naive' for uid in counts.index.tolist()}

        for uid, n_obs in counts.items():
            y = df.loc[df['unique_id'] == uid, 'y'].to_numpy(dtype=float)
            prof = _series_profile(y)
            bucket = _bucket_series(prof, season_length=int(self.season_length), min_arima_len=min_arima_len)

            # Heuristic selections for very short series (skip CV entirely)
            if bucket == 'short' or int(n_obs) < min_len:
                if int(self.season_length) >= 2 and int(n_obs) >= int(self.season_length) + 1 and bucket != 'intermittent':
                    best_by_uid[uid] = 'SeasonalNaive'
                else:
                    best_by_uid[uid] = 'Naive'
                continue

            buckets.setdefault(bucket, []).append(uid)

        def _candidate_keys_for_bucket(bucket: str, n_obs: int) -> list[str]:
            # Keep candidate sets small for speed.
            if bucket == 'intermittent':
                return ['croston_optimized', 'adida', 'naive']
            if bucket == 'seasonal':
                return ['seasonal_naive', 'auto_ets', 'theta', 'optimized_theta', 'naive']
            if bucket == 'trend':
                keys = ['auto_ets', 'theta', 'optimized_theta', 'naive']
                if n_obs >= min_arima_len:
                    keys.insert(0, 'auto_arima')
                return keys
            # smooth
            keys = ['auto_ets', 'theta', 'optimized_theta', 'naive']
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
                season_length=int(self.season_length),
            )
            if not model_specs:
                continue

            # Collect per-uid per-model scores
            metric_scores: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            rmse_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}
            mae_scores_map: dict[str, dict[str, float]] = {uid: {} for uid in uids}

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
                else:
                    if not metric_scores[uid]:
                        continue
                    best_by_uid[uid] = str(min(metric_scores[uid].items(), key=lambda kv: kv[1])[0])

        # Forecast per chosen model in batches.
        by_model: dict[str, list[str]] = {}
        for uid, model_name in best_by_uid.items():
            by_model.setdefault(model_name, []).append(uid)

        parts: list[pd.DataFrame] = []
        # Use the full set of factories so we can forecast whatever was selected
        # in any bucket.
        all_model_specs = _build_candidate_model_factories(self.season_length)
        for model_name, factory in all_model_specs:
            uids = by_model.get(model_name)
            if not uids:
                continue
            sf_one = StatsForecast(models=[factory()], freq=self.freq, n_jobs=1)
            fcst = sf_one.forecast(df=df[df['unique_id'].isin(uids)], h=h)
            if model_name not in fcst.columns:
                raise RuntimeError(f"Expected forecast column '{model_name}' not found")
            part = fcst.loc[:, ['unique_id', 'ds']].copy()
            part['yhat'] = fcst[model_name].to_numpy(dtype=float)
            part['model_used'] = model_name
            parts.append(part)

        if not parts:
            raise RuntimeError('Failed to generate forecasts for any series')

        out = pd.concat(parts, ignore_index=True)
        out['yhat'] = out['yhat'].clip(lower=0.0)
        return out.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    def auto_model_forecast_single(
        self,
        item_hist: pd.DataFrame,
        h: int,
        metric: str = 'robust',
        cv_h: Optional[int] = None,
        n_windows: int = 1,
    ) -> tuple[np.ndarray, str]:
        """Auto-select a model for a single series; returns (forecast_path, model_used)."""
        df = item_hist.rename(columns={'day': 'ds', 'actual_sale': 'y'}).loc[:, ['ds', 'y']].copy()
        df['unique_id'] = 'item'
        panel = self.auto_model_forecast_panel(df, h=h, metric=metric, cv_h=cv_h, n_windows=n_windows)
        model_used = str(panel['model_used'].iloc[0])
        path = panel['yhat'].to_numpy(dtype=float)
        return path, model_used

    # ---------- Public: daily path ----------
    def daily_path(self, item_hist: pd.DataFrame, periods: int) -> np.ndarray:
        """
        Returns a length=periods np.array of forecasts (float),
        aligned to the next period after the last 'day' in item_hist.
        """
        if self.mode == 'timegpt':
            fcst = self._timegpt_forecast_path(item_hist, periods)
        else:
            if self.local_model in ('auto_model', 'automodel'):
                path, _ = self.auto_model_forecast_single(item_hist, h=periods)
                return np.maximum(path, 0.0)
            fcst = self._local_forecast_path(item_hist, periods)

        path = fcst['yhat'].to_numpy(dtype=float)
        return np.maximum(path, 0.0)

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

        path = self.daily_path(item_hist, periods)
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
