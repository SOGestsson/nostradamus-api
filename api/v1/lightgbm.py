"""LightGBM endpoints.

These endpoints are intentionally separate (like LightGPT), because LightGBM:
- trains locally on provided history
- stores artifacts + metadata in a DuckDB model store
- serves batch forecasts for many items
"""

from __future__ import annotations

import os
import traceback
import uuid
import time
import threading
from typing import Any

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
import httpx
import json
import hmac
import hashlib

from api.models import LightGBMBatchForecastRequest, LightGBMTrainRequest
from inventory_algorithm.lightgbm_forecasts import LightGBMForecast


router = APIRouter()


def _debug_log(payload: dict[str, Any]) -> None:
    try:
        with open("/Users/palmipetursson/Dropbox/Nostradamus/nostradamus-api/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass

# Background job store (Redis preferred; in-memory fallback)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
try:
    import redis

    _redis_sync = redis.Redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    _redis_sync = None

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _job_key(job_id: str) -> str:
    return f"lgbm_job:{job_id}"


def _now_ts() -> int:
    return int(time.time())


def _set_job(job_id: str, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload.setdefault('updated_at', _now_ts())
    if _redis_sync is not None:
        try:
            _redis_sync.hset(_job_key(job_id), mapping={k: json.dumps(v) for k, v in payload.items()})
            _redis_sync.expire(_job_key(job_id), 60 * 60 * 24)
            return
        except Exception:
            # Fall back to in-memory storage if Redis is unavailable.
            pass
    with _JOBS_LOCK:
        _JOBS[job_id] = payload


def _update_job(job_id: str, **fields: Any) -> None:
    fields = dict(fields)
    fields['updated_at'] = _now_ts()
    if _redis_sync is not None:
        try:
            _redis_sync.hset(_job_key(job_id), mapping={k: json.dumps(v) for k, v in fields.items()})
            _redis_sync.expire(_job_key(job_id), 60 * 60 * 24)
            return
        except Exception:
            # Fall back to in-memory storage if Redis is unavailable.
            pass
    with _JOBS_LOCK:
        cur = _JOBS.get(job_id, {})
        cur.update(fields)
        _JOBS[job_id] = cur


def _get_job(job_id: str) -> dict[str, Any] | None:
    if _redis_sync is not None:
        try:
            raw = _redis_sync.hgetall(_job_key(job_id))
            if not raw:
                # Fall back to in-memory if not in Redis
                with _JOBS_LOCK:
                    return dict(_JOBS.get(job_id)) if job_id in _JOBS else None
            out: dict[str, Any] = {}
            for k, v in raw.items():
                try:
                    out[k] = json.loads(v)
                except Exception:
                    out[k] = v
            return out
        except Exception:
            # Fall back to in-memory storage if Redis is unavailable
            pass
    with _JOBS_LOCK:
        return dict(_JOBS.get(job_id)) if job_id in _JOBS else None


def _send_webhook_with_retries(job_id: str, webhook: str, payload: dict[str, Any], max_attempts: int = 5) -> bool:
    delays = [1, 5, 30, 120, 600]
    attempt = 0
    last_exc: str | None = None
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    secret_bytes = webhook_secret.encode() if webhook_secret else None

    while attempt < max_attempts:
        attempt += 1
        try:
            payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode()
            headers: dict[str, str] = {'Content-Type': 'application/json'}
            if secret_bytes:
                ts = str(int(time.time()))
                signed = hmac.new(secret_bytes, ts.encode() + b'.' + payload_bytes, hashlib.sha256).hexdigest()
                headers['X-Signature'] = f"sha256={signed}"
                headers['X-Signature-Timestamp'] = ts

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(webhook, content=payload_bytes, headers=headers)
            if 200 <= resp.status_code < 300:
                _update_job(job_id, webhook_attempts=attempt)
                return True
            last_exc = f"status={resp.status_code}, body={resp.text}"
        except Exception as e:
            last_exc = str(e)

        _update_job(job_id, webhook_attempts=attempt, webhook_last_error=last_exc)
        time.sleep(delays[min(attempt - 1, len(delays) - 1)])

    return False


def _default_store_root() -> str:
    return os.getenv("NOSTRADAMUS_STORE_ROOT", "./nostradamus_store")


@router.post("/train")
def train_lightgbm(request: LightGBMTrainRequest):
    try:
        df_hist = pd.DataFrame(request.sim_input_his)
        # item_id is an opaque identifier (often non-numeric)
        if "item_id" in df_hist.columns:
            df_hist["item_id"] = df_hist["item_id"].astype(str)
        df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
        df_drivers = pd.DataFrame(request.external_drivers) if request.external_drivers else None

        forecaster = LightGBMForecast(store_root=request.store_root or _default_store_root(), customer_id=request.customer_id)
        result = forecaster.train_and_register(
            df_hist,
            item_attributes=df_items,
            drivers=df_drivers,
            freq=request.freq,
            horizon=request.forecast_periods,
            exogenous_columns=request.exogenous_columns,
            detrend_method=getattr(request, "detrend_method", "none"),
            model_version=request.model_version,
            status=request.status,
            notes=request.notes,
            min_history_points=request.min_history_points,
            min_improvement=request.min_improvement,
            lgbm_min_data_in_leaf=getattr(request, "lgbm_min_data_in_leaf", 50),
            lgbm_min_data_in_bin=getattr(request, "lgbm_min_data_in_bin", 1),
            tune_hyperparameters=getattr(request, "tune_hyperparameters", False),
            optuna_n_trials=getattr(request, "optuna_n_trials", 30),
            optuna_timeout=getattr(request, "optuna_timeout", 300),
        )

        return {
            "customer_id": result.customer_id,
            "model_version": result.model_version,
            "status": result.status,
            "artifact_root": result.artifact_root,
            "feature_spec_hash": result.feature_spec_hash,
            "items_trained": result.items_trained,
            "rows_trained": result.rows_trained,
            "metrics_summary": result.metrics_summary,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightGBM train error: {str(e)}")


@router.post("/train_async")
def train_lightgbm_async(
    request: LightGBMTrainRequest,
    background_tasks: BackgroundTasks,
    webhook_url: str | None = None,
):
    """Start LightGBM training as a background job.

    Returns a `job_id` immediately; poll `/jobs/{job_id}` for progress.
    Optionally sends a completion webhook.
    """

    job_id = str(uuid.uuid4())
    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "lightgbm_train_async",
        "hypothesisId": "J1",
        "location": "api/v1/lightgbm.py:train_async:entry",
        "message": "Train_async request received",
        "data": {
            "job_id": job_id,
            "customer_id": request.customer_id,
            "status": request.status,
            "forecast_periods": int(request.forecast_periods),
            "sim_rows": len(request.sim_input_his or []),
            "item_attr_rows": len(request.item_attributes or []),
            "external_driver_rows": len(request.external_drivers or []),
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion
    _set_job(
        job_id,
        {
            'status': 'queued',
            'phase': 'queued',
            'started_at': _now_ts(),
            'customer_id': request.customer_id,
            'model_version': request.model_version,
            'total_horizons': int(request.forecast_periods),
        },
    )

    def _runner() -> None:
        try:
            _update_job(job_id, status='running', phase='starting')

            df_hist = pd.DataFrame(request.sim_input_his)
            if "item_id" in df_hist.columns:
                df_hist["item_id"] = df_hist["item_id"].astype(str)
            df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
            df_drivers = pd.DataFrame(request.external_drivers) if request.external_drivers else None
            # #region agent log
            _debug_log({
                "sessionId": "debug-session",
                "runId": "lightgbm_train_async",
                "hypothesisId": "J2",
                "location": "api/v1/lightgbm.py:train_async:runner_df",
                "message": "Train_async built dataframes",
                "data": {
                    "job_id": job_id,
                    "hist_rows": int(getattr(df_hist, "shape", [0, 0])[0]),
                    "hist_cols": int(getattr(df_hist, "shape", [0, 0])[1]),
                    "item_rows": int(getattr(df_items, "shape", [0, 0])[0]) if df_items is not None else 0,
                    "driver_rows": int(getattr(df_drivers, "shape", [0, 0])[0]) if df_drivers is not None else 0,
                },
                "timestamp": int(time.time() * 1000),
            })
            # #endregion

            forecaster = LightGBMForecast(store_root=request.store_root or _default_store_root(), customer_id=request.customer_id)

            def progress_hook(evt: dict[str, Any]) -> None:
                # evt includes phase + optional current_horizon, etc.
                fields = dict(evt)
                if 'phase' in fields:
                    _update_job(job_id, **fields)
                else:
                    _update_job(job_id, phase='running', **fields)

            result = forecaster.train_and_register(
                df_hist,
                item_attributes=df_items,
                drivers=df_drivers,
                freq=request.freq,
                horizon=request.forecast_periods,
                exogenous_columns=request.exogenous_columns,
                detrend_method=getattr(request, "detrend_method", "none"),
                model_version=request.model_version,
                status=request.status,
                notes=request.notes,
                min_history_points=request.min_history_points,
                min_improvement=request.min_improvement,
                lgbm_min_data_in_leaf=getattr(request, "lgbm_min_data_in_leaf", 50),
                lgbm_min_data_in_bin=getattr(request, "lgbm_min_data_in_bin", 1),
                tune_hyperparameters=getattr(request, "tune_hyperparameters", False),
                optuna_n_trials=getattr(request, "optuna_n_trials", 30),
                optuna_timeout=getattr(request, "optuna_timeout", 300),
                progress_hook=progress_hook,
            )

            _update_job(
                job_id,
                status='finished',
                phase='done',
                finished_at=_now_ts(),
                model_version=result.model_version,
                artifact_root=result.artifact_root,
                feature_spec_hash=result.feature_spec_hash,
                items_trained=result.items_trained,
                rows_trained=result.rows_trained,
                metrics_summary=result.metrics_summary,
            )

            if webhook_url:
                _send_webhook_with_retries(
                    job_id,
                    webhook_url,
                    {
                        'job_id': job_id,
                        'status': 'finished',
                        'customer_id': request.customer_id,
                        'model_version': result.model_version,
                        'result': {
                            'artifact_root': result.artifact_root,
                            'feature_spec_hash': result.feature_spec_hash,
                            'items_trained': result.items_trained,
                            'rows_trained': result.rows_trained,
                            'metrics_summary': result.metrics_summary,
                        },
                    },
                )

        except Exception as e:
            # #region agent log
            _debug_log({
                "sessionId": "debug-session",
                "runId": "lightgbm_train_async",
                "hypothesisId": "J3",
                "location": "api/v1/lightgbm.py:train_async:runner_error",
                "message": "Train_async runner failed",
                "data": {
                    "job_id": job_id,
                    "error": str(e),
                },
                "timestamp": int(time.time() * 1000),
            })
            # #endregion
            _update_job(job_id, status='failed', phase='failed', finished_at=_now_ts(), error=str(e), traceback=traceback.format_exc())
            if webhook_url:
                _send_webhook_with_retries(
                    job_id,
                    webhook_url,
                    {
                        'job_id': job_id,
                        'status': 'failed',
                        'customer_id': request.customer_id,
                        'error': str(e),
                    },
                )

    background_tasks.add_task(_runner)

    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "lightgbm_train_async",
        "hypothesisId": "J4",
        "location": "api/v1/lightgbm.py:train_async:return",
        "message": "Train_async response queued",
        "data": {
            "job_id": job_id,
            "status_url": f"/api/v1/lightgbm/jobs/{job_id}",
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion
    return {
        'job_id': job_id,
        'status': 'queued',
        'status_url': f"/api/v1/lightgbm/jobs/{job_id}",
    }


@router.get("/jobs/{job_id}")
def get_lightgbm_job(job_id: str):
    # #region agent log
    _debug_log({
        "sessionId": "debug-session",
        "runId": "lightgbm_job_get",
        "hypothesisId": "J5",
        "location": "api/v1/lightgbm.py:get_job:entry",
        "message": "Job status request received",
        "data": {
            "job_id": job_id,
            "redis_enabled": bool(_redis_sync),
        },
        "timestamp": int(time.time() * 1000),
    })
    # #endregion
    try:
        job = _get_job(job_id)
        if not job:
            # #region agent log
            _debug_log({
                "sessionId": "debug-session",
                "runId": "lightgbm_job_get",
                "hypothesisId": "J6",
                "location": "api/v1/lightgbm.py:get_job:not_found",
                "message": "Job not found",
                "data": {
                    "job_id": job_id,
                },
                "timestamp": int(time.time() * 1000),
            })
            # #endregion
            raise HTTPException(status_code=404, detail='job not found')
        # #region agent log
        _debug_log({
            "sessionId": "debug-session",
            "runId": "lightgbm_job_get",
            "hypothesisId": "J7",
            "location": "api/v1/lightgbm.py:get_job:success",
            "message": "Job status returned",
            "data": {
                "job_id": job_id,
                "status": job.get("status"),
                "phase": job.get("phase"),
            },
            "timestamp": int(time.time() * 1000),
        })
        # #endregion
        return job
    except HTTPException:
        raise
    except Exception as e:
        _debug_log({
            "sessionId": "debug-session",
            "runId": "lightgbm_job_get",
            "hypothesisId": "J8",
            "location": "api/v1/lightgbm.py:get_job:error",
            "message": f"Error fetching job: {str(e)}",
            "data": {"job_id": job_id, "error": str(e), "traceback": traceback.format_exc()},
            "timestamp": int(time.time() * 1000),
        })
        raise HTTPException(status_code=500, detail=f"Error fetching job: {str(e)}")


@router.post("/batch")
def batch_forecast_lightgbm(request: LightGBMBatchForecastRequest):
    """Batch forecast for many items.

    Uses the active model version in DuckDB (status='prod' by default), unless
    `model_version` is provided.

    Routing:
    - if eligibility says ML allowed: predict via LightGBM
    - else: naive fallback (repeat last value)

    (We can extend fallback to StatsForecast models once you decide what the default should be.)
    """

    try:
        # #region agent log
        try:
            with open("/Users/palmipetursson/Dropbox/Nostradamus/nostradamus-api/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "lightgbm_batch",
                    "hypothesisId": "H1",
                    "location": "api/v1/lightgbm.py:batch:entry",
                    "message": "Batch request received",
                    "data": {
                        "customer_id": request.customer_id,
                        "status": request.status,
                        "model_version": request.model_version,
                        "store_root": request.store_root,
                        "sim_rows": len(request.sim_input_his or []),
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion
        df_hist = pd.DataFrame(request.sim_input_his)
        df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
        df_drivers = pd.DataFrame(request.external_drivers) if request.external_drivers else None

        forecaster = LightGBMForecast(store_root=request.store_root or _default_store_root(), customer_id=request.customer_id)

        # If we want eligibility-based routing, pull it first
        model_version = request.model_version
        if model_version is None:
            model_version = forecaster.store.get_active_model_version(status=request.status)
        if not model_version:
            raise ValueError(f"No active model version found for status='{request.status}'")

        # #region agent log
        try:
            with open("/Users/palmipetursson/Dropbox/Nostradamus/nostradamus-api/.cursor/debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "lightgbm_batch",
                    "hypothesisId": "H2",
                    "location": "api/v1/lightgbm.py:batch:model_version",
                    "message": "Resolved model_version for batch",
                    "data": {
                        "resolved_model_version": model_version,
                        "status": request.status,
                        "store_root": request.store_root or _default_store_root(),
                    },
                    "timestamp": int(time.time() * 1000),
                }) + "\n")
        except Exception:
            pass
        # #endregion

        unique_ids = [str(x) for x in df_hist["item_id"].astype(str).unique().tolist()]
        eligibility = forecaster.store.get_eligibility(model_version=model_version, unique_ids=unique_ids)

        # ML batch prediction for all; we'll decide per item whether to use it.
        # NOTE: Keep sync /batch fast to avoid Cloudflare timeouts.
        # Deep SHAP should run via /batch_async.
        fcst_df, meta = forecaster.batch_forecast(
            df_hist,
            forecast_periods=request.forecast_periods,
            freq=request.freq,
            item_attributes=df_items,
            drivers=df_drivers,
            exogenous_columns=request.exogenous_columns,
            model_version=model_version,
            status=request.status,
            deep_shap=False,
        )

        # Build naive fallback
        df_hist["day"] = pd.to_datetime(df_hist["day"], errors="coerce")
        df_hist = df_hist.dropna(subset=["item_id", "day", "actual_sale"])
        df_hist["item_id"] = df_hist["item_id"].astype(str)
        last_vals = (
            df_hist.sort_values(["item_id", "day"], kind="mergesort")
            .groupby("item_id", sort=False)
            .tail(1)
            .set_index("item_id")["actual_sale"]
        )

        # Convert ML results to per-item format
        out: list[dict[str, Any]] = []
        if fcst_df.empty:
            fcst_df = pd.DataFrame(columns=["item_id", "day", "yhat", "upper_70", "upper_90", "upper_95"])

        for item_id in unique_ids:
            elig = eligibility.get(str(item_id)) or {}
            ml_allowed = bool(elig.get("ml_allowed")) if elig.get("ml_allowed") is not None else False
            winner = elig.get("winner_model") or ("lgbm_direct" if ml_allowed else "naive")
            reason = elig.get("reason_code") or ("OK" if ml_allowed else "NO_ELIGIBILITY")
            confidence = elig.get("confidence")

            use_ml = ml_allowed and str(winner).startswith("lgbm")
            if use_ml:
                item_fcst = fcst_df[fcst_df["item_id"].astype(str) == str(item_id)].sort_values("day")
                if len(item_fcst) == int(request.forecast_periods):
                    yhat = item_fcst["yhat"].to_numpy(dtype=float).tolist()
                    upper_70 = None
                    if "upper_70" in item_fcst.columns:
                        upper_70 = item_fcst["upper_70"].to_numpy(dtype=float).tolist()
                    upper_90 = None
                    if "upper_90" in item_fcst.columns:
                        upper_90 = item_fcst["upper_90"].to_numpy(dtype=float).tolist()
                    upper_95 = None
                    if "upper_95" in item_fcst.columns:
                        upper_95 = item_fcst["upper_95"].to_numpy(dtype=float).tolist()
                    # Month-start semantics: YYYY-MM-01 represents that month.
                    days = [
                        pd.to_datetime(d).to_period("M").to_timestamp(how="start").strftime("%Y-%m-%d")
                        for d in item_fcst["day"].tolist()
                    ]
                    adjustments_list = []
                    if "adjustments" in item_fcst.columns:
                        try:
                            adjustments_list = [
                                (a if isinstance(a, dict) else {}) for a in item_fcst["adjustments"].tolist()
                            ]
                        except Exception:
                            adjustments_list = []
                    out.append(
                        {
                            "item_id": item_id,
                            "forecast": yhat,
                            "upper_70": upper_70,
                            "upper_90": upper_90,
                            "upper_95": upper_95,
                            "forecast_dates": days,
                            "model_used": str(winner),
                            "reason_code": reason,
                            "confidence": confidence,
                            "adjustments": adjustments_list,
                        }
                    )
                    continue

                # If ML is allowed but we don't have predictions (e.g. not enough history
                # for that item), fall back to naive but keep an explicit reason.
                reason = "ML_NO_PREDICTION"

                last = float(last_vals.get(str(item_id), 0.0)) if len(last_vals) else 0.0
                # naive dates: just increment by freq offset from max day
                last_day = df_hist[df_hist["item_id"].astype(str) == str(item_id)]["day"].max()
                if pd.isna(last_day):
                    days = []
                else:
                    start = pd.to_datetime(last_day).to_period('M').to_timestamp(how='start')
                    future = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=request.forecast_periods, freq='MS')
                    days = [d.strftime("%Y-%m-%d") for d in future]

                out.append(
                    {
                        "item_id": item_id,
                        "forecast": [last] * int(request.forecast_periods),
                        "forecast_dates": days,
                        "model_used": "naive",
                        "reason_code": reason,
                        "confidence": confidence,
                        "skipped": True,
                    }
                )
                continue
            if not ml_allowed:
                last = float(last_vals.get(str(item_id), 0.0)) if len(last_vals) else 0.0
                last_day = df_hist[df_hist["item_id"].astype(str) == str(item_id)]["day"].max()
                if pd.isna(last_day):
                    days = []
                else:
                    start = pd.to_datetime(last_day).to_period('M').to_timestamp(how='start')
                    future = pd.date_range(
                        start=start + pd.offsets.MonthBegin(1),
                        periods=request.forecast_periods,
                        freq='MS',
                    )
                    days = [d.strftime("%Y-%m-%d") for d in future]

                out.append(
                    {
                        "item_id": item_id,
                        "forecast": [last] * int(request.forecast_periods),
                        "forecast_dates": days,
                        "model_used": "naive",
                        "reason_code": reason,
                        "confidence": confidence,
                        "skipped": True,
                    }
                )

        return {
            "forecasts": out,
            "total_items": len(unique_ids),
            "forecast_type": "batch",
            "periods": request.forecast_periods,
            "freq": meta.get("freq"),
            "model_version": meta.get("model_version"),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        raise HTTPException(status_code=500, detail=f"LightGBM forecast error: {str(e)}")


@router.post("/batch_async")
def batch_forecast_lightgbm_async(
    request: LightGBMBatchForecastRequest,
    background_tasks: BackgroundTasks,
    webhook_url: str | None = None,
):
    """Start batch forecast as a background job.

    Returns a `job_id` immediately; poll `/jobs/{job_id}` for progress.
    Optionally sends a completion webhook.
    """
    job_id = str(uuid.uuid4())
    
    _set_job(
        job_id,
        {
            'status': 'queued',
            'phase': 'queued',
            'started_at': _now_ts(),
            'customer_id': request.customer_id,
            'model_version': request.model_version,
            'status_filter': request.status,
        },
    )

    def _batch_runner() -> None:
        try:
            _update_job(job_id, status='running', phase='processing')

            df_hist = pd.DataFrame(request.sim_input_his)
            df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
            df_drivers = pd.DataFrame(request.external_drivers) if request.external_drivers else None

            forecaster = LightGBMForecast(store_root=request.store_root or _default_store_root(), customer_id=request.customer_id)

            # Resolve model version
            model_version = request.model_version
            if model_version is None:
                model_version = forecaster.store.get_active_model_version(status=request.status)
            if not model_version:
                raise ValueError(f"No active model version found for status='{request.status}'")

            _update_job(job_id, phase='forecasting', model_version=model_version)

            unique_ids = [str(x) for x in df_hist["item_id"].astype(str).unique().tolist()]
            eligibility = forecaster.store.get_eligibility(model_version=model_version, unique_ids=unique_ids)

            # ML batch prediction for all; we'll decide per item whether to use it.
            fcst_df, meta = forecaster.batch_forecast(
                df_hist,
                forecast_periods=request.forecast_periods,
                freq=request.freq,
                item_attributes=df_items,
                drivers=df_drivers,
                exogenous_columns=request.exogenous_columns,
                model_version=model_version,
                status=request.status,
                deep_shap=bool(getattr(request, "deep_shap", True)),
            )

            _update_job(job_id, phase='formatting')

            # Build naive fallback
            df_hist["day"] = pd.to_datetime(df_hist["day"], errors="coerce")
            df_hist = df_hist.dropna(subset=["item_id", "day", "actual_sale"])
            df_hist["item_id"] = df_hist["item_id"].astype(str)
            last_vals = (
                df_hist.sort_values(["item_id", "day"], kind="mergesort")
                .groupby("item_id", sort=False)
                .tail(1)
                .set_index("item_id")["actual_sale"]
            )

            # Convert ML results to per-item format
            out: list[dict[str, Any]] = []
            if fcst_df.empty:
                fcst_df = pd.DataFrame(columns=["item_id", "day", "yhat", "upper_70", "upper_90", "upper_95"])

            for item_id in unique_ids:
                elig = eligibility.get(str(item_id)) or {}
                ml_allowed = bool(elig.get("ml_allowed")) if elig.get("ml_allowed") is not None else False
                winner = elig.get("winner_model") or ("lgbm_direct" if ml_allowed else "naive")
                reason = elig.get("reason_code") or ("OK" if ml_allowed else "NO_ELIGIBILITY")
                confidence = elig.get("confidence")

                use_ml = ml_allowed and str(winner).startswith("lgbm")
                if use_ml:
                    item_fcst = fcst_df[fcst_df["item_id"].astype(str) == str(item_id)].sort_values("day")
                    if len(item_fcst) == int(request.forecast_periods):
                        yhat = item_fcst["yhat"].to_numpy(dtype=float).tolist()
                        upper_70 = None
                        if "upper_70" in item_fcst.columns:
                            upper_70 = item_fcst["upper_70"].to_numpy(dtype=float).tolist()
                        upper_90 = None
                        if "upper_90" in item_fcst.columns:
                            upper_90 = item_fcst["upper_90"].to_numpy(dtype=float).tolist()
                        upper_95 = None
                        if "upper_95" in item_fcst.columns:
                            upper_95 = item_fcst["upper_95"].to_numpy(dtype=float).tolist()
                        days = [
                            pd.to_datetime(d).to_period("M").to_timestamp(how="start").strftime("%Y-%m-%d")
                            for d in item_fcst["day"].tolist()
                        ]
                        adjustments_list = []
                        if "adjustments" in item_fcst.columns:
                            try:
                                adjustments_list = [
                                    (a if isinstance(a, dict) else {}) for a in item_fcst["adjustments"].tolist()
                                ]
                            except Exception:
                                adjustments_list = []
                        out.append(
                            {
                                "item_id": item_id,
                                "forecast": yhat,
                                "upper_70": upper_70,
                                "upper_90": upper_90,
                                "upper_95": upper_95,
                                "forecast_dates": days,
                                "model_used": str(winner),
                                "reason_code": reason,
                                "confidence": confidence,
                                "adjustments": adjustments_list,
                            }
                        )
                        continue

                    reason = "ML_NO_PREDICTION"

                last = float(last_vals.get(str(item_id), 0.0)) if len(last_vals) else 0.0
                last_day = df_hist[df_hist["item_id"].astype(str) == str(item_id)]["day"].max()
                if pd.isna(last_day):
                    days = []
                else:
                    start = pd.to_datetime(last_day).to_period('M').to_timestamp(how='start')
                    future = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=request.forecast_periods, freq='MS')
                    days = [d.strftime("%Y-%m-%d") for d in future]

                out.append(
                    {
                        "item_id": item_id,
                        "forecast": [last] * int(request.forecast_periods),
                        "forecast_dates": days,
                        "model_used": "naive",
                        "reason_code": reason,
                        "confidence": confidence,
                        "skipped": True,
                    }
                )

            result = {
                "forecasts": out,
                "total_items": len(unique_ids),
                "forecast_type": "batch",
                "periods": request.forecast_periods,
                "freq": meta.get("freq"),
                "model_version": meta.get("model_version"),
                "deep_shap": meta.get("deep_shap"),
                "deep_shap_rows": meta.get("deep_shap_rows") if bool(getattr(request, "deep_shap", True)) else [],
            }

            _update_job(
                job_id,
                status='finished',
                phase='done',
                finished_at=_now_ts(),
                result=result,
            )

            if webhook_url:
                _send_webhook_with_retries(
                    job_id,
                    webhook_url,
                    {
                        'job_id': job_id,
                        'status': 'finished',
                        'customer_id': request.customer_id,
                        'result': result,
                    },
                )

        except Exception as e:
            _update_job(
                job_id,
                status='failed',
                phase='failed',
                finished_at=_now_ts(),
                error=str(e),
                traceback=traceback.format_exc(),
            )
            if webhook_url:
                _send_webhook_with_retries(
                    job_id,
                    webhook_url,
                    {
                        'job_id': job_id,
                        'status': 'failed',
                        'customer_id': request.customer_id,
                        'error': str(e),
                    },
                )

    background_tasks.add_task(_batch_runner)

    return {
        'job_id': job_id,
        'status': 'queued',
        'status_url': f"/api/v1/lightgbm/jobs/{job_id}",
    }


@router.get("/diagnostics")
def get_lightgbm_diagnostics(
    customer_id: str,
    model_version: str | None = None,
    status: str = "prod",
    item_ids: str | None = None,
    limit: int = 200,
    store_root: str | None = None,
):
    """Fetch per-item diagnostics produced at training time.

    This is designed to be called after training/forecasting without re-uploading history.

    Query params:
    - customer_id: required
    - model_version: optional (defaults to active version for status)
    - status: used if model_version is omitted
    - item_ids: optional comma-separated list
    - limit: max rows if item_ids not specified
    - store_root: optional override
    """

    try:
        # DEBUG: Log request params
        print(f"[DEBUG /diagnostics] customer_id={customer_id}, model_version={model_version}, status={status}")

        forecaster = LightGBMForecast(store_root=store_root or _default_store_root(), customer_id=customer_id)

        mv = model_version
        if mv is None:
            mv = forecaster.store.get_active_model_version(status=status)
            print(f"[DEBUG /diagnostics] resolved model_version from status={status}: {mv}")
        if not mv:
            raise ValueError(f"No active model version found for status='{status}'")

        unique_ids: list[str] | None
        if item_ids:
            unique_ids = [s.strip() for s in item_ids.split(",") if s.strip()]
        else:
            unique_ids = None

        explain = forecaster.store.get_explain_summary(model_version=mv, unique_ids=unique_ids, limit=int(limit))
        # DEBUG: Log results
        print(f"[DEBUG /diagnostics] explain returned {len(explain)} items")
        name_token_vocab = None
        name_token_columns = None
        name_token_buckets = None
        name_cluster_k = None
        try:
            _, spec = forecaster._load_spec(mv)
            name_token_vocab = spec.get("name_token_vocab")
            name_token_columns = spec.get("name_token_columns")
            name_token_buckets = spec.get("name_token_buckets")
            name_cluster_k = spec.get("name_cluster_k")
        except Exception:
            pass
        return {
            "customer_id": customer_id,
            "model_version": mv,
            "name_token_vocab": name_token_vocab,
            "name_token_columns": name_token_columns,
            "name_token_buckets": name_token_buckets,
            "name_cluster_k": name_cluster_k,
            "items": [
                {
                    "item_id": uid,
                    **payload,
                }
                for uid, payload in explain.items()
            ],
            "total_items": len(explain),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightGBM diagnostics error: {str(e)}")
