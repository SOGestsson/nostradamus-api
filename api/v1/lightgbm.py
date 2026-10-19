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
import math

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
import httpx
import json
import hmac
import hashlib
import logging

from api.models import LightGBMBatchForecastRequest, LightGBMTrainRequest
from inventory_algorithm.lightgbm_forecasts import LightGBMForecast
from inventory_algorithm.lgbm_origin import CONFIDENCE_DEFINITION

logger = logging.getLogger("nostradamus.lightgbm")


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

    _redis_sync = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2.0,
        socket_timeout=2.0,
    )
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
    with _JOBS_LOCK:
        _JOBS[job_id] = payload
    if _redis_sync is not None:
        try:
            _redis_sync.hset(_job_key(job_id), mapping={k: json.dumps(v) for k, v in payload.items()})
            _redis_sync.expire(_job_key(job_id), 60 * 60 * 24)
        except Exception:
            pass


def _update_job(job_id: str, **fields: Any) -> None:
    fields = dict(fields)
    fields['updated_at'] = _now_ts()
    with _JOBS_LOCK:
        cur = _JOBS.get(job_id, {})
        cur.update(fields)
        _JOBS[job_id] = cur
    if _redis_sync is not None:
        try:
            _redis_sync.hset(_job_key(job_id), mapping={k: json.dumps(v) for k, v in fields.items()})
            _redis_sync.expire(_job_key(job_id), 60 * 60 * 24)
        except Exception:
            pass


def _get_job(job_id: str) -> dict[str, Any] | None:
    if _redis_sync is not None:
        try:
            raw = _redis_sync.hgetall(_job_key(job_id))
            if raw:
                out: dict[str, Any] = {}
                for k, v in raw.items():
                    try:
                        out[k] = json.loads(v)
                    except Exception:
                        out[k] = v
                return out
        except Exception:
            pass
    with _JOBS_LOCK:
        return dict(_JOBS.get(job_id)) if job_id in _JOBS else None


def _start_job_thread(name: str, fn) -> threading.Thread:
    """Run the job in a dedicated thread so FastAPI can still serve GET /jobs."""
    t = threading.Thread(target=fn, name=name, daemon=True)
    t.start()
    return t


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
                logger.info(
                    "webhook sent",
                    extra={"job_id": job_id, "customer_id": payload.get("customer_id"), "attempt": attempt},
                )
                return True
            last_exc = f"status={resp.status_code}, body={resp.text}"
        except Exception as e:
            last_exc = str(e)

        _update_job(job_id, webhook_attempts=attempt, webhook_last_error=last_exc)
        if attempt < max_attempts:
            time.sleep(delays[min(attempt - 1, len(delays) - 1)])

    dlq_entry = {
        "job_id": job_id,
        "webhook": webhook,
        "payload": payload,
        "attempts": attempt,
        "last_error": last_exc,
    }
    if _redis_sync is not None:
        try:
            _redis_sync.rpush("webhook:dlq", json.dumps(dlq_entry, default=str))
        except Exception:
            logger.error("DLQ push failed", extra={"job_id": job_id})
    else:
        logger.error("webhook failed after retries (no redis DLQ)", extra={"job_id": job_id, "error": last_exc})
    return False


def _default_store_root() -> str:
    return os.getenv("NOSTRADAMUS_STORE_ROOT", "./nostradamus_store")


def _future_month_starts(last_day: Any, periods: int) -> list[str]:
    if pd.isna(last_day):
        return []
    start = pd.to_datetime(last_day).to_period("M").to_timestamp(how="start")
    future = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=int(periods), freq="MS")
    return [d.strftime("%Y-%m-%d") for d in future]


def _requested_item_ids(df_hist: pd.DataFrame, df_items: pd.DataFrame | None) -> list[str]:
    ids: list[str] = []
    if df_hist is not None and not df_hist.empty and "item_id" in df_hist.columns:
        ids.extend(df_hist["item_id"].astype(str).tolist())
    if df_items is not None and not df_items.empty and "item_id" in df_items.columns:
        ids.extend(df_items["item_id"].astype(str).tolist())
    out: list[str] = []
    seen: set[str] = set()
    for item_id in ids:
        if item_id not in seen:
            seen.add(item_id)
            out.append(item_id)
    return out


def _failed_item_payload(
    *,
    item_id: str,
    forecast_periods: int,
    reason: str,
    confidence: Any,
    dates: list[str] | None = None,
) -> dict[str, Any]:
    n = int(forecast_periods)
    return {
        "item_id": item_id,
        "forecast": [0.0] * n,
        "upper_70": [0.0] * n,
        "upper_90": [0.0] * n,
        "upper_95": [0.0] * n,
        "forecast_dates": dates or [],
        "model_used": "failed",
        "reason_code": reason,
        "confidence": confidence,
        "skipped": True,
        "failed": True,
        "adjustments": [{"failed": True, "reason_code": str(reason)} for _ in range(n)],
    }


def _dispatch_fallback(
    *,
    df_hist: pd.DataFrame,
    item_id: str,
    forecast_periods: int,
    reason: str,
    confidence: Any,
    method: str | None,
) -> dict[str, Any]:
    """Dispatch the eligibility fallback: last-value naive or seasonal naive.

    History is monthly-aggregated with the same sum contract as training
    (Jan 40+60 → 100, not 60). Upper bounds are heuristic, not calibrated.
    """
    from inventory_algorithm.lightgbm_forecasts import _normalize_monthly_history
    from inventory_algorithm.lgbm_grid import complete_monthly_grid, seasonal_naive_value

    method_key = str(method or "naive").strip().lower()
    h = df_hist[df_hist["item_id"].astype(str) == str(item_id)].copy() if not df_hist.empty else pd.DataFrame()
    if h.empty:
        return _failed_item_payload(
            item_id=item_id,
            forecast_periods=forecast_periods,
            reason=reason or "NO_HISTORY",
            confidence=confidence,
            dates=[],
        )
    try:
        monthly = _normalize_monthly_history(h)
        monthly = complete_monthly_grid(monthly)
    except Exception:
        return _failed_item_payload(
            item_id=item_id,
            forecast_periods=forecast_periods,
            reason=reason or "NO_HISTORY",
            confidence=confidence,
            dates=[],
        )
    if monthly is None or monthly.empty:
        return _failed_item_payload(
            item_id=item_id,
            forecast_periods=forecast_periods,
            reason=reason or "NO_HISTORY",
            confidence=confidence,
            dates=[],
        )
    monthly = monthly.sort_values("ds", kind="mergesort")
    last_day = monthly["ds"].max()
    days = _future_month_starts(last_day, int(forecast_periods))
    y = pd.to_numeric(monthly["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    t = len(y) - 1
    last_val = float(y[t]) if t >= 0 else 0.0

    fcst: list[float] = []
    u70: list[float] = []
    u90: list[float] = []
    u95: list[float] = []
    adj: list[dict[str, Any]] = []
    spread = float(max(1.0, 0.35 * abs(last_val)))
    used = "naive"
    for hi, _d in enumerate(days, start=1):
        if method_key in {"seasonal_naive", "snaive", "seasonal"}:
            yhat = float(max(0.0, seasonal_naive_value(y, t, hi)))
            used = "seasonal_naive"
        else:
            yhat = float(max(0.0, last_val))
            used = "naive"
        q70 = float(max(0.0, yhat + 0.30 * spread))
        q90 = float(max(q70, yhat + 0.60 * spread))
        q95 = float(max(q90, yhat + 0.90 * spread))
        fcst.append(yhat)
        u70.append(q70)
        u90.append(q90)
        u95.append(q95)
        adj.append(
            {
                "fallback": used,
                "reason_code": str(reason),
                "bound_type": "heuristic_not_calibrated",
            }
        )

    return {
        "item_id": item_id,
        "forecast": fcst,
        "upper_70": u70,
        "upper_90": u90,
        "upper_95": u95,
        "forecast_dates": days,
        "model_used": used,
        "reason_code": reason,
        "confidence": confidence,
        "skipped": True,
        "adjustments": adj,
    }


def _json_num_list(values) -> list:
    out = []
    for v in list(values):
        if v is None:
            out.append(None)
            continue
        try:
            fv = float(v)
        except Exception:
            out.append(None)
            continue
        out.append(fv if math.isfinite(fv) else None)
    return out


def _format_batch_results(
    *,
    unique_ids: list[str],
    fcst_df: pd.DataFrame,
    eligibility: dict[str, Any],
    df_hist: pd.DataFrame,
    forecast_periods: int,
    meta: dict[str, Any],
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    if fcst_df is None or fcst_df.empty:
        fcst_df = pd.DataFrame(columns=["item_id", "day", "yhat", "upper_70", "upper_90", "upper_95"])

    for item_id in unique_ids:
        elig = eligibility.get(str(item_id)) or {}
        ml_allowed = bool(elig.get("ml_allowed")) if elig.get("ml_allowed") is not None else False
        winner = elig.get("winner_model") or ("lgbm_direct" if ml_allowed else "naive")
        reason = elig.get("reason_code") or ("OK" if ml_allowed else "NO_ELIGIBILITY")
        confidence = elig.get("confidence")
        fallback_method = elig.get("fallback_model") or "naive"
        logger.info(
            "item routing",
            extra={
                "customer_id": meta.get("customer_id"),
                "model_version": meta.get("model_version"),
                "item_id": str(item_id),
                "reason_code": str(reason),
            },
        )

        item_fcst = fcst_df[fcst_df["item_id"].astype(str) == str(item_id)].sort_values("day") if "item_id" in fcst_df.columns else pd.DataFrame()
        item_failed = bool(
            not item_fcst.empty and "failed" in item_fcst.columns and bool(item_fcst["failed"].fillna(False).any())
        )
        use_ml = ml_allowed and str(winner).startswith("lgbm") and not item_failed
        if use_ml and len(item_fcst) == int(forecast_periods):
            days = [
                pd.to_datetime(d).to_period("M").to_timestamp(how="start").strftime("%Y-%m-%d")
                for d in item_fcst["day"].tolist()
            ]
            if len(days) != len(set(days)):
                item_failed = True
            yhat = _json_num_list(item_fcst["yhat"].tolist())
            if any(v is None for v in yhat):
                item_failed = True
            if not item_failed:
                upper_70 = _json_num_list(item_fcst["upper_70"].tolist()) if "upper_70" in item_fcst.columns else None
                upper_90 = _json_num_list(item_fcst["upper_90"].tolist()) if "upper_90" in item_fcst.columns else None
                upper_95 = _json_num_list(item_fcst["upper_95"].tolist()) if "upper_95" in item_fcst.columns else None
                adjustments_list: list[Any] = []
                if "adjustments" in item_fcst.columns:
                    try:
                        adjustments_list = [(a if isinstance(a, (dict, list)) else {}) for a in item_fcst["adjustments"].tolist()]
                    except Exception:
                        adjustments_list = []
                payload: dict[str, Any] = {
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
                if "yhat_raw" in item_fcst.columns:
                    payload["yhat_raw"] = _json_num_list(item_fcst["yhat_raw"].tolist())
                if "mean_head" in item_fcst.columns:
                    payload["mean_head"] = item_fcst["mean_head"].tolist()
                if "planning_ceiling" in item_fcst.columns:
                    payload["planning_ceiling"] = _json_num_list(item_fcst["planning_ceiling"].tolist())
                out.append(payload)
                continue

        if item_failed:
            fail_rows = item_fcst
            if "failed" in item_fcst.columns:
                fail_rows = item_fcst[item_fcst["failed"].fillna(False).astype(bool)]
            if fail_rows.empty:
                fail_rows = item_fcst
            fail_reason = "PREDICT_FAILED"
            if "reason_code" in fail_rows.columns and not fail_rows.empty:
                fail_reason = str(fail_rows["reason_code"].iloc[0])
            out.append(
                _failed_item_payload(
                    item_id=str(item_id),
                    forecast_periods=int(forecast_periods),
                    reason=fail_reason,
                    confidence=confidence,
                    dates=[
                        pd.to_datetime(d).to_period("M").to_timestamp(how="start").strftime("%Y-%m-%d")
                        for d in item_fcst["day"].tolist()
                    ] if "day" in item_fcst.columns else [],
                )
            )
            continue

        out.append(
            _dispatch_fallback(
                df_hist=df_hist,
                item_id=str(item_id),
                forecast_periods=int(forecast_periods),
                reason="ML_NO_PREDICTION" if ml_allowed else str(reason),
                confidence=confidence,
                method=str(fallback_method if not ml_allowed else fallback_method),
            )
        )

    meta = dict(meta or {})
    meta.setdefault("confidence_definition", CONFIDENCE_DEFINITION)
    meta["job_runner"] = "fastapi_background_tasks"
    return {
        "forecasts": out,
        "total_items": len(unique_ids),
        "forecast_type": "batch",
        "periods": forecast_periods,
        "freq": meta.get("freq"),
        "model_version": meta.get("model_version"),
        "meta": meta,
    }
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
            required_driver_columns=getattr(request, "required_driver_columns", None),
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
                required_driver_columns=getattr(request, "required_driver_columns", None),
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

    background_tasks.add_task(_start_job_thread, f"lgbm-train-{job_id}", _runner)

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
    try:
        job = _get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='job not found')
        return job
    except HTTPException:
        raise
    except Exception as e:
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

        unique_ids = _requested_item_ids(df_hist, df_items)
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
        )
        meta = dict(meta or {})
        meta["customer_id"] = request.customer_id

        df_hist["day"] = pd.to_datetime(df_hist["day"], errors="coerce")
        df_hist = df_hist.dropna(subset=["item_id", "day", "actual_sale"])
        df_hist["item_id"] = df_hist["item_id"].astype(str)

        return _format_batch_results(
            unique_ids=unique_ids,
            fcst_df=fcst_df,
            eligibility=eligibility,
            df_hist=df_hist,
            forecast_periods=int(request.forecast_periods),
            meta=meta,
        )

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

            unique_ids = _requested_item_ids(df_hist, df_items)
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
            )
            meta = dict(meta or {})
            meta["customer_id"] = request.customer_id
            meta["job_id"] = job_id

            _update_job(job_id, phase='formatting')

            df_hist["day"] = pd.to_datetime(df_hist["day"], errors="coerce")
            df_hist = df_hist.dropna(subset=["item_id", "day", "actual_sale"])
            df_hist["item_id"] = df_hist["item_id"].astype(str)

            result = _format_batch_results(
                unique_ids=unique_ids,
                fcst_df=fcst_df,
                eligibility=eligibility,
                df_hist=df_hist,
                forecast_periods=int(request.forecast_periods),
                meta=meta,
            )

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

    background_tasks.add_task(_start_job_thread, f"lgbm-batch-{job_id}", _batch_runner)

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
        forecaster = LightGBMForecast(store_root=store_root or _default_store_root(), customer_id=customer_id)

        mv = model_version
        if mv is None:
            mv = forecaster.store.get_active_model_version(status=status)
        if not mv:
            raise ValueError(f"No active model version found for status='{status}'")

        unique_ids: list[str] | None
        if item_ids:
            unique_ids = [s.strip() for s in item_ids.split(",") if s.strip()]
        else:
            unique_ids = None

        explain = forecaster.store.get_explain_summary(model_version=mv, unique_ids=unique_ids, limit=int(limit))
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
