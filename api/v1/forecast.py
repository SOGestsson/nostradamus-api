# api/v1/forecast.py - Updated with detailed parameter descriptions

"""
Forecast endpoints using ClassicalForecasts.
"""
import traceback
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

import httpx
import json
import os
import hmac
import hashlib
import time

from fastapi import APIRouter, HTTPException, Request, Depends, Header
import logging

from api.models import ForecastRequest
from inventory_algorithm.classical_forecasts import ClassicalForecasts

router = APIRouter()

# Logging
logger = logging.getLogger("nostradamus")
logging.basicConfig(level=logging.INFO)

# Redis-backed job store
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
try:
    import redis.asyncio as aioredis
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    redis_client = None

# Fallback in-memory store if redis isn't available (useful for local/dev)
JOBS: Dict[str, Dict[str, Any]] = {}


async def send_webhook_with_retries(job_id: str, webhook: str, payload: Dict[str, Any], max_attempts: int = 5):
    delays = [1, 5, 30, 120, 600]
    attempt = 0
    last_exc = None
    webhook_secret = os.getenv('WEBHOOK_SECRET')
    secret_bytes = webhook_secret.encode() if webhook_secret else None
    while attempt < max_attempts:
        attempt += 1
        try:
            payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode()
            headers = {}
            if secret_bytes:
                ts = str(int(time.time()))
                signed = hmac.new(secret_bytes, ts.encode() + b'.' + payload_bytes, hashlib.sha256).hexdigest()
                headers['X-Signature'] = f"sha256={signed}"
                headers['X-Signature-Timestamp'] = ts

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook, content=payload_bytes, headers=headers)

            if 200 <= resp.status_code < 300:
                logger.info("webhook sent", extra={'job_id': job_id, 'attempt': attempt, 'status_code': resp.status_code})
                if redis_client:
                    try:
                        await redis_client.hset(_job_key(job_id), mapping={'webhook_attempts': attempt})
                        await redis_client.incr('metrics:webhook:success')
                    except Exception:
                        # Redis is best-effort for metrics; don't fail webhook delivery.
                        pass
                else:
                    if job_id in JOBS:
                        JOBS[job_id]['webhook_attempts'] = attempt
                return True
            else:
                last_exc = f"status={resp.status_code}, body={resp.text}"
        except Exception as e:
            last_exc = str(e)
            logger.warning("webhook attempt failed", extra={'job_id': job_id, 'attempt': attempt, 'error': last_exc})

        if redis_client:
            try:
                await redis_client.hset(_job_key(job_id), mapping={'webhook_attempts': attempt, 'webhook_last_error': last_exc or ''})
                await redis_client.incr('metrics:webhook:attempts')
            except Exception:
                pass
        else:
            if job_id in JOBS:
                JOBS[job_id]['webhook_attempts'] = attempt
                JOBS[job_id]['webhook_last_error'] = last_exc or ''

        await asyncio.sleep(delays[min(attempt-1, len(delays)-1)])

    dlq_entry = {
        'job_id': job_id,
        'webhook': webhook,
        'payload': payload,
        'attempts': attempt,
        'last_error': last_exc
    }
    try:
        if redis_client:
            try:
                await redis_client.rpush('webhook:dlq', json.dumps(dlq_entry))
                await redis_client.incr('metrics:webhook:dlq')
            except Exception as e:
                logger.exception("Failed to push to DLQ", exc_info=e)
        else:
            logger.error("DLQ push (no redis)", extra={'dlq_entry': dlq_entry})
    except Exception as e:
        logger.exception("Failed to push to DLQ", exc_info=e)

    if redis_client:
        try:
            await redis_client.incr('metrics:webhook:failed')
        except Exception:
            pass
    return False


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def api_key_header(x_api_key: Optional[str] = Header(None)) -> bool:
    """Simple API key dependency. If `API_KEY` env var is set, require matching header."""
    expected = os.getenv('API_KEY')
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API Key")
    return True


@router.post("/generate")
def generate_forecast(request: ForecastRequest):
    """
    Generate forecasts for items using classical forecasting models or TimeGPT.
    
    ## Forecasting Modes
    - **local**: StatsForecast models (AutoARIMA, ETS, Croston, etc.) - No API key needed
    - **timegpt**: Nixtla TimeGPT cloud API - Requires API key
    
    ## Parameters
    
    **sim_input_his** (required): Historical sales data for forecasting
    - Format: List of dicts with `item_id`, `actual_sale`, `day` (YYYY-MM-DD)
    - Minimum: 5-30 data points depending on model complexity
    - Example: [{"item_id": 100, "actual_sale": 120, "day": "2023-01-01"}]
    
    **forecast_periods** (default: 30): Number of future periods to forecast
    - Range: 1-365 for daily, 1-24 for monthly
    - Example: 7 = forecast next 7 days/months
    
    **mode** (default: 'local'): Forecasting engine
    - 'local': Local StatsForecast models
    - 'timegpt': TimeGPT cloud API
    
    **local_model** (default: 'auto_arima'): Statistical model (only for mode='local')
    - 'auto_arima': Automated ARIMA - best for trend data (20+ points)
    - 'auto_ets': Exponential Smoothing - best for seasonal data (30+ points)
    - 'naive': Simple forecast (last value) - works with any data
    - 'seasonal_naive': Repeats seasonal pattern - needs 1 full season
    - 'croston_optimized': For intermittent/sparse demand
    - 'adida': Adaptive intermittent demand
    - 'theta': Theta method - balanced approach
    - 'optimized_theta': Optimized theta
    - 'auto_ces': Complex exponential smoothing
    - 'auto_model': Automatically selects the best StatsForecast model per item via cross-validation (defaults to a robust RMSE+MAE rank aggregation; excludes TimeGPT/LightGPT)
    
    **season_length** (default: 12): Length of one seasonal cycle
    - 7 = weekly seasonality (for daily data)
    - 12 = yearly seasonality (for monthly data)
    - 24 = daily seasonality (for hourly data)
    - 52 = yearly seasonality (for weekly data)
    
    **freq** (default: 'D'): Frequency/interval of time series data
    - 'D': Daily
    - 'MS': Month Start (monthly data)
    - 'W': Weekly
    - 'H': Hourly
    - 'Q': Quarterly
    - 'Y': Yearly
    - Must match your input data frequency
    
    **api_key** (optional): Nixtla API key for TimeGPT mode
    - Required when mode='timegpt'
    - Alternative: Set NIXTLA_API_KEY environment variable
    
    **quantiles** (optional): Quantile levels for uncertainty intervals (TimeGPT only)
    - Format: List of floats between 0 and 1
    - Example: [0.1, 0.5, 0.9] for 10th, 50th, 90th percentiles
    - Use for safety stock calculations and confidence intervals
    
    ## Quick Guide by Use Case
    - **Daily demand, next week**: freq='D', season_length=7, forecast_periods=7
    - **Monthly sales, next 6 months**: freq='MS', season_length=12, forecast_periods=6
    - **Sparse/intermittent demand**: local_model='croston_optimized'
    - **Need uncertainty estimates**: mode='timegpt', quantiles=[0.1, 0.5, 0.9]
    
    ## Returns
    Dictionary containing:
    - forecasts: List of forecast results per item
    - total_items: Number of items processed
    - mode: Forecasting mode used
    - model: Model used
    - periods: Number of periods forecasted
    - frequency: Data frequency
    
    ## Example Request
    ```json
    {
      "sim_input_his": [
        {"item_id": 100, "actual_sale": 120, "day": "2023-01-01"},
        {"item_id": 100, "actual_sale": 115, "day": "2023-02-01"}
      ],
      "forecast_periods": 6,
      "mode": "local",
      "local_model": "auto_ets",
      "season_length": 12,
      "freq": "MS"
    }
    ```
    """
    try:
        print(f"Starting forecast generation with mode: {request.mode}")
        
        # Convert input data to DataFrame
        df_his = pd.DataFrame(request.sim_input_his)
        
        # Validate required columns
        required_cols = ['item_id', 'actual_sale', 'day']
        missing_cols = [col for col in required_cols if col not in df_his.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column to datetime
        df_his['day'] = pd.to_datetime(df_his['day'])
        
        # Initialize forecaster
        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles,
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )
        
        print(f"Forecaster initialized: {request.local_model if request.mode == 'local' else 'TimeGPT'}")
        
        results = []
        unique_items = df_his['item_id'].unique()

        print(f"Generating forecasts for {len(unique_items)} items")

        # Auto-select best StatsForecast model per item (explicitly not TimeGPT/LightGPT)
        if request.mode == 'local' and request.local_model in ('auto_model', 'automodel'):
            panel_fcst = forecaster.auto_model_forecast_panel(df_his, h=request.forecast_periods, metric='robust')
            id_map = {str(v): v for v in unique_items}
            for uid, grp in panel_fcst.groupby('unique_id', sort=False):
                try:
                    original_item_id = id_map.get(str(uid), uid)
                    item_id_out = int(original_item_id) if isinstance(original_item_id, (int, float)) else str(original_item_id)

                    grp = grp.sort_values('ds')
                    model_used = str(grp['model_used'].iloc[0])
                    results.append({
                        'item_id': item_id_out,
                        'forecast': grp['yhat'].to_numpy(dtype=float).tolist(),
                        'forecast_dates': [pd.to_datetime(d).strftime('%Y-%m-%d') for d in grp['ds'].tolist()],
                        'model_used': model_used,
                        'periods_forecasted': int(len(grp))
                    })
                    print(f"  ✓ Item {uid}: auto_model -> {model_used}")
                except Exception as e:
                    print(f"  ✗ Item {uid}: {str(e)}")
                    results.append({
                        'item_id': str(uid),
                        'error': str(e),
                        'forecast': [],
                        'forecast_dates': []
                    })
        else:
            for item_id in unique_items:
                try:
                    item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)

                    forecast_values = forecaster.daily_path(item_data, periods=request.forecast_periods)

                    last_date = item_data['day'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1 if request.freq == 'D' else 0),
                        periods=request.forecast_periods,
                        freq=request.freq
                    )

                    results.append({
                        'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                        'forecast': forecast_values.tolist(),
                        'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                        'model_used': request.local_model if request.mode == 'local' else 'timegpt',
                        'periods_forecasted': len(forecast_values)
                    })

                    print(f"  ✓ Item {item_id}: forecast generated")

                except Exception as e:
                    print(f"  ✗ Item {item_id}: {str(e)}")
                    results.append({
                        'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                        'error': str(e),
                        'forecast': [],
                        'forecast_dates': []
                    })
        
        print(f"Forecast generation completed: {len(results)} items processed")
        
        return {
            'forecasts': results,
            'total_items': len(unique_items),
            'mode': request.mode,
            'model': request.local_model if request.mode == 'local' else 'timegpt',
            'periods': request.forecast_periods,
            'frequency': request.freq
        }
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@router.post("/generate_async")
async def generate_forecast_async(request: ForecastRequest):
    """
    Async version of `/generate`. Runs per-item forecasting off the event loop
    using `asyncio.to_thread` so the server remains responsive. Parameters
    and response structure are identical to `/generate`.
    """
    try:
        print(f"Starting async forecast generation with mode: {request.mode}")

        df_his = pd.DataFrame(request.sim_input_his)

        required_cols = ['item_id', 'actual_sale', 'day']
        missing_cols = [col for col in required_cols if col not in df_his.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_his['day'] = pd.to_datetime(df_his['day'])

        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles,
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )

        print(f"Forecaster initialized (async): {request.local_model if request.mode == 'local' else 'TimeGPT'}")

        results = []
        unique_items = df_his['item_id'].unique()

        print(f"Generating async forecasts for {len(unique_items)} items")

        if request.mode == 'local' and request.local_model in ('auto_model', 'automodel'):
            panel_fcst = await asyncio.to_thread(
                forecaster.auto_model_forecast_panel,
                df_his,
                request.forecast_periods,
                'robust'
            )
            id_map = {str(v): v for v in unique_items}
            for uid, grp in panel_fcst.groupby('unique_id', sort=False):
                try:
                    original_item_id = id_map.get(str(uid), uid)
                    item_id_out = int(original_item_id) if isinstance(original_item_id, (int, float)) else str(original_item_id)

                    grp = grp.sort_values('ds')
                    model_used = str(grp['model_used'].iloc[0])
                    results.append({
                        'item_id': item_id_out,
                        'forecast': grp['yhat'].to_numpy(dtype=float).tolist(),
                        'forecast_dates': [pd.to_datetime(d).strftime('%Y-%m-%d') for d in grp['ds'].tolist()],
                        'model_used': model_used,
                        'periods_forecasted': int(len(grp))
                    })
                    print(f"  ✓ Item {uid}: async auto_model -> {model_used}")
                except Exception as e:
                    print(f"  ✗ Item {uid}: {str(e)}")
                    results.append({
                        'item_id': str(uid),
                        'error': str(e),
                        'forecast': [],
                        'forecast_dates': []
                    })
        else:
            for item_id in unique_items:
                try:
                    item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)

                    forecast_values = await asyncio.to_thread(forecaster.daily_path, item_data, request.forecast_periods)

                    last_date = item_data['day'].max()
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1 if request.freq == 'D' else 0),
                        periods=request.forecast_periods,
                        freq=request.freq
                    )

                    results.append({
                        'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                        'forecast': forecast_values.tolist(),
                        'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                        'model_used': request.local_model if request.mode == 'local' else 'timegpt',
                        'periods_forecasted': len(forecast_values)
                    })

                    print(f"  ✓ Item {item_id}: async forecast generated")

                except Exception as e:
                    print(f"  ✗ Item {item_id}: {str(e)}")
                    results.append({
                        'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                        'error': str(e),
                        'forecast': [],
                        'forecast_dates': []
                    })

        print(f"Async forecast generation completed: {len(results)} items processed")

        return {
            'forecasts': results,
            'total_items': len(unique_items),
            'mode': request.mode,
            'model': request.local_model if request.mode == 'local' else 'timegpt',
            'periods': request.forecast_periods,
            'frequency': request.freq
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full async error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Async forecast error: {str(e)}")


@router.post("/generate_job")
async def generate_forecast_job(request: ForecastRequest, req: Request, webhook_url: Optional[str] = None, auth: bool = Depends(api_key_header)):
    """
    Submit an asynchronous forecast job. Returns immediately with a `job_id` and
    a status URL. The job runs in the background and the result can be polled
    via `GET /jobs/{job_id}`. Optionally, provide `webhook_url` as a query
    parameter to receive a POST callback on completion.
    """
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    job_record = {
        'status': 'pending',
        'result': '',
        'error': '',
        'created_at': now,
        'finished_at': ''
    }

    global redis_client
    if redis_client:
        try:
            # store as hash; result/error stored as JSON string when present
            await redis_client.hset(_job_key(job_id), mapping=job_record)
            # set a default TTL (7 days)
            await redis_client.expire(_job_key(job_id), 60 * 60 * 24 * 7)
        except Exception as e:
            # If Redis is misconfigured/unavailable, fall back to in-memory store.
            logger.warning("Redis unavailable; falling back to in-memory JOBS", extra={'error': str(e)})
            redis_client = None
            JOBS[job_id] = job_record
    else:
        JOBS[job_id] = job_record

    async def _run_job(job_id: str, request_obj: ForecastRequest, webhook: Optional[str]):
        # update status -> running
        global redis_client
        if redis_client:
            try:
                await redis_client.hset(_job_key(job_id), mapping={'status': 'running'})
            except Exception:
                redis_client = None
                if job_id in JOBS:
                    JOBS[job_id]['status'] = 'running'
        else:
            if job_id in JOBS:
                JOBS[job_id]['status'] = 'running'

        # use module-level send_wits pebhook_with_retries defined above

        try:
            # Run the existing synchronous generator in a thread to avoid blocking
            result = await asyncio.to_thread(generate_forecast, request_obj)

            finished_at = datetime.utcnow().isoformat()
            if redis_client:
                try:
                    await redis_client.hset(_job_key(job_id), mapping={
                        'status': 'finished',
                        'result': json.dumps(result),
                        'finished_at': finished_at
                    })
                except Exception:
                    redis_client = None
                    if job_id in JOBS:
                        JOBS[job_id]['status'] = 'finished'
                        JOBS[job_id]['result'] = result
                        JOBS[job_id]['finished_at'] = finished_at
            else:
                if job_id in JOBS:
                    JOBS[job_id]['status'] = 'finished'
                    JOBS[job_id]['result'] = result
                    JOBS[job_id]['finished_at'] = finished_at

            if webhook:
                payload = {'job_id': job_id, 'status': 'finished', 'result': result}
                sent = await send_webhook_with_retries(job_id, webhook, payload)
                if not sent:
                    print(f"Failed to deliver webhook after retries for job {job_id}")

        except Exception as e:
            finished_at = datetime.utcnow().isoformat()
            if redis_client:
                try:
                    await redis_client.hset(_job_key(job_id), mapping={
                        'status': 'failed',
                        'error': str(e),
                        'finished_at': finished_at
                    })
                except Exception:
                    redis_client = None
                    if job_id in JOBS:
                        JOBS[job_id]['status'] = 'failed'
                        JOBS[job_id]['error'] = str(e)
                        JOBS[job_id]['finished_at'] = finished_at
            else:
                if job_id in JOBS:
                    JOBS[job_id]['status'] = 'failed'
                    JOBS[job_id]['error'] = str(e)
                    JOBS[job_id]['finished_at'] = finished_at

            if webhook:
                payload = {'job_id': job_id, 'status': 'failed', 'error': str(e)}
                sent = await send_webhook_with_retries(job_id, webhook, payload)
                if not sent:
                    print(f"Failed to deliver failure webhook after retries for job {job_id}")

    # Schedule background task and return job metadata
    asyncio.create_task(_run_job(job_id, request, webhook_url))

    # NOTE: This router is mounted at /api/v1/forecast
    status_path = f"/api/v1/forecast/jobs/{job_id}"
    base_url = str(req.base_url).rstrip('/')
    if redis_client:
        try:
            status = await redis_client.hget(_job_key(job_id), 'status')
        except Exception:
            redis_client = None
            status = JOBS[job_id]['status']
    else:
        status = JOBS[job_id]['status']
    return {
        'job_id': job_id,
        'status_url': f"{base_url}{status_path}",
        'status': status
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, auth: bool = Depends(api_key_header)):
    """Return status and result (if finished) for a submitted job."""
    global redis_client
    if redis_client:
        try:
            data = await redis_client.hgetall(_job_key(job_id))
        except Exception:
            redis_client = None
            data = None
        if data:
            if data.get('result'):
                try:
                    data['result'] = json.loads(data['result'])
                except Exception:
                    pass
            return data

    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/webhook/dlq")
async def list_webhook_dlq(limit: int = 100, auth: bool = Depends(api_key_header)):
    """List entries in the webhook dead-letter queue (requires API key if set)."""
    global redis_client
    if not redis_client:
        return {'count': 0, 'dlq': []}
    try:
        items = await redis_client.lrange('webhook:dlq', 0, limit - 1)
    except Exception:
        redis_client = None
        return {'count': 0, 'dlq': []}
    parsed = []
    for it in items:
        try:
            parsed.append(json.loads(it))
        except Exception:
            parsed.append({'raw': it})
    return {'count': len(parsed), 'dlq': parsed}


@router.post("/webhook/dlq/requeue")
async def requeue_webhook_dlq(job_id: str, auth: bool = Depends(api_key_header)):
    """Requeue a DLQ entry by `job_id`. This will remove the DLQ entry and
    attempt delivery again in the background."""
    global redis_client
    if not redis_client:
        raise HTTPException(status_code=500, detail='Redis not configured')

    try:
        items = await redis_client.lrange('webhook:dlq', 0, -1)
    except Exception:
        redis_client = None
        raise HTTPException(status_code=500, detail='Redis not configured')
    found_raw = None
    for raw in items:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if obj.get('job_id') == job_id:
            found_raw = raw
            entry = obj
            break

    if not found_raw:
        raise HTTPException(status_code=404, detail='DLQ entry not found')

    # remove single occurrence
    await redis_client.lrem('webhook:dlq', 1, found_raw)

    # re-attempt delivery asynchronously
    webhook = entry.get('webhook')
    payload = entry.get('payload')
    asyncio.create_task(send_webhook_with_retries(job_id, webhook, payload))

    return {'status': 'requeued', 'job_id': job_id}


@router.post("/leadtime_quantile")
def calculate_leadtime_quantile(request: ForecastRequest):
    """
    Calculate lead-time demand quantiles for safety stock calculations.
    
    Uses Monte Carlo simulation to estimate demand variability over lead time,
    accounting for forecast uncertainty. Useful for inventory optimization and
    determining safety stock levels.
    
    ## Parameters
    
    Same as /generate endpoint, with key parameters:
    
    **sim_input_his** (required): Historical sales data
    - Minimum 20-30 data points recommended for accurate quantile estimation
    
    **forecast_periods**: Lead time in periods
    - Example: 7 = 7-day lead time (for daily data)
    - Example: 2 = 2-month lead time (for monthly data)
    
    **mode**: Forecasting engine ('local' or 'timegpt')
    
    **local_model**: Statistical model for local mode
    - Recommended: 'auto_arima' or 'auto_ets' for quantile calculations
    
    ## Returns
    Dictionary containing:
    - leadtime_quantiles: List of quantile results per item
    - Each item includes quantiles at 90%, 95%, and 99% service levels
    - total_items: Number of items processed
    - lead_time_periods: Lead time used
    - mode: Forecasting mode used
    
    ## Service Level Quantiles
    - **q_90**: 90% service level (10% stockout risk)
    - **q_95**: 95% service level (5% stockout risk)
    - **q_99**: 99% service level (1% stockout risk)
    
    ## Use Cases
    - Safety stock calculation
    - Reorder point determination
    - Service level planning
    - Risk assessment
    
    ## Example Request
    ```json
    {
      "sim_input_his": [
        {"item_id": 100, "actual_sale": 120, "day": "2023-01-01"},
        ...more data points...
      ],
      "forecast_periods": 7,
      "mode": "local",
      "local_model": "auto_arima",
      "freq": "D"
    }
    ```
    
    ## Example Response
    ```json
    {
      "leadtime_quantiles": [
        {
          "item_id": 100,
          "lead_time_periods": 7,
          "quantiles": {
            "q_90": 850.5,
            "q_95": 920.3,
            "q_99": 1050.2
          }
        }
      ]
    }
    ```
    """
    try:
        print(f"Starting lead-time quantile calculation")
        
        # Convert input data to DataFrame
        df_his = pd.DataFrame(request.sim_input_his)
        df_his['day'] = pd.to_datetime(df_his['day'])
        
        # Initialize forecaster
        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles or [0.5, 0.9, 0.95, 0.99],
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )
        
        print(f"Calculating quantiles for lead-time: {request.forecast_periods} periods")
        
        # Calculate quantiles for each item
        results = []
        unique_items = df_his['item_id'].unique()
        
        for item_id in unique_items:
            try:
                item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)
                
                # Calculate quantiles for different service levels
                quantiles = {}
                for service_level in [0.90, 0.95, 0.99]:
                    q_value = forecaster.leadtime_total_quantile(
                        item_data,
                        L=request.forecast_periods,
                        serv_lev=service_level,
                        trials=2000
                    )
                    quantiles[f'q_{int(service_level*100)}'] = round(q_value, 2)
                
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'lead_time_periods': request.forecast_periods,
                    'quantiles': quantiles
                })
                
                print(f"  ✓ Item {item_id}: quantiles calculated")
                
            except Exception as e:
                print(f"  ✗ Item {item_id}: {str(e)}")
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'error': str(e)
                })
        
        print(f"Quantile calculation completed: {len(results)} items processed")
        
        return {
            'leadtime_quantiles': results,
            'total_items': len(unique_items),
            'lead_time_periods': request.forecast_periods,
            'mode': request.mode
        }
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Quantile calculation error: {str(e)}")