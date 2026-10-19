"""API key authentication for the versioned API.

Applied as a router-level dependency in `api/v1/__init__.py`, so every `/api/v1`
endpoint is covered without each route opting in. `/` and `/health` live on the
app itself and stay public for container health checks.

Local forecasts stay public: classical generate/jobs and LightGBM. Those models
run in-process and do not use a Nixtla/TimeGPT key. LightGPT, simulation, and
the rest of `/api/v1` still require `X-Api-Key`.
"""
import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger("uvicorn.error")

_MISSING_KEY_LOGGED = False

# Classical generate (sync/async/job). Job status uses the prefix below.
LOCAL_FORECAST_PATHS = frozenset({
    "/api/v1/forecast/generate",
    "/api/v1/forecast/generate_daily",
    "/api/v1/forecast/generate_async",
    "/api/v1/forecast/generate_job",
})

LOCAL_FORECAST_PREFIXES = (
    "/api/v1/forecast/jobs/",
    "/api/v1/lightgbm/",
)


def _is_local_forecast_path(path: str) -> bool:
    p = (path or "").rstrip("/")
    if p in LOCAL_FORECAST_PATHS:
        return True
    return any(p.startswith(prefix) for prefix in LOCAL_FORECAST_PREFIXES)


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> bool:
    """Reject the request unless `X-Api-Key` matches the configured `API_KEY`.

    Fails closed: an unset `API_KEY` rejects every request rather than
    disabling authentication. Local classical and LightGBM forecasts are the
    exception — they do not use a Nixtla/TimeGPT key.
    """
    global _MISSING_KEY_LOGGED

    if _is_local_forecast_path(request.url.path):
        return True

    expected = (os.getenv("API_KEY") or "").strip()
    if not expected:
        if not _MISSING_KEY_LOGGED:
            logger.error(
                "API_KEY is not set — refusing all /api/v1 requests. "
                "Set it in the environment to enable the API."
            )
            _MISSING_KEY_LOGGED = True
        raise HTTPException(status_code=503, detail="API authentication is not configured")

    provided = (x_api_key or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True
