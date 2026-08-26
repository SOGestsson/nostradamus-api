"""API key authentication for the versioned API.

Applied as a router-level dependency in `api/v1/__init__.py`, so every `/api/v1`
endpoint is covered without each route opting in. `/` and `/health` live on the
app itself and stay public for container health checks.
"""
import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException

logger = logging.getLogger("uvicorn.error")

_MISSING_KEY_LOGGED = False


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> bool:
    """Reject the request unless `X-Api-Key` matches the configured `API_KEY`.

    Fails closed: an unset `API_KEY` rejects every request rather than
    disabling authentication.
    """
    global _MISSING_KEY_LOGGED

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
