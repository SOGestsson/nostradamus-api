"""API key authentication for the versioned API.

Applied as a router-level dependency in `api/v1/__init__.py`, so every `/api/v1`
endpoint is covered without each route opting in. `/` and `/health` live on the
app itself and stay public for container health checks.

Local classical forecasts and LightGBM stay **public** when no partner keys are
configured (`API_KEYS` unset). Once `API_KEYS` is set, those routes require
`X-Api-Key` matching `API_KEY` (internal) or a partner entry. LightGPT,
simulation, and the rest of `/api/v1` always require a valid key when `API_KEY` is set.

Environment:
  API_KEY  — internal systems (Fagrihvammur / pipeline gateway). Client id ``internal``.
  API_KEYS — optional external partners: ``partner_id:secret,other:secret``
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger("uvicorn.error")

_MISSING_KEY_LOGGED = False

INTERNAL_CLIENT_ID = "internal"

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


def _parse_api_keys_env(raw: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            logger.warning("Skipping API_KEYS entry without partner_id prefix")
            continue
        partner_id, key = part.split(":", 1)
        partner_id = partner_id.strip()
        key = key.strip()
        if partner_id and key:
            entries.append((partner_id, key))
    return entries


def configured_api_clients() -> list[tuple[str, str]]:
    clients: list[tuple[str, str]] = []
    main = (os.getenv("API_KEY") or "").strip()
    if main:
        clients.append((INTERNAL_CLIENT_ID, main))
    extra = (os.getenv("API_KEYS") or "").strip()
    if extra:
        clients.extend(_parse_api_keys_env(extra))
    return clients


def partner_keys_configured() -> bool:
    return bool((os.getenv("API_KEYS") or "").strip())


def resolve_api_client(provided_key: str) -> str | None:
    matched: str | None = None
    for client_id, expected in configured_api_clients():
        if secrets.compare_digest(provided_key, expected):
            matched = client_id
    return matched


def _is_local_forecast_path(path: str) -> bool:
    p = (path or "").rstrip("/")
    if p in LOCAL_FORECAST_PATHS:
        return True
    return any(p.startswith(prefix) for prefix in LOCAL_FORECAST_PREFIXES)


def _validate_api_key(request: Request, x_api_key: Optional[str]) -> bool:
    global _MISSING_KEY_LOGGED

    clients = configured_api_clients()
    if not clients:
        if not _MISSING_KEY_LOGGED:
            logger.error(
                "API_KEY is not set — refusing protected /api/v1 requests. "
                "Set API_KEY (and optional API_KEYS for partners) in the environment."
            )
            _MISSING_KEY_LOGGED = True
        raise HTTPException(status_code=503, detail="API authentication is not configured")

    provided = (x_api_key or "").strip()
    if not provided:
        raise HTTPException(status_code=401, detail="Invalid API key")

    client_id = resolve_api_client(provided)
    if client_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request.state.api_client_id = client_id
    return True


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> bool:
    path = request.url.path
    if _is_local_forecast_path(path) and not partner_keys_configured():
        return True
    return _validate_api_key(request, x_api_key)
