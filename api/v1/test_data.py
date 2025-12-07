"""Test data export endpoints."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from create_test_data.services.forecast_input_builder import ForecastInputForecastBuilder
from create_test_data.services.json_exporter import SandboxJSONExporter

router = APIRouter()


class InputAndForecastRequest(BaseModel):
    """Request body for generating forecast input and calling the forecast endpoint."""

    item_number: int
    forecast_periods: int = 6
    mode: str = "local"
    local_model: str = "auto_ets"
    season_length: int = 12
    freq: str = "MS"
    quantiles: Optional[List[float]] = None
    endpoint: Optional[str] = None
    timeout: int = 30
    history_base_url: Optional[str] = None
    history_timeout: Optional[int] = None
    overrides: Optional[Dict[str, Any]] = None


def _build_exporter(minimum_date: Optional[date], status: Optional[str]) -> SandboxJSONExporter:
    """Create a SandboxJSONExporter with optional overrides."""
    kwargs = {}
    if minimum_date is not None:
        kwargs["minimum_date"] = minimum_date
    if status is not None:
        kwargs["status"] = status

    try:
        return SandboxJSONExporter(**kwargs)
    except Exception as exc:  # pragma: no cover - safe-guard for runtime misconfig
        raise HTTPException(status_code=500, detail=f"Failed to initialise exporter: {exc}") from exc


@router.get("/history", name="test-data:history")
def read_history(minimum_date: Optional[date] = None, status: Optional[str] = None):
    """Return historical sales records in the sandbox dataset."""
    exporter = _build_exporter(minimum_date, status)
    payload = json.loads(exporter.history_as_json())
    return {"history": payload}


@router.get("/history/{item_id}", name="test-data:history-item")
def read_history_for_item(
    item_id: int,
    minimum_date: Optional[date] = None,
    status: Optional[str] = None,
):
    """Return historical sales records for a single item."""
    exporter = _build_exporter(minimum_date, status)
    try:
        payload = json.loads(exporter.history_for_item_as_json(item_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "metadata": {
            "item": item_id,
            "minimum_date": (minimum_date.isoformat() if minimum_date else None),
            "status": status,
        },
        "history": payload,
    }


@router.get("/products", name="test-data:products")
def read_products(status: Optional[str] = None):
    """Return product metadata from the sandbox dataset."""
    exporter = _build_exporter(None, status)
    payload = json.loads(exporter.products_as_json())
    return {"products": payload}


@router.get("/combined", name="test-data:combined")
def read_combined(minimum_date: Optional[date] = None, status: Optional[str] = None):
    """Return combined history and product data for the sandbox dataset."""
    exporter = _build_exporter(minimum_date, status)
    return exporter.combined_payload()


@router.post("/input-and-forecast", name="test-data:input-and-forecast")
def create_input_and_forecast(request: InputAndForecastRequest):
    """Fetch test history, build forecast payload, and call the forecast API."""

    builder_kwargs: Dict[str, Any] = {
        "item_number": request.item_number,
        "forecast_periods": request.forecast_periods,
        "mode": request.mode,
        "local_model": request.local_model,
        "season_length": request.season_length,
        "freq": request.freq,
    }

    if request.history_base_url:
        builder_kwargs["base_url"] = request.history_base_url
    if request.history_timeout is not None:
        builder_kwargs["request_timeout"] = request.history_timeout

    builder = ForecastInputForecastBuilder(**builder_kwargs)

    try:
        history = builder.fetch_history()
        payload = builder.build_payload(history)
        endpoint = request.endpoint or "https://api.nostradamus-api.com/api/v1/forecast/generate"
        forecast_result = builder.generate_forecast(
            payload=payload,
            endpoint=endpoint,
            timeout=request.timeout,
            quantiles=request.quantiles,
            **(request.overrides or {}),
        )
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "item_number": request.item_number,
        "history": history,
        "payload": forecast_result["payload"],
        "forecast": forecast_result["forecast"],
    }
