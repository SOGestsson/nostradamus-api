"""Test data export endpoints."""
from __future__ import annotations

import json
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException

from create_test_data.services.json_exporter import SandboxJSONExporter

router = APIRouter()


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
