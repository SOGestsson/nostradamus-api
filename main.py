"""
Inventory Simulation API - Main Entry Point

FastAPI application for inventory optimization and forecasting simulations.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from api.v1 import router as v1_router

VERSION = "2.2.0"  # 👉 uppfærð útgáfa

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Inventory Simulation API",
    description="API for inventory optimization and forecasting simulations",
    version=VERSION,
)


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError):
    # 422s are otherwise invisible in server logs, which makes client-side
    # payload bugs hard to diagnose. Log loc/type/msg but not the (huge) input.
    summary = [
        {k: v for k, v in err.items() if k != 'input'}
        for err in exc.errors()[:10]
    ]
    logger.warning("422 validation error on %s %s: %s", request.method, request.url.path, summary)
    return await request_validation_exception_handler(request, exc)

# Include v1 router
app.include_router(v1_router, prefix="/api/v1")


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Inventory Simulation API",
        "version": VERSION,   # 👉 nota sömu breytu hér
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}
