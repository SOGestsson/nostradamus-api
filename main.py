"""
Inventory Simulation API - Main Entry Point

FastAPI application for inventory optimization and forecasting simulations.
"""
from fastapi import FastAPI
from api.v1 import router as v1_router

VERSION = "2.1.0"  # 👉 uppfærð útgáfa

app = FastAPI(
    title="Inventory Simulation API",
    description="API for inventory optimization and forecasting simulations",
    version=VERSION,
)

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
