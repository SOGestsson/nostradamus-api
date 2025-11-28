"""
API v1 router - aggregates all v1 endpoints.
"""
from fastapi import APIRouter
from api.v1 import simulation, forecast

router = APIRouter()


# Include simulation routes
router.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

# Include forecast routes
router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])
