"""
API v1 router - aggregates all v1 endpoints.
"""
from fastapi import APIRouter
from api.v1 import forecast, lightgpt, simulation, test_data

router = APIRouter()


# Include simulation routes
router.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

# Include forecast routes
router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])

# Include LightGPT routes
router.include_router(lightgpt.router, prefix="/lightgpt", tags=["lightgpt"])
 
# Include sandbox test data routes
router.include_router(test_data.router, prefix="/test-data", tags=["test-data"])
