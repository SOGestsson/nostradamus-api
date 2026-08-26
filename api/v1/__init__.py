"""
API v1 router - aggregates all v1 endpoints.
"""
from fastapi import APIRouter, Depends

from api.security import require_api_key
from api.v1 import forecast, lightgpt, lightgbm, simulation, test_data

router = APIRouter(dependencies=[Depends(require_api_key)])


# Include simulation routes
router.include_router(simulation.router, prefix="/simulation", tags=["simulation"])

# Include forecast routes
router.include_router(forecast.router, prefix="/forecast", tags=["forecast"])

# Include LightGPT routes
router.include_router(lightgpt.router, prefix="/lightgpt", tags=["lightgpt"])

# Include LightGBM routes
router.include_router(lightgbm.router, prefix="/lightgbm", tags=["lightgbm"])
 
# Include sandbox test data routes
router.include_router(test_data.router, prefix="/test-data", tags=["test-data"])
