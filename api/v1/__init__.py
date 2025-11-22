"""
API v1 router - aggregates all v1 endpoints.
"""
from fastapi import APIRouter
from api.v1 import simulation

router = APIRouter()

# Include simulation endpoints
router.include_router(
    simulation.router,
    prefix="/simulation",
    tags=["simulation"]
)
