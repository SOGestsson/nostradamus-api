"""
Pydantic models for API requests and responses.
"""
from typing import List, Dict, Any
from pydantic import BaseModel


class SimInput(BaseModel):
    """Input model for simulation data containing historical sales and item configuration."""
    sim_input_his: List[Dict[str, Any]]
    sim_rio_items: List[Dict[str, Any]]
    sim_rio_item_details: List[Dict[str, Any]]
    sim_rio_on_order: List[Dict[str, Any]]


class SimulationRequest(BaseModel):
    """Request model for simulation execution with configurable parameters."""
    sim_input_his: List[Dict[str, Any]]
    sim_rio_items: List[Dict[str, Any]]
    sim_rio_item_details: List[Dict[str, Any]]
    sim_rio_on_order: List[Dict[str, Any]]
    number_of_days: int = 900
    number_of_simulations: int = 1000
    service_level: float = 0.95
