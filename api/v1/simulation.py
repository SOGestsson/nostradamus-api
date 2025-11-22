"""
Inventory simulation endpoints.
"""
import traceback

from fastapi import APIRouter, HTTPException

import inventory_algorithm.inventory_opt_and_forecasting_package as inv
from api.models import SimInput, SimulationRequest
from api.deps import build_dataframes

router = APIRouter()


@router.post("/simulate")
def run_inventory_simulation(request: SimulationRequest):
    """
    Run inventory simulation with input prep using JSON data.
    
    Executes the inventory simulator with historical sales data and item configuration,
    returning simulation results including purchase suggestions and inventory levels.
    
    Args:
        request: SimulationRequest with data and simulation parameters
        
    Returns:
        Dictionary containing simulation results
        
    Raises:
        HTTPException: If simulation fails
    """
    try:
        print("Starting simulation...")
        
        # Build DataFrames from JSON
        dfs = build_dataframes(SimInput(
            sim_input_his=request.sim_input_his,
            sim_rio_items=request.sim_rio_items,
            sim_rio_item_details=request.sim_rio_item_details,
            sim_rio_on_order=request.sim_rio_on_order
        ))
        
        print("DataFrames built successfully")
        
        sim_input_his = dfs["sim_input_his"]
        sim_rio_items = dfs["sim_rio_items"]
        sim_rio_item_details = dfs["sim_rio_item_details"]
        sim_rio_on_order = dfs["sim_rio_on_order"]
        
        print("Running simulation...")
        
        # Run the simulation (buy_freq and del_time are already in the JSON/DataFrame)
        inv_sim = inv.inventory_simulator_with_input_prep(
            sim_input_his, 
            sim_rio_items, 
            sim_rio_on_order, 
            sim_rio_item_details,
            request.number_of_days, 
            request.number_of_simulations, 
            request.service_level
        )
        
        print("Simulation completed")
        
        # Return results
        result = inv_sim.sim_result
        if hasattr(result, 'to_dict'):
            return {"sim_result": result.to_dict(orient="records")}
        else:
            return {"sim_result": result}
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.post("/histo_buy")
def run_histogram_buy_freq(request: SimulationRequest):
    """
    Run simulation and return histogram_buy frequency data.
    
    Executes inventory simulation and returns the buying frequency histogram,
    which shows the distribution of purchase quantities needed over the simulation period.
    
    Args:
        request: SimulationRequest with data and simulation parameters
        
    Returns:
        Dictionary containing histogram_buy data in serializable format
        
    Raises:
        HTTPException: If simulation fails
    """
    try:
        print("Starting histogram_buy simulation...")
        
        # Build DataFrames from JSON
        dfs = build_dataframes(SimInput(
            sim_input_his=request.sim_input_his,
            sim_rio_items=request.sim_rio_items,
            sim_rio_item_details=request.sim_rio_item_details,
            sim_rio_on_order=request.sim_rio_on_order
        ))
        
        print("DataFrames built successfully")
        
        sim_input_his = dfs["sim_input_his"]
        sim_rio_items = dfs["sim_rio_items"]
        sim_rio_item_details = dfs["sim_rio_item_details"]
        sim_rio_on_order = dfs["sim_rio_on_order"]
        
        print("Running simulation...")
        
        # Run the simulation
        inv_sim = inv.inventory_simulator_with_input_prep(
            sim_input_his, 
            sim_rio_items, 
            sim_rio_on_order, 
            sim_rio_item_details,
            request.number_of_days, 
            request.number_of_simulations, 
            request.service_level
        )
        
        print("Simulation completed")
        
        # Return histogram_buy results with improved serialization
        result = inv_sim.histogram_buy
        
        print(f"histogram_buy type: {type(result)}")
        print(f"histogram_buy value: {result}")
        
        # Convert to serializable format
        if hasattr(result, 'to_dict'):
            # If it's a DataFrame
            return {"histogram_buy": result.to_dict(orient="records")}
        elif hasattr(result, 'tolist'):
            # If it's a numpy array
            return {"histogram_buy": result.tolist()}
        elif isinstance(result, dict):
            # If it's a dict, check if values are serializable
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):
                    serializable_result[key] = value.tolist()
                elif hasattr(value, 'to_dict'):
                    serializable_result[key] = value.to_dict(orient="records")
                else:
                    serializable_result[key] = str(value)
            return {"histogram_buy": serializable_result}
        elif isinstance(result, (list, int, float, str, bool, type(None))):
            # If it's a basic Python type
            return {"histogram_buy": result}
        else:
            # Last resort: convert to string
            return {"histogram_buy": str(result)}
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Histogram buy error: {str(e)}")
