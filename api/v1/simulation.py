"""
Inventory simulation endpoints.
"""
import json
import random
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException

import inventory_algorithm.inventory_opt_and_forecasting_package as inv
import inventory_algorithm.simulator_class as sim
from api.models import CoreSimInput, SimInput, SimulationRequest
from api.deps import build_dataframes

router = APIRouter()

# Monte Carlo uses global numpy/random state — serialize multi-sim workers.
_sim_mc_lock = threading.Lock()


def _apply_random_seed(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def _resolve_item_seed(request: SimulationRequest) -> int | None:
    if request.random_seed is not None:
        return int(request.random_seed)
    if request.sim_input_his:
        try:
            return int(request.sim_input_his[0]["item_id"])
        except (TypeError, ValueError, KeyError, IndexError):
            return None
    return None


def coerce_sim_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns/dtypes to what inventory_algorithm.simulator_class expects.

    This avoids `None` values leaking into numeric arrays (e.g. actual_sale), which
    can crash when the simulator calls math.isnan().
    """
    required_cols = [
        "item_id",
        "day",
        "forecast",
        "variance",
        "actual_sale",
        "order_day",
        "delivery",
        "extra_params",
    ]

    out = df.copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = None

    out = out[required_cols]

    # Strings
    out["day"] = out["day"].astype("object")
    out["extra_params"] = out["extra_params"].fillna("").astype(str)

    # Floats (None -> NaN)
    for col in ["forecast", "variance", "actual_sale"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # Ints
    for col in ["item_id", "order_day", "delivery"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")

    return out



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


@router.post("/histo_lead")
def run_histogram_lead_freq(request: SimulationRequest):
    """
    Run simulation and return histogram_lead frequency data.

    Executes inventory simulation and returns the lead time frequency histogram,
    which shows the distribution of lead times observed over the simulation period.

    Args:
        request: SimulationRequest with data and simulation parameters

    Returns:
        Dictionary containing histogram_lead data in serializable format

    Raises:
        HTTPException: If simulation fails
    """
    try:
        print("Starting histogram_lead simulation...")

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

        result = inv_sim.histogram_lead

        print(f"histogram_lead type: {type(result)}")
        print(f"histogram_lead value: {result}")

        if hasattr(result, 'to_dict'):
            return {"histogram_lead": result.to_dict(orient="records")}
        elif hasattr(result, 'tolist'):
            return {"histogram_lead": result.tolist()}
        elif isinstance(result, dict):
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):
                    serializable_result[key] = value.tolist()
                elif hasattr(value, 'to_dict'):
                    serializable_result[key] = value.to_dict(orient="records")
                else:
                    serializable_result[key] = str(value)
            return {"histogram_lead": serializable_result}
        elif isinstance(result, (list, int, float, str, bool, type(None))):
            return {"histogram_lead": result}
        else:
            return {"histogram_lead": str(result)}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Histogram lead error: {str(e)}")


@router.post("/sim_prep")
def run_sim_prep(request: SimulationRequest):
    """
    Run simulation setup and return initial Monte Carlo parameters and prepared simulator input.

    Executes the input preparation steps without running the full simulation, returning:
    - serv_level_value_lead: Monte Carlo service-level forecast for lead time
    - serv_level_value_buy: Monte Carlo service-level forecast for buy frequency
    - histo_with_cum_lead: lead time histogram with cumulative distribution
    - histo_with_cum_buy: buy frequency histogram with cumulative distribution
    - simulator_input_his: the prepared input DataFrame fed into the simulator

    Args:
        request: SimulationRequest with data and simulation parameters

    Returns:
        Dictionary containing the initial parameters in serializable format

    Raises:
        HTTPException: If preparation fails
    """
    try:
        print("Starting sim_prep...")
        _apply_random_seed(_resolve_item_seed(request))

        dfs = build_dataframes(SimInput(
            sim_input_his=request.sim_input_his,
            sim_rio_items=request.sim_rio_items,
            sim_rio_item_details=request.sim_rio_item_details,
            sim_rio_on_order=request.sim_rio_on_order
        ))

        sim_input_his = dfs["sim_input_his"]
        sim_rio_items = dfs["sim_rio_items"]
        sim_rio_item_details = dfs["sim_rio_item_details"]
        sim_rio_on_order = dfs["sim_rio_on_order"]

        inv_sim = inv.inventory_simulator_with_input_prep(
            sim_input_his,
            sim_rio_items,
            sim_rio_on_order,
            sim_rio_item_details,
            request.number_of_days,
            request.number_of_simulations,
            request.service_level
        )

        def to_serializable(val):
            if hasattr(val, 'to_dict'):
                return val.to_dict(orient="records")
            elif hasattr(val, 'tolist'):
                return val.tolist()
            elif hasattr(val, 'item'):
                return val.item()
            elif isinstance(val, (list, int, float, str, bool, type(None))):
                return val
            else:
                return str(val)

        return {
            "histogram_lead": to_serializable(inv_sim.histogram_lead),
            "serv_level_value_lead": to_serializable(inv_sim.serv_level_value_lead),
            "histo_with_cum_lead": to_serializable(inv_sim.histo_with_cum_lead),
            "histogram_buy": to_serializable(inv_sim.histogram_buy),
            "serv_level_value_buy": to_serializable(inv_sim.serv_level_value_buy),
            "histo_with_cum_buy": to_serializable(inv_sim.histo_with_cum_buy),
            "simulator_input_his": to_serializable(inv_sim.simulator_input_his),
            "sim_result": to_serializable(inv_sim.sim_result),
        }

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Sim prep error: {str(e)}")


@router.post("/multi-sim")
def run_multi_sim(requests: List[SimulationRequest]):
    """
    Run simulation for multiple part numbers and return combined results.

    Accepts a list of SimulationRequest objects (the output of the sim-prep endpoint)
    and runs inventory_simulator_with_input_prep for each one. Returns the combined
    sim_result across all part numbers and purchase suggestions.

    Args:
        requests: List of SimulationRequest, one per part number

    Returns:
        Dictionary with sim_result (all rows combined) and purchase_suggestions

    Raises:
        HTTPException: If any simulation fails
    """
    def _run_single(request: SimulationRequest):
        label = request.sim_rio_items[0].get("pn") if request.sim_rio_items else "unknown"
        try:
            # Global numpy/random is not thread-safe; pin seed per item like Deep Dive.
            with _sim_mc_lock:
                _apply_random_seed(_resolve_item_seed(request))
                dfs = build_dataframes(SimInput(
                    sim_input_his=request.sim_input_his,
                    sim_rio_items=request.sim_rio_items,
                    sim_rio_item_details=request.sim_rio_item_details,
                    sim_rio_on_order=request.sim_rio_on_order,
                ))
                inv_sim = inv.inventory_simulator_with_input_prep(
                    dfs["sim_input_his"],
                    dfs["sim_rio_items"],
                    dfs["sim_rio_on_order"],
                    dfs["sim_rio_item_details"],
                    request.number_of_days,
                    request.number_of_simulations,
                    request.service_level,
                )
            return label, inv_sim.sim_result, None
        except Exception as e:
            print(f"Error on item_id {label}:\n{traceback.format_exc()}")
            return label, None, str(e)

    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_run_single, req): req for req in requests}
        for future in as_completed(futures):
            item_id, sim_result, error = future.result()
            if error:
                errors.append({"item_id": item_id, "error": error})
            else:
                results.append(sim_result)

    if not results:
        return {"sim_result": [], "purchase_suggestions": [], "errors": errors}

    final_res = pd.concat(results, ignore_index=True)

    if final_res.empty:
        return {"sim_result": [], "purchase_suggestions": [], "errors": errors}

    purchase_suggestions = pd.DataFrame()
    if "item_id" in final_res.columns and "sim_date" in final_res.columns and "purchase_qty" in final_res.columns:
        oldest = final_res.loc[final_res.groupby("item_id")["sim_date"].idxmin()]
        purchase_suggestions = oldest[["item_id", "purchase_qty"]].copy()
        purchase_suggestions["current_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "sim_result": final_res.to_dict(orient="records"),
        "purchase_suggestions": purchase_suggestions.to_dict(orient="records"),
        "errors": errors,
    }


@router.post("/raw_simulate")
def run_raw_simulation(request: CoreSimInput):
    """Run inventory simulation directly from a prepared simulator input DataFrame.

    Expects `request.inv_df` to represent the simulator input dataset (records).
    """
    try:
        print("Starting raw simulation...")

        inv_df = pd.DataFrame(request.inv_df)
        inv_df = coerce_sim_input_df(inv_df)

        required_cols = {"item_id", "day", "forecast", "order_day", "delivery", "extra_params"}
        missing = sorted(required_cols - set(inv_df.columns))
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"inv_df is missing required columns: {missing}",
            )

        # Run simulator
        sim_result = sim.inventory_simulator(inv_df)

        if not hasattr(sim_result, "simulator_output"):
            return {"simulator_output": sim_result}

        output = sim_result.simulator_output
        try:
            print(output.head(100))
        except Exception:
            pass

        if hasattr(output, "to_dict"):
            return {"simulator_output": output.to_dict(orient="records")}
        return {"simulator_output": output}

    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Raw simulation error: {str(e)}")
