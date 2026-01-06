# api/v1/lightgpt.py - New file

"""
LightGPT forecast endpoints for batch processing with drivers and attributes.
"""
import traceback
from typing import Dict, Any, List, Optional
import pandas as pd

from fastapi import APIRouter, HTTPException, Header

from api.models import LightGPTForecastRequest
from inventory_algorithm.lightgpt_forecasts import LightGPTForecast

router = APIRouter()


@router.post("/batch")
def batch_forecast_with_drivers(
    request: LightGPTForecastRequest,
    x_nixtla_api_key: str | None = Header(default=None, alias="X-Nixtla-Api-Key"),
):
    """
    Batch forecast for multiple items with external drivers and item attributes.
    
    Supports:
    - Multiple items in single request
    - External drivers (price, promotion, seasonality)
    - Item attributes (brand, category, supplier)
    - Cross-item learning
    
    **sim_input_his**: Historical sales data
    - Columns: item_id, day, actual_sale (required) + optional driver columns
    
    **item_attributes**: Item metadata
    - Columns: item_id, brand, category, supplier, margin, etc.
    - Used for grouping and cross-learning
    
    **external_drivers**: External regressors
    - Columns: item_id, day, driver_name, driver_value
    - Or global: day, driver_name, driver_value
    
    **freq**: Input/output frequency
    - 'D' for daily
    - 'MS'/'M'/'monthly' for monthly (handled as month-start internally)

    **exogenous_columns**: Which driver columns to use
    - Example: ['price', 'promotion', 'seasonality_index']
    """
    try:
        print(f"Starting LightGPT batch forecast")
        
        # Convert to DataFrames
        df_hist = pd.DataFrame(request.sim_input_his)
        df_hist['day'] = pd.to_datetime(df_hist['day'])
        
        df_items = None
        if request.item_attributes:
            df_items = pd.DataFrame(request.item_attributes)
        
        df_drivers = None
        if request.external_drivers:
            df_drivers = pd.DataFrame(request.external_drivers)
            if 'day' in df_drivers.columns:
                df_drivers['day'] = pd.to_datetime(df_drivers['day'])
        
        # Initialize forecaster (requires Nixtla key)
        api_key = request.api_key or x_nixtla_api_key
        model = request.model or 'timegpt-1'
        forecaster = LightGPTForecast(freq=request.freq, api_key=api_key, model=model)
        freq_used = forecaster.freq_label
        
        # Generate forecasts
        result = forecaster.batch_forecast_with_drivers(
            hist=df_hist,
            item_attributes=df_items,
            drivers=df_drivers,
            forecast_periods=request.forecast_periods,
            exogenous_columns=request.exogenous_columns
        )
        
        return {
            'forecasts': result.to_dict(orient='records'),
            'total_items': df_hist['item_id'].nunique(),
            'forecast_type': 'batch',
            'periods': request.forecast_periods,
            'freq': freq_used,
        }
        
    except ValueError as e:
        # Common case: missing NIXTLA_API_KEY (either env var or request.api_key)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"LightGPT forecast error: {str(e)}")


@router.post("/cross_learning")
def cross_learning_forecast(
    request: LightGPTForecastRequest,
    x_nixtla_api_key: str | None = Header(default=None, alias="X-Nixtla-Api-Key"),
):
    """
    Forecast with cross-learning by item groups (brand, category, etc.).
    Items in same group share information for better forecasts.
    """
    try:
        if not request.group_column:
            raise ValueError("group_column required for cross_learning")
        
        df_hist = pd.DataFrame(request.sim_input_his)
        if 'day' in df_hist.columns:
            df_hist['day'] = pd.to_datetime(df_hist['day'])
        df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
        
        if df_items is None:
            raise ValueError("item_attributes required for cross_learning")
        
        api_key = request.api_key or x_nixtla_api_key
        model = request.model or 'timegpt-1'
        forecaster = LightGPTForecast(freq=request.freq, api_key=api_key, model=model)
        freq_used = forecaster.freq_label
        
        results = forecaster.forecast_with_cross_learning(
            hist=df_hist,
            item_attributes=df_items,
            group_column=request.group_column,
            forecast_periods=request.forecast_periods
        )
        
        # Combine results
        combined = pd.concat(results.values(), ignore_index=True)
        
        return {
            'forecasts': combined.to_dict(orient='records'),
            'total_items': df_hist['item_id'].nunique(),
            'groups': list(results.keys()),
            'forecast_type': 'cross_learning',
            'group_column': request.group_column,
            'freq': freq_used,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Cross-learning error: {str(e)}")


@router.post("/hierarchical")
def hierarchical_forecast(
    request: LightGPTForecastRequest,
    x_nixtla_api_key: str | None = Header(default=None, alias="X-Nixtla-Api-Key"),
):
    """
    Hierarchical forecast respecting category structure.
    Ensures forecasts are coherent across hierarchy levels.
    """
    try:
        if not request.hierarchy:
            raise ValueError("hierarchy required (e.g., ['brand', 'category', 'item_id'])")
        
        df_hist = pd.DataFrame(request.sim_input_his)
        if 'day' in df_hist.columns:
            df_hist['day'] = pd.to_datetime(df_hist['day'])
        df_items = pd.DataFrame(request.item_attributes) if request.item_attributes else None
        
        api_key = request.api_key or x_nixtla_api_key
        model = request.model or 'timegpt-1'
        forecaster = LightGPTForecast(freq=request.freq, api_key=api_key, model=model)
        freq_used = forecaster.freq_label
        
        result = forecaster.hierarchical_forecast(
            hist=df_hist,
            item_attributes=df_items,
            hierarchy=request.hierarchy,
            forecast_periods=request.forecast_periods
        )
        
        return {
            'forecasts': result.to_dict(orient='records'),
            'total_items': df_hist['item_id'].nunique(),
            'hierarchy': request.hierarchy,
            'forecast_type': 'hierarchical',
            'freq': freq_used,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Hierarchical forecast error: {str(e)}")