# api/v1/forecast.py - New file for forecast endpoints

"""
Forecast endpoints using ClassicalForecasts.
"""
import traceback
from typing import Dict, Any, List
import pandas as pd

from fastapi import APIRouter, HTTPException

from api.models import ForecastRequest
from inventory_algorithm.classical_forecasts import ClassicalForecasts

router = APIRouter()


@router.post("/generate")
def generate_forecast(request: ForecastRequest):
    """
    Generate forecasts for items using classical forecasting models or TimeGPT.
    
    Supports multiple forecasting modes:
    - 'local': StatsForecast models (AutoARIMA, ETS, Croston, etc.)
    - 'timegpt': Nixtla TimeGPT (requires API key)
    
    Args:
        request: ForecastRequest with historical data and forecast parameters
        
    Returns:
        Dictionary containing forecasts for each item
        
    Raises:
        HTTPException: If forecast generation fails
    """
    try:
        print(f"Starting forecast generation with mode: {request.mode}")
        
        # Convert input data to DataFrame
        df_his = pd.DataFrame(request.sim_input_his)
        
        # Validate required columns
        required_cols = ['item_id', 'actual_sale', 'day']
        missing_cols = [col for col in required_cols if col not in df_his.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column to datetime
        df_his['day'] = pd.to_datetime(df_his['day'])
        
        # Initialize forecaster
        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles,
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )
        
        print(f"Forecaster initialized: {request.local_model if request.mode == 'local' else 'TimeGPT'}")
        
        # Generate forecasts for each item
        results = []
        unique_items = df_his['item_id'].unique()
        
        print(f"Generating forecasts for {len(unique_items)} items")
        
        for item_id in unique_items:
            try:
                # Filter data for this item
                item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)
                
                # Generate forecast
                forecast_values = forecaster.daily_path(item_data, periods=request.forecast_periods)
                
                # Generate future dates
                last_date = item_data['day'].max()
                future_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1 if request.freq == 'D' else 0),
                    periods=request.forecast_periods,
                    freq=request.freq
                )
                
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'forecast': forecast_values.tolist(),
                    'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'model_used': request.local_model if request.mode == 'local' else 'timegpt',
                    'periods_forecasted': len(forecast_values)
                })
                
                print(f"  ✓ Item {item_id}: forecast generated")
                
            except Exception as e:
                print(f"  ✗ Item {item_id}: {str(e)}")
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'error': str(e),
                    'forecast': [],
                    'forecast_dates': []
                })
        
        print(f"Forecast generation completed: {len(results)} items processed")
        
        return {
            'forecasts': results,
            'total_items': len(unique_items),
            'mode': request.mode,
            'model': request.local_model if request.mode == 'local' else 'timegpt',
            'periods': request.forecast_periods,
            'frequency': request.freq
        }
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@router.post("/leadtime_quantile")
def calculate_leadtime_quantile(request: ForecastRequest):
    """
    Calculate lead-time demand quantiles for safety stock calculations.
    
    Uses Monte Carlo simulation to estimate demand variability over lead time,
    accounting for forecast uncertainty.
    
    Args:
        request: ForecastRequest with historical data and parameters
        
    Returns:
        Dictionary containing lead-time quantiles for each item
        
    Raises:
        HTTPException: If calculation fails
    """
    try:
        print(f"Starting lead-time quantile calculation")
        
        # Convert input data to DataFrame
        df_his = pd.DataFrame(request.sim_input_his)
        df_his['day'] = pd.to_datetime(df_his['day'])
        
        # Initialize forecaster
        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles or [0.5, 0.9, 0.95, 0.99],
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )
        
        print(f"Calculating quantiles for lead-time: {request.forecast_periods} periods")
        
        # Calculate quantiles for each item
        results = []
        unique_items = df_his['item_id'].unique()
        
        for item_id in unique_items:
            try:
                item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)
                
                # Calculate quantiles for different service levels
                quantiles = {}
                for service_level in [0.90, 0.95, 0.99]:
                    q_value = forecaster.leadtime_total_quantile(
                        item_data,
                        L=request.forecast_periods,
                        serv_lev=service_level,
                        trials=2000
                    )
                    quantiles[f'q_{int(service_level*100)}'] = round(q_value, 2)
                
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'lead_time_periods': request.forecast_periods,
                    'quantiles': quantiles
                })
                
                print(f"  ✓ Item {item_id}: quantiles calculated")
                
            except Exception as e:
                print(f"  ✗ Item {item_id}: {str(e)}")
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'error': str(e)
                })
        
        print(f"Quantile calculation completed: {len(results)} items processed")
        
        return {
            'leadtime_quantiles': results,
            'total_items': len(unique_items),
            'lead_time_periods': request.forecast_periods,
            'mode': request.mode
        }
            
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Full error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Quantile calculation error: {str(e)}")