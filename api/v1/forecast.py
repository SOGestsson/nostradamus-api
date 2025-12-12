# api/v1/forecast.py - Updated with detailed parameter descriptions

"""
Forecast endpoints using ClassicalForecasts.
"""
import traceback
import asyncio
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
    
    ## Forecasting Modes
    - **local**: StatsForecast models (AutoARIMA, ETS, Croston, etc.) - No API key needed
    - **timegpt**: Nixtla TimeGPT cloud API - Requires API key
    
    ## Parameters
    
    **sim_input_his** (required): Historical sales data for forecasting
    - Format: List of dicts with `item_id`, `actual_sale`, `day` (YYYY-MM-DD)
    - Minimum: 5-30 data points depending on model complexity
    - Example: [{"item_id": 100, "actual_sale": 120, "day": "2023-01-01"}]
    
    **forecast_periods** (default: 30): Number of future periods to forecast
    - Range: 1-365 for daily, 1-24 for monthly
    - Example: 7 = forecast next 7 days/months
    
    **mode** (default: 'local'): Forecasting engine
    - 'local': Local StatsForecast models
    - 'timegpt': TimeGPT cloud API
    
    **local_model** (default: 'auto_arima'): Statistical model (only for mode='local')
    - 'auto_arima': Automated ARIMA - best for trend data (20+ points)
    - 'auto_ets': Exponential Smoothing - best for seasonal data (30+ points)
    - 'naive': Simple forecast (last value) - works with any data
    - 'seasonal_naive': Repeats seasonal pattern - needs 1 full season
    - 'croston_optimized': For intermittent/sparse demand
    - 'adida': Adaptive intermittent demand
    - 'theta': Theta method - balanced approach
    - 'optimized_theta': Optimized theta
    - 'auto_ces': Complex exponential smoothing
    
    **season_length** (default: 12): Length of one seasonal cycle
    - 7 = weekly seasonality (for daily data)
    - 12 = yearly seasonality (for monthly data)
    - 24 = daily seasonality (for hourly data)
    - 52 = yearly seasonality (for weekly data)
    
    **freq** (default: 'D'): Frequency/interval of time series data
    - 'D': Daily
    - 'MS': Month Start (monthly data)
    - 'W': Weekly
    - 'H': Hourly
    - 'Q': Quarterly
    - 'Y': Yearly
    - Must match your input data frequency
    
    **api_key** (optional): Nixtla API key for TimeGPT mode
    - Required when mode='timegpt'
    - Alternative: Set NIXTLA_API_KEY environment variable
    
    **quantiles** (optional): Quantile levels for uncertainty intervals (TimeGPT only)
    - Format: List of floats between 0 and 1
    - Example: [0.1, 0.5, 0.9] for 10th, 50th, 90th percentiles
    - Use for safety stock calculations and confidence intervals
    
    ## Quick Guide by Use Case
    - **Daily demand, next week**: freq='D', season_length=7, forecast_periods=7
    - **Monthly sales, next 6 months**: freq='MS', season_length=12, forecast_periods=6
    - **Sparse/intermittent demand**: local_model='croston_optimized'
    - **Need uncertainty estimates**: mode='timegpt', quantiles=[0.1, 0.5, 0.9]
    
    ## Returns
    Dictionary containing:
    - forecasts: List of forecast results per item
    - total_items: Number of items processed
    - mode: Forecasting mode used
    - model: Model used
    - periods: Number of periods forecasted
    - frequency: Data frequency
    
    ## Example Request
    ```json
    {
      "sim_input_his": [
        {"item_id": 100, "actual_sale": 120, "day": "2023-01-01"},
        {"item_id": 100, "actual_sale": 115, "day": "2023-02-01"}
      ],
      "forecast_periods": 6,
      "mode": "local",
      "local_model": "auto_ets",
      "season_length": 12,
      "freq": "MS"
    }
    ```
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


@router.post("/generate_async")
async def generate_forecast_async(request: ForecastRequest):
    """
    Async version of `/generate`. Runs per-item forecasting off the event loop
    using `asyncio.to_thread` so the server remains responsive. Parameters
    and response structure are identical to `/generate`.
    """
    try:
        print(f"Starting async forecast generation with mode: {request.mode}")

        df_his = pd.DataFrame(request.sim_input_his)

        required_cols = ['item_id', 'actual_sale', 'day']
        missing_cols = [col for col in required_cols if col not in df_his.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df_his['day'] = pd.to_datetime(df_his['day'])

        forecaster = ClassicalForecasts(
            mode=request.mode,
            api_key=request.api_key,
            quantiles=request.quantiles,
            local_model=request.local_model,
            season_length=request.season_length,
            freq=request.freq
        )

        print(f"Forecaster initialized (async): {request.local_model if request.mode == 'local' else 'TimeGPT'}")

        results = []
        unique_items = df_his['item_id'].unique()

        print(f"Generating async forecasts for {len(unique_items)} items")

        for item_id in unique_items:
            try:
                item_data = df_his[df_his['item_id'] == item_id].sort_values('day').reset_index(drop=True)

                # Run potentially blocking forecasting call in a thread
                forecast_values = await asyncio.to_thread(forecaster.daily_path, item_data, request.forecast_periods)

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

                print(f"  ✓ Item {item_id}: async forecast generated")

            except Exception as e:
                print(f"  ✗ Item {item_id}: {str(e)}")
                results.append({
                    'item_id': int(item_id) if isinstance(item_id, (int, float)) else str(item_id),
                    'error': str(e),
                    'forecast': [],
                    'forecast_dates': []
                })

        print(f"Async forecast generation completed: {len(results)} items processed")

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
        print(f"Full async error traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Async forecast error: {str(e)}")


@router.post("/leadtime_quantile")
def calculate_leadtime_quantile(request: ForecastRequest):
    """
    Calculate lead-time demand quantiles for safety stock calculations.
    
    Uses Monte Carlo simulation to estimate demand variability over lead time,
    accounting for forecast uncertainty. Useful for inventory optimization and
    determining safety stock levels.
    
    ## Parameters
    
    Same as /generate endpoint, with key parameters:
    
    **sim_input_his** (required): Historical sales data
    - Minimum 20-30 data points recommended for accurate quantile estimation
    
    **forecast_periods**: Lead time in periods
    - Example: 7 = 7-day lead time (for daily data)
    - Example: 2 = 2-month lead time (for monthly data)
    
    **mode**: Forecasting engine ('local' or 'timegpt')
    
    **local_model**: Statistical model for local mode
    - Recommended: 'auto_arima' or 'auto_ets' for quantile calculations
    
    ## Returns
    Dictionary containing:
    - leadtime_quantiles: List of quantile results per item
    - Each item includes quantiles at 90%, 95%, and 99% service levels
    - total_items: Number of items processed
    - lead_time_periods: Lead time used
    - mode: Forecasting mode used
    
    ## Service Level Quantiles
    - **q_90**: 90% service level (10% stockout risk)
    - **q_95**: 95% service level (5% stockout risk)
    - **q_99**: 99% service level (1% stockout risk)
    
    ## Use Cases
    - Safety stock calculation
    - Reorder point determination
    - Service level planning
    - Risk assessment
    
    ## Example Request
    ```json
    {
      "sim_input_his": [
        {"item_id": 100, "actual_sale": 120, "day": "2023-01-01"},
        ...more data points...
      ],
      "forecast_periods": 7,
      "mode": "local",
      "local_model": "auto_arima",
      "freq": "D"
    }
    ```
    
    ## Example Response
    ```json
    {
      "leadtime_quantiles": [
        {
          "item_id": 100,
          "lead_time_periods": 7,
          "quantiles": {
            "q_90": 850.5,
            "q_95": 920.3,
            "q_99": 1050.2
          }
        }
      ]
    }
    ```
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