# Inventory Simulation API - Testing Guide

## Overview

The Inventory Simulation API provides endpoints for running Monte Carlo simulations to optimize inventory levels and forecast purchasing requirements.

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Health & Status

#### GET `/`
Root endpoint with API information.

**Response:**
```json
{
  "status": "ok",
  "message": "Inventory Simulation API",
  "version": "2.0.0",
  "docs": "/docs"
}
```

#### GET `/health`
Health check for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### Simulation Endpoints

#### POST `/api/v1/simulation/simulate`
Run full inventory simulation with complete results.

**Request Body:**
```json
{
  "sim_input_his": [...],
  "sim_rio_items": [...],
  "sim_rio_item_details": [...],
  "sim_rio_on_order": [...],
  "number_of_days": 900,
  "number_of_simulations": 1000,
  "service_level": 0.95
}
```

**Parameters:**
- `sim_input_his`: Historical sales data (item_id, actual_sale, day)
- `sim_rio_items`: Item configuration (pn, description, stock levels, lead times, etc.)
- `sim_rio_item_details`: Vendor/supplier details
- `sim_rio_on_order`: Outstanding orders (pn, est_deliv_date, est_deliv_qty)
- `number_of_days`: Simulation period (default: 900)
- `number_of_simulations`: Number of Monte Carlo runs (default: 1000)
- `service_level`: Target service level (default: 0.95 = 95%)

**Response:**
```json
{
  "histogram_data": [...],
  "histogram_info": {
    "mean": 42.5,
    "median": 41.0,
    "mode": 40,
    "std_dev": 5.2,
    "min": 30,
    "max": 55
  },
  "full_results": {...}
}
```

#### POST `/api/v1/simulation/histo_buy`
Run simulation and return only purchase frequency histogram (lighter response).

**Request Body:** Same as `/simulate`

**Response:**
```json
{
  "histogram_data": [...],
  "histogram_info": {
    "mean": 42.5,
    "median": 41.0,
    "mode": 40,
    "std_dev": 5.2,
    "min": 30,
    "max": 55
  }
}
```

## Running the Server

### 1. Start the server
```bash
pipenv run python -m uvicorn main:app --reload --port 8000
```

### 2. Access API documentation
Open your browser to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing with curl

### Health check
```bash
curl http://localhost:8000/health
```

### Full simulation
```bash
curl -X POST http://localhost:8000/api/v1/simulation/simulate \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data.v2.json
```

### Histogram only
```bash
curl -X POST http://localhost:8000/api/v1/simulation/histo_buy \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data.v2.json
```

## Testing with the test script

Make the script executable:
```bash
chmod +x test_api.sh
```

Run all tests:
```bash
./test_api.sh
```

## Project Structure

```
Nostradamus_api/
├── main.py                   # FastAPI app entry point
├── api/
│   ├── __init__.py          # Package marker
│   ├── deps.py              # Shared dependencies (build_dataframes)
│   ├── models.py            # Pydantic request/response models
│   └── v1/
│       ├── __init__.py      # V1 router aggregator
│       └── simulation.py    # Simulation endpoints
├── inventory_algorithm/     # Core simulation logic
└── all_sim_input_data.v2.json  # Clean JSON input data
```

## JSON Schema

### V2 JSON Format (Recommended)
- Integer IDs (not floats like 31.0)
- No CSV artifacts (Unnamed columns)
- Proper data types throughout

Example:
```json
{
  "sim_input_his": [
    {
      "item_id": 31,
      "actual_sale": 2,
      "day": "2019-01-01"
    }
  ],
  "sim_rio_items": [
    {
      "pn": 31,
      "description": "Item Name",
      "actual_stock": 100,
      "ideal_stock": 150,
      "station": 1,
      "del_time": 14,
      "buy_freq": 30,
      "purchasing_method": "auto",
      "min": 50,
      "max": 200
    }
  ],
  "sim_rio_item_details": [
    {
      "pn": 31,
      "vendor_name": "Vendor ABC"
    }
  ],
  "sim_rio_on_order": []
}
```

## Adding New Endpoints

To add new simulation endpoints:

1. Create new functions in `api/v1/simulation.py`
2. Add routes using `@router.post()` or `@router.get()`
3. Import dependencies from `api.deps`
4. Use models from `api.models`

Example:
```python
@router.post("/my-endpoint")
def my_new_endpoint(request: SimulationRequest):
    from api.deps import build_dataframes
    dfs = build_dataframes(request)
    # ... your logic here
    return {"result": "data"}
```

## Troubleshooting

### uvicorn not found
Use: `pipenv run python -m uvicorn main:app --reload --port 8000`

### Import errors
Ensure you're running from the project root directory.

### JSON file not found
Use absolute paths or run from the project directory where the JSON file is located.
