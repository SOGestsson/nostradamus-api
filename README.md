# Inventory Simulation API

FastAPI application for inventory optimization and forecasting using Monte Carlo simulations.

## Features

- **Modular API Structure**: Clean separation of concerns with `api/` directory
- **Type Safety**: Pydantic models for request/response validation
- **Versioned Endpoints**: API v1 at `/api/v1/simulation/*`
- **Comprehensive Documentation**: Auto-generated OpenAPI docs at `/docs`
- **Clean JSON Schema**: V2 format with proper integer types and no CSV artifacts

## Project Structure

```
Nostradamus_api/
├── main.py                      # FastAPI app entry point
├── api/
│   ├── __init__.py             # Package marker
│   ├── deps.py                 # Shared dependencies (build_dataframes)
│   ├── models.py               # Pydantic request/response models
│   └── v1/
│       ├── __init__.py         # V1 router aggregator
│       └── simulation.py       # Simulation endpoints
├── inventory_algorithm/         # Core simulation logic
│   ├── inventory_opt_and_forecasting_package.py
│   ├── inventory_simulator.py
│   └── ...
├── all_sim_input_data.v2.json  # Clean JSON input data
├── test_api.sh                 # API test script
├── API_TESTING.md              # Detailed testing guide
├── Pipfile                     # Dependencies
└── requirements.txt            # Requirements export
```

## Quick Start

### 1. Install Dependencies

```bash
pipenv install
```

### 2. Start the Server

```bash
pipenv run python -m uvicorn main:app --reload --port 8000
```

### 3. Access API Documentation

Open your browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


### 4. Test the API

```bash
./test_api.sh
```

## API Endpoints

### Health & Status

- `GET /` - Root endpoint with API info
- `GET /health` - Health check

### Simulation (v1)

- `POST /api/v1/simulation/simulate` - Full simulation with complete results
- `POST /api/v1/simulation/histo_buy` - Simulation with histogram only

## Usage Examples

### Full Simulation

```bash
curl -X POST http://localhost:8000/api/v1/simulation/simulate \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data.v2.json
```

Response includes:
- `histogram_data`: Purchase frequency distribution
- `histogram_info`: Statistical summary (mean, median, mode, std_dev)
- `full_results`: Complete simulation output

### Histogram Only

```bash
curl -X POST http://localhost:8000/api/v1/simulation/histo_buy \
  -H "Content-Type: application/json" \
  -d @all_sim_input_data.v2.json
```

Lighter response with only histogram data.

## Request Parameters

```json
{
  "sim_input_his": [...],           // Historical sales data
  "sim_rio_items": [...],           // Item configuration
  "sim_rio_item_details": [...],    // Vendor details
  "sim_rio_on_order": [...],        // Outstanding orders
  "number_of_days": 900,            // Simulation period
  "number_of_simulations": 1000,    // Monte Carlo runs
  "service_level": 0.95             // Target service level (95%)
}
```

## JSON Schema V2

The V2 JSON format uses clean data types:

- ✅ Integer IDs: `31` (not `31.0`)
- ✅ No CSV artifacts: No `Unnamed: 0` columns
- ✅ Proper types: Dates as strings, numbers as integers/floats

To convert V1 to V2:

```bash
pipenv run python scripts/convert_to_v2.py
```

## Development

### Adding New Endpoints

1. Create functions in `api/v1/simulation.py`
2. Use `@router.post()` or `@router.get()` decorators
3. Import dependencies from `api.deps`
4. Define models in `api.models`

Example:

```python
from fastapi import APIRouter
from api.deps import build_dataframes
from api.models import SimulationRequest

router = APIRouter()

@router.post("/my-endpoint")
def my_endpoint(request: SimulationRequest):
    dfs = build_dataframes(request)
    # ... your logic
    return {"result": "data"}
```

### Running Tests

```bash
# Make script executable (first time only)
chmod +x test_api.sh

# Run tests
./test_api.sh
```

## Architecture

### main.py
Slim FastAPI application entry point:
- Creates app instance
- Includes v1 router with `/api/v1` prefix
- Defines root and health endpoints

### api/deps.py
Shared utilities:
- `build_dataframes()`: Converts JSON to pandas DataFrames with type coercion

### api/models.py
Pydantic models:
- `SimInput`: Base simulation input
- `SimulationRequest`: Full request with parameters

### api/v1/simulation.py
Simulation endpoints:
- `/simulate`: Full results
- `/histo_buy`: Histogram only

### api/v1/__init__.py
Router aggregation:
- Combines all v1 endpoints
- Exports single `router` for main.py

## Dependencies

- **FastAPI 0.121.3**: Web framework
- **Pandas 2.3.3**: Data manipulation
- **Pydantic 2.12.4**: Data validation
- **uvicorn 0.38.0**: ASGI server

## Documentation

- [API_TESTING.md](API_TESTING.md) - Comprehensive testing guide
- [/docs](http://localhost:8000/docs) - Interactive Swagger UI
- [/redoc](http://localhost:8000/redoc) - ReDoc documentation

## Troubleshooting

### uvicorn not found
Use: `pipenv run python -m uvicorn main:app --reload --port 8000`

### Import errors
Run from project root directory where `main.py` is located.

### JSON file not found in curl
Use absolute path: `-d @/full/path/to/all_sim_input_data.v2.json`

## License

[Add your license here]

## Authors

[Add authors here]
