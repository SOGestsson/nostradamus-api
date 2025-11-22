# Architecture Diagram

## API Request Flow

```
Client Request
    ↓
http://localhost:8000/api/v1/simulation/simulate
    ↓
main.py (FastAPI app instance)
    ↓
app.include_router(v1_router, prefix="/api/v1")
    ↓
api/v1/__init__.py (V1 router aggregator)
    ↓
router.include_router(simulation_router, prefix="/simulation")
    ↓
api/v1/simulation.py
    ↓
@router.post("/simulate")
def run_inventory_simulation(request: SimulationRequest)
    ↓
    ├─→ api.deps.build_dataframes(request)
    │   ├─→ Validate with api.models.SimInput
    │   ├─→ Convert JSON to DataFrames
    │   ├─→ Type coercion (int, float, datetime)
    │   └─→ Return dict of DataFrames
    │
    └─→ inventory_algorithm.inventory_simulator_with_input_prep()
        ├─→ Monte Carlo simulation
        ├─→ Inventory optimization
        └─→ Generate results
            ↓
        Response JSON
            ↓
        Client
```

## File Dependency Graph

```
main.py
├── imports FastAPI
├── imports api.v1.router
│
api/v1/__init__.py
├── imports APIRouter from fastapi
├── imports simulation_router from api.v1.simulation
│
api/v1/simulation.py
├── imports APIRouter from fastapi
├── imports HTTPException from fastapi
├── imports build_dataframes from api.deps
├── imports SimulationRequest from api.models
├── imports inventory_algorithm package
│
api/deps.py
├── imports pandas
├── imports SimInput from api.models
│
api/models.py
├── imports BaseModel from pydantic
├── imports List, Dict, Any from typing
```

## Directory Structure (Detailed)

```
Nostradamus_api/
│
├── main.py                          # FastAPI app entry (33 lines)
│   └── Responsibilities:
│       ├── Create FastAPI instance
│       ├── Include v1 router with /api/v1 prefix
│       └── Define root and health endpoints
│
├── api/                             # API package
│   │
│   ├── __init__.py                  # Package marker
│   │
│   ├── models.py                    # Pydantic models (24 lines)
│   │   └── Classes:
│   │       ├── SimInput (base data model)
│   │       └── SimulationRequest (full request with params)
│   │
│   ├── deps.py                      # Shared dependencies (113 lines)
│   │   └── Functions:
│   │       └── build_dataframes(data: SimInput) -> Dict[str, DataFrame]
│   │           ├── Convert JSON to DataFrames
│   │           ├── Type coercion (int, float, datetime)
│   │           ├── Remove CSV artifacts
│   │           └── Return dict with 4 DataFrames
│   │
│   └── v1/                          # API version 1
│       │
│       ├── __init__.py              # V1 router aggregator (14 lines)
│       │   └── Combines:
│       │       └── simulation_router with /simulation prefix
│       │
│       └── simulation.py            # Simulation endpoints (164 lines)
│           └── Routes:
│               ├── POST /simulate
│               │   └── Full simulation with all results
│               │
│               └── POST /histo_buy
│                   └── Simulation with histogram only
│
├── inventory_algorithm/             # Core simulation logic (unchanged)
│   ├── __init__.py
│   ├── inventory_opt_and_forecasting_package.py
│   ├── inventory_simulator.py
│   ├── part_number_container.py
│   └── simulator_class.py
│
├── services/                        # Additional services
│   └── data_frames.py
│
├── scripts/                         # Utility scripts
│   └── convert_to_v2.py            # JSON V1 to V2 converter
│
├── all_sim_input_data.v2.json      # Clean input data
├── test_api.sh                      # Automated test script
├── API_TESTING.md                   # Testing documentation
├── README.md                        # Project documentation
├── RESTRUCTURING.md                 # Restructuring summary
├── Pipfile                          # Dependencies
└── requirements.txt                 # Requirements export
```

## Module Relationships

```
┌─────────────────────────────────────────────┐
│              main.py                        │
│  (FastAPI app instance + root endpoints)    │
└──────────────────┬──────────────────────────┘
                   │ includes
                   ↓
┌─────────────────────────────────────────────┐
│         api/v1/__init__.py                  │
│     (V1 router with /simulation prefix)     │
└──────────────────┬──────────────────────────┘
                   │ includes
                   ↓
┌─────────────────────────────────────────────┐
│       api/v1/simulation.py                  │
│  (Simulation endpoints: /simulate, /histo)  │
└─────┬──────────────────────────┬────────────┘
      │ imports                  │ imports
      ↓                          ↓
┌──────────────┐          ┌─────────────────┐
│  api/deps.py │          │  api/models.py  │
│ (utilities)  │          │  (Pydantic)     │
└──────┬───────┘          └─────────────────┘
       │ imports
       ↓
┌─────────────────┐
│  api/models.py  │
│   (SimInput)    │
└─────────────────┘

All modules import from:
┌────────────────────────────────────────┐
│   inventory_algorithm/                 │
│   (Core simulation logic - unchanged)  │
└────────────────────────────────────────┘
```

## Endpoint Hierarchy

```
http://localhost:8000
│
├── GET  /                          (main.py: read_root)
│        → API info and links
│
├── GET  /health                    (main.py: health_check)
│        → Health status
│
├── GET  /docs                      (FastAPI auto-generated)
│        → Swagger UI
│
├── GET  /redoc                     (FastAPI auto-generated)
│        → ReDoc documentation
│
└── /api
    └── /v1                         (version 1 API)
        └── /simulation             (simulation router)
            │
            ├── POST /simulate      (api/v1/simulation.py)
            │        → Full simulation with complete results
            │        → Request: SimulationRequest
            │        → Response: {histogram_data, histogram_info, full_results}
            │
            └── POST /histo_buy     (api/v1/simulation.py)
                     → Simulation with histogram only
                     → Request: SimulationRequest
                     → Response: {histogram_data, histogram_info}
```

## Data Flow Example

```
1. Client sends POST to /api/v1/simulation/simulate
   {
     "sim_input_his": [...],
     "sim_rio_items": [...],
     "sim_rio_item_details": [...],
     "sim_rio_on_order": [...],
     "number_of_days": 900,
     "number_of_simulations": 1000,
     "service_level": 0.95
   }
   ↓
2. FastAPI routes to api/v1/simulation.py:run_inventory_simulation()
   ↓
3. Pydantic validates request against SimulationRequest model
   ↓
4. Call build_dataframes() from api.deps
   ↓
5. Convert JSON to 4 DataFrames:
   - df_his (historical sales)
   - df_items (item config)
   - df_item_details (vendor info)
   - df_on_order (outstanding orders)
   ↓
6. Call inventory_simulator_with_input_prep() from inventory_algorithm
   ↓
7. Monte Carlo simulation runs 1000 times over 900 days
   ↓
8. Generate results:
   - histogram_data (purchase frequency distribution)
   - histogram_info (statistics: mean, median, mode, etc.)
   - full_results (complete simulation output)
   ↓
9. Return JSON response to client
   {
     "histogram_data": [...],
     "histogram_info": {...},
     "full_results": {...}
   }
```

## Testing Flow

```
./test_api.sh
│
├─→ Test 1: GET /health
│   └─→ Expect: {"status": "healthy"}
│
├─→ Test 2: GET /
│   └─→ Expect: API info with version
│
├─→ Test 3: POST /api/v1/simulation/simulate
│   ├─→ Send: all_sim_input_data.v2.json
│   └─→ Expect: histogram_info with statistics
│
└─→ Test 4: POST /api/v1/simulation/histo_buy
    ├─→ Send: all_sim_input_data.v2.json
    └─→ Expect: histogram_info with statistics
```

## Version Management

```
Current:
    api/v1/  (simulation endpoints)

Future:
    api/v2/  (enhanced simulation endpoints)
    api/v3/  (real-time optimization)

Migration strategy:
    - Keep v1 running for backward compatibility
    - Add new features to v2
    - Deprecate v1 after migration period
    - Remove v1 when all clients upgraded
```
