import sys
import os
import asyncio
import importlib.util
import numpy as np

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load forecast module directly (bypass package __init__ side-effects)
forecast_path = os.path.join(ROOT, 'api', 'v1', 'forecast.py')
spec = importlib.util.spec_from_file_location('temp_forecast', forecast_path)
forecast_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_mod)

# Import request model
from api.models import ForecastRequest

# Monkeypatch ClassicalForecasts.daily_path to avoid heavy deps
import inventory_algorithm.classical_forecasts as cf

def fake_daily_path(self, item_hist, periods):
    try:
        last = float(item_hist['actual_sale'].iloc[-1])
    except Exception:
        last = 0.0
    return np.array([last for _ in range(periods)], dtype=float)

cf.ClassicalForecasts.daily_path = fake_daily_path

# Build a small test request
req = ForecastRequest(
    sim_input_his=[
        {"item_id": 1, "actual_sale": 10, "day": "2023-01-01"},
        {"item_id": 1, "actual_sale": 12, "day": "2023-01-02"},
        {"item_id": 2, "actual_sale": 5, "day": "2023-01-01"},
        {"item_id": 2, "actual_sale": 7, "day": "2023-01-02"}
    ],
    forecast_periods=3,
    mode='local',
    local_model='naive',
    freq='D',
    season_length=7
)

# Run the async endpoint function directly
result = asyncio.run(forecast_mod.generate_forecast_async(req))

print('--- Async endpoint result ---')
import json
print(json.dumps(result, indent=2))
