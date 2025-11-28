# test_forecast_api.sh

#!/bin/bash

echo "Testing Forecast API Endpoints"
echo "================================"

# Test 1: Generate local forecast
echo -e "\n1. Testing /api/v1/forecast/generate (local mode)"
curl -X POST "http://localhost:8000/api/v1/forecast/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_input_his": [
      {"item_id": 31, "actual_sale": 10, "day": "2024-01-01"},
      {"item_id": 31, "actual_sale": 12, "day": "2024-01-02"},
      {"item_id": 31, "actual_sale": 15, "day": "2024-01-03"},
      {"item_id": 31, "actual_sale": 11, "day": "2024-01-04"},
      {"item_id": 31, "actual_sale": 13, "day": "2024-01-05"}
    ],
    "forecast_periods": 7,
    "mode": "local",
    "local_model": "naive",
    "freq": "D"
  }' | python3 -m json.tool

# Test 2: Calculate lead-time quantiles
echo -e "\n2. Testing /api/v1/forecast/leadtime_quantile"
curl -X POST "http://localhost:8000/api/v1/forecast/leadtime_quantile" \
  -H "Content-Type: application/json" \
  -d '{
    "sim_input_his": [
      {"item_id": 31, "actual_sale": 10, "day": "2024-01-01"},
      {"item_id": 31, "actual_sale": 12, "day": "2024-01-02"},
      {"item_id": 31, "actual_sale": 15, "day": "2024-01-03"},
      {"item_id": 31, "actual_sale": 11, "day": "2024-01-04"},
      {"item_id": 31, "actual_sale": 13, "day": "2024-01-05"}
    ],
    "forecast_periods": 7,
    "mode": "local",
    "local_model": "naive",
    "freq": "D"
  }' | python3 -m json.tool

echo -e "\n================================"
echo "Tests completed!"