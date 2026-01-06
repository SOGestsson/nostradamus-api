from api.models import LightGPTForecastRequest


def test_lightgpt_request_accepts_api_key_aliases():
    payload = {
        "sim_input_his": [
            {"item_id": 1, "actual_sale": 10, "day": "2025-12-01"},
        ],
        "forecast_periods": 3,
        "apiKey": "k-test",
        "exogenousColumns": ["price"],
    }

    req = LightGPTForecastRequest.model_validate(payload)
    assert req.api_key == "k-test"
    assert req.exogenous_columns == ["price"]
