from inventory_algorithm.lightgpt_forecasts import LightGPTForecast


def test_timegpt_default_model_is_timegpt_1():
    f = LightGPTForecast(api_key="test-key")
    assert f.model == "timegpt-1"
