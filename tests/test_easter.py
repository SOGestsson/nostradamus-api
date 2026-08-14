from datetime import date

import pandas as pd

from inventory_algorithm.easter import (
    EasterPlan,
    apply_easter_to_forecasts,
    apply_easter_to_item_results,
    easter_sunday,
    monthly_easter_weights,
    prepare_easter_history,
)


def test_known_easter_sundays():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_early_easter_is_all_march():
    w = monthly_easter_weights(2024)
    assert w[3] == 1.0
    assert w.get(4, 0.0) == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_late_easter_is_mostly_april():
    w = monthly_easter_weights(2025)
    assert abs(w[3] - 2 / 21) < 1e-12
    assert abs(w[4] - 19 / 21) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_mid_easter_splits_toward_march():
    w = monthly_easter_weights(2026)
    assert abs(w[3] - 17 / 21) < 1e-12
    assert abs(w[4] - 4 / 21) < 1e-12


def _hist(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "actual_sale": [r[2] for r in rows],
        }
    )


def test_strip_zeros_out_a_pure_easter_month_when_weights_match():
    hist = _hist(
        [
            ("egg", "2024-03-01", 100.0),
            ("egg", "2024-04-01", 0.0),
            ("egg", "2024-05-01", 3.0),
        ]
    )

    out, plan = prepare_easter_history(hist, {"egg"})

    by_month = out.set_index("day")["actual_sale"].to_dict()
    assert by_month[pd.Timestamp("2024-03-01")] == 0.0
    assert by_month[pd.Timestamp("2024-04-01")] == 0.0
    assert by_month[pd.Timestamp("2024-05-01")] == 3.0
    assert plan.event_forecast["egg"] == 100.0


def test_non_paskavara_history_is_returned_unchanged():
    hist = _hist([("plain", "2024-03-01", 50.0), ("plain", "2024-04-01", 40.0)])

    out, plan = prepare_easter_history(hist, {"egg"})

    assert out is hist
    assert plan.is_empty


def test_input_history_is_not_mutated():
    hist = _hist([("egg", "2024-03-01", 100.0)])
    original = hist["actual_sale"].tolist()

    prepare_easter_history(hist, {"egg"})

    assert hist["actual_sale"].tolist() == original


def test_place_puts_the_event_on_this_years_months():
    hist = _hist([("egg", "2024-03-01", 105.0)])
    _, plan = prepare_easter_history(hist, {"egg"})

    fcst = pd.DataFrame(
        {
            "item_id": ["egg", "egg", "egg"],
            "forecast_date": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"]),
            "forecast": [1.0, 2.0, 3.0],
            "forecast_upper_95": [4.0, 5.0, 6.0],
        }
    )
    out = apply_easter_to_forecasts(fcst, plan)
    w = monthly_easter_weights(2026)
    assert abs(out["forecast"].iloc[0] - (1.0 + 105.0 * w[3])) < 1e-9
    assert abs(out["forecast"].iloc[1] - (2.0 + 105.0 * w[4])) < 1e-9
    assert out["forecast"].iloc[2] == 3.0
    assert abs((out["forecast_upper_95"].iloc[0] - out["forecast"].iloc[0]) - 3.0) < 1e-9


def test_place_spans_two_easters_in_the_horizon():
    plan = EasterPlan(event_forecast={"egg": 100.0})
    fcst = pd.DataFrame(
        {
            "item_id": ["egg", "egg"],
            "forecast_date": pd.to_datetime(["2026-03-01", "2028-04-01"]),
            "forecast": [0.0, 0.0],
        }
    )
    out = apply_easter_to_forecasts(fcst, plan)
    w2026 = monthly_easter_weights(2026)
    w2028 = monthly_easter_weights(2028)
    assert abs(out["forecast"].iloc[0] - 100.0 * w2026[3]) < 1e-9
    assert abs(out["forecast"].iloc[1] - 100.0 * w2028.get(4, 0.0)) < 1e-9
    assert w2028.get(4, 0.0) > 0.0


def test_place_leaves_other_items_alone():
    plan = EasterPlan(event_forecast={"egg": 100.0})
    fcst = pd.DataFrame(
        {
            "item_id": ["plain", "plain"],
            "forecast_date": pd.to_datetime(["2026-03-01", "2026-04-01"]),
            "forecast": [7.0, 8.0],
        }
    )
    out = apply_easter_to_forecasts(fcst, plan)
    assert list(out["forecast"]) == [7.0, 8.0]


def test_recency_weights_recent_easters_more():
    hist = _hist(
        [
            ("egg", "2022-04-01", 10.0),
            ("egg", "2024-03-01", 100.0),
        ]
    )
    _, plan = prepare_easter_history(hist, {"egg"})
    assert abs(plan.event_forecast["egg"] - (100.0 + 10.0 * 0.5) / 1.5) < 1e-9


def test_empty_easter_ids_is_a_no_op():
    hist = _hist([("egg", "2024-03-01", 100.0)])
    out, plan = prepare_easter_history(hist, set())
    assert out is hist
    assert plan.is_empty


def test_apply_easter_to_item_results_matches_long_form():
    plan = EasterPlan(event_forecast={"egg": 100.0})
    w = monthly_easter_weights(2026)
    results = [
        {
            "item_id": "egg",
            "forecast": [1.0, 2.0, 3.0],
            "forecast_dates": ["2026-03-01", "2026-04-01", "2026-05-01"],
            "upper_95": [4.0, 5.0, 6.0],
        },
        {
            "item_id": "plain",
            "forecast": [7.0, 8.0],
            "forecast_dates": ["2026-03-01", "2026-04-01"],
        },
    ]
    out = apply_easter_to_item_results(results, plan)
    assert abs(out[0]["forecast"][0] - (1.0 + 100.0 * w[3])) < 1e-9
    assert abs(out[0]["forecast"][1] - (2.0 + 100.0 * w[4])) < 1e-9
    assert out[0]["forecast"][2] == 3.0
    assert abs((out[0]["upper_95"][0] - out[0]["forecast"][0]) - 3.0) < 1e-9
    assert out[1]["forecast"] == [7.0, 8.0]


def test_forecast_request_accepts_paskavara_alias():
    from api.models import ForecastRequest

    req = ForecastRequest.model_validate(
        {
            "sim_input_his": [{"item_id": "egg", "actual_sale": 1, "day": "2024-03-01"}],
            "paskavara_item_ids": ["egg"],
            "freq": "M",
        }
    )
    assert req.easter_item_ids == ["egg"]
