from datetime import date

import pandas as pd

from inventory_algorithm.easter import (
    EasterPlan,
    apply_easter_to_forecasts,
    apply_easter_to_item_results,
    clear_unsold_months,
    easter_sunday,
    monthly_easter_weights,
    positive_months_by_item,
    prepare_easter_history,
)


def test_known_easter_sundays():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_early_easter_splits_february_and_march():
    w = monthly_easter_weights(2024)
    assert abs(w[2] - 26 / 56) < 1e-12
    assert abs(w[3] - 30 / 56) < 1e-12
    assert w.get(4, 0.0) == 0.0
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_late_easter_keeps_february_small():
    w = monthly_easter_weights(2025)
    assert abs(w[2] - 6 / 56) < 1e-12
    assert abs(w[3] - 31 / 56) < 1e-12
    assert abs(w[4] - 19 / 56) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_mid_easter_reaches_february():
    w = monthly_easter_weights(2026)
    assert abs(w[2] - 21 / 56) < 1e-12
    assert abs(w[3] - 31 / 56) < 1e-12
    assert abs(w[4] - 4 / 56) < 1e-12


def _hist(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "day": pd.to_datetime([r[1] for r in rows]),
            "actual_sale": [r[2] for r in rows],
        }
    )


def test_strip_zeros_february_and_march_when_weights_match():
    hist = _hist(
        [
            ("egg", "2024-02-01", 26.0),
            ("egg", "2024-03-01", 30.0),
            ("egg", "2024-05-01", 3.0),
        ]
    )

    out, plan = prepare_easter_history(hist, {"egg"})

    by_month = out.set_index("day")["actual_sale"].to_dict()
    assert by_month[pd.Timestamp("2024-02-01")] == 0.0
    assert by_month[pd.Timestamp("2024-03-01")] == 0.0
    assert by_month[pd.Timestamp("2024-05-01")] == 3.0
    assert plan.event_forecast["egg"] == 56.0


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
            "item_id": ["egg", "egg", "egg", "egg"],
            "forecast_date": pd.to_datetime(["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"]),
            "forecast": [0.0, 1.0, 2.0, 3.0],
            "forecast_upper_95": [3.0, 4.0, 5.0, 6.0],
        }
    )
    out = apply_easter_to_forecasts(fcst, plan)
    w = monthly_easter_weights(2026)
    assert abs(out["forecast"].iloc[0] - (105.0 * w[2])) < 1e-9
    assert abs(out["forecast"].iloc[1] - (1.0 + 105.0 * w[3])) < 1e-9
    assert abs(out["forecast"].iloc[2] - (2.0 + 105.0 * w[4])) < 1e-9
    assert out["forecast"].iloc[3] == 3.0
    assert abs((out["forecast_upper_95"].iloc[1] - out["forecast"].iloc[1]) - 3.0) < 1e-9


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
            "forecast": [0.0, 1.0, 2.0, 3.0],
            "forecast_dates": ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"],
            "upper_95": [3.0, 4.0, 5.0, 6.0],
        },
        {
            "item_id": "plain",
            "forecast": [7.0, 8.0],
            "forecast_dates": ["2026-03-01", "2026-04-01"],
        },
    ]
    out = apply_easter_to_item_results(results, plan)
    assert abs(out[0]["forecast"][0] - (100.0 * w[2])) < 1e-9
    assert abs(out[0]["forecast"][1] - (1.0 + 100.0 * w[3])) < 1e-9
    assert abs(out[0]["forecast"][2] - (2.0 + 100.0 * w[4])) < 1e-9
    assert out[0]["forecast"][3] == 3.0
    assert abs((out[0]["upper_95"][1] - out[0]["forecast"][1]) - 3.0) < 1e-9
    assert out[1]["forecast"] == [7.0, 8.0]


def test_clear_unsold_months_keeps_real_seasons_and_other_items():
    hist = _hist(
        [
            ("egg", "2024-02-01", 40.0),
            ("egg", "2024-03-01", 80.0),
            ("egg", "2025-12-01", 15.0),
            ("egg", "2025-05-01", -2.0),
            ("plain", "2024-07-01", 9.0),
        ]
    )
    sold = positive_months_by_item(hist, ["egg"])
    assert sold["egg"] == {2, 3, 12}
    assert "plain" not in sold

    results = [
        {
            "item_id": "egg",
            "forecast": [6.0, 6.0, 6.0, 6.0],
            "forecast_dates": ["2027-02-01", "2027-05-01", "2027-07-01", "2027-12-01"],
            "upper_95": [8.0, 8.0, 8.0, 8.0],
        },
        {
            "item_id": "plain",
            "forecast": [4.0, 5.0],
            "forecast_dates": ["2027-07-01", "2027-08-01"],
        },
    ]
    cleared = clear_unsold_months(results, sold)
    assert cleared[0]["forecast"] == [6.0, 0.0, 0.0, 6.0]
    assert cleared[0]["upper_95"] == [8.0, 0.0, 0.0, 8.0]
    assert cleared[1]["forecast"] == [4.0, 5.0]

    placed = apply_easter_to_item_results(cleared, EasterPlan(event_forecast={"egg": 100.0}))
    assert placed[0]["forecast"][1] == 0.0
    assert placed[0]["forecast"][2] == 0.0
    assert placed[0]["forecast"][0] > 6.0
    assert placed[1]["forecast"] == [4.0, 5.0]


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
