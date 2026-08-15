import pandas as pd

from api.deps import build_dataframes
from api.models import SimInput
from inventory_algorithm.inventory_opt_and_forecasting_package import (
    inventory_simulator_with_input_prep,
    split_on_order_by_due_date,
)


def test_on_order_mixed_iso_dates_map_to_sim_delivery():
    sim_input = SimInput(
        sim_input_his=[
            {"item_id": 1, "actual_sale": 0, "day": "2026-06-24"},
            {"item_id": 1, "actual_sale": -1000001, "day": "2026-06-28"},
        ],
        sim_rio_items=[
            {
                "pn": "TEST",
                "description": "Test",
                "actual_stock": 0,
                "ideal_stock": 0,
                "station": 0,
                "del_time": 52,
                "buy_freq": 10,
                "purchasing_method": "min-max",
                "min": 20,
                "max": 40,
            }
        ],
        sim_rio_item_details=[{"id": "TEST", "vendor_name": "Vendor"}],
        sim_rio_on_order=[
            {"item_number": "TEST", "est_deliv_date": "2026-06-24T19:00:40.360000", "est_deliv_qty": 10},
            {"item_number": "TEST", "est_deliv_date": "2026-06-28T14:48:00", "est_deliv_qty": 30},
        ],
    )

    dfs = build_dataframes(sim_input)
    on_order = dfs["sim_rio_on_order"]
    assert not on_order["est_deliv_date"].isna().any()

    hist = pd.DataFrame(
        {
            "day": pd.to_datetime(["2026-06-24"]),
            "item_id": [1],
            "actual_sale": [0],
        }
    )
    opt = inventory_simulator_with_input_prep.__new__(inventory_simulator_with_input_prep)
    opt._on_order_due_stock = 0.0
    mapped = opt.step_2_add_on_order_to_sim_input(on_order, hist)
    by_day = dict(zip(mapped["day"].dt.strftime("%Y-%m-%d"), mapped["delivery"]))
    assert opt._on_order_due_stock == 10
    assert by_day.get("2026-06-24", 0) == 0
    assert by_day["2026-06-28"] == 30


def test_on_order_due_today_counts_as_stock_not_future_delivery():
    due_qty, future = split_on_order_by_due_date(
        pd.DataFrame(
            [
                {"pn": "TEST", "est_deliv_date": "2026-06-24", "est_deliv_qty": 8},
                {"pn": "TEST", "est_deliv_date": "2026-07-01", "est_deliv_qty": 2},
            ]
        ),
        as_of_day="2026-06-24",
    )
    assert due_qty == 8
    assert len(future) == 1
    assert float(future.iloc[0]["est_deliv_qty"]) == 2


def test_on_order_due_today_suppresses_minmax_purchase_suggestion():
    sim_input = SimInput(
        sim_input_his=[
            {"item_id": 1, "actual_sale": 0, "day": "2026-06-24"},
            {"item_id": 1, "actual_sale": -1000001, "day": "2026-06-28"},
        ],
        sim_rio_items=[
            {
                "pn": "TEST",
                "description": "Test",
                "actual_stock": 0,
                "ideal_stock": 0,
                "station": 0,
                "del_time": 1,
                "buy_freq": 10,
                "purchasing_method": "min-max",
                "min": 4,
                "max": 4,
            }
        ],
        sim_rio_item_details=[{"id": "TEST", "vendor_name": "Vendor"}],
        sim_rio_on_order=[
            {"item_number": "TEST", "est_deliv_date": "2026-06-24T08:00:00", "est_deliv_qty": 8},
        ],
    )

    dfs = build_dataframes(sim_input)
    sim = inventory_simulator_with_input_prep(
        dfs["sim_input_his"],
        dfs["sim_rio_items"],
        dfs["sim_rio_on_order"],
        pd.DataFrame(sim_input.sim_rio_item_details),
        periods=5,
        number_of_trials=20,
        serv_level=0.95,
    )
    first_purchase = float(sim.sim_result.iloc[0]["purchase_qty"])
    assert first_purchase == 0
