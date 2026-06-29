import pandas as pd

from api.deps import build_dataframes
from api.models import SimInput
from inventory_algorithm.inventory_opt_and_forecasting_package import inventory_opt_and_forecasting_package


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
            "day": pd.to_datetime(["2026-06-24", "2026-06-28"]),
            "item_id": [1, 1],
            "actual_sale": [0, -1000001],
        }
    )
    opt = inventory_opt_and_forecasting_package()
    mapped = opt.step_2_add_on_order_to_sim_input(on_order, hist)
    by_day = dict(zip(mapped["day"].dt.strftime("%Y-%m-%d"), mapped["delivery"]))
    assert by_day["2026-06-24"] == 10
    assert by_day["2026-06-28"] == 30
