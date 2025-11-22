import json
from pathlib import Path
from datetime import datetime

root = Path(__file__).resolve().parents[1]
input_path = root / "all_sim_input_data.json"
output_path = root / "all_sim_input_data.v2.json"


def to_int_if_possible(x):
    try:
        # Handle floats that are whole numbers (e.g., 31.0)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        # Handle numeric strings
        if isinstance(x, str):
            f = float(x)
            if f.is_integer():
                return int(f)
            return f
        return x
    except Exception:
        return x


def main():
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}

    # sim_input_his: normalize types
    his = data.get("sim_input_his", [])
    norm_his = []
    for row in his:
        item_id = row.get("item_id")
        actual_sale = row.get("actual_sale")
        day = row.get("day")
        # normalize item_id and actual_sale
        item_id = to_int_if_possible(item_id)
        actual_sale = to_int_if_possible(actual_sale)
        # normalize day to ISO date string
        if isinstance(day, str):
            day = day[:10]
        norm_his.append({
            "item_id": item_id,
            "actual_sale": actual_sale,
            "day": day,
        })
    out["sim_input_his"] = norm_his

    # sim_rio_items: map to clean schema, drop CSV artifacts
    items = data.get("sim_rio_items", [])
    norm_items = []
    if items:
        src = items[0]
        norm_items.append({
            "pn": src.get("pn"),
            "description": src.get("description"),
            "actual_stock": to_int_if_possible(src.get("actual_stock", 0)),
            "ideal_stock": to_int_if_possible(src.get("ideal_stock", 0)),
            "station": to_int_if_possible(src.get("station", 0)),
            "del_time": to_int_if_possible(src.get("del_time", 0)),
            "buy_freq": to_int_if_possible(src.get("buy_freq", 0)),
            "purchasing_method": src.get("purchasing_method", "low_sale"),
            "min": to_int_if_possible(src.get("min", 0)),
            "max": to_int_if_possible(src.get("max", 0)),
        })
    out["sim_rio_items"] = norm_items

    # sim_rio_item_details: keep vendor_name, passthrough other fields
    details = data.get("sim_rio_item_details", [])
    norm_details = []
    for d in details:
        nd = {k: v for k, v in d.items()}
        if "vendor_name" in nd and nd["vendor_name"] is not None:
            nd["vendor_name"] = str(nd["vendor_name"])
        norm_details.append(nd)
    out["sim_rio_item_details"] = norm_details

    # sim_rio_on_order: use [] if placeholder detected
    on_order = data.get("sim_rio_on_order", [])
    if on_order and isinstance(on_order, list) and set(on_order[0].keys()) == {"Empty", "DataFrame"}:
        out["sim_rio_on_order"] = []
    else:
        # normalize if provided
        norm_oo = []
        for r in on_order or []:
            nd = {k: v for k, v in r.items()}
            if "est_deliv_date" in nd and isinstance(nd["est_deliv_date"], str):
                nd["est_deliv_date"] = nd["est_deliv_date"][:10]
            if "est_deliv_qty" in nd:
                nd["est_deliv_qty"] = to_int_if_possible(nd["est_deliv_qty"]) or 0
            if "pn" in nd and nd["pn"] is not None:
                nd["pn"] = str(nd["pn"])
            norm_oo.append(nd)
        out["sim_rio_on_order"] = norm_oo

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {output_path.relative_to(root)} with V2 schema")


if __name__ == "__main__":
    main()
