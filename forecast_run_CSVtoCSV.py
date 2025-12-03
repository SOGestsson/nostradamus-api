#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import requests


def read_input_csv(path: Path):
    """
    Read CSV with columns: item, value, date
    Returns list of dicts in API format:
    { "item_id": ..., "actual_sale": ..., "day": ... }
    """
    records = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"item", "value", "date"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row in reader:
            item_id = row["item"]
            # allow empty/NaN values to be skipped
            value_str = row["value"]
            if value_str is None or value_str == "":
                continue
            actual_sale = float(value_str)
            day = row["date"]
            records.append(
                {
                    "item_id": str(item_id),
                    "actual_sale": actual_sale,
                    "day": day,
                }
            )
    if not records:
        raise ValueError("No usable rows found in CSV.")
    return records


def build_request_payload(
    sim_input_his,
    forecast_periods: int,
    mode: str,
    local_model: str,
    season_length: int,
    freq: str,
    api_key: str | None = None,
):
    payload = {
        "sim_input_his": sim_input_his,
        "forecast_periods": forecast_periods,
        "mode": mode,
        "local_model": local_model,
        "season_length": season_length,
        "freq": freq,
    }
    if api_key:
        payload["api_key"] = api_key
    return payload


def call_forecast_api(
    base_url: str,
    payload: dict,
    timeout: int = 60,
):
    """
    Calls POST /api/v1/forecast/generate
    Returns parsed JSON.
    """
    url = base_url.rstrip("/") + "/api/v1/forecast/generate"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Helpful debug
        raise SystemExit(
            f"API error {resp.status_code}: {resp.text}"
        ) from e
    return resp.json()


def forecasts_to_rows(response_json: dict):
    """
    Convert API response into rows:
    { "item": ..., "value": forecast_value, "date": forecast_date }

    Response structure (from your API):

    {
      "forecasts": [
        {
          "item_id": "11250",
          "forecast": [0.0, 0.0, ...],
          "forecast_dates": ["2022-12-01", "2023-01-01", ...],
          "model_used": "auto_ets",
          "periods_forecasted": 6
        },
        ...
      ],
      "total_items": 954,
      "mode": "local",
      "model": "auto_ets",
      "periods": 6,
      "frequency": "MS"
    }
    """
    rows = []

    forecasts_list = response_json.get("forecasts", [])
    if not isinstance(forecasts_list, list):
        raise ValueError("Expected 'forecasts' to be a list in the API response.")

    for item_block in forecasts_list:
        item_id = item_block.get("item_id")

        # Handle possible error entries like:
        # { "item_id": "95020", "error": "'fitted'", "forecast": [], "forecast_dates": [] }
        fc_vals = item_block.get("forecast") or []
        fc_dates = item_block.get("forecast_dates") or []
        error_msg = item_block.get("error")

        if not fc_vals or not fc_dates:
            if error_msg:
                print(f"Skipping item {item_id} due to model error: {error_msg}")
            continue

        # Pair each forecast value with its corresponding date
        for date, value in zip(fc_dates, fc_vals):
            rows.append(
                {
                    "item": item_id,
                    "value": value,
                    "date": date,
                }
            )

    return rows


def write_output_csv(path: Path, rows):
    """
    Write rows with columns item, value, date.
    """
    fieldnames = ["item", "value", "date"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read a CSV (item,value,date), call the forecast API, "
            "and write a forecast CSV with the same column layout."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to input CSV file")
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        help="Path for output CSV (default: <input>_forecast.csv)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the Inventory Simulation API "
        "(default: http://localhost:8000)",
    )
    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=6,
        help="Number of future periods to forecast (default: 6)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "timegpt"],
        default="local",
        help="Forecast mode (local or timegpt). Default: local",
    )
    parser.add_argument(
        "--local-model",
        default="auto_arima",
        help="Local model to use when mode=local "
        "(e.g. auto_arima, auto_ets, croston_optimized, ...)",
    )
    parser.add_argument(
        "--season-length",
        type=int,
        default=12,
        help="Season length, e.g. 12 for monthly data with yearly seasonality",
    )
    parser.add_argument(
        "--freq",
        default="MS",
        help="Pandas-style frequency string (D, MS, W, H, Q, Y). "
        "Your example file looks like monthly start, so default is MS.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Nixtla TimeGPT API key (required if mode=timegpt, "
        "optional otherwise)",
    )
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = args.input_csv.with_name(
            args.input_csv.stem + "_forecast.csv"
        )

    # 1) Read CSV and build sim_input_his
    sim_input_his = read_input_csv(args.input_csv)

    # 2) Build request payload
    payload = build_request_payload(
        sim_input_his=sim_input_his,
        forecast_periods=args.forecast_periods,
        mode=args.mode,
        local_model=args.local_model,
        season_length=args.season_length,
        freq=args.freq,
        api_key=args.api_key,
    )

    # 3) Call API
    response_json = call_forecast_api(args.base_url, payload)
    print("DEBUG FULL RESPONSE:")
    print(json.dumps(response_json, indent=2))


    # Optional: print summary for debugging
    print("API response (truncated):")
    print(json.dumps(response_json, indent=2)[:2000], "...\n")

    # 4) Convert forecasts into rows with item,value,date
    rows = forecasts_to_rows(response_json)

    # 5) Write output CSV
    write_output_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} forecast rows to {args.output_csv}")


if __name__ == "__main__":
    main()
