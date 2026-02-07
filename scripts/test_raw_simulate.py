"""Test client for the /api/v1/simulation/raw_simulate endpoint.

Reads sim_input_modified.json (list of records for inv_df) and posts it to the API.

Usage:
  python3 scripts/test_raw_simulate.py
    python3 scripts/test_raw_simulate.py --url http://localhost:8000/api/v1/simulation/raw_simulate
  python3 scripts/test_raw_simulate.py --limit 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="POST sim_input_modified.json to /raw_simulate")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/simulation/raw_simulate",
        help="Endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        default="sim_input_modified.json",
        help="Input JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If set, only send the first N rows",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    input_path = Path(args.file)
    data = _load_json(input_path)

    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and "inv_df" in data:
        rows = data["inv_df"]
    else:
        raise SystemExit(
            "Expected JSON to be a list of records, or an object with an 'inv_df' key"
        )

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    payload = {"inv_df": rows}

    print(f"POST {args.url}")
    print(f"Rows: {len(rows)}")

    resp = requests.post(args.url, json=payload, timeout=args.timeout)
    print(f"Status: {resp.status_code}")

    if resp.status_code == 404:
        print(
            "Hint: in this repo, simulation routes are mounted under /api/v1/simulation. "
            "Try: http://localhost:8000/api/v1/simulation/raw_simulate"
        )

    try:
        body = resp.json()
    except Exception:
        print(resp.text[:2000])
        return 1

    if resp.status_code >= 400:
        print(json.dumps(body, indent=2)[:4000])
        return 1

    out = body.get("simulator_output")
    if isinstance(out, list):
        print(f"simulator_output rows: {len(out)}")
        print("First 100 rows:")
        print(json.dumps(out[:100], indent=2)[:4000])
    else:
        print("simulator_output:")
        print(json.dumps(out, indent=2)[:4000])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
