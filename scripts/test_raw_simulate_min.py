import json

import requests

inv_df = json.load(open("sim_input_modified.json", "r", encoding="utf-8"))
r = requests.post(
    "http://localhost:8000/api/v1/simulation/raw_simulate",
    json={"inv_df": inv_df},
    timeout=120,
)
print(f"Status: {r.status_code}")

try:
    data = r.json()
except Exception:
    print(r.text)
    raise SystemExit(1)

out = data.get("simulator_output")
if isinstance(out, list):
    print(f"simulator_output rows: {len(out)}")
    print(json.dumps(out[:200], indent=2))
else:
    print(json.dumps(data, indent=2))
