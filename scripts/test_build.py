import json
import sys
from pathlib import Path

# Try to ensure project root is on sys.path so json_to_panda.py can be imported
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from json_to_panda import build_dataframes, SimInput  # noqa: E402

root = Path(__file__).resolve().parents[1]

print("Testing with V1 JSON...")
with (root / 'all_sim_input_data.json').open('r', encoding='utf-8') as f:
    data_v1 = json.load(f)

dfs_v1 = build_dataframes(SimInput(**data_v1))
for name, df in dfs_v1.items():
    print(name, df.shape, list(df.columns)[:8])

if (root / 'all_sim_input_data.v2.json').exists():
    print("\nTesting with V2 JSON...")
    with (root / 'all_sim_input_data.v2.json').open('r', encoding='utf-8') as f:
        data_v2 = json.load(f)
    dfs_v2 = build_dataframes(SimInput(**data_v2))
    for name, df in dfs_v2.items():
        print('V2', name, df.shape, list(df.columns)[:8])
