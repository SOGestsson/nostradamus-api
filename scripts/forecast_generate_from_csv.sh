#!/usr/bin/env bash
# POST /api/v1/forecast/generate from CSV columns: item, value, date
set -euo pipefail

CSV="${1:?Usage: $0 path/to.csv [X-Api-Key]}"
API_KEY="${2:-${X_API_KEY:-${SIM_API_KEY:-}}}"
BASE_URL="${FORECAST_API_BASE:-https://api.nostradamus-api.com}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "Pass X-Api-Key as arg 2, or export X_API_KEY / SIM_API_KEY" >&2
  exit 1
fi

HIST=$(tail -n +2 "$CSV" | jq -R -s -c '
  split("\n") | map(select(length > 0)) |
  map(split(",")) |
  map({item_id: (.[0]|tonumber), actual_sale: (.[1]|tonumber), day: (.[2]|gsub("\r";""))})
')

BODY=$(jq -n \
  --argjson sim_input_his "$HIST" \
  '{
    sim_input_his: $sim_input_his,
    forecast_periods: 6,
    mode: "local",
    local_model: "auto_ets",
    season_length: 12,
    freq: "M"
  }')

curl -sS -X POST "${BASE_URL%/}/api/v1/forecast/generate" \
  -H 'Content-Type: application/json' \
  -H "X-Api-Key: ${API_KEY}" \
  --data "$BODY"
