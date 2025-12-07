"""Helper class for building forecast payloads from Nostradamus test data."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import requests


@dataclass
class ForecastInputForecastBuilder:
    """Compose the forecast request payload and fetch forecasts from the API."""

    item_number: int = 11500
    forecast_periods: int = 6
    mode: str = "local"
    local_model: str = "auto_ets"
    season_length: int = 12
    freq: str = "MS"
    base_url: str = "https://api.nostradamus-api.com/api/v1/test-data/history/{item_id}"
    request_timeout: int = 15
    session: requests.Session = field(default_factory=requests.Session)

    _field_aliases: Dict[str, List[str]] = field(
        init=False,
        repr=False,
        default_factory=lambda: {
            "item_id": ["item_id", "itemid", "item", "sku", "sku_id", "product_id"],
            "actual_sale": [
                "actual_sale",
                "actualsales",
                "actual_sales",
                "sales",
                "sale",
                "quantity",
                "qty",
                "qty_sold",
                "units",
                "value",
            ],
            "day": ["day", "date", "ds", "timestamp", "period", "period_start", "period_end"],
        },
    )

    def _resolve(self, entry: Dict[str, object], target: str) -> Optional[object]:
        normalized = {key.lower().replace("-", "_"): value for key, value in entry.items()}
        for alias in self._field_aliases[target]:
            candidate = normalized.get(alias)
            if candidate is not None:
                return candidate
        return None

    def fetch_history(self) -> List[Dict[str, object]]:
        """Retrieve the raw history list for the configured item."""
        url = self.base_url.format(item_id=self.item_number)
        response = self.session.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        payload = response.json()
        history = payload.get("history")
        if history is None:
            raise KeyError("Response JSON does not contain a 'history' field.")
        if not isinstance(history, list):
            raise TypeError("Response JSON history field must be a list.")
        return history

    def history_to_dataframe(self, history: Optional[List[Dict[str, object]]] = None) -> pd.DataFrame:
        """Convert history entries into a tidy DataFrame."""
        history = history if history is not None else self.fetch_history()
        return pd.json_normalize(history)

    def build_sim_input(self, history: Optional[List[Dict[str, object]]] = None) -> List[Dict[str, object]]:
        """Prepare the `sim_input_his` structure expected by the forecast API."""
        history = history if history is not None else self.fetch_history()
        sim_input_his: List[Dict[str, object]] = []
        for idx, entry in enumerate(history):
            if not isinstance(entry, dict):
                raise TypeError(f"History entry at index {idx} is not a dictionary.")
            item_id = self._resolve(entry, "item_id")
            actual_sale = self._resolve(entry, "actual_sale")
            day = self._resolve(entry, "day")
            missing = [
                key
                for key, value in {"item_id": item_id, "actual_sale": actual_sale, "day": day}.items()
                if value is None
            ]
            if missing:
                raise KeyError(
                    "History entry at index {idx} is missing required field(s): {fields}".format(
                        idx=idx, fields=", ".join(missing)
                    )
                )
            sim_input_his.append({"item_id": item_id, "actual_sale": actual_sale, "day": day})
        return sim_input_his

    def build_payload(self, history: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
        """Assemble the full payload JSON body for the forecast API."""
        sim_input_his = self.build_sim_input(history)
        return {
            "sim_input_his": sim_input_his,
            "forecast_periods": self.forecast_periods,
            "mode": self.mode,
            "local_model": self.local_model,
            "season_length": self.season_length,
            "freq": self.freq,
        }

    def create_payload_from_api(self) -> Dict[str, object]:
        """Fetch history and construct the payload in a single call."""
        history = self.fetch_history()
        return self.build_payload(history)

    def fetch_and_build(self) -> Dict[str, object]:
        """Alias for create_payload_from_api to keep naming flexible."""
        return self.create_payload_from_api()

    def generate_forecast(
        self,
        *,
        payload: Optional[Dict[str, object]] = None,
        endpoint: str = "https://api.nostradamus-api.com/api/v1/forecast/generate",
        timeout: int = 30,
        quantiles: Optional[List[float]] = None,
        **overrides: object,
    ) -> Dict[str, object]:
        """Return both the request payload and the forecast produced by the API."""

        payload = payload if payload is not None else self.create_payload_from_api()
        body = json.loads(json.dumps(payload))

        if quantiles is not None:
            body["quantiles"] = quantiles

        for key, value in overrides.items():
            if value is not None:
                body[key] = value

        response = self.session.post(endpoint, json=body, timeout=timeout)
        response.raise_for_status()
        forecast = response.json()

        return {"payload": body, "forecast": forecast}


def _main() -> None:
    builder = ForecastInputForecastBuilder()
    result = builder.generate_forecast(quantiles=[0.1, 0.5, 0.9])
    print("Payload:")
    print(json.dumps(result["payload"], indent=2))
    print("\nForecast:")
    print(json.dumps(result["forecast"], indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    _main()
