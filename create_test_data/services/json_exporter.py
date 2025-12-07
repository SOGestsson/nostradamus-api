"""Utilities for exporting sandbox data readers to JSON."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crud.read import HistoryByItemReader, HistoryReader, ProductMasterReader


@dataclass
class HistoryRecord:
    item: int
    value: float | None
    date: str


@dataclass
class ProductRecord:
    item: int
    status: str | None
    name: str | None
    category_id: int | None
    category_name: str | None


class SandboxJSONExporter:
    """Serialize HistoryReader and ProductMasterReader payloads as JSON."""

    def __init__(
        self,
        *,
        minimum_date: date = HistoryReader().minimum_date,
        status: str = HistoryReader().status,
        items: tuple[int, ...] | None = None,
    ) -> None:
        self._minimum_date = minimum_date
        self._status = status
        if items:
            self._history_reader = HistoryByItemReader(
                items=items, minimum_date=minimum_date, status=status
            )
        else:
            self._history_reader = HistoryReader(minimum_date=minimum_date, status=status)
        self._product_reader = ProductMasterReader(status=status)

    def history_as_json(self) -> str:
        """Return history rows (with product context) as JSON."""
        history_rows = self._history_reader.fetch()
        payload = [
            HistoryRecord(
                item=history.item,
                value=history.value,
                date=history.date.isoformat(),
            )
            for history, product in history_rows
        ]
        return json.dumps([asdict(record) for record in payload], ensure_ascii=False, indent=2)

    def history_for_item_as_json(self, item: int) -> str:
        """Return history rows filtered to a single item number."""
        reader = HistoryByItemReader(
            items=(item,),
            minimum_date=self._minimum_date,
            status=self._status,
        )
        history_rows = reader.fetch()
        payload = [
            HistoryRecord(
                item=history.item,
                value=history.value,
                date=history.date.isoformat(),
            )
            for history, _ in history_rows
        ]
        return json.dumps([asdict(record) for record in payload], ensure_ascii=False, indent=2)

    def products_as_json(self) -> str:
        """Return filtered products as JSON."""
        products = self._product_reader.fetch()
        payload = [
            ProductRecord(
                item=product.vornumer,
                status=product.stada,
                name=product.voruheiti,
                category_id=product.voruflokkur,
                category_name=product.voruflokkaheiti,
            )
            for product in products
        ]
        return json.dumps([asdict(record) for record in payload], ensure_ascii=False, indent=2)

    def combined_payload(self) -> dict[str, Any]:
        """Return a combined dictionary of history and product data."""
        history_json = json.loads(self.history_as_json())
        product_json = json.loads(self.products_as_json())
        return {
            "metadata": {
                "minimum_date": self._minimum_date.isoformat(),
                "status": self._status,
                "items": list(self._history_reader.items) if hasattr(self._history_reader, "items") else None,
            },
            "history": history_json,
            "products": product_json,
        }

    def combined_as_json(self) -> str:
        """Return the combined payload serialized to JSON."""
        return json.dumps(self.combined_payload(), ensure_ascii=False, indent=2)


def main() -> None:
    exporter = SandboxJSONExporter()
    print(exporter.history_for_item_as_json(11401))


if __name__ == "__main__":
    main()
