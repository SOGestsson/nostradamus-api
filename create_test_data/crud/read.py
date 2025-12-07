"""Read helpers for sandbox data stored in MySQL."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.orm_models import HistoryTest, ProductMasterTest, session_scope

DEFAULT_MIN_DATE = date(2020, 1, 1)
DEFAULT_STATUS = "Virkt"


class HistoryReader:
    """Fetch history rows filtered by product status and date."""

    def __init__(self, minimum_date: date = DEFAULT_MIN_DATE, status: str = DEFAULT_STATUS) -> None:
        self.minimum_date = minimum_date
        self.status = status

    def fetch(self) -> list[tuple[HistoryTest, ProductMasterTest]]:
        """Return history entries for active products since the minimum date."""
        stmt = (
            select(HistoryTest, ProductMasterTest)
            .join(ProductMasterTest, HistoryTest.item == ProductMasterTest.vornumer)
            .where(HistoryTest.date >= self.minimum_date)
            .where(ProductMasterTest.stada == self.status)
            .order_by(HistoryTest.item, HistoryTest.date)
        )

        with session_scope() as session:
            result = session.execute(stmt)
            return list(result.all())


class HistoryByItemReader:
    """Fetch history rows for specific item numbers."""

    def __init__(
        self,
        items: Sequence[int],
        minimum_date: date = DEFAULT_MIN_DATE,
        status: str = DEFAULT_STATUS,
    ) -> None:
        if not items:
            raise ValueError("items must contain at least one item number")
        self.items = tuple(int(item) for item in items)
        self.minimum_date = minimum_date
        self.status = status

    def fetch(self) -> list[tuple[HistoryTest, ProductMasterTest]]:
        stmt = (
            select(HistoryTest, ProductMasterTest)
            .join(ProductMasterTest, HistoryTest.item == ProductMasterTest.vornumer)
            .where(HistoryTest.item.in_(self.items))
            .where(HistoryTest.date >= self.minimum_date)
            .where(ProductMasterTest.stada == self.status)
            .order_by(HistoryTest.item, HistoryTest.date)
        )

        with session_scope() as session:
            result = session.execute(stmt)
            return list(result.all())


class ProductMasterReader:
    """Fetch product master rows filtered by status."""

    def __init__(self, status: str = DEFAULT_STATUS) -> None:
        self.status = status

    def fetch(self) -> list[ProductMasterTest]:
        """Return product rows that match the configured status."""
        stmt = (
            select(ProductMasterTest)
            .where(ProductMasterTest.stada == self.status)
            .order_by(ProductMasterTest.vornumer)
        )

        with session_scope() as session:
            result = session.execute(stmt)
            return [row[0] for row in result.all()]


def main() -> None:
    reader = HistoryReader()
    rows: Iterable[tuple[HistoryTest, ProductMasterTest]] = reader.fetch()
    for history, product in rows[:10]:  # limit preview
        print(
            f"item={history.item}, date={history.date}, value={history.value}, "
            f"status={product.stada}, name={product.voruheiti}"
        )

    product_reader = ProductMasterReader()
    products = product_reader.fetch()[:10]
    print("\nActive products:")
    for product in products:
        print(f"item={product.vornumer}, status={product.stada}, name={product.voruheiti}")


if __name__ == "__main__":
    main()
