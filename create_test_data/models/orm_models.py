"""SQLAlchemy configuration and ORM models for the sandbox MySQL database."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import BigInteger, Column, Date, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Allow overriding credentials through environment variables while keeping sensible defaults.
HOST = os.getenv("SANDBOX_DB_HOST", "raspberrypi.local")
PORT = int(os.getenv("SANDBOX_DB_PORT", "4406"))
USER = os.getenv("SANDBOX_DB_USER", "root")
PASSWORD = os.getenv("SANDBOX_DB_PASSWORD", "Superman")
DATABASE = os.getenv("SANDBOX_DB_NAME", "sandbox")

DATABASE_URL = f"mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

Base = declarative_base()


def get_engine(echo: bool = False) -> Engine:
    """Create an engine bound to the sandbox database."""
    return create_engine(DATABASE_URL, echo=echo, future=True)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=get_engine(),
    expire_on_commit=False,
    future=True,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for database operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Return a new Session instance."""
    return SessionLocal()


class HistoryTest(Base):
    """Model mapping the sandbox.history_test table."""

    __tablename__ = "history_test"

    item = Column(Integer, primary_key=True, autoincrement=False)
    value = Column(Float, nullable=True)
    date = Column(Date, primary_key=True)

    def __repr__(self) -> str:
        return f"HistoryTest(item={self.item!r}, value={self.value!r}, date={self.date!r})"


class ProductMasterTest(Base):
    """Model mapping the sandbox.product_master_test table."""

    __tablename__ = "product_master_test"

    vornumer = Column("Vörunúmer", BigInteger, primary_key=True, autoincrement=False)
    stada = Column("staða", String(255), nullable=True)
    voruheiti = Column("Vöruheiti", String(255), nullable=True)
    voruflokkur = Column("Vöruflokkur", BigInteger, nullable=True)
    voruflokkaheiti = Column("Vöruflokkaheiti", String(255), nullable=True)

    def __repr__(self) -> str:
        return (
            "ProductMasterTest("
            f"Vörunúmer={self.vornumer!r}, "
            f"staða={self.stada!r}, "
            f"vöruheiti={self.voruheiti!r}, "
            f"Vöruflokkur={self.voruflokkur!r}, "
            f"Vöruflokkaheiti={self.voruflokkaheiti!r})"
        )


def main() -> None:
    """Quick smoke test that fetches a few rows from each table."""
    with session_scope() as session:
        history_sample = session.query(HistoryTest).limit(5).all()
        product_sample = session.query(ProductMasterTest).limit(5).all()

        print("history_test sample:")
        for row in history_sample:
            print(row)

        print("\nproduct_master_test sample:")
        for row in product_sample:
            print(row)


if __name__ == "__main__":
    main()
