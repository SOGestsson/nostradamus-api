"""DuckDB-backed model registry + evidence locker.

This module provides a small, file-based model store that is:
- portable (single .duckdb file per customer)
- queryable (DuckDB)
- compatible with artifact files on disk (LightGBM models, JSON specs, etc.)

The store is intentionally minimal and future-proof.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable


try:
    import duckdb  # type: ignore

    _HAS_DUCKDB = True
except Exception:
    duckdb = None
    _HAS_DUCKDB = False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class ModelVersionRow:
    customer_id: str
    model_version: str
    created_at: datetime
    status: str
    freq: str
    horizon: int
    target: str
    train_end_ds: str | None
    artifact_root: str
    notes: str | None = None
    created_by: str | None = None
    git_sha: str | None = None


class DuckDBModelStore:
    """A small DuckDB-backed registry for ML model releases."""

    def __init__(self, store_root: str, customer_id: str):
        if not _HAS_DUCKDB:
            raise RuntimeError("duckdb is not installed. Add `duckdb` to requirements.")
        self.store_root = os.path.abspath(store_root)
        self.customer_id = str(customer_id)

    def customer_dir(self) -> str:
        return os.path.join(self.store_root, self.customer_id)

    def db_path(self) -> str:
        return os.path.join(self.customer_dir(), "store.duckdb")

    def connect(self):
        os.makedirs(self.customer_dir(), exist_ok=True)
        con = duckdb.connect(self.db_path())
        self.ensure_schema(con)
        return con

    def ensure_schema(self, con) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              created_at TIMESTAMP NOT NULL,
              created_by TEXT,
              status TEXT NOT NULL,
              freq TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              target TEXT NOT NULL,
              train_end_ds DATE,
              artifact_root TEXT NOT NULL,
              notes TEXT,
              git_sha TEXT,
              PRIMARY KEY (customer_id, model_version)
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_specs (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              spec_json JSON NOT NULL,
              spec_hash TEXT NOT NULL,
              created_at TIMESTAMP NOT NULL,
              PRIMARY KEY (customer_id, model_version)
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_metrics_item (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              unique_id TEXT NOT NULL,
              model_name TEXT NOT NULL,
              metric_name TEXT NOT NULL,
              metric_value DOUBLE,
              n_folds INTEGER,
              eval_start_ds DATE,
              eval_end_ds DATE,
              updated_at TIMESTAMP NOT NULL
            );
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_backtest_item
            ON backtest_metrics_item(customer_id, model_version, unique_id);
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS eligibility_item (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              unique_id TEXT NOT NULL,
              winner_model TEXT,
              ml_allowed BOOLEAN,
              ml_preferred BOOLEAN,
              fallback_model TEXT,
              reason_code TEXT,
              confidence DOUBLE,
              min_history_points INTEGER,
              requires_features_json JSON,
              updated_at TIMESTAMP NOT NULL,
              PRIMARY KEY (customer_id, model_version, unique_id)
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS explain_item_summary (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              unique_id TEXT NOT NULL,
              top_features_json JSON,
              group_contrib_json JSON,
              support_share DOUBLE,
              updated_at TIMESTAMP NOT NULL,
              PRIMARY KEY (customer_id, model_version, unique_id)
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS cv_residuals (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              unique_id TEXT NOT NULL,
              ds DATE NOT NULL,
              horizon INTEGER NOT NULL,
              y DOUBLE,
              yhat DOUBLE,
              residual DOUBLE,
              positive_excess DOUBLE,
              archetype TEXT,
              fold_id INTEGER,
              created_at TIMESTAMP NOT NULL
            );
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS residual_quantiles (
              customer_id TEXT NOT NULL,
              model_version TEXT NOT NULL,
              archetype TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              scale_bucket TEXT,
              q95_excess DOUBLE,
              n INTEGER,
              updated_at TIMESTAMP NOT NULL,
              PRIMARY KEY (customer_id, model_version, archetype, horizon, scale_bucket)
            );
            """
        )
        # Migration: multi-quantile residual calibration (older rows only had q95_excess)
        try:
            con.execute("ALTER TABLE residual_quantiles ADD COLUMN IF NOT EXISTS q70_excess DOUBLE;")
            con.execute("ALTER TABLE residual_quantiles ADD COLUMN IF NOT EXISTS q90_excess DOUBLE;")
            con.execute("ALTER TABLE residual_quantiles ADD COLUMN IF NOT EXISTS margin_q70 DOUBLE;")
            con.execute("ALTER TABLE residual_quantiles ADD COLUMN IF NOT EXISTS margin_q90 DOUBLE;")
            con.execute("ALTER TABLE residual_quantiles ADD COLUMN IF NOT EXISTS margin_q95 DOUBLE;")
        except Exception:
            pass

    def create_model_version(self, row: ModelVersionRow) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO model_versions (
                  customer_id, model_version, created_at, created_by, status,
                  freq, horizon, target, train_end_ds, artifact_root, notes, git_sha
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (customer_id, model_version) DO UPDATE SET
                  created_at=excluded.created_at,
                  created_by=excluded.created_by,
                  status=excluded.status,
                  freq=excluded.freq,
                  horizon=excluded.horizon,
                  target=excluded.target,
                  train_end_ds=excluded.train_end_ds,
                  artifact_root=excluded.artifact_root,
                  notes=excluded.notes,
                  git_sha=excluded.git_sha;
                """,
                [
                    row.customer_id,
                    row.model_version,
                    row.created_at,
                    row.created_by,
                    row.status,
                    row.freq,
                    int(row.horizon),
                    row.target,
                    row.train_end_ds,
                    row.artifact_root,
                    row.notes,
                    row.git_sha,
                ],
            )

    def set_feature_spec(self, model_version: str, spec: dict[str, Any]) -> str:
        spec_text = _canonical_json(spec)
        spec_hash = _sha256_text(spec_text)
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO feature_specs (customer_id, model_version, spec_json, spec_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (customer_id, model_version) DO UPDATE SET
                  spec_json=excluded.spec_json,
                  spec_hash=excluded.spec_hash,
                  created_at=excluded.created_at;
                """,
                [self.customer_id, model_version, spec_text, spec_hash, datetime.now(UTC)],
            )
        return spec_hash

    def get_active_model_version(self, status: str = "prod") -> str | None:
        with self.connect() as con:
            res = con.execute(
                """
                SELECT model_version
                FROM model_versions
                WHERE customer_id=? AND status=?
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                [self.customer_id, status],
            ).fetchone()
        return res[0] if res else None

    def get_model_artifact_root(self, model_version: str) -> str | None:
        with self.connect() as con:
            res = con.execute(
                """
                SELECT artifact_root
                FROM model_versions
                WHERE customer_id=? AND model_version=?
                LIMIT 1;
                """,
                [self.customer_id, model_version],
            ).fetchone()
        return res[0] if res else None

    def upsert_backtest_metrics(self, rows: Iterable[dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        now = datetime.now(UTC)
        values = []
        for r in rows_list:
            values.append(
                (
                    self.customer_id,
                    r["model_version"],
                    str(r["unique_id"]),
                    str(r["model_name"]),
                    str(r["metric_name"]),
                    float(r["metric_value"]) if r.get("metric_value") is not None else None,
                    int(r.get("n_folds") or 0) if r.get("n_folds") is not None else None,
                    r.get("eval_start_ds"),
                    r.get("eval_end_ds"),
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO backtest_metrics_item (
                  customer_id, model_version, unique_id, model_name, metric_name,
                  metric_value, n_folds, eval_start_ds, eval_end_ds, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                values,
            )

    def insert_cv_residuals(self, rows: Iterable[dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        now = datetime.now(UTC)
        values = []
        for r in rows_list:
            values.append(
                (
                    self.customer_id,
                    r["model_version"],
                    str(r["unique_id"]),
                    r["ds"],
                    int(r["horizon"]),
                    r.get("y"),
                    r.get("yhat"),
                    r.get("residual"),
                    r.get("positive_excess"),
                    r.get("archetype"),
                    int(r.get("fold_id", 0)),
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO cv_residuals (
                  customer_id, model_version, unique_id, ds, horizon,
                  y, yhat, residual, positive_excess, archetype, fold_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                values,
            )

    def upsert_residual_quantiles(self, rows: Iterable[dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        now = datetime.now(UTC)
        values = []
        for r in rows_list:
            q95 = r.get("q95_excess")
            q70 = r.get("q70_excess")
            q90 = r.get("q90_excess")
            if q95 is not None:
                q95f = float(q95)
                if q70 is None:
                    q70 = 0.4 * q95f
                if q90 is None:
                    q90 = 0.8 * q95f
            values.append(
                (
                    self.customer_id,
                    r["model_version"],
                    r["archetype"],
                    int(r["horizon"]),
                    r.get("scale_bucket"),
                    q70,
                    q90,
                    q95,
                    r.get("margin_q70"),
                    r.get("margin_q90"),
                    r.get("margin_q95"),
                    int(r.get("n", 0)),
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO residual_quantiles (
                  customer_id, model_version, archetype, horizon, scale_bucket,
                  q70_excess, q90_excess, q95_excess,
                  margin_q70, margin_q90, margin_q95,
                  n, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (customer_id, model_version, archetype, horizon, scale_bucket) DO UPDATE SET
                  q70_excess=excluded.q70_excess,
                  q90_excess=excluded.q90_excess,
                  q95_excess=excluded.q95_excess,
                  margin_q70=COALESCE(excluded.margin_q70, residual_quantiles.margin_q70),
                  margin_q90=COALESCE(excluded.margin_q90, residual_quantiles.margin_q90),
                  margin_q95=COALESCE(excluded.margin_q95, residual_quantiles.margin_q95),
                  n=excluded.n,
                  updated_at=excluded.updated_at;
                """,
                values,
            )

    def get_residual_quantiles(self, model_version: str) -> dict[tuple[str, int], dict[str, float]]:
        """Per (archetype, horizon): q70/90/95 excess + optional conformal margin_q*."""
        with self.connect() as con:
            res = con.execute(
                """
                SELECT archetype, horizon, q70_excess, q90_excess, q95_excess,
                  margin_q70, margin_q90, margin_q95
                FROM residual_quantiles
                WHERE customer_id=? AND model_version=?;
                """,
                [self.customer_id, model_version],
            ).fetchall()
        out: dict[tuple[str, int], dict[str, float]] = {}
        for row in res or []:
            (
                archetype,
                horizon,
                q70_excess,
                q90_excess,
                q95_excess,
                margin_q70,
                margin_q90,
                margin_q95,
            ) = row
            if archetype is None or horizon is None:
                continue
            key = (str(archetype), int(horizon))
            q95f = float(q95_excess or 0.0)
            q70f = float(q70_excess) if q70_excess is not None else float(0.4 * q95f)
            q90f = float(q90_excess) if q90_excess is not None else float(0.8 * q95f)
            out[key] = {
                "q70": q70f,
                "q90": q90f,
                "q95": q95f,
                "margin_q70": float(margin_q70) if margin_q70 is not None else 0.0,
                "margin_q90": float(margin_q90) if margin_q90 is not None else 0.0,
                "margin_q95": float(margin_q95) if margin_q95 is not None else 0.0,
            }
        return out
    def upsert_eligibility(self, rows: Iterable[dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        now = datetime.now(UTC)
        values = []
        for r in rows_list:
            values.append(
                (
                    self.customer_id,
                    r["model_version"],
                    str(r["unique_id"]),
                    r.get("winner_model"),
                    bool(r.get("ml_allowed")) if r.get("ml_allowed") is not None else None,
                    bool(r.get("ml_preferred")) if r.get("ml_preferred") is not None else None,
                    r.get("fallback_model"),
                    r.get("reason_code"),
                    float(r.get("confidence")) if r.get("confidence") is not None else None,
                    int(r.get("min_history_points")) if r.get("min_history_points") is not None else None,
                    _canonical_json(r.get("requires_features", [])),
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO eligibility_item (
                  customer_id, model_version, unique_id,
                  winner_model, ml_allowed, ml_preferred, fallback_model,
                  reason_code, confidence, min_history_points, requires_features_json,
                  updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (customer_id, model_version, unique_id) DO UPDATE SET
                  winner_model=excluded.winner_model,
                  ml_allowed=excluded.ml_allowed,
                  ml_preferred=excluded.ml_preferred,
                  fallback_model=excluded.fallback_model,
                  reason_code=excluded.reason_code,
                  confidence=excluded.confidence,
                  min_history_points=excluded.min_history_points,
                  requires_features_json=excluded.requires_features_json,
                  updated_at=excluded.updated_at;
                """,
                values,
            )

    def upsert_explain_summary(self, rows: Iterable[dict[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        now = datetime.now(UTC)
        values = []
        for r in rows_list:
            values.append(
                (
                    self.customer_id,
                    r["model_version"],
                    str(r["unique_id"]),
                    _canonical_json(r.get("top_features", [])),
                    _canonical_json(r.get("group_contrib", {})),
                    float(r.get("support_share")) if r.get("support_share") is not None else None,
                    now,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO explain_item_summary (
                  customer_id, model_version, unique_id,
                  top_features_json, group_contrib_json, support_share, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (customer_id, model_version, unique_id) DO UPDATE SET
                  top_features_json=excluded.top_features_json,
                  group_contrib_json=excluded.group_contrib_json,
                  support_share=excluded.support_share,
                  updated_at=excluded.updated_at;
                """,
                values,
            )

    def get_eligibility(self, model_version: str, unique_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not unique_ids:
            return {}
        placeholders = ",".join(["?"] * len(unique_ids))
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT unique_id, winner_model, ml_allowed, fallback_model, reason_code, confidence
                FROM eligibility_item
                WHERE customer_id=? AND model_version=? AND unique_id IN ({placeholders});
                """,
                [self.customer_id, model_version, *unique_ids],
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for uid, winner_model, ml_allowed, fallback_model, reason_code, confidence in rows:
            out[str(uid)] = {
                "winner_model": winner_model,
                "ml_allowed": bool(ml_allowed) if ml_allowed is not None else None,
                "fallback_model": fallback_model,
                "reason_code": reason_code,
                "confidence": float(confidence) if confidence is not None else None,
            }
        return out

    def get_explain_summary(
        self,
        model_version: str,
        unique_ids: list[str] | None = None,
        limit: int = 500,
    ) -> dict[str, dict[str, Any]]:
        """Fetch per-item explanation/diagnostics JSON stored in explain_item_summary.

        Returns mapping: unique_id -> {top_features, group_contrib, support_share, updated_at}
        """
        with self.connect() as con:
            if unique_ids:
                placeholders = ",".join(["?"] * len(unique_ids))
                rows = con.execute(
                    f"""
                    SELECT unique_id, top_features_json, group_contrib_json, support_share, updated_at
                    FROM explain_item_summary
                    WHERE customer_id=? AND model_version=? AND unique_id IN ({placeholders});
                    """,
                    [self.customer_id, model_version, *unique_ids],
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT unique_id, top_features_json, group_contrib_json, support_share, updated_at
                    FROM explain_item_summary
                    WHERE customer_id=? AND model_version=?
                    ORDER BY updated_at DESC
                    LIMIT ?;
                    """,
                    [self.customer_id, model_version, int(limit)],
                ).fetchall()

        out: dict[str, dict[str, Any]] = {}
        for uid, topj, groupj, support_share, updated_at in rows:
            try:
                top = json.loads(topj) if isinstance(topj, str) and topj else topj
            except Exception:
                top = topj
            try:
                group = json.loads(groupj) if isinstance(groupj, str) and groupj else groupj
            except Exception:
                group = groupj
            out[str(uid)] = {
                "top_features": top or [],
                "group_contrib": group or {},
                "support_share": float(support_share) if support_share is not None else None,
                "updated_at": str(updated_at) if updated_at is not None else None,
            }
        return out
