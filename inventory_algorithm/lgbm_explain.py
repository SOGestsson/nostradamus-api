"""SHAP reconciliation and payload keys for the served statistical head."""

from __future__ import annotations

from typing import Any

import numpy as np


def shap_raw_score(bias: float, contributions: np.ndarray) -> float:
    return float(bias) + float(np.sum(np.asarray(contributions, dtype=float)))


def independent_raw_score(booster: Any, X) -> float:
    """LightGBM raw_score for one row. Must not be reconstructed from pred_contrib."""
    raw = booster.predict(X, raw_score=True)
    return float(np.asarray(raw).ravel()[0])


def pred_contrib_payload(booster: Any, X, *, feature_names: list[str] | None = None) -> dict[str, Any]:
    contrib = np.asarray(booster.predict(X, pred_contrib=True)[0], dtype=float)
    raw_score = independent_raw_score(booster, X)
    bias = float(contrib[-1])
    feat_contrib = contrib[:-1]
    names = list(feature_names or booster.feature_name())
    recon = reconcile_shap(bias=bias, contributions=feat_contrib, raw_score=raw_score)
    pairs = list(zip(names, feat_contrib))
    pairs_sorted = sorted(pairs, key=lambda kv: abs(float(kv[1])), reverse=True)
    return {
        "bias": bias,
        "contributions": feat_contrib,
        "feature_names": names,
        "pairs_sorted": pairs_sorted,
        "contrib_map": {str(n): float(v) for n, v in pairs},
        "raw_score": raw_score,
        "margin_sum": float(np.sum(contrib)),
        "reconcile": recon,
    }


def reconcile_shap(
    *,
    bias: float,
    contributions: np.ndarray,
    raw_score: float,
    atol: float = 1e-4,
) -> dict[str, Any]:
    recon = shap_raw_score(bias, contributions)
    ok = bool(np.isfinite(recon) and np.isfinite(raw_score) and abs(recon - float(raw_score)) <= atol)
    return {
        "bias": float(bias),
        "sum_contributions": float(np.sum(np.asarray(contributions, dtype=float))),
        "reconstructed_raw_score": recon,
        "model_raw_score": float(raw_score),
        "reconciled": ok,
        "atol": float(atol),
    }


def explain_keys(
    *,
    model_version: str,
    spec_hash: str | None,
    context_version: str | None,
    item_id: str,
    origin,
    forecast_ds,
    horizon: int,
    scenario_id: str = "baseline",
) -> dict[str, Any]:
    return {
        "model_version": str(model_version),
        "spec_hash": spec_hash,
        "context_version": context_version,
        "item_id": str(item_id),
        "origin": str(origin),
        "forecast_ds": str(forecast_ds),
        "horizon": int(horizon),
        "scenario_id": str(scenario_id),
    }
