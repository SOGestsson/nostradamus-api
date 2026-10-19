"""Shared prediction composition: hurdle vs tweedie, versioned adjustments, quantiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from inventory_algorithm.lgbm_origin import wape_with_undefined

MEAN_HEAD_TWEEDIE = "tweedie"
MEAN_HEAD_HURDLE = "hurdle"


@dataclass
class AdjustmentStep:
    name: str
    before: float
    after: float
    delta: float
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": float(self.before),
            "after": float(self.after),
            "delta": float(self.delta),
            **self.detail,
        }


@dataclass
class AdjustmentPolicy:
    """Planning rules. Statistical audit should use enabled=False for all rules."""

    enabled: bool = False
    occurrence_blend: bool = False
    nonzero_level_anchor: bool = False
    recent_level_blend: bool = False
    seasonal_floor: bool = False
    croston_floor: bool = False
    month_cap: bool = False
    non_negative: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "occurrence_blend": bool(self.occurrence_blend),
            "nonzero_level_anchor": bool(self.nonzero_level_anchor),
            "recent_level_blend": bool(self.recent_level_blend),
            "seasonal_floor": bool(self.seasonal_floor),
            "croston_floor": bool(self.croston_floor),
            "month_cap": bool(self.month_cap),
            "non_negative": bool(self.non_negative),
        }


AUDIT_POLICY = AdjustmentPolicy(enabled=False, non_negative=True)
PLANNING_POLICY = AdjustmentPolicy(
    enabled=True,
    occurrence_blend=False,
    nonzero_level_anchor=True,
    recent_level_blend=True,
    seasonal_floor=True,
    croston_floor=True,
    month_cap=True,
    non_negative=True,
)
# Serve/eligibility policy for seasonal and intermittent SKUs. Same-month history
# can lift a near-zero head (December-only items). recent_level_blend stays off so
# off-season recency does not shrink the peak. month_cap/croston stay off so this
# is not the full planning stack.
SEASONAL_SERVE_POLICY = AdjustmentPolicy(
    enabled=True,
    occurrence_blend=False,
    nonzero_level_anchor=True,
    recent_level_blend=False,
    seasonal_floor=True,
    croston_floor=False,
    month_cap=False,
    non_negative=True,
)


def quantile_to_absolute(
    q_raw: float | None,
    yhat_mean: float,
    *,
    direct_demand: bool,
) -> float | None:
    """Schema v2 quantile heads are absolute demand. v1 heads are excess over the mean."""
    if q_raw is None:
        return None
    if direct_demand:
        return float(q_raw)
    return float(yhat_mean) + float(q_raw)


def hurdle_expectation(p_occ: float, amount_if_positive: float) -> float:
    p = float(np.clip(p_occ, 0.0, 1.0))
    return float(p * max(0.0, amount_if_positive))


def compose_mean(
    *,
    head: str,
    yhat_tweedie: float,
    yhat_positive: float | None,
    p_occ: float | None,
) -> tuple[float, str]:
    """Return (mean, head_used). Never multiply an unconditional Tweedie mean by occurrence."""
    if head == MEAN_HEAD_HURDLE:
        if yhat_positive is None or p_occ is None:
            return float(max(0.0, yhat_tweedie)), MEAN_HEAD_TWEEDIE
        return hurdle_expectation(p_occ, yhat_positive), MEAN_HEAD_HURDLE
    return float(max(0.0, yhat_tweedie)), MEAN_HEAD_TWEEDIE


def select_mean_head(
    y_true: np.ndarray,
    yhat_tweedie: np.ndarray,
    yhat_hurdle: np.ndarray,
) -> str:
    """Pick hurdle only when calibration has positive demand and hurdle is not worse on those months.

    An all-zero off-season window makes hurdle look good because it predicts zeros. Serving that
    head then zeros the seasonal peak (e.g. December-only items).
    """
    y_true = np.asarray(y_true, dtype=float)
    yhat_tweedie = np.asarray(yhat_tweedie, dtype=float)
    yhat_hurdle = np.asarray(yhat_hurdle, dtype=float)
    if len(y_true) == 0 or float(np.sum(np.abs(y_true))) <= 1e-12:
        return MEAN_HEAD_TWEEDIE
    pos = y_true > 1e-12
    if int(np.sum(pos)) > 0:
        w_t_pos = wape_with_undefined(y_true[pos], yhat_tweedie[pos])
        w_h_pos = wape_with_undefined(y_true[pos], yhat_hurdle[pos])
        if w_h_pos.get("wape") is None:
            return MEAN_HEAD_TWEEDIE
        if w_t_pos.get("wape") is not None and float(w_h_pos["wape"]) > float(w_t_pos["wape"]):
            return MEAN_HEAD_TWEEDIE
    w_t = wape_with_undefined(y_true, yhat_tweedie)
    w_h = wape_with_undefined(y_true, yhat_hurdle)
    if w_h.get("wape") is not None and (
        w_t.get("wape") is None or float(w_h["wape"]) < float(w_t["wape"])
    ):
        return MEAN_HEAD_HURDLE
    return MEAN_HEAD_TWEEDIE


def predict_from_heads(
    *,
    head: str,
    yhat_tweedie: float,
    yhat_positive: float | None,
    p_occ: float | None,
    policy: AdjustmentPolicy,
    q70: float | None = None,
    q90: float | None = None,
    q95: float | None = None,
    month_mean: float = 0.0,
    month_max: float = 0.0,
    recent_level: float = 0.0,
    croston: float = 0.0,
    month_cap: float | None = None,
) -> Prediction:
    raw, head_used = compose_mean(
        head=head,
        yhat_tweedie=yhat_tweedie,
        yhat_positive=yhat_positive,
        p_occ=p_occ,
    )
    yhat, trace = apply_adjustment_policy(
        raw,
        policy,
        p_occ=p_occ,
        month_mean=month_mean,
        month_max=month_max,
        recent_level=recent_level,
        croston=croston,
        month_cap=month_cap,
    )
    u70 = u90 = u95 = None
    ceiling = None
    if q70 is not None and q90 is not None and q95 is not None:
        u70, u90, u95 = enforce_quantile_order(yhat, q70, q90, q95, allow_below_mean=True)
        if policy.month_cap and month_cap is not None and month_cap > 0.0:
            ceiling = float(month_cap)
    return Prediction(
        yhat=float(yhat),
        yhat_raw=float(raw),
        head=head_used,
        p_occ=None if p_occ is None else float(p_occ),
        yhat_positive=yhat_positive,
        adjustments=trace,
        upper_70=u70,
        upper_90=u90,
        upper_95=u95,
        planning_ceiling=ceiling,
    )


def _step(name: str, before: float, after: float, **detail: Any) -> AdjustmentStep:
    return AdjustmentStep(name=name, before=float(before), after=float(after), delta=float(after - before), detail=detail)


def apply_adjustment_policy(
    yhat: float,
    policy: AdjustmentPolicy,
    *,
    p_occ: float | None = None,
    month_mean: float = 0.0,
    month_max: float = 0.0,
    recent_level: float = 0.0,
    croston: float = 0.0,
    month_cap: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    trace: list[AdjustmentStep] = []
    cur = float(yhat)
    if policy.non_negative:
        nxt = max(0.0, cur)
        if nxt != cur:
            trace.append(_step("non_negative", cur, nxt))
            cur = nxt
    if not policy.enabled:
        return cur, [s.as_dict() for s in trace]

    if policy.occurrence_blend and p_occ is not None:
        # Explicit opt-in only. Default planning policy keeps this off so we do not
        # multiply an unconditional mean by occurrence.
        nxt = float(np.clip(p_occ, 0.0, 1.0)) * cur
        trace.append(_step("occurrence_blend", cur, nxt, p_occ=float(p_occ)))
        cur = nxt

    if policy.nonzero_level_anchor and month_mean > 0.0 and cur < 0.5 * month_mean:
        nxt = 0.5 * float(month_mean)
        trace.append(_step("nonzero_level_anchor", cur, nxt, month_mean=float(month_mean)))
        cur = nxt

    if policy.recent_level_blend and recent_level > 0.0:
        nxt = 0.7 * cur + 0.3 * float(recent_level)
        trace.append(_step("recent_level_blend", cur, nxt, recent_level=float(recent_level)))
        cur = nxt

    if policy.seasonal_floor and month_mean > 0.0:
        floor = 0.3 * float(month_mean)
        if cur < floor:
            trace.append(_step("seasonal_floor", cur, floor, month_mean=float(month_mean)))
            cur = floor

    if policy.croston_floor and croston > 0.0 and cur <= 0.0:
        trace.append(_step("croston_floor", cur, float(croston), croston=float(croston)))
        cur = float(croston)

    statistical = cur
    if policy.month_cap and month_cap is not None and month_cap > 0.0 and cur > float(month_cap):
        trace.append(_step("month_cap", cur, float(month_cap), cap=float(month_cap), planning_constraint=True))
        cur = float(month_cap)

    if policy.non_negative:
        nxt = max(0.0, cur)
        if nxt != cur:
            trace.append(_step("non_negative", cur, nxt))
            cur = nxt
    return cur, [s.as_dict() for s in trace]


@dataclass
class Prediction:
    yhat: float
    yhat_raw: float
    head: str
    p_occ: float | None
    yhat_positive: float | None
    adjustments: list[dict[str, Any]]
    upper_70: float | None = None
    upper_90: float | None = None
    upper_95: float | None = None
    planning_ceiling: float | None = None
    failed: bool = False
    reason_code: str | None = None


def enforce_quantile_order(
    yhat: float,
    q70: float,
    q90: float,
    q95: float,
    *,
    allow_below_mean: bool = True,
) -> tuple[float, float, float]:
    """Enforce q70 <= q90 <= q95. Quantiles may sit below the mean when appropriate."""
    qs = [float(q70), float(q90), float(q95)]
    qs.sort()
    if not allow_below_mean:
        qs = [max(float(yhat), q) for q in qs]
        qs.sort()
    return qs[0], qs[1], qs[2]


def coverage_audit(
    y_true: np.ndarray,
    q_pred: np.ndarray,
    *,
    after_caps: bool = False,
) -> dict[str, Any]:
    if after_caps:
        return {
            "coverage": None,
            "coverage_undefined": True,
            "reason": "coverage is not claimed after arbitrary planning caps",
            "n": int(len(y_true)),
        }
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    if len(y_true) == 0:
        return {"coverage": None, "coverage_undefined": True, "n": 0}
    return {
        "coverage": float(np.mean(y_true <= q_pred)),
        "coverage_undefined": False,
        "n": int(len(y_true)),
    }


def cumulative_quantile_not_sum(monthly: np.ndarray) -> dict[str, Any]:
    """Document that the quantile of a sum is not the sum of quantiles."""
    monthly = np.asarray(monthly, dtype=float)
    summed = float(np.sum(monthly))
    return {
        "sum_of_monthly_quantiles": summed,
        "note": "Do not use the sum of monthly quantiles as a lead-time+review quantile of cumulative demand.",
    }
