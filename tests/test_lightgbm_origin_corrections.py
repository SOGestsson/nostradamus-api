"""Stage tests for origin-safe LightGBM forecasts (schema v2)."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from inventory_algorithm.lgbm_drivers import MissingRequiredDriver, attach_target_month_drivers, driver_value_at_target
from inventory_algorithm.lgbm_explain import reconcile_shap
from inventory_algorithm.lgbm_grid import calendar_lag_value, complete_monthly_grid, months_since_last_sale, seasonal_naive_value
from inventory_algorithm.lgbm_origin import (
    FEATURE_SPEC_SCHEMA_VERSION,
    apply_expanding_te_to_rows,
    decide_item_eligibility,
    expanding_target_encoding_maps,
    label_is_observable,
    month_start,
    origin_split_masks,
    split_rolling_origins,
    wape_with_undefined,
)
from inventory_algorithm.lgbm_predict import (
    AUDIT_POLICY,
    MEAN_HEAD_HURDLE,
    MEAN_HEAD_TWEEDIE,
    PLANNING_POLICY,
    SEASONAL_SERVE_POLICY,
    apply_adjustment_policy,
    compose_mean,
    coverage_audit,
    cumulative_quantile_not_sum,
    enforce_quantile_order,
    predict_from_heads,
    select_mean_head,
)
from inventory_algorithm.lgbm_sf_join import join_sf_features, normalize_sf_cv_frame, normalize_sf_forecast_frame
from inventory_algorithm.lightgbm_forecasts import (
    _assemble_feature_row,
    _build_direct_rows_for_item_all_h,
    _detect_lifecycle_phase,
    _frozen_peer_lookup,
    _policy_from_spec,
    _trailing_same_month_stats,
    deserialize_peer_context,
    deserialize_segment_stats,
    load_peer_context_or_raise,
    serialize_peer_context,
    serialize_segment_stats,
)
from api.v1.lightgbm import _dispatch_fallback, _format_batch_results, _requested_item_ids, _send_webhook_with_retries, _start_job_thread


def test_optuna_all_nonfinite_fails(monkeypatch):
    from inventory_algorithm import lightgbm_forecasts as m
    from inventory_algorithm.lightgbm_forecasts import _tune_lgbm_hyperparameters

    X_train = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y_train = np.array([1.0, 2.0, 3.0])
    X_val = pd.DataFrame({"f": [1.5, 2.5]})
    y_val = np.array([1.5, 2.5])

    class _Trial:
        value = float("inf")

    class _Study:
        trials = [_Trial(), _Trial()]

        def optimize(self, *args, **kwargs):
            return None

        @property
        def best_params(self):
            raise ValueError("no completed trials")

    fake = MagicMock()
    fake.create_study.return_value = _Study()
    fake.exceptions.TrialPruned = type("TrialPruned", (Exception,), {})
    monkeypatch.setattr(m, "_HAS_OPTUNA", True)
    monkeypatch.setattr(m, "optuna", fake)
    monkeypatch.setattr(m, "lgb", MagicMock())
    with pytest.raises(ValueError, match="all trials nonfinite"):
        _tune_lgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials=2, timeout_seconds=1)


def _panel_with_gap() -> pd.DataFrame:
    rows = []
    for uid, start in [("a", "2022-01-01"), ("b", "2022-01-01")]:
        ds = pd.date_range(start, periods=15, freq="MS")
        for i, d in enumerate(ds):
            if uid == "a" and d == pd.Timestamp("2022-06-01"):
                continue
            rows.append({"unique_id": uid, "item_id": uid, "ds": d, "y": float(10 + i)})
    return pd.DataFrame(rows)


def test_origin_safe_labels_future_mutation_does_not_change_origin_t():
    ds = pd.date_range("2022-01-01", periods=24, freq="MS").to_numpy()
    y = np.arange(24, dtype=float) + 3.0
    y_mut = y.copy()
    y_mut[20:] = 999.0
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    t = 12
    r1 = _assemble_feature_row(
        unique_id="sku",
        t=t,
        h=1,
        ds=ds,
        y_model=y,
        y_orig=y,
        lags=[1, 12],
        roll_windows=[3],
        static={},
        exo={},
        same_month_stats=_trailing_same_month_stats(y, months, t),
        lifecycle_feats=_detect_lifecycle_phase(y[: t + 1]),
        calendar_cols=[],
        trend_cols=[],
        segment_cols=[],
        global_monthly=None,
        global_yoy=None,
        segment_stats=None,
    )
    r2 = _assemble_feature_row(
        unique_id="sku",
        t=t,
        h=1,
        ds=ds,
        y_model=y_mut,
        y_orig=y_mut,
        lags=[1, 12],
        roll_windows=[3],
        static={},
        exo={},
        same_month_stats=_trailing_same_month_stats(y_mut, months, t),
        lifecycle_feats=_detect_lifecycle_phase(y_mut[: t + 1]),
        calendar_cols=[],
        trend_cols=[],
        segment_cols=[],
        global_monthly=None,
        global_yoy=None,
        segment_stats=None,
    )
    for k in ("lag_1", "lag_12", "zero_ratio_12", "roll_mean_3"):
        assert r1[k] == pytest.approx(r2[k])
    assert label_is_observable(ds[t], 1, ds[-1])


def test_expanding_te_does_not_see_own_future_y():
    hist = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "flavour": ["vanilla", "vanilla", "vanilla"],
            "y": [10.0, 10.0, 1000.0],
        }
    )
    maps_feb = expanding_target_encoding_maps(hist, pd.Timestamp("2022-02-01"), ["flavour"], prior=1.0)
    maps_full = expanding_target_encoding_maps(hist, pd.Timestamp("2022-03-01"), ["flavour"], prior=1.0)
    assert maps_feb["flavour"]["mapping"]["vanilla"] != pytest.approx(maps_full["flavour"]["mapping"]["vanilla"])
    rows = apply_expanding_te_to_rows(
        [{"decision_ds": pd.Timestamp("2022-02-01"), "flavour": "vanilla"}],
        hist,
        ["flavour"],
        prior=1.0,
    )
    assert "flavour_te" in rows[0]
    assert rows[0]["flavour_te"] == pytest.approx(maps_feb["flavour"]["mapping"]["vanilla"])


def test_rolling_origin_splits_are_disjoint():
    splits = split_rolling_origins(pd.Timestamp("2024-12-01"), tune_months=2, calibration_months=2, audit_months=2)
    tune = set(splits.tune_origins)
    cal = set(splits.calibration_origins)
    audit = set(splits.audit_origins)
    assert tune.isdisjoint(cal)
    assert cal.isdisjoint(audit)
    assert tune.isdisjoint(audit)
    assert splits.history_end == month_start("2024-12-01")
    assert FEATURE_SPEC_SCHEMA_VERSION == 2


def test_statsforecast_join_cardinality_and_duplicate_rejection():
    rows = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "decision_ds": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "forecast_ds": [pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01")],
            "lag_1": [1.0, 2.0],
        }
    )
    cv = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "cutoff": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
            "ds": [pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01")],
            "CrostonOptimized": [3.0, 4.0],
        }
    )
    sf = normalize_sf_cv_frame(cv)
    joined = join_sf_features(rows, sf)
    assert len(joined) == len(rows)
    assert joined.loc[0, "sf_croston_optimized"] == pytest.approx(3.0)
    dup = pd.concat([sf, sf], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        join_sf_features(rows, dup)


def test_sf_forecast_frame_has_origin_key():
    fcst = pd.DataFrame(
        {
            "unique_id": ["a"],
            "ds": [pd.Timestamp("2023-02-01")],
            "CrostonOptimized": [1.5],
        }
    )
    out = normalize_sf_forecast_frame(fcst, decision_ds=pd.Timestamp("2023-01-01"))
    assert out.loc[0, "decision_ds"] == month_start("2023-01-01")
    assert int(out.loc[0, "horizon"]) == 1


def test_missing_month_calendar_lag_12():
    hist = _panel_with_gap()
    grid = complete_monthly_grid(hist)
    a = grid[grid["unique_id"] == "a"].sort_values("ds")
    assert pd.Timestamp("2022-06-01") in set(pd.to_datetime(a["ds"]))
    june = a[a["ds"] == pd.Timestamp("2022-06-01")].iloc[0]
    assert june["y"] == 0.0
    assert int(june["y_observed"]) == 1
    y = a["y"].to_numpy(dtype=float)
    ds = a["ds"].to_numpy()
    t = int(np.where(pd.to_datetime(ds) == pd.Timestamp("2023-01-01"))[0][0]) if (pd.to_datetime(ds) == pd.Timestamp("2023-01-01")).any() else len(y) - 1
    lag12 = calendar_lag_value(y, ds, t, 12)
    assert np.isfinite(lag12)


def test_zero_ratio_uses_y_orig_not_detrended():
    ds = pd.date_range("2022-01-01", periods=12, freq="MS").to_numpy()
    y_orig = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
    y_model = y_orig - 3.0
    t = 11
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    r = _assemble_feature_row(
        unique_id="sku",
        t=t,
        h=1,
        ds=ds,
        y_model=y_model,
        y_orig=y_orig,
        lags=[1],
        roll_windows=[3],
        static={},
        exo={},
        same_month_stats=_trailing_same_month_stats(y_orig, months, t),
        lifecycle_feats=_detect_lifecycle_phase(y_orig[: t + 1]),
        calendar_cols=[],
        trend_cols=[],
        segment_cols=[],
        global_monthly=None,
        global_yoy=None,
        segment_stats=None,
    )
    assert r["zero_ratio_12"] == pytest.approx(float(np.mean(y_orig == 0.0)))
    assert r["zero_ratio_12"] != pytest.approx(float(np.mean(y_model == 0.0)))
    assert r["months_since_last_sale"] == pytest.approx(months_since_last_sale(y_orig, t))


def test_batch_invariance_peer_lookup_frozen():
    from inventory_algorithm.lightgbm_forecasts import _frozen_peer_lookup

    frozen = {
        "global_monthly": {1: 10.0, 2: 12.0},
        "global_yoy": {},
        "segment_stats": {},
    }
    lookup = _frozen_peer_lookup(frozen)
    a = lookup(pd.Timestamp("2024-01-01"))
    b = lookup(pd.Timestamp("2024-06-01"))
    assert a["global_monthly"][1] == b["global_monthly"][1] == 10.0


def test_target_month_driver_and_future_mutation():
    drivers = pd.DataFrame(
        {
            "item_id": ["sku", "sku"],
            "ds": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
            "promo": [1.0, 9.0],
            "vintage": ["observed", "known_future"],
        }
    )
    val = driver_value_at_target(
        drivers, item_id="sku", forecast_ds=pd.Timestamp("2023-02-01"), column="promo", origin=pd.Timestamp("2023-01-01")
    )
    assert val == pytest.approx(9.0)
    observed_only = drivers.copy()
    observed_only["vintage"] = "observed"
    leaked = driver_value_at_target(
        observed_only,
        item_id="sku",
        forecast_ds=pd.Timestamp("2023-02-01"),
        column="promo",
        origin=pd.Timestamp("2023-01-01"),
    )
    assert leaked is None
    row = {"unique_id": "sku", "decision_ds": pd.Timestamp("2023-01-01"), "forecast_ds": pd.Timestamp("2023-02-01")}
    attach_target_month_drivers(row, drivers, columns=["promo"], required=["promo"])
    assert row["promo"] == pytest.approx(9.0)
    with pytest.raises(MissingRequiredDriver):
        attach_target_month_drivers(
            {"unique_id": "sku", "decision_ds": pd.Timestamp("2023-01-01"), "forecast_ds": pd.Timestamp("2023-03-01")},
            drivers,
            columns=["promo"],
            required=["promo"],
        )


def test_compose_mean_never_multiplies_unconditional_tweedie():
    tweedie, head = compose_mean(head=MEAN_HEAD_TWEEDIE, yhat_tweedie=10.0, yhat_positive=20.0, p_occ=0.2)
    assert head == MEAN_HEAD_TWEEDIE
    assert tweedie == pytest.approx(10.0)
    hurdle, head_h = compose_mean(head=MEAN_HEAD_HURDLE, yhat_tweedie=10.0, yhat_positive=20.0, p_occ=0.2)
    assert head_h == MEAN_HEAD_HURDLE
    assert hurdle == pytest.approx(4.0)


def test_adjustment_trace_on_policy():
    yhat, trace = apply_adjustment_policy(5.0, PLANNING_POLICY, month_mean=20.0, recent_level=8.0, croston=3.0)
    assert trace
    assert all({"name", "before", "after", "delta"} <= set(s) for s in trace)
    y2, t2 = apply_adjustment_policy(-1.0, AUDIT_POLICY)
    assert y2 == 0.0
    assert t2[0]["name"] == "non_negative"


def test_quantile_order_may_sit_below_mean():
    q70, q90, q95 = enforce_quantile_order(10.0, 8.0, 12.0, 9.0, allow_below_mean=True)
    assert q70 <= q90 <= q95
    assert q70 < 10.0
    capped = coverage_audit(np.array([1.0]), np.array([2.0]), after_caps=True)
    assert capped["coverage"] is None
    assert capped["coverage_undefined"] is True
    note = cumulative_quantile_not_sum(np.array([10.0, 20.0, 30.0]))
    assert note["sum_of_monthly_quantiles"] == pytest.approx(60.0)
    assert "not" in note["note"].lower()


def test_shap_reconciliation():
    recon = reconcile_shap(bias=1.5, contributions=np.array([0.2, -0.1, 0.4]), raw_score=2.0)
    assert recon["reconciled"] is True
    bad = reconcile_shap(bias=1.0, contributions=np.array([1.0]), raw_score=9.0)
    assert bad["reconciled"] is False


def test_wape_undefined_when_all_zero():
    out = wape_with_undefined(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
    assert out["wape"] is None
    assert out["wape_undefined"] is True


def test_predict_from_heads_validation_serving_parity():
    a = predict_from_heads(
        head=MEAN_HEAD_TWEEDIE,
        yhat_tweedie=5.0,
        yhat_positive=8.0,
        p_occ=0.4,
        policy=AUDIT_POLICY,
        q70=6.0,
        q90=7.0,
        q95=8.0,
    )
    b = predict_from_heads(
        head=MEAN_HEAD_TWEEDIE,
        yhat_tweedie=5.0,
        yhat_positive=8.0,
        p_occ=0.4,
        policy=AUDIT_POLICY,
        q70=6.0,
        q90=7.0,
        q95=8.0,
    )
    assert a.yhat == pytest.approx(b.yhat)
    assert a.adjustments == b.adjustments


def test_requested_item_ids_include_attributes_only():
    hist = pd.DataFrame({"item_id": ["a"], "day": ["2023-01-01"], "actual_sale": [1.0]})
    attrs = pd.DataFrame({"item_id": ["a", "b"]})
    ids = _requested_item_ids(hist, attrs)
    assert ids == ["a", "b"]


def test_fallback_dispatch_naive_not_seasonal_blend():
    hist = pd.DataFrame(
        {
            "item_id": ["x", "x"],
            "day": ["2023-01-01", "2023-02-01"],
            "actual_sale": [10.0, 12.0],
        }
    )
    out = _dispatch_fallback(
        df_hist=hist, item_id="x", forecast_periods=2, reason="ML_WORSE_THAN_TS", confidence=0.1, method="naive"
    )
    assert out["model_used"] == "naive"
    assert out["forecast"][0] == pytest.approx(12.0)
    assert out["adjustments"][0]["bound_type"] == "heuristic_not_calibrated"


def test_format_batch_every_requested_item():
    fcst = pd.DataFrame(
        {
            "item_id": ["a", "a"],
            "day": ["2023-03-01", "2023-04-01"],
            "yhat": [1.0, 2.0],
            "upper_70": [1.1, 2.1],
            "upper_90": [1.2, 2.2],
            "upper_95": [1.3, 2.3],
            "adjustments": [[{"name": "non_negative", "before": 1.0, "after": 1.0, "delta": 0.0}], []],
            "failed": [False, False],
        }
    )
    hist = pd.DataFrame({"item_id": ["a"], "day": ["2023-02-01"], "actual_sale": [5.0]})
    payload = _format_batch_results(
        unique_ids=["a", "missing"],
        fcst_df=fcst,
        eligibility={
            "a": {"ml_allowed": True, "winner_model": "lgbm_direct", "reason_code": "OK", "confidence": 0.5},
            "missing": {"ml_allowed": False, "winner_model": "naive", "fallback_model": "naive", "reason_code": "NO_HISTORY", "confidence": 0.0},
        },
        df_hist=hist,
        forecast_periods=2,
        meta={"model_version": "v1", "freq": "MS"},
    )
    assert payload["total_items"] == 2
    by_id = {r["item_id"]: r for r in payload["forecasts"]}
    assert by_id["a"]["model_used"] == "lgbm_direct"
    assert by_id["missing"]["failed"] is True or by_id["missing"]["skipped"] is True
    assert payload["meta"]["confidence_definition"]


def test_webhook_does_not_sleep_after_last_retry():
    with patch("api.v1.lightgbm.httpx.Client") as client_cls, patch("api.v1.lightgbm.time.sleep") as sleep:
        client = MagicMock()
        client.__enter__.return_value.post.return_value.status_code = 500
        client.__enter__.return_value.post.return_value.text = "nope"
        client_cls.return_value = client
        ok = _send_webhook_with_retries("job", "http://example.invalid", {"job_id": "job"}, max_attempts=2)
        assert ok is False
        assert sleep.call_count == 1


def test_direct_rows_skip_unobservable_and_keep_origin():
    ds = pd.date_range("2022-01-01", periods=8, freq="MS").to_numpy()
    y = np.ones(8, dtype=float)
    rows = _build_direct_rows_for_item_all_h(
        unique_id="sku",
        ds=ds,
        y=y,
        y_orig=y,
        trend_params=None,
        exo={},
        static={},
        horizon=3,
        lags=[1],
        roll_windows=[3],
        history_end=pd.Timestamp(ds[-1]),
    )
    for h, lst in rows.items():
        for r in lst:
            assert label_is_observable(r["decision_ds"], h, ds[-1])
            assert r["horizon"] == h


def test_origin_masks_small_n_never_overlap():
    splits = split_rolling_origins(pd.Timestamp("2024-12-01"), tune_months=1, calibration_months=1, audit_months=1)
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=12, freq="MS")):
        rows.append(
            {
                "decision_ds": d,
                "forecast_ds": d + pd.offsets.MonthBegin(1),
                "y": float(i),
            }
        )
    frame = pd.DataFrame(rows)
    is_train, is_tune, is_cal, is_audit = origin_split_masks(frame, splits)
    assert int(is_train.sum()) < 50
    assert not np.any(is_train & is_tune)
    assert not np.any(is_train & is_cal)
    assert not np.any(is_train & is_audit)
    assert not np.any(is_tune & is_cal)
    assert not np.any(is_cal & is_audit)


def test_label_not_observable_at_fit_cutoff():
    origin = pd.Timestamp("2024-01-01")
    assert label_is_observable(origin, 12, pd.Timestamp("2025-01-01"))
    assert not label_is_observable(origin, 12, pd.Timestamp("2024-10-01"))
    splits = split_rolling_origins(pd.Timestamp("2025-01-01"), tune_months=1, calibration_months=1, audit_months=1)
    frame = pd.DataFrame(
        [
            {
                "decision_ds": origin,
                "forecast_ds": pd.Timestamp("2025-01-01"),
                "y": 1.0,
            }
        ]
    )
    is_train, *_ = origin_split_masks(frame, OriginSplits_with_cutoff(splits, pd.Timestamp("2024-10-01")))
    assert not bool(is_train[0])


def OriginSplits_with_cutoff(splits, cutoff):
    from inventory_algorithm.lgbm_origin import OriginSplits

    return OriginSplits(
        history_end=splits.history_end,
        fit_cutoff=month_start(cutoff),
        tune_origins=splits.tune_origins,
        calibration_origins=splits.calibration_origins,
        audit_origins=splits.audit_origins,
    )


def test_seasonal_naive_calendar_aligned_1_to_24():
    y = np.arange(1, 25, dtype=float)
    t = 23
    assert seasonal_naive_value(y, t, 1) == pytest.approx(13.0)
    assert seasonal_naive_value(y, t, 6) == pytest.approx(18.0)
    assert seasonal_naive_value(y, t, 12) == pytest.approx(24.0)
    ds = pd.date_range("2022-01-01", periods=24, freq="MS").to_numpy()
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    for h, expected in ((1, 13.0), (6, 18.0), (12, 24.0)):
        r = _assemble_feature_row(
            unique_id="sku",
            t=t,
            h=h,
            ds=ds,
            y_model=y,
            y_orig=y,
            lags=[1, 12],
            roll_windows=[3],
            static={},
            exo={},
            same_month_stats=_trailing_same_month_stats(y, months, t),
            lifecycle_feats=_detect_lifecycle_phase(y[: t + 1]),
            calendar_cols=[],
            trend_cols=[],
            segment_cols=[],
            global_monthly=None,
            global_yoy=None,
            segment_stats=None,
        )
        assert r["yhat_snaive"] == pytest.approx(expected)
        assert r["lag_12"] == pytest.approx(13.0)


def test_peer_json_roundtrip_preserves_january_segment():
    stats = {
        "flavour": {
            "vanilla": {1: {"mean": 150.0, "std": 50.0, "count": 1.0, "nonzero_rate": 0.8}},
            5: {1: {"mean": 9.0, "std": 0.0, "count": 1.0, "nonzero_rate": 1.0}},
            "5": {1: {"mean": 3.0, "std": 0.0, "count": 1.0, "nonzero_rate": 1.0}},
        }
    }
    blob = serialize_peer_context(
        customer_id="c",
        context_version="abc",
        origin=pd.Timestamp("2024-01-01"),
        global_monthly={1: 10.0},
        global_yoy={},
        segment_stats=stats,
    )
    restored = deserialize_peer_context(json.loads(json.dumps(blob)))
    jan = restored["segment_stats"]["flavour"]["vanilla"][1]
    assert jan["mean"] == pytest.approx(150.0)
    assert jan["std"] == pytest.approx(50.0)
    assert jan["count"] == pytest.approx(1.0)
    assert jan["nonzero_rate"] == pytest.approx(0.8)
    assert restored["segment_stats"]["flavour"][5][1]["mean"] == pytest.approx(9.0)
    assert restored["segment_stats"]["flavour"]["5"][1]["mean"] == pytest.approx(3.0)
    from inventory_algorithm.lightgbm_forecasts import _segment_features_for_row

    feats = _segment_features_for_row(
        1, {"flavour": "vanilla"}, ["flavour"], restored["segment_stats"], item_same_month_mean=120.0
    )
    assert feats["flavour_seg_month_mean"] == pytest.approx(150.0)


def test_peer_batch_invariance_after_json_roundtrip():
    blob = serialize_peer_context(
        customer_id="c",
        context_version="v",
        origin=pd.Timestamp("2024-06-01"),
        global_monthly={1: 10.0, 6: 12.0},
        global_yoy={},
        segment_stats={"flavour": {"vanilla": {1: {"mean": 150.0, "std": 50.0, "count": 1.0, "nonzero_rate": 0.8}}}},
    )
    loaded = json.loads(json.dumps(blob))
    lookup = _frozen_peer_lookup(loaded)
    a = lookup(pd.Timestamp("2024-01-01"))
    b = lookup(pd.Timestamp("2024-06-01"))
    assert a["global_monthly"][1] == b["global_monthly"][1] == 10.0
    assert a["segment_stats"]["flavour"]["vanilla"][1]["mean"] == pytest.approx(150.0)


def test_v2_missing_peer_context_is_hard_error(tmp_path):
    spec = {"schema_version": 2, "peer_context_file": "peer_context.json"}
    with pytest.raises(ValueError, match="Missing required peer_context"):
        load_peer_context_or_raise(str(tmp_path), spec)


def test_assemble_exo_uses_origin_not_future():
    ds = pd.date_range("2022-01-01", periods=6, freq="MS").to_numpy()
    y = np.ones(6)
    exo = {"promo": np.array([1.0, 2.0, 3.0, 4.0, 99.0, 99.0])}
    months = np.array([pd.Timestamp(d).month for d in ds], dtype=int)
    t = 2
    r = _assemble_feature_row(
        unique_id="sku",
        t=t,
        h=2,
        ds=ds,
        y_model=y,
        y_orig=y,
        lags=[1],
        roll_windows=[3],
        static={},
        exo=exo,
        same_month_stats=_trailing_same_month_stats(y, months, t),
        lifecycle_feats=_detect_lifecycle_phase(y[: t + 1]),
        calendar_cols=[],
        trend_cols=[],
        segment_cols=[],
        global_monthly=None,
        global_yoy=None,
        segment_stats=None,
    )
    assert r["promo"] == pytest.approx(3.0)


def test_shap_corrupt_contrib_fails_independent_raw_score():
    from inventory_algorithm.lightgbm_forecasts import _HAS_LGBM, lgb

    if not _HAS_LGBM or lgb is None:
        pytest.skip("lightgbm unavailable")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f": rng.normal(size=40), "g": rng.normal(size=40)})
    y = (0.5 * X["f"] + 0.2 * X["g"]).to_numpy()
    dtrain = lgb.Dataset(X, label=y)
    booster = lgb.train({"objective": "regression", "verbosity": -1}, dtrain, num_boost_round=20)
    X1 = X.iloc[[0]]
    contrib = np.asarray(booster.predict(X1, pred_contrib=True)[0], dtype=float)
    raw = float(np.asarray(booster.predict(X1, raw_score=True)).ravel()[0])
    ok = reconcile_shap(bias=float(contrib[-1]), contributions=contrib[:-1], raw_score=raw)
    assert ok["reconciled"] is True
    corrupted = contrib[:-1].copy()
    corrupted[0] += 10.0
    bad = reconcile_shap(bias=float(contrib[-1]), contributions=corrupted, raw_score=raw)
    assert bad["reconciled"] is False
    tautology = reconcile_shap(
        bias=float(contrib[-1]),
        contributions=corrupted,
        raw_score=float(contrib[-1] + np.sum(corrupted)),
    )
    assert tautology["reconciled"] is True


def test_quantile_v1_excess_adapter():
    from inventory_algorithm.lgbm_predict import quantile_to_absolute

    assert quantile_to_absolute(2.0, 10.0, direct_demand=True) == pytest.approx(2.0)
    assert quantile_to_absolute(2.0, 10.0, direct_demand=False) == pytest.approx(12.0)
    assert quantile_to_absolute(None, 10.0, direct_demand=True) is None


def test_fallback_monthly_sum_not_last_raw_row():
    hist = pd.DataFrame(
        {
            "item_id": ["x", "x"],
            "day": ["2023-01-15", "2023-01-20"],
            "actual_sale": [40.0, 60.0],
        }
    )
    out = _dispatch_fallback(
        df_hist=hist, item_id="x", forecast_periods=1, reason="ML_WORSE_THAN_TS", confidence=0.1, method="naive"
    )
    assert out["forecast"][0] == pytest.approx(100.0)


def test_failed_reason_from_failed_row_not_iloc0():
    fcst = pd.DataFrame(
        {
            "item_id": ["a", "a"],
            "day": ["2023-03-01", "2023-04-01"],
            "yhat": [1.0, 0.0],
            "upper_70": [1.1, 0.0],
            "upper_90": [1.2, 0.0],
            "upper_95": [1.3, 0.0],
            "failed": [False, True],
            "reason_code": ["OK", "MISSING_REQUIRED_DRIVER"],
        }
    )
    hist = pd.DataFrame({"item_id": ["a"], "day": ["2023-02-01"], "actual_sale": [5.0]})
    payload = _format_batch_results(
        unique_ids=["a"],
        fcst_df=fcst,
        eligibility={"a": {"ml_allowed": True, "winner_model": "lgbm_direct", "reason_code": "OK", "confidence": 0.5}},
        df_hist=hist,
        forecast_periods=2,
        meta={"model_version": "v1", "freq": "MS"},
    )
    assert payload["forecasts"][0]["failed"] is True
    assert payload["forecasts"][0]["reason_code"] == "MISSING_REQUIRED_DRIVER"


def test_unknown_name_cluster_not_md5():
    from inventory_algorithm.lightgbm_forecasts import _apply_name_cluster

    base = pd.DataFrame({"name": ["alpha", "unseen-sku"]})
    out = _apply_name_cluster(base, {"mapping": {"alpha": 3}, "__default__": -1, "k": 8, "unseen": "unknown_cluster"})
    assert int(out.loc[out["name"] == "alpha", "name_cluster_id"].iloc[0]) == 3
    assert int(out.loc[out["name"] == "unseen-sku", "name_cluster_id"].iloc[0]) == -1


def test_optuna_sets_categorical_on_dataset_not_train(monkeypatch):
    from inventory_algorithm import lightgbm_forecasts as m
    from inventory_algorithm.lightgbm_forecasts import _tune_lgbm_hyperparameters

    captured: dict = {}

    class _DS:
        def __init__(self, data, label=None, weight=None, categorical_feature=None, reference=None, free_raw_data=True):
            captured.setdefault("dataset_cats", []).append(categorical_feature)

        def construct(self):
            return None

    class _Model:
        def predict(self, X):
            return np.ones(len(X), dtype=float)

    def fake_train(params, dtrain, **kwargs):
        captured["train_kwargs"] = kwargs
        return _Model()

    class _Trial:
        def suggest_float(self, *args, **kwargs):
            return 0.05

        def suggest_int(self, *args, **kwargs):
            return 15

        value = 0.1

    class _Study:
        trials = [_Trial()]
        best_params = {"learning_rate": 0.05, "num_leaves": 15, "feature_fraction": 0.8, "bagging_fraction": 0.8, "min_data_in_leaf": 5, "lambda_l1": 0.0, "lambda_l2": 0.0}

        def optimize(self, fn, **kwargs):
            fn(_Trial())

    fake_optuna = MagicMock()
    fake_optuna.create_study.return_value = _Study()
    fake_optuna.exceptions.TrialPruned = type("TrialPruned", (Exception,), {})
    fake_optuna.logging = MagicMock()
    fake_optuna.samplers.TPESampler.return_value = object()
    fake_lgb = MagicMock()
    fake_lgb.Dataset = _DS
    fake_lgb.train = fake_train
    fake_lgb.early_stopping.return_value = None
    monkeypatch.setattr(m, "_HAS_OPTUNA", True)
    monkeypatch.setattr(m, "optuna", fake_optuna)
    monkeypatch.setattr(m, "lgb", fake_lgb)
    monkeypatch.setattr(m, "LightGBMPruningCallback", None)
    X_train = pd.DataFrame({"f": [1.0, 2.0, 3.0], "cat": [0, 1, 0]})
    y_train = np.array([1.0, 2.0, 3.0])
    X_val = pd.DataFrame({"f": [1.5, 2.5], "cat": [1, 0]})
    y_val = np.array([1.5, 2.5])
    _tune_lgbm_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=1, timeout_seconds=1, categorical_feature=["cat"]
    )
    assert captured["dataset_cats"]
    assert captured["dataset_cats"][0] == ["cat"]
    assert "categorical_feature" not in (captured.get("train_kwargs") or {})


def test_optuna_smoke_real_lightgbm():
    from inventory_algorithm.lightgbm_forecasts import _HAS_LGBM, _HAS_OPTUNA, _tune_lgbm_hyperparameters

    if not _HAS_LGBM or not _HAS_OPTUNA:
        pytest.skip("lightgbm or optuna unavailable")
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame({"f": rng.normal(size=60), "cat": rng.integers(0, 3, size=60)})
    y_train = np.maximum(0.1, 2.0 + X_train["f"].to_numpy())
    X_val = pd.DataFrame({"f": rng.normal(size=20), "cat": rng.integers(0, 3, size=20)})
    y_val = np.maximum(0.1, 2.0 + X_val["f"].to_numpy())
    params = _tune_lgbm_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=1, timeout_seconds=15, categorical_feature=["cat"]
    )
    assert "learning_rate" in params
    assert "num_leaves" in params


def test_start_job_thread_returns_before_work_finishes():
    started = threading.Event()
    released = threading.Event()

    def _fn():
        started.set()
        released.wait(timeout=2.0)

    t0 = time.perf_counter()
    t = _start_job_thread("lgbm-test-job", _fn)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.2
    assert started.wait(timeout=1.0)
    released.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_eligibility_gates_against_last_value_not_snaive():
    elig = decide_item_eligibility(
        series_len=36,
        min_history_points=24,
        n_eval=6,
        wape_ml=0.40,
        wape_naive=0.80,
        wape_snaive=0.20,
        min_improvement=0.02,
    )
    assert elig["ml_allowed"] is True
    assert elig["reason_code"] == "OK"
    assert elig["fallback_model"] == "seasonal_naive"


def test_eligibility_undefined_wape_allows_ml_not_naive_zero():
    elig = decide_item_eligibility(
        series_len=36,
        min_history_points=24,
        n_eval=6,
        wape_ml=None,
        wape_naive=None,
        wape_snaive=None,
        min_improvement=0.02,
    )
    assert elig["ml_allowed"] is True
    assert elig["winner_model"] == "lgbm_direct"
    assert elig["reason_code"] == "OK_SPARSE_HOLDOUT"
    assert elig["fallback_model"] != "naive"


def test_eligibility_no_eval_allows_ml_when_history_ok():
    elig = decide_item_eligibility(
        series_len=36,
        min_history_points=24,
        n_eval=0,
        wape_ml=None,
        wape_naive=None,
        min_improvement=0.02,
    )
    assert elig["ml_allowed"] is True
    assert elig["reason_code"] == "OK_SPARSE_HOLDOUT"


def test_eligibility_short_history_still_naive():
    elig = decide_item_eligibility(
        series_len=10,
        min_history_points=24,
        n_eval=6,
        wape_ml=0.1,
        wape_naive=0.9,
        min_improvement=0.0,
    )
    assert elig["ml_allowed"] is False
    assert elig["reason_code"] == "SHORT_HISTORY"


def test_eligibility_ml_worse_than_last_value_falls_back_to_snaive():
    elig = decide_item_eligibility(
        series_len=36,
        min_history_points=24,
        n_eval=6,
        wape_ml=0.90,
        wape_naive=0.40,
        wape_snaive=0.20,
        min_improvement=0.02,
    )
    assert elig["ml_allowed"] is False
    assert elig["fallback_model"] == "seasonal_naive"
    assert elig["reason_code"] == "ML_WORSE_THAN_TS"


def test_hurdle_not_selected_on_all_zero_cal():
    y = np.zeros(6)
    tweedie = np.array([0.1, 0.2, 0.0, 0.1, 0.0, 0.0])
    hurdle = np.zeros(6)
    assert select_mean_head(y, tweedie, hurdle) == MEAN_HEAD_TWEEDIE


def test_hurdle_not_selected_when_worse_on_positive_months():
    y = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    tweedie = np.array([40.0, 40.0, 80.0, 40.0, 40.0, 40.0])
    hurdle = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0])
    assert select_mean_head(y, tweedie, hurdle) == MEAN_HEAD_TWEEDIE


def test_hurdle_selected_when_better_on_positive_months():
    y = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    tweedie = np.array([40.0, 40.0, 80.0, 40.0, 40.0, 40.0])
    hurdle = np.array([5.0, 5.0, 95.0, 5.0, 5.0, 5.0])
    assert select_mean_head(y, tweedie, hurdle) == MEAN_HEAD_HURDLE


def test_seasonal_serve_policy_lifts_december_zero():
    yhat, trace = apply_adjustment_policy(0.0, SEASONAL_SERVE_POLICY, month_mean=1000.0)
    assert yhat == pytest.approx(500.0)
    names = [s["name"] for s in trace]
    assert "nonzero_level_anchor" in names
    y_off, _ = apply_adjustment_policy(0.0, SEASONAL_SERVE_POLICY, month_mean=0.0)
    assert y_off == pytest.approx(0.0)


def test_v2_audit_spec_remaps_to_seasonal_serve():
    policy = _policy_from_spec({"schema_version": 2, "adjustment_policy": AUDIT_POLICY.as_dict()})
    assert policy.as_dict() == SEASONAL_SERVE_POLICY.as_dict()
    explicit = _policy_from_spec({"schema_version": 2, "adjustment_policy": PLANNING_POLICY.as_dict()})
    assert explicit.as_dict() == PLANNING_POLICY.as_dict()

