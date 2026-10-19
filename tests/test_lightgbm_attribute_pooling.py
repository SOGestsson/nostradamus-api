"""Pooling by product attributes: no SKU id in the booster, TE on training rows."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from inventory_algorithm.lightgbm_forecasts import (
    IDENTITY_CAT_COLS,
    LightGBMForecast,
    _HAS_LGBM,
    _apply_target_encodings_to_rows,
    _categorical_feature_columns,
    _compute_target_encodings,
    _model_feature_columns,
    _prepare_lgbm_X,
    classify_attribute,
    route_static_columns,
)


def test_unique_id_and_name_are_not_model_features():
    cols = ["unique_id", "item_id", "name", "flavour", "flavour_te", "lag_1", "y"]
    out = _model_feature_columns(cols)
    assert "unique_id" not in out
    assert "item_id" not in out
    assert "name" not in out
    assert "y" not in out
    assert "flavour" in out
    assert "flavour_te" in out
    assert "lag_1" in out


def test_sku_identity_is_not_a_lightgbm_categorical():
    cats = _categorical_feature_columns(
        ["unique_id", "name", "flavour", "item_group", "flavour_te"]
    )
    assert "unique_id" not in cats
    assert "name" not in cats
    assert "flavour_te" not in cats
    assert "flavour" in cats
    assert "item_group" in cats
    assert IDENTITY_CAT_COLS == {"unique_id", "item_id", "name"}


def test_flavour_and_groups_route_as_segments():
    assert classify_attribute("flavour") == "segment"
    assert classify_attribute("item_group") == "segment"
    assert classify_attribute("item_group2") == "segment"
    assert classify_attribute("brand") == "segment"
    assert classify_attribute("colour") == "segment"

    base = pd.DataFrame(
        {
            "flavour": ["vanilla"],
            "item_group": ["dairy"],
            "name": ["Ice cream vanilla 1L"],
            "flavour_te": [8.0],
        }
    )
    routed = route_static_columns(
        ["flavour", "item_group", "name", "flavour_te"],
        base,
    )
    assert "flavour" in routed["segment"]
    assert "item_group" in routed["segment"]
    assert "name" not in routed["segment"]
    assert "flavour_te" not in routed["segment"]


def test_target_encoding_is_attached_to_training_rows():
    hist = pd.DataFrame(
        {
            "flavour": ["vanilla", "vanilla", "chocolate", "chocolate"],
            "y": [10.0, 12.0, 4.0, 6.0],
        }
    )
    te_maps, _ = _compute_target_encodings(hist, ["flavour"], prior=10.0)
    rows = [
        {"unique_id": "a", "flavour": "vanilla", "lag_1": 1.0, "y": 10.0},
        {"unique_id": "b", "flavour": "chocolate", "lag_1": 1.0, "y": 4.0},
        {"unique_id": "c", "flavour": "unknown", "lag_1": 1.0, "y": 0.0},
    ]
    out = _apply_target_encodings_to_rows(rows, ["flavour"], te_maps)
    assert all("flavour_te" in r for r in out)
    vanilla = next(r for r in out if r["flavour"] == "vanilla")
    choc = next(r for r in out if r["flavour"] == "chocolate")
    unknown = next(r for r in out if r["flavour"] == "unknown")
    assert vanilla["flavour_te"] > choc["flavour_te"]
    global_mean = float(te_maps["flavour"]["__global__"])
    assert abs(unknown["flavour_te"] - global_mean) < 1e-9


def test_prepare_lgbm_x_drops_identity_and_keeps_interactions():
    df = pd.DataFrame(
        {
            "unique_id": ["a", "b"],
            "flavour": ["vanilla", "chocolate"],
            "month_sin_1": [0.5, 0.1],
            "month_cos_1": [0.5, 0.9],
            "month_sin_2": [0.2, 0.3],
            "month_cos_2": [0.4, 0.5],
            "month_sin_3": [0.1, 0.2],
            "month_cos_3": [0.3, 0.4],
            "lag_1": [1.0, 2.0],
        }
    )
    feature_cols = [
        "flavour",
        "lag_1",
        "month_sin_1",
        "month_cos_1",
        "month_sin_2",
        "month_cos_2",
        "month_sin_3",
        "month_cos_3",
    ]
    X = _prepare_lgbm_X(
        df,
        feature_cols,
        cat_cols=["flavour"],
        mappings={"flavour": {"vanilla": 0, "chocolate": 1}},
        static_interaction_cols=["flavour"],
    )
    assert "unique_id" not in X.columns
    assert "flavour_month_sin_1" in X.columns
    assert list(X.columns[: len(feature_cols)]) == feature_cols


def test_item_profile_shap_patterns_catch_pooling_features():
    patterns = ["name_tok_", "name_cluster", "_te", "item_vs_global", "_seg_", "_vs_segment"]

    def _item_profile(feat: str) -> bool:
        fl = feat.lower()
        return any(p.lower() in fl for p in patterns)

    assert _item_profile("flavour_te")
    assert _item_profile("item_group_seg_month_mean")
    assert _item_profile("brand_vs_segment")
    assert not _item_profile("lag_1")


def test_trained_spec_excludes_sku_id_and_includes_flavour():
    if not _HAS_LGBM:
        pytest.skip("lightgbm import failed in this environment (often missing libomp on macOS)")

    rng = np.random.default_rng(7)
    n_items, n_months = 16, 24
    ds = pd.date_range("2022-01-01", periods=n_months, freq="MS")
    flavours = ["vanilla", "chocolate"]
    groups = ["dairy", "bakery"]

    hist_rows = []
    attr_rows = []
    for item_id in range(1, n_items + 1):
        flavour = flavours[item_id % 2]
        group = groups[0 if item_id <= n_items // 2 else 1]
        level = 20.0 if flavour == "vanilla" else 6.0
        attr_rows.append(
            {
                "item_id": item_id,
                "name": f"{flavour} {group} {item_id}",
                "flavour": flavour,
                "item_group": group,
            }
        )
        for i, d in enumerate(ds):
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (i / 12.0))
            y = max(0.0, level * seasonal + float(rng.normal(0, 1.0)))
            hist_rows.append(
                {
                    "item_id": item_id,
                    "day": d.strftime("%Y-%m-%d"),
                    "actual_sale": y,
                }
            )

    for d in ds[-3:]:
        hist_rows.append(
            {
                "item_id": 99,
                "day": d.strftime("%Y-%m-%d"),
                "actual_sale": 18.0,
            }
        )
    attr_rows.append(
        {
            "item_id": 99,
            "name": "vanilla dairy 99",
            "flavour": "vanilla",
            "item_group": "dairy",
        }
    )

    hist = pd.DataFrame(hist_rows)
    attrs = pd.DataFrame(attr_rows)

    with tempfile.TemporaryDirectory() as td:
        f = LightGBMForecast(store_root=td, customer_id="pool_test")
        train = f.train_and_register(
            hist,
            item_attributes=attrs,
            freq="M",
            horizon=2,
            status="prod",
            min_history_points=12,
            min_improvement=0.0,
            val_months=2,
            lgbm_min_data_in_leaf=5,
        )
        spec_path = os.path.join(
            f.store.customer_dir(), "artifacts", train.model_version, "feature_spec.json"
        )
        with open(spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
        assert spec.get("schema_version") == 2
        assert spec.get("artifact_complete") is True
        assert spec.get("confidence_definition")
        peer_path = os.path.join(
            f.store.customer_dir(), "artifacts", train.model_version, "peer_context.json"
        )
        assert os.path.exists(peer_path)

        features = spec["feature_columns"]
        cats = spec["categorical_columns"]
        assert "unique_id" not in features
        assert "item_id" not in features
        assert "name" not in features
        assert "unique_id" not in cats
        assert "flavour" in features
        assert "flavour" in cats
        assert "item_group" in features
        assert any(c.endswith("_seg_month_mean") and "flavour" in c for c in features)
        assert any(c.endswith("_te") for c in features) or "flavour" in features

        short_id = 99
        one_month_id = 100
        one_month_rows = [
            {
                "item_id": one_month_id,
                "day": ds[-1].strftime("%Y-%m-%d"),
                "actual_sale": 18.0,
            }
        ]
        hist_with_short = pd.concat(
            [hist, pd.DataFrame(one_month_rows)],
            ignore_index=True,
        )
        attrs_with_short = pd.concat(
            [
                attrs,
                pd.DataFrame(
                    [
                        {
                            "item_id": one_month_id,
                            "name": "vanilla dairy 100",
                            "flavour": "vanilla",
                            "item_group": "dairy",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        fcst_df, meta = f.batch_forecast(
            hist_with_short,
            item_attributes=attrs_with_short,
            forecast_periods=2,
            freq="M",
            status="prod",
        )
        ids = set(fcst_df["item_id"].astype(str))
        assert str(short_id) in ids
        assert str(one_month_id) in ids
        assert meta["batch_coverage"]["n_skipped_empty_history"] == 0
        short_fcst = fcst_df[fcst_df["item_id"].astype(str) == str(short_id)]
        assert len(short_fcst) == 2
        assert (short_fcst["yhat"] >= 0).all()
