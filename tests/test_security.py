"""Local classical and LightGBM forecasts stay public; Nixtla/simulation fail closed."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import security


def _req(path: str):
    return SimpleNamespace(url=SimpleNamespace(path=path), state=SimpleNamespace())


@pytest.fixture(autouse=True)
def _reset_missing_key_log():
    security._MISSING_KEY_LOGGED = False
    yield
    security._MISSING_KEY_LOGGED = False


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/forecast/generate",
        "/api/v1/forecast/generate/",
        "/api/v1/forecast/generate_daily",
        "/api/v1/forecast/generate_async",
        "/api/v1/forecast/generate_job",
        "/api/v1/forecast/jobs/abc-123",
        "/api/v1/forecast/jobs/abc-123/",
        "/api/v1/lightgbm/train",
        "/api/v1/lightgbm/train_async",
        "/api/v1/lightgbm/batch",
        "/api/v1/lightgbm/batch_async",
        "/api/v1/lightgbm/diagnostics",
        "/api/v1/lightgbm/jobs/abc-123",
        "/api/v1/lightgbm/jobs/abc-123/",
    ],
)
def test_local_forecast_paths_are_public_without_api_key(monkeypatch, path):
    monkeypatch.delenv("API_KEY", raising=False)
    assert security.require_api_key(_req(path), x_api_key=None) is True


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/simulation/run",
        "/api/v1/forecast/webhook/dlq",
        "/api/v1/lightgpt/generate",
        "/api/v1/lightgpt/batch",
    ],
)
def test_nixtla_and_simulation_fail_closed_without_api_key(monkeypatch, path):
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc:
        security.require_api_key(_req(path), x_api_key=None)
    assert exc.value.status_code == 503
    assert exc.value.detail == "API authentication is not configured"


def test_non_local_path_accepts_matching_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert security.require_api_key(_req("/api/v1/simulation/run"), x_api_key="secret") is True


def test_non_local_path_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        security.require_api_key(_req("/api/v1/simulation/run"), x_api_key="nope")
    assert exc.value.status_code == 401


def test_local_forecast_requires_key_when_partner_keys_configured(monkeypatch):
    monkeypatch.setenv("API_KEY", "internal-secret")
    monkeypatch.setenv("API_KEYS", "partner_a:partner-a-key")
    with pytest.raises(HTTPException) as exc:
        security.require_api_key(_req("/api/v1/forecast/generate"), x_api_key=None)
    assert exc.value.status_code == 401


def test_local_forecast_accepts_internal_key_with_partners(monkeypatch):
    monkeypatch.setenv("API_KEY", "internal-secret")
    monkeypatch.setenv("API_KEYS", "partner_a:partner-a-key")
    req = _req("/api/v1/forecast/generate")
    assert security.require_api_key(req, x_api_key="internal-secret") is True
    assert req.state.api_client_id == "internal"


def test_local_forecast_accepts_partner_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "internal-secret")
    monkeypatch.setenv("API_KEYS", "partner_a:partner-a-key")
    req = _req("/api/v1/forecast/generate")
    assert security.require_api_key(req, x_api_key="partner-a-key") is True
    assert req.state.api_client_id == "partner_a"
