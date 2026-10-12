"""Tests for the tiny FastAPI backend."""

from fastapi.testclient import TestClient

from aegis.api import app, reset_store


def _client() -> TestClient:
    reset_store()
    return TestClient(app)


def test_health_ok():
    assert _client().get("/health").json() == {"status": "ok", "events": 0}


def test_create_and_get_event():
    client = _client()
    created = client.post(
        "/events",
        json={"camera_id": "cam-north", "kind": "red-light-run", "confidence": 0.9},
    ).json()
    assert created["id"] == 1
    assert client.get("/events").json()[0]["id"] == 1
    assert client.get("/events/1").json()["camera_id"] == "cam-north"


def test_missing_event_404():
    assert _client().get("/events/999").status_code == 404


def test_bad_payload_422():
    assert _client().post("/events", json={"camera_id": ""}).status_code == 422
