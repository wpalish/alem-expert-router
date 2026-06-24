"""Интеграционные тесты HTTP-слоя (FastAPI TestClient)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, store


@pytest.fixture(autouse=True)
def clean_store():
    store.reset()
    yield
    store.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_dashboard_served_as_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Alem Expert Router" in r.text


def test_create_and_list_expert(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 5}, "capacity": 2}
    r = client.post("/experts", json=body)
    assert r.status_code == 200
    assert r.json()["current_load"] == 0  # системное поле выставлено сервером
    assert len(client.get("/experts").json()) == 1


def test_create_expert_cannot_set_current_load(client: TestClient) -> None:
    """current_load — служебное поле; клиент не может его задать через схему."""
    body = {"id": "e1", "name": "A", "skills": {"python": 5},
            "capacity": 2, "current_load": 99}
    r = client.post("/experts", json=body)
    assert r.status_code == 200
    assert r.json()["current_load"] == 0  # игнорируется, не 99


def test_invalid_skill_level_rejected(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 9}, "capacity": 2}
    assert client.post("/experts", json=body).status_code == 422


def test_seed_assign_dashboard_flow(client: TestClient) -> None:
    assert client.post("/seed").json() == {"experts": 5, "requests": 6}
    assert len(client.get("/experts").json()) == 5

    assign = client.get("/assign").json()
    assert assign["matched"] >= 4

    dash = client.get("/dashboard").json()
    assert dash["matched"] == assign["matched"]
    assert dash["total"] == 6
    assert len(dash["experts"]) == 5
    assert dash["assignments"]  # назначения присутствуют


def test_reset_clears_store(client: TestClient) -> None:
    client.post("/seed")
    client.post("/reset")
    assert client.get("/experts").json() == []
    assert client.get("/requests").json() == []
