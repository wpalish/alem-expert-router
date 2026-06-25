"""Интеграционные тесты HTTP-слоя (FastAPI TestClient + SQLite)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app, db


@pytest.fixture(autouse=True)
def clean_db():
    db.reset()
    yield
    db.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_dashboard_served_as_html(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Alem Expert Router" in r.text


def test_create_and_persist_expert(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 5}, "capacity": 2}
    assert client.post("/experts", json=body).json()["current_load"] == 0
    assert len(client.get("/experts").json()) == 1


def test_create_expert_cannot_set_current_load(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 5},
            "capacity": 2, "current_load": 99}
    assert client.post("/experts", json=body).json()["current_load"] == 0


def test_invalid_skill_level_rejected(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 9}, "capacity": 2}
    assert client.post("/experts", json=body).status_code == 422


def test_delete_expert(client: TestClient) -> None:
    client.post("/experts", json={"id": "e1", "name": "A", "skills": {"python": 5}})
    assert client.delete("/experts/e1").json() == {"deleted": True}
    assert client.get("/experts").json() == []


def test_seed_assign_dashboard_flow(client: TestClient) -> None:
    assert client.post("/seed").json() == {"experts": 5, "requests": 6}
    dash = client.get("/dashboard").json()
    assert dash["matched"] >= 4
    assert dash["total"] == 6
    assert len(dash["experts"]) == 5
    assert dash["assignments"]


def test_ai_parse_keywords_fallback(client: TestClient) -> None:
    """AI-эндпоинт извлекает компетенции из текста (fallback по ключевым словам)."""
    r = client.post("/requests/parse",
                    json={"text": "Нужен чат-бот на Python с ML по базе знаний"})
    data = r.json()
    assert "python" in data["skills"]
    assert "ml" in data["skills"]
    assert data["method"] in ("keywords", "llm")


def test_request_with_parsed_skills_then_assign(client: TestClient) -> None:
    client.post("/experts", json={"id": "e1", "name": "Pro",
                                  "skills": {"python": 5, "ml": 5}, "capacity": 2, "rating": 5})
    parsed = client.post("/requests/parse",
                         json={"text": "ассистент на Python с машинным обучением"}).json()
    client.post("/requests", json={"id": "r1", "title": "Бот",
                                   "required_skills": parsed["skills"], "priority": 5})
    assert client.get("/assign").json()["matched"] == 1


def test_reset_clears_db(client: TestClient) -> None:
    client.post("/seed")
    client.post("/reset")
    assert client.get("/experts").json() == []
    assert client.get("/requests").json() == []
