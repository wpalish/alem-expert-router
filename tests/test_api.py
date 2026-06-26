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


def test_request_without_skills_rejected(client: TestClient) -> None:
    r = client.post("/requests", json={"id": "r1", "title": "X",
                                       "required_skills": {}, "priority": 3})
    assert r.status_code == 400


def test_expert_without_skills_rejected(client: TestClient) -> None:
    r = client.post("/experts", json={"id": "e1", "name": "X", "skills": {}})
    assert r.status_code == 400


def test_seed_assign_dashboard_flow(client: TestClient) -> None:
    assert client.post("/seed").json() == {"experts": 5, "requests": 6, "announcements": 2}
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


def test_create_expert_generates_id_when_missing(client: TestClient) -> None:
    r = client.post("/experts", json={"name": "NoId", "skills": {"python": 5}})
    assert r.status_code == 201
    assert r.json()["id"].startswith("e")


def test_duplicate_id_conflicts(client: TestClient) -> None:
    body = {"id": "e1", "name": "A", "skills": {"python": 5}}
    assert client.post("/experts", json=body).status_code == 201
    assert client.post("/experts", json=body).status_code == 409  # POST не перезаписывает


def test_put_updates_expert(client: TestClient) -> None:
    client.post("/experts", json={"id": "e1", "name": "Old", "skills": {"python": 5}})
    r = client.put("/experts/e1", json={"name": "New", "skills": {"python": 4}, "rating": 5})
    assert r.status_code == 200
    assert r.json()["name"] == "New"
    assert client.get("/experts").json()[0]["name"] == "New"


def test_put_missing_expert_404(client: TestClient) -> None:
    r = client.put("/experts/nope", json={"name": "X", "skills": {"python": 5}})
    assert r.status_code == 404


def test_announcements_crud(client: TestClient) -> None:
    r = client.post("/announcements", json={"title": "Нужен ментор", "body": "ML", "type": "expert_needed"})
    assert r.status_code == 201
    aid = r.json()["id"]
    assert any(a["id"] == aid for a in client.get("/announcements").json())
    assert client.delete(f"/announcements/{aid}").json() == {"deleted": True}
    assert all(a["id"] != aid for a in client.get("/announcements").json())


def test_manual_assignment_overrides_auto(client: TestClient) -> None:
    client.post("/experts", json={"id": "ea", "name": "A", "skills": {"python": 5}, "capacity": 2, "rating": 5})
    client.post("/experts", json={"id": "eb", "name": "B", "skills": {"python": 5}, "capacity": 2, "rating": 3})
    client.post("/requests", json={"id": "r1", "title": "t", "required_skills": {"python": 3}, "priority": 3})
    # вручную закрепляем менее рейтингового B
    r = client.post("/assign/manual", json={"request_id": "r1", "expert_id": "eb"})
    assert r.status_code == 201 and r.json()["manual"] is True
    dash = client.get("/dashboard").json()
    a = next(x for x in dash["assignments"] if x["request_id"] == "r1")
    assert a["expert_id"] == "eb" and a["manual"] is True
    # снимаем -> авто вернёт лучшего (A, рейтинг выше)
    assert client.delete("/assign/manual/r1").json() == {"deleted": True}
    dash2 = client.get("/dashboard").json()
    a2 = next(x for x in dash2["assignments"] if x["request_id"] == "r1")
    assert a2["expert_id"] == "ea" and a2["manual"] is False


def test_manual_assignment_unknown_request_404(client: TestClient) -> None:
    r = client.post("/assign/manual", json={"request_id": "nope", "expert_id": "x"})
    assert r.status_code == 404
