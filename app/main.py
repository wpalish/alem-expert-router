"""REST API системы умного распределения экспертов (FastAPI).

Запуск:  uvicorn app.main:app --reload
Документация:  http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from . import seed
from .matching import Weights, assign_all
from .models import AssignmentResult, Expert, Request
from .store import Store

app = FastAPI(
    title="Alem Expert Router",
    description="Умное распределение экспертов, менторов и фрилансеров "
    "по входящим заявкам — с учётом компетенций, загрузки и приоритетов.",
    version="0.1.0",
)
store = Store()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --- эксперты ---
@app.post("/experts", response_model=Expert)
def add_expert(expert: Expert) -> Expert:
    return store.add_expert(expert)


@app.get("/experts", response_model=list[Expert])
def list_experts() -> list[Expert]:
    return store.experts()


# --- заявки ---
@app.post("/requests", response_model=Request)
def add_request(request: Request) -> Request:
    return store.add_request(request)


@app.get("/requests", response_model=list[Request])
def list_requests() -> list[Request]:
    return store.requests()


# --- распределение ---
@app.post("/assign", response_model=AssignmentResult)
def assign() -> AssignmentResult:
    """Запустить умное распределение по текущему пулу заявок и исполнителей."""
    return assign_all(store.requests(), store.experts(), Weights())


@app.get("/dashboard")
def dashboard() -> dict:
    """Прозрачная картина ресурсов: загрузка каждого исполнителя и итог матчинга."""
    result = assign_all(store.requests(), store.experts(), Weights())
    load: dict[str, int] = {e.id: 0 for e in store.experts()}
    for a in result.assignments:
        load[a.expert_id] = load.get(a.expert_id, 0) + 1
    experts = [
        {
            "id": e.id,
            "name": e.name,
            "role": e.role,
            "assigned_now": load.get(e.id, 0),
            "capacity": e.capacity,
            "utilization": round(load.get(e.id, 0) / e.capacity, 2),
        }
        for e in store.experts()
    ]
    return {
        "matched": result.matched,
        "total": result.total,
        "unassigned": [u.model_dump() for u in result.unassigned],
        "experts": experts,
    }


@app.post("/seed")
def load_seed() -> dict[str, int]:
    """Загрузить демо-данные (пул и заявки в духе Alem School)."""
    store.reset()
    for e in seed.EXPERTS:
        store.add_expert(e)
    for r in seed.REQUESTS:
        store.add_request(r)
    return {"experts": len(seed.EXPERTS), "requests": len(seed.REQUESTS)}


@app.post("/reset")
def reset() -> dict[str, str]:
    store.reset()
    return {"status": "cleared"}
