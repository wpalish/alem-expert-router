"""REST API системы умного распределения экспертов (FastAPI).

Запуск:  uvicorn app.main:app --reload
Дашборд:  http://localhost:8000/   ·   Документация:  /docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import ai, seed
from .db import Database
from .matching import Weights, assign_all
from .models import (
    AssignmentResult,
    DashboardResponse,
    Expert,
    ExpertCreate,
    ExpertLoad,
    ParseInput,
    ParseResult,
    Request,
    RequestCreate,
)

db = Database()
_STATIC = Path(__file__).parent / "static"


def _seed_store() -> None:
    db.reset()
    for expert in seed.EXPERTS:
        db.upsert_expert(expert)
    for request in seed.REQUESTS:
        db.upsert_request(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if db.count() == (0, 0):  # первый запуск — наполнить демо-данными
        _seed_store()
    yield


app = FastAPI(
    title="Alem Expert Router",
    description="Умное распределение экспертов, менторов и фрилансеров "
    "по входящим заявкам — с учётом компетенций, загрузки и приоритетов. "
    "AI-слой извлекает компетенции из свободного текста заявки.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "testclient", "localhost"}


def local_only(request: HttpRequest) -> None:
    host = request.client.host if request.client else None
    if host not in _LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Доступно только локально")


# --------------------------------------------------------------------------- #
#  Страница / служебное
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def dashboard_page() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health", tags=["Dev"])
def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
#  Эксперты (CRUD)
# --------------------------------------------------------------------------- #
@app.post("/experts", response_model=Expert, tags=["Experts"])
def save_expert(payload: ExpertCreate) -> Expert:
    """Создать или обновить исполнителя (по id)."""
    if not payload.skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы один навык исполнителя")
    return db.upsert_expert(Expert(**payload.model_dump()))


@app.get("/experts", response_model=list[Expert], tags=["Experts"])
def list_experts() -> list[Expert]:
    return db.experts()


@app.delete("/experts/{expert_id}", tags=["Experts"])
def remove_expert(expert_id: str) -> dict[str, bool]:
    return {"deleted": db.delete_expert(expert_id)}


# --------------------------------------------------------------------------- #
#  Заявки (CRUD + AI)
# --------------------------------------------------------------------------- #
@app.post("/requests", response_model=Request, tags=["Requests"])
def save_request(payload: RequestCreate) -> Request:
    """Создать или обновить заявку (по id)."""
    if not payload.required_skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы одну требуемую компетенцию")
    return db.upsert_request(Request(**payload.model_dump()))


@app.get("/requests", response_model=list[Request], tags=["Requests"])
def list_requests() -> list[Request]:
    return db.requests()


@app.delete("/requests/{request_id}", tags=["Requests"])
def remove_request(request_id: str) -> dict[str, bool]:
    return {"deleted": db.delete_request(request_id)}


@app.post("/requests/parse", response_model=ParseResult, tags=["Requests"])
def parse_request(payload: ParseInput) -> ParseResult:
    """AI: извлечь требуемые компетенции из свободного текста заявки."""
    skills, method = ai.extract_skills(payload.text)
    return ParseResult(skills=skills, method=method)


# --------------------------------------------------------------------------- #
#  Распределение
# --------------------------------------------------------------------------- #
@app.get("/assign", response_model=AssignmentResult, tags=["Matching"])
def assign() -> AssignmentResult:
    return assign_all(db.requests(), db.experts(), Weights())


@app.get("/dashboard", response_model=DashboardResponse, tags=["Matching"])
def dashboard() -> DashboardResponse:
    """Единый прозрачный срез: назначения + загрузка каждого исполнителя."""
    snapshot = db.experts()
    result = assign_all(db.requests(), snapshot, Weights())

    load: dict[str, int] = {e.id: 0 for e in snapshot}
    for a in result.assignments:
        load[a.expert_id] = load.get(a.expert_id, 0) + 1

    experts = [
        ExpertLoad(
            id=e.id, name=e.name, role=e.role,
            assigned_now=load.get(e.id, 0), capacity=e.capacity,
            utilization=round(load.get(e.id, 0) / e.capacity, 2),
        )
        for e in snapshot
    ]
    return DashboardResponse(
        matched=result.matched, total=result.total,
        assignments=result.assignments, unassigned=result.unassigned,
        experts=experts,
    )


# --------------------------------------------------------------------------- #
#  Dev-операции (только локально)
# --------------------------------------------------------------------------- #
@app.post("/seed", tags=["Dev"], dependencies=[Depends(local_only)])
def load_seed() -> dict[str, int]:
    _seed_store()
    return {"experts": len(seed.EXPERTS), "requests": len(seed.REQUESTS)}


@app.post("/reset", tags=["Dev"], dependencies=[Depends(local_only)])
def reset() -> dict[str, str]:
    db.reset()
    return {"status": "cleared"}
