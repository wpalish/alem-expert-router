"""REST API системы умного распределения экспертов (FastAPI).

Запуск:  uvicorn app.main:app --reload
Документация:  http://localhost:8000/docs
Дашборд:  http://localhost:8000/
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import seed
from .matching import Weights, assign_all
from .models import (
    AssignmentResult,
    DashboardResponse,
    Expert,
    ExpertCreate,
    ExpertLoad,
    Request,
    RequestCreate,
)
from .store import Store

store = Store()
_STATIC = Path(__file__).parent / "static"


def _seed_store() -> None:
    store.reset()
    for expert in seed.EXPERTS:
        store.add_expert(expert)
    for request in seed.REQUESTS:
        store.add_request(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _seed_store()  # данные доступны сразу после старта (демо «из коробки»)
    yield


app = FastAPI(
    title="Alem Expert Router",
    description="Умное распределение экспертов, менторов и фрилансеров "
    "по входящим заявкам — с учётом компетенций, загрузки и приоритетов.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_LOCAL_HOSTS = {"127.0.0.1", "::1", "testclient", "localhost"}


def local_only(request: HttpRequest) -> None:
    """Разрешить write-операции (seed/reset) только с локального хоста."""
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
#  Эксперты
# --------------------------------------------------------------------------- #
@app.post("/experts", response_model=Expert, tags=["Experts"])
def add_expert(payload: ExpertCreate) -> Expert:
    return store.add_expert(Expert(**payload.model_dump()))


@app.get("/experts", response_model=list[Expert], tags=["Experts"])
def list_experts() -> list[Expert]:
    return store.experts()


# --------------------------------------------------------------------------- #
#  Заявки
# --------------------------------------------------------------------------- #
@app.post("/requests", response_model=Request, tags=["Requests"])
def add_request(payload: RequestCreate) -> Request:
    return store.add_request(Request(**payload.model_dump()))


@app.get("/requests", response_model=list[Request], tags=["Requests"])
def list_requests() -> list[Request]:
    return store.requests()


# --------------------------------------------------------------------------- #
#  Распределение
# --------------------------------------------------------------------------- #
@app.get("/assign", response_model=AssignmentResult, tags=["Matching"])
def assign() -> AssignmentResult:
    """Умное распределение по текущему пулу заявок и исполнителей."""
    return assign_all(store.requests(), store.experts(), Weights())


@app.get("/dashboard", response_model=DashboardResponse, tags=["Matching"])
def dashboard() -> DashboardResponse:
    """Единый прозрачный срез: назначения + загрузка каждого исполнителя."""
    snapshot = store.experts()  # один снимок — без гонок с seed/reset
    result = assign_all(store.requests(), snapshot, Weights())

    load: dict[str, int] = {e.id: 0 for e in snapshot}
    for a in result.assignments:
        load[a.expert_id] = load.get(a.expert_id, 0) + 1

    experts = [
        ExpertLoad(
            id=e.id,
            name=e.name,
            role=e.role,
            assigned_now=load.get(e.id, 0),
            capacity=e.capacity,
            utilization=round(load.get(e.id, 0) / e.capacity, 2),
        )
        for e in snapshot
    ]
    return DashboardResponse(
        matched=result.matched,
        total=result.total,
        assignments=result.assignments,
        unassigned=result.unassigned,
        experts=experts,
    )


# --------------------------------------------------------------------------- #
#  Dev-операции (только локально)
# --------------------------------------------------------------------------- #
@app.post("/seed", tags=["Dev"], dependencies=[Depends(local_only)])
def load_seed() -> dict[str, int]:
    """Загрузить демо-данные (пул и заявки в духе Alem School)."""
    _seed_store()
    return {"experts": len(seed.EXPERTS), "requests": len(seed.REQUESTS)}


@app.post("/reset", tags=["Dev"], dependencies=[Depends(local_only)])
def reset() -> dict[str, str]:
    store.reset()
    return {"status": "cleared"}
