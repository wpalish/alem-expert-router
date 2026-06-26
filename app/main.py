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
from . import __version__
from .db import Database
from .matching import Weights, assign_all
from .models import (
    Announcement,
    AnnouncementCreate,
    Assignment,
    AssignmentResult,
    DashboardResponse,
    Expert,
    ExpertCreate,
    ExpertLoad,
    ManualAssignInput,
    ParseInput,
    ParseResult,
    Request,
    RequestCreate,
)

db = Database()
_STATIC = Path(__file__).parent / "static"

# Ручные назначения координатора: request_id -> expert_id. Перебивают
# автоматическое распределение для конкретной заявки.
_manual: dict[str, str] = {}


def _new_id(prefix: str) -> str:
    import uuid

    return f"{prefix}{uuid.uuid4().hex[:8]}"


def _seed_store() -> None:
    db.reset()
    _manual.clear()
    for expert in seed.EXPERTS:
        db.upsert_expert(expert)
    for request in seed.REQUESTS:
        db.upsert_request(request)
    for ann in seed.ANNOUNCEMENTS:
        db.upsert_announcement(ann)


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
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
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
@app.post("/experts", response_model=Expert, status_code=201, tags=["Experts"])
def create_expert(payload: ExpertCreate) -> Expert:
    """Создать исполнителя. id можно не задавать — сервер сгенерирует.

    Если id задан и уже занят — 409 (POST не перезаписывает; для обновления PUT).
    """
    if not payload.skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы один навык исполнителя")
    data = payload.model_dump()
    if not data.get("id"):
        data["id"] = _new_id("e")
    elif any(e.id == data["id"] for e in db.experts()):
        raise HTTPException(status_code=409, detail=f"Исполнитель {data['id']} уже существует — используйте PUT")
    return db.upsert_expert(Expert(**data))


@app.put("/experts/{expert_id}", response_model=Expert, tags=["Experts"])
def update_expert(expert_id: str, payload: ExpertCreate) -> Expert:
    """Обновить существующего исполнителя (id из пути)."""
    if not payload.skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы один навык исполнителя")
    if not any(e.id == expert_id for e in db.experts()):
        raise HTTPException(status_code=404, detail="Исполнитель не найден")
    data = {**payload.model_dump(), "id": expert_id}
    return db.upsert_expert(Expert(**data))


@app.get("/experts", response_model=list[Expert], tags=["Experts"])
def list_experts() -> list[Expert]:
    return db.experts()


@app.delete("/experts/{expert_id}", tags=["Experts"])
def remove_expert(expert_id: str) -> dict[str, bool]:
    # снять ручные назначения на этого исполнителя
    for rid, eid in list(_manual.items()):
        if eid == expert_id:
            _manual.pop(rid, None)
    return {"deleted": db.delete_expert(expert_id)}


# --------------------------------------------------------------------------- #
#  Заявки (CRUD + AI)
# --------------------------------------------------------------------------- #
@app.post("/requests", response_model=Request, status_code=201, tags=["Requests"])
def create_request(payload: RequestCreate) -> Request:
    """Создать заявку. id можно не задавать — сервер сгенерирует.

    Если id задан и уже занят — 409 (для обновления используйте PUT).
    """
    if not payload.required_skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы одну требуемую компетенцию")
    data = payload.model_dump()
    if not data.get("id"):
        data["id"] = _new_id("r")
    elif any(r.id == data["id"] for r in db.requests()):
        raise HTTPException(status_code=409, detail=f"Заявка {data['id']} уже существует — используйте PUT")
    return db.upsert_request(Request(**data))


@app.put("/requests/{request_id}", response_model=Request, tags=["Requests"])
def update_request(request_id: str, payload: RequestCreate) -> Request:
    """Обновить существующую заявку (id из пути)."""
    if not payload.required_skills:
        raise HTTPException(status_code=400, detail="Укажите хотя бы одну требуемую компетенцию")
    if not any(r.id == request_id for r in db.requests()):
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    data = {**payload.model_dump(), "id": request_id}
    return db.upsert_request(Request(**data))


@app.get("/requests", response_model=list[Request], tags=["Requests"])
def list_requests() -> list[Request]:
    return db.requests()


@app.delete("/requests/{request_id}", tags=["Requests"])
def remove_request(request_id: str) -> dict[str, bool]:
    _manual.pop(request_id, None)
    return {"deleted": db.delete_request(request_id)}


@app.post("/requests/parse", response_model=ParseResult, tags=["Requests"])
async def parse_request(payload: ParseInput) -> ParseResult:
    """AI: извлечь требуемые компетенции из свободного текста заявки."""
    skills, method = await ai.extract_skills(payload.text)
    return ParseResult(skills=skills, method=method)


# --------------------------------------------------------------------------- #
#  Распределение
# --------------------------------------------------------------------------- #
@app.get("/assign", response_model=AssignmentResult, tags=["Matching"])
def assign() -> AssignmentResult:
    return assign_all(db.requests(), db.experts(), Weights())


@app.post("/assign/manual", response_model=Assignment, status_code=201, tags=["Matching"])
def assign_manual(payload: ManualAssignInput) -> Assignment:
    """Координатор вручную закрепляет исполнителя за заявкой.

    Перебивает автоматическое распределение для этой заявки (видно в /dashboard).
    """
    req = next((r for r in db.requests() if r.id == payload.request_id), None)
    if req is None:
        raise HTTPException(status_code=404, detail="Заявка не найдена")
    exp = next((e for e in db.experts() if e.id == payload.expert_id), None)
    if exp is None:
        raise HTTPException(status_code=404, detail="Исполнитель не найден")
    _manual[payload.request_id] = payload.expert_id
    return Assignment(
        request_id=req.id, request_title=req.title,
        expert_id=exp.id, expert_name=exp.name, score=0.0, manual=True,
        reasons=["Назначено вручную координатором"],
    )


@app.delete("/assign/manual/{request_id}", tags=["Matching"])
def clear_manual(request_id: str) -> dict[str, bool]:
    """Снять ручное назначение — заявка вернётся в авто-распределение."""
    return {"deleted": _manual.pop(request_id, None) is not None}


def _dashboard() -> DashboardResponse:
    """Срез: авто-распределение + ручные назначения + загрузка."""
    snapshot = db.experts()
    requests = db.requests()
    by_id = {e.id: e for e in snapshot}
    req_by_id = {r.id: r for r in requests}

    # действующие ручные назначения (только для существующих заявок/исполнителей)
    manual = {rid: eid for rid, eid in _manual.items()
              if rid in req_by_id and eid in by_id}

    # авто-распределение по заявкам БЕЗ ручного назначения
    auto_requests = [r for r in requests if r.id not in manual]
    result = assign_all(auto_requests, snapshot, Weights())

    assignments = list(result.assignments)
    for rid, eid in manual.items():
        r, e = req_by_id[rid], by_id[eid]
        assignments.append(Assignment(
            request_id=r.id, request_title=r.title,
            expert_id=e.id, expert_name=e.name, score=0.0, manual=True,
            reasons=["Назначено вручную координатором"],
        ))

    load: dict[str, int] = {e.id: 0 for e in snapshot}
    for a in assignments:
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
        matched=len(assignments), total=len(requests),
        assignments=assignments, unassigned=result.unassigned,
        experts=experts,
    )


@app.get("/dashboard", response_model=DashboardResponse, tags=["Matching"])
def dashboard() -> DashboardResponse:
    """Единый прозрачный срез: назначения (авто+ручные) + загрузка."""
    return _dashboard()


# --------------------------------------------------------------------------- #
#  Объявления (CRUD)
# --------------------------------------------------------------------------- #
@app.post("/announcements", response_model=Announcement, status_code=201, tags=["Announcements"])
def create_announcement(payload: AnnouncementCreate) -> Announcement:
    """Опубликовать объявление (поиск эксперта / задача / мероприятие)."""
    if not payload.title.strip():
        raise HTTPException(status_code=400, detail="Укажите тему объявления")
    return db.upsert_announcement(Announcement(id=_new_id("a"), **payload.model_dump()))


@app.get("/announcements", response_model=list[Announcement], tags=["Announcements"])
def list_announcements() -> list[Announcement]:
    return db.announcements()


@app.delete("/announcements/{ann_id}", tags=["Announcements"])
def remove_announcement(ann_id: str) -> dict[str, bool]:
    return {"deleted": db.delete_announcement(ann_id)}


# --------------------------------------------------------------------------- #
#  Dev-операции (только локально)
# --------------------------------------------------------------------------- #
@app.post("/seed", tags=["Dev"], dependencies=[Depends(local_only)])
def load_seed() -> dict[str, int]:
    _seed_store()
    return {
        "experts": len(seed.EXPERTS),
        "requests": len(seed.REQUESTS),
        "announcements": len(seed.ANNOUNCEMENTS),
    }


@app.post("/reset", tags=["Dev"], dependencies=[Depends(local_only)])
def reset() -> dict[str, str]:
    db.reset()
    _manual.clear()
    return {"status": "cleared"}
