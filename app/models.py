"""Доменные модели системы умного распределения экспертов.

Разделены входные схемы (`*Create`) и доменные/выходные модели: клиент не
может задать служебные поля вроде `current_load` или `status` напрямую.
Уровни владения навыком ограничены 1..5, загрузка не превышает ёмкость.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


Role = Literal["expert", "mentor", "freelancer"]
RequestStatus = Literal["new", "assigned", "unassigned"]
AnnouncementType = Literal["expert_needed", "task_available", "event"]

# уровень владения навыком 1 (базовый) .. 5 (экспертный)
SkillLevel = Annotated[int, Field(ge=1, le=5)]
SkillMap = dict[str, SkillLevel]  # компетенция -> уровень


# --------------------------------------------------------------------------- #
#  Исполнители
# --------------------------------------------------------------------------- #
class ExpertCreate(BaseModel):
    """Входная схема создания исполнителя (без служебных полей).

    `id` необязателен: если не задан, сервер сгенерирует уникальный.
    """

    id: Optional[str] = None
    name: str
    role: Role = "expert"
    skills: SkillMap = Field(default_factory=dict)
    capacity: int = Field(3, ge=1, description="Сколько заявок ведёт одновременно")
    available: bool = True
    rating: float = Field(4.0, ge=0, le=5)


class Expert(ExpertCreate):
    """Доменная модель исполнителя (с текущей загрузкой)."""

    current_load: int = Field(0, ge=0, description="Сколько ведёт сейчас")

    @property
    def free_capacity(self) -> int:
        return max(self.capacity - self.current_load, 0)

    @model_validator(mode="after")
    def _load_within_capacity(self) -> "Expert":
        if self.current_load > self.capacity:
            raise ValueError("current_load не может превышать capacity")
        return self


# --------------------------------------------------------------------------- #
#  Заявки
# --------------------------------------------------------------------------- #
class RequestCreate(BaseModel):
    """Входная схема создания заявки (статус выставляет система).

    `id` необязателен: если не задан, сервер сгенерирует уникальный.
    """

    id: Optional[str] = None
    title: str
    description: str = ""
    required_skills: SkillMap = Field(default_factory=dict)
    priority: int = Field(3, ge=1, le=5, description="5 — наивысший приоритет")
    deadline: Optional[date] = None


class Request(RequestCreate):
    """Доменная модель заявки."""

    status: RequestStatus = "new"


# --------------------------------------------------------------------------- #
#  Результаты распределения
# --------------------------------------------------------------------------- #
class Assignment(BaseModel):
    """Назначение исполнителя на заявку с объяснением выбора."""

    request_id: str
    request_title: str
    expert_id: str
    expert_name: str
    score: float = Field(..., description="Итоговая оценка соответствия 0..100")
    reasons: list[str] = Field(default_factory=list)
    assigned_at: datetime = Field(default_factory=_utcnow)
    partial: bool = Field(False, description="Назначено с частичным покрытием навыков")
    manual: bool = Field(False, description="Назначено вручную координатором")


class Unassigned(BaseModel):
    """Заявка без подходящего свободного исполнителя."""

    request_id: str
    request_title: str
    reason: str


class AssignmentResult(BaseModel):
    """Итог прогона распределения."""

    assignments: list[Assignment] = Field(default_factory=list)
    unassigned: list[Unassigned] = Field(default_factory=list)
    matched: int = 0
    total: int = 0


class ExpertLoad(BaseModel):
    """Загрузка исполнителя для дашборда."""

    id: str
    name: str
    role: Role
    assigned_now: int
    capacity: int
    utilization: float


class DashboardResponse(BaseModel):
    """Единый ответ дашборда: матчинг + назначения + загрузка."""

    matched: int
    total: int
    assignments: list[Assignment] = Field(default_factory=list)
    unassigned: list[Unassigned] = Field(default_factory=list)
    experts: list[ExpertLoad] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Объявления (вторая часть задачи ИКТ: «управление заявками и объявлениями»)
# --------------------------------------------------------------------------- #
class AnnouncementCreate(BaseModel):
    """Входная схема объявления (id и дату выставляет система)."""

    title: str
    body: str = ""
    type: AnnouncementType = "expert_needed"


class Announcement(AnnouncementCreate):
    """Доменная модель объявления."""

    id: str
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
#  Ручное назначение
# --------------------------------------------------------------------------- #
class ManualAssignInput(BaseModel):
    """Координатор вручную закрепляет исполнителя за заявкой."""

    request_id: str
    expert_id: str


class ParseInput(BaseModel):
    """Свободный текст заявки для AI-извлечения компетенций."""

    text: str


class ParseResult(BaseModel):
    """Результат AI-разбора: извлечённые навыки и метод (llm/keywords)."""

    skills: SkillMap = Field(default_factory=dict)
    method: str
