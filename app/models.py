"""Доменные модели системы умного распределения экспертов.

Заявка (Request) описывает входящую задачу с требуемыми компетенциями,
приоритетом и сроком. Исполнитель (Expert) — эксперт / ментор / фрилансер
с набором навыков, ёмкостью и текущей загрузкой. Assignment — результат
назначения с машинно-читаемым объяснением (требование «прозрачность»).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from pydantic import BaseModel, Field

Role = Literal["expert", "mentor", "freelancer"]
SkillMap = dict[str, int]  # компетенция -> уровень владения 1..5


class Expert(BaseModel):
    """Исполнитель: эксперт, ментор или фрилансер."""

    id: str
    name: str
    role: Role = "expert"
    # компетенция -> уровень владения (1 — базовый, 5 — экспертный)
    skills: SkillMap = Field(default_factory=dict)
    capacity: int = Field(3, ge=1, description="Сколько заявок может вести одновременно")
    current_load: int = Field(0, ge=0, description="Сколько ведёт сейчас")
    available: bool = True
    rating: float = Field(4.0, ge=0, le=5)

    @property
    def free_capacity(self) -> int:
        return max(self.capacity - self.current_load, 0)


class Request(BaseModel):
    """Входящая заявка / объявление в рамках программы."""

    id: str
    title: str
    description: str = ""
    # требуемая компетенция -> минимальный уровень
    required_skills: SkillMap = Field(default_factory=dict)
    priority: int = Field(3, ge=1, le=5, description="5 — наивысший приоритет")
    deadline: Optional[date] = None
    status: Literal["new", "assigned", "unassigned"] = "new"


class Assignment(BaseModel):
    """Назначение исполнителя на заявку с объяснением выбора."""

    request_id: str
    request_title: str
    expert_id: str
    expert_name: str
    score: float = Field(..., description="Итоговая оценка соответствия 0..100")
    reasons: list[str] = Field(default_factory=list, description="Почему выбран именно он")
    assigned_at: datetime = Field(default_factory=_utcnow)


class Unassigned(BaseModel):
    """Заявка, для которой не нашлось подходящего свободного исполнителя."""

    request_id: str
    request_title: str
    reason: str


class AssignmentResult(BaseModel):
    """Итог прогона распределения."""

    assignments: list[Assignment] = Field(default_factory=list)
    unassigned: list[Unassigned] = Field(default_factory=list)
    matched: int = 0
    total: int = 0
