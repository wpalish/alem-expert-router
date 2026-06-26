"""Демо-данные в духе Alem School: пул менторов/экспертов и входящие заявки."""
from __future__ import annotations

from .models import Announcement, Expert, Request

EXPERTS: list[Expert] = [
    Expert(id="e1", name="Айгерим", role="expert",
           skills={"python": 5, "fastapi": 5, "ml": 4}, capacity=3, rating=4.9),
    Expert(id="e2", name="Данияр", role="mentor",
           skills={"python": 4, "django": 5, "sql": 4}, capacity=4, rating=4.6),
    Expert(id="e3", name="Жанель", role="freelancer",
           skills={"react": 5, "typescript": 5, "ui": 4}, capacity=2, rating=4.7),
    Expert(id="e4", name="Тимур", role="expert",
           skills={"devops": 5, "docker": 5, "python": 3}, capacity=3, rating=4.4),
    Expert(id="e5", name="Сабина", role="mentor",
           skills={"ml": 5, "python": 5, "data": 5}, capacity=2, rating=5.0),
]

REQUESTS: list[Request] = [
    Request(id="r1", title="Чат-бот поддержки студентов",
            description="LLM-ассистент по базе знаний школы",
            required_skills={"python": 4, "ml": 3}, priority=5),
    Request(id="r2", title="Веб-портал заявок",
            description="Фронтенд кабинета подачи заявок",
            required_skills={"react": 4, "typescript": 4}, priority=4),
    Request(id="r3", title="Бэкенд распределения экспертов",
            description="API сопоставления заявок и менторов",
            required_skills={"python": 4, "fastapi": 4}, priority=5),
    Request(id="r4", title="CI/CD и контейнеризация",
            required_skills={"devops": 4, "docker": 4}, priority=2),
    Request(id="r5", title="Аналитика загрузки менторов",
            required_skills={"sql": 3, "data": 3}, priority=3),
    Request(id="r6", title="Мобильное приложение",
            required_skills={"flutter": 4}, priority=3),  # нет такого навыка -> unassigned
]

ANNOUNCEMENTS: list[Announcement] = [
    Announcement(
        id="a1", title="Требуется ментор по Data Science",
        body="Нужен наставник для группы из 8 студентов на летний интенсив.",
        type="expert_needed",
    ),
    Announcement(
        id="a2", title="Хакатон «EdTech Solutions»",
        body="Регистрация до 15 июля. Приглашаем экспертов в жюри.",
        type="event",
    ),
]
