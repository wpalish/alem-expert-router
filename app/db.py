"""SQLite-хранилище с CRUD. Данные переживают перезапуск.

Интерфейс намеренно повторяет репозиторий: вся работа с БД изолирована здесь,
остальной код зависит только от методов, а не от того, что внутри SQLite.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from datetime import datetime

from .models import Announcement, Expert, Request

_DEFAULT = Path(__file__).resolve().parent.parent / "alem.db"


class Database:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = str(path or os.getenv("DB_PATH", _DEFAULT))
        self._lock = threading.Lock()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        # WAL — конкурентные чтения не блокируются записью; foreign_keys — на будущее
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS experts (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT,
                    skills TEXT, capacity INTEGER, current_load INTEGER,
                    available INTEGER, rating REAL)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT,
                    required_skills TEXT, priority INTEGER, deadline TEXT,
                    status TEXT)"""
            )
            c.execute(
                """CREATE TABLE IF NOT EXISTS announcements (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT,
                    type TEXT, created_at TEXT)"""
            )

    # ----- сериализация -----
    @staticmethod
    def _row_to_expert(r: sqlite3.Row) -> Expert:
        return Expert(
            id=r["id"], name=r["name"], role=r["role"],
            skills=json.loads(r["skills"] or "{}"), capacity=r["capacity"],
            current_load=r["current_load"], available=bool(r["available"]),
            rating=r["rating"],
        )

    @staticmethod
    def _row_to_request(r: sqlite3.Row) -> Request:
        return Request(
            id=r["id"], title=r["title"], description=r["description"] or "",
            required_skills=json.loads(r["required_skills"] or "{}"),
            priority=r["priority"], deadline=r["deadline"] or None,
            status=r["status"],
        )

    # ----- эксперты -----
    def upsert_expert(self, e: Expert) -> Expert:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO experts
                       (id, name, role, skills, capacity, current_load,
                        available, rating)
                   VALUES (:id,:name,:role,:skills,:capacity,
                       :current_load,:available,:rating)
                   ON CONFLICT(id) DO UPDATE SET name=:name, role=:role,
                       skills=:skills, capacity=:capacity, current_load=:current_load,
                       available=:available, rating=:rating""",
                {**e.model_dump(), "skills": json.dumps(e.skills),
                 "available": int(e.available)},
            )
        return e

    def experts(self) -> list[Expert]:
        with self._lock, self._conn() as c:
            return [self._row_to_expert(r) for r in c.execute(
                "SELECT * FROM experts ORDER BY rating DESC")]

    def delete_expert(self, expert_id: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM experts WHERE id=?", (expert_id,)).rowcount > 0

    # ----- заявки -----
    def upsert_request(self, r: Request) -> Request:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO requests
                       (id, title, description, required_skills, priority,
                        deadline, status)
                   VALUES (:id,:title,:description,
                       :required_skills,:priority,:deadline,:status)
                   ON CONFLICT(id) DO UPDATE SET title=:title, description=:description,
                       required_skills=:required_skills, priority=:priority,
                       deadline=:deadline, status=:status""",
                {**r.model_dump(), "required_skills": json.dumps(r.required_skills),
                 "deadline": r.deadline.isoformat() if r.deadline else None},
            )
        return r

    def requests(self) -> list[Request]:
        with self._lock, self._conn() as c:
            return [self._row_to_request(r) for r in c.execute(
                "SELECT * FROM requests ORDER BY priority DESC")]

    def delete_request(self, request_id: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM requests WHERE id=?", (request_id,)).rowcount > 0

    # ----- объявления -----
    @staticmethod
    def _row_to_announcement(r: sqlite3.Row) -> Announcement:
        return Announcement(
            id=r["id"], title=r["title"], body=r["body"] or "",
            type=r["type"], created_at=datetime.fromisoformat(r["created_at"]),
        )

    def upsert_announcement(self, a: Announcement) -> Announcement:
        with self._lock, self._conn() as c:
            c.execute(
                """INSERT INTO announcements (id, title, body, type, created_at)
                   VALUES (:id,:title,:body,:type,:created_at)
                   ON CONFLICT(id) DO UPDATE SET title=:title, body=:body,
                       type=:type, created_at=:created_at""",
                {**a.model_dump(), "created_at": a.created_at.isoformat()},
            )
        return a

    def announcements(self) -> list[Announcement]:
        with self._lock, self._conn() as c:
            return [self._row_to_announcement(r) for r in c.execute(
                "SELECT * FROM announcements ORDER BY created_at DESC")]

    def delete_announcement(self, ann_id: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute(
                "DELETE FROM announcements WHERE id=?", (ann_id,)).rowcount > 0

    # ----- обслуживание -----
    def reset(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM experts")
            c.execute("DELETE FROM requests")
            c.execute("DELETE FROM announcements")

    def count(self) -> tuple[int, int]:
        with self._lock, self._conn() as c:
            e = c.execute("SELECT COUNT(*) FROM experts").fetchone()[0]
            r = c.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
            return e, r
