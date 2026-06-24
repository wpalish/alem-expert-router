"""Простое потокобезопасное хранилище в памяти.

Для MVP достаточно in-memory; интерфейс намеренно повторяет паттерн
репозитория, поэтому замена на PostgreSQL/SQLite — это новая реализация
тех же методов, без изменения остального кода.
"""
from __future__ import annotations

from threading import Lock

from .models import Expert, Request


class Store:
    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}
        self._requests: dict[str, Request] = {}
        self._lock = Lock()

    # --- эксперты ---
    def add_expert(self, expert: Expert) -> Expert:
        with self._lock:
            self._experts[expert.id] = expert
        return expert

    def experts(self) -> list[Expert]:
        return list(self._experts.values())

    # --- заявки ---
    def add_request(self, request: Request) -> Request:
        with self._lock:
            self._requests[request.id] = request
        return request

    def requests(self) -> list[Request]:
        return list(self._requests.values())

    def reset(self) -> None:
        with self._lock:
            self._experts.clear()
            self._requests.clear()
