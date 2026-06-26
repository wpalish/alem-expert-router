"""Ядро системы: оценка соответствия и распределение заявок.

Алгоритм объяснимый и детерминированный:

1.  Жёсткие фильтры (эксперт недоступен / нет свободной ёмкости /
    не покрывает требуемые навыки) — кандидат отсеивается.
2.  Среди подходящих считается оценка 0..100 как взвешенная сумма:
    компетентность + балансировка загрузки + рейтинг.
3.  Строится матрица стоимости (request x expert), решается задача о
    назначениях через scipy.optimize.linear_sum_assignment -- даёт
    **глобальный оптимум**, а не жадную эвристику.

Каждое назначение сопровождается списком причин (требование «прозрачность
управления ресурсами»). Веса вынесены в конфиг -- их можно настраивать
под политику школы, не трогая код.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import (
    Assignment,
    AssignmentResult,
    Expert,
    Request,
    Unassigned,
)


@dataclass(frozen=True)
class Weights:
    """Веса критериев оценки (в сумме нормируются автоматически)."""

    competency: float = 0.60  # насколько глубоко эксперт владеет нужными навыками
    load_balance: float = 0.25  # приоритет менее загруженным (равномерность)
    rating: float = 0.15  # репутация / качество


@dataclass
class Candidate:
    """Подходящий исполнитель с посчитанной оценкой и объяснением."""

    expert: Expert
    score: float
    reasons: list[str] = field(default_factory=list)


def _covers(expert: Expert, request: Request) -> tuple[bool, list[str]]:
    """Покрывает ли эксперт все требуемые навыки на нужном уровне."""
    missing: list[str] = []
    matched: list[str] = []
    for skill, min_level in request.required_skills.items():
        have = expert.skills.get(skill, 0)
        if have >= min_level:
            matched.append(f"{skill} ({have}>= {min_level})")
        else:
            missing.append(f"{skill} (нужно {min_level}, есть {have or 0})")
    return (len(missing) == 0, missing if missing else matched)


def score_candidate(
    expert: Expert, request: Request, weights: Weights | None = None
) -> Candidate | None:
    """Оценить пару (эксперт, заявка). None -- если кандидат не подходит."""
    weights = weights or Weights()
    if not expert.available:
        return None
    if expert.free_capacity <= 0:
        return None

    ok, details = _covers(expert, request)
    if not ok:
        return None

    # --- компетентность: средний уровень по требуемым навыкам ---
    keys = list(request.required_skills.keys())
    competency = (
        sum(expert.skills.get(s, 0) for s in keys) / (len(keys) * 5)
        if keys
        else 0.5
    )

    # --- балансировка загрузки (абсолютные свободные слоты) ---
    load_balance = expert.free_capacity / max(expert.capacity, 1)  # [0..1]

    # --- рейтинг ---
    rating_norm = expert.rating / 5  # [0..1]

    # --- итог ---
    tw = weights.competency + weights.load_balance + weights.rating
    score = (
        weights.competency * competency
        + weights.load_balance * load_balance
        + weights.rating * rating_norm
    ) / tw * 100

    reasons = [
        f"Покрывает навыки: {', '.join(details)}",
        f"Загрузка {expert.current_load}/{expert.capacity} -- свободно {expert.free_capacity}",
        f"Рейтинг {expert.rating}/5",
        f"Роль: {expert.role}",
    ]

    return Candidate(expert=expert, score=score, reasons=reasons)


def assign_all(
    requests: list[Request],
    experts: list[Expert],
    weights: Weights | None = None,
) -> AssignmentResult:
    """Распределить заявки по исполнителям.

    Использует scipy.optimize.linear_sum_assignment для глобального
    оптимума, а не жадную эвристику.

    Алгоритм:
    1. Строим матрицу стоимости: cost[i][j] = 100 - score(expert_j, request_i).
       Недопустимые пары получают штраф 1e9 (не будут назначены).
    2. linear_sum_assignment находит оптимальное назначение с минимальной
       суммарной стоимостью = максимальным суммарным качеством.
    3. Генерируем объяснения для каждого назначения.
    """
    weights = weights or Weights()

    if not requests:
        return AssignmentResult(total=0)
    if not experts:
        empty = AssignmentResult(total=len(requests))
        for r in requests:
            empty.unassigned.append(
                Unassigned(request_id=r.id, request_title=r.title,
                           reason="В системе нет ни одного исполнителя")
            )
        return empty

    # рабочие копии, чтобы не мутировать вход
    pool = {e.id: e.model_copy(deep=True) for e in experts}

    # --- П.2 FIX: кортеж, не конкатенация строк ---
    def sort_key(r: Request) -> tuple[int, str, str]:
        deadline = r.deadline.isoformat() if r.deadline else "9999-12-31"
        return (-r.priority, deadline, r.id)

    ordered = sorted(requests, key=sort_key)
    n_req = len(ordered)
    n_exp = len(experts)
    pool_list = list(pool.values())

    # --- П.1 FIX: матрица стоимости для linear_sum_assignment ---
    cost_matrix = np.full((n_req, n_exp), 1e9)
    score_matrix: list[list[Candidate | None]] = [
        [None] * n_exp for _ in range(n_req)
    ]

    for i, req in enumerate(ordered):
        for j, exp in enumerate(pool_list):
            candidate = score_candidate(exp, req, weights)
            if candidate is not None:
                cost_matrix[i, j] = 100 - candidate.score
                score_matrix[i][j] = candidate

    # Решаем задачу о назначениях
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    result = AssignmentResult(total=n_req)

    for i, j in zip(row_ind, col_ind):
        candidate = score_matrix[i][j]
        # Недопустимая пара (cost >= 1e8)
        if candidate is None or cost_matrix[i, j] >= 1e8:
            req = ordered[i]
            result.unassigned.append(
                Unassigned(
                    request_id=req.id,
                    request_title=req.title,
                    reason=_explain_no_match(req, pool),
                )
            )
            continue

        req = ordered[i]
        exp = candidate.expert

        reasons = list(candidate.reasons)
        if req.priority >= 4:
            reasons.append(
                f"Заявка высокого приоритета ({req.priority}/5) -- "
                "обработана в первую очередь"
            )

        # Сравнение со вторым кандидатом для прозрачности
        all_candidates = [
            score_matrix[i][k]
            for k in range(n_exp)
            if score_matrix[i][k] is not None and cost_matrix[i, k] < 1e8
        ]
        if len(all_candidates) >= 2:
            sorted_cands = sorted(all_candidates, key=lambda c: -c.score)
            second = sorted_cands[1]
            diff = round(candidate.score - second.score, 1)
            if diff < 1:
                reasons.append(
                    f"Незначительное преимущество ({diff} баллов) "
                    f"перед {second.expert.name}"
                )
            elif candidate.expert.rating > second.expert.rating:
                reasons.append(
                    f"Преимущество по рейтингу ({candidate.expert.rating} vs {second.expert.rating}) "
                    f"перед {second.expert.name}"
                )
            else:
                best_free = candidate.expert.free_capacity
                sec_free = second.expert.free_capacity
                if best_free > sec_free:
                    reasons.append(
                        f"Больше свободной ёмкости ({best_free} vs {sec_free}) "
                        f"по сравнению с {second.expert.name}"
                    )

        result.assignments.append(
            Assignment(
                request_id=req.id,
                request_title=req.title,
                expert_id=exp.id,
                expert_name=exp.name,
                score=candidate.score,
                reasons=reasons,
            )
        )
        pool[exp.id].current_load += 1

    # заявки, не попавшие в строки назначения (n_req > n_exp), — в неназначенные
    assigned_rows = {int(x) for x in row_ind}
    for i, req in enumerate(ordered):
        if i not in assigned_rows:
            result.unassigned.append(
                Unassigned(request_id=req.id, request_title=req.title,
                           reason=_explain_no_match(req, pool))
            )

    result.matched = len(result.assignments)
    return result


def _explain_no_match(request: Request, pool: dict[str, Expert]) -> str:
    """Человеко-понятная причина, почему заявка осталась без исполнителя."""
    skilled = [
        e
        for e in pool.values()
        if all(e.skills.get(s, 0) >= lvl for s, lvl in request.required_skills.items())
    ]
    if not skilled:
        need = ", ".join(f"{s}>={lvl}" for s, lvl in request.required_skills.items())
        return f"Нет исполнителя с компетенциями: {need}"
    if not any(e.available for e in skilled):
        return "Все подходящие исполнители помечены как недоступны"
    return "Все подходящие исполнители заняты (нет свободной ёмкости)"