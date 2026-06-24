"""Ядро системы: оценка соответствия и распределение заявок.

Алгоритм объяснимый и детерминированный:

1.  Жёсткие фильтры (эксперт недоступен / нет свободной ёмкости /
    не покрывает требуемые навыки) — кандидат отсеивается.
2.  Среди подходящих считается оценка 0..100 как взвешенная сумма:
    компетентность + балансировка загрузки + рейтинг.
3.  Заявки обрабатываются в порядке приоритета (затем по сроку), и каждой
    отдаётся лучший свободный исполнитель. Ёмкость уменьшается «на лету»,
    поэтому загрузка распределяется равномерно.

Каждое назначение сопровождается списком причин (требование «прозрачность
управления ресурсами»). Веса вынесены в конфиг — их можно настраивать
под политику школы, не трогая код.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
            matched.append(f"{skill} ({have}≥{min_level})")
        else:
            missing.append(f"{skill} (нужно {min_level}, есть {have or 0})")
    return (len(missing) == 0, missing if missing else matched)


def score_candidate(
    expert: Expert, request: Request, weights: Weights = Weights()
) -> Candidate | None:
    """Оценить пару (эксперт, заявка). None — если кандидат не подходит."""
    if not expert.available:
        return None
    if expert.free_capacity <= 0:
        return None

    covers, detail = _covers(expert, request)
    if not covers:
        return None  # не покрывает требуемые компетенции

    # --- компетентность: средний уровень владения требуемыми навыками / 5 ---
    if request.required_skills:
        levels = [expert.skills.get(s, 0) for s in request.required_skills]
        competency = sum(levels) / (len(levels) * 5)
    else:
        competency = 0.5

    # --- балансировка загрузки: чем свободнее, тем лучше ---
    load_balance = expert.free_capacity / expert.capacity

    # --- рейтинг ---
    rating = expert.rating / 5

    total_w = weights.competency + weights.load_balance + weights.rating
    raw = (
        weights.competency * competency
        + weights.load_balance * load_balance
        + weights.rating * rating
    ) / total_w
    score = round(raw * 100, 1)

    reasons = [
        f"Покрывает требуемые навыки: {', '.join(detail)}",
        f"Загрузка {expert.current_load}/{expert.capacity} — "
        f"свободно {expert.free_capacity}",
        f"Рейтинг {expert.rating}/5",
        f"Роль: {expert.role}",
    ]
    return Candidate(expert=expert, score=score, reasons=reasons)


def assign_all(
    requests: list[Request],
    experts: list[Expert],
    weights: Weights = Weights(),
) -> AssignmentResult:
    """Распределить заявки по исполнителям.

    Заявки идут по приоритету (выше — раньше), затем по сроку. Загрузка
    исполнителей уменьшается по ходу, поэтому система не перегружает одного
    и распределяет нагрузку справедливо.
    """
    # рабочие копии, чтобы не мутировать вход (immutability)
    pool = {e.id: e.model_copy(deep=True) for e in experts}

    def sort_key(r: Request) -> tuple[int, str]:
        # приоритет по убыванию, затем ближайший срок, затем id
        deadline = r.deadline.isoformat() if r.deadline else "9999-12-31"
        return (-r.priority, deadline + r.id)

    ordered = sorted(requests, key=sort_key)
    result = AssignmentResult(total=len(requests))

    for req in ordered:
        candidates = [
            c
            for e in pool.values()
            if (c := score_candidate(e, req, weights)) is not None
        ]
        if not candidates:
            result.unassigned.append(
                Unassigned(
                    request_id=req.id,
                    request_title=req.title,
                    reason=_explain_no_match(req, pool),
                )
            )
            continue

        # лучший: по оценке, затем рейтинг, затем больше свободной ёмкости
        best = max(
            candidates,
            key=lambda c: (c.score, c.expert.rating, c.expert.free_capacity),
        )
        reasons = list(best.reasons)
        if req.priority >= 4:
            reasons.append(f"Заявка высокого приоритета ({req.priority}/5) — "
                           "обработана в первую очередь")
        result.assignments.append(
            Assignment(
                request_id=req.id,
                request_title=req.title,
                expert_id=best.expert.id,
                expert_name=best.expert.name,
                score=best.score,
                reasons=reasons,
            )
        )
        # занять ёмкость выбранного исполнителя
        pool[best.expert.id].current_load += 1

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
        need = ", ".join(f"{s}≥{lvl}" for s, lvl in request.required_skills.items())
        return f"Нет исполнителя с компетенциями: {need}"
    if not any(e.available for e in skilled):
        return "Все подходящие исполнители помечены как недоступны"
    return "Все подходящие исполнители заняты (нет свободной ёмкости)"
