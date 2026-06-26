"""Ядро системы: оценка соответствия и распределение заявок.

Алгоритм объяснимый и детерминированный:

1.  Жёсткие фильтры (эксперт недоступен / нет свободной ёмкости /
    не покрывает требуемые навыки) — кандидат отсеивается.
2.  Среди подходящих считается оценка 0..100 как взвешенная сумма:
    компетентность + балансировка загрузки + рейтинг.
3.  Распределение ищет ГЛОБАЛЬНЫЙ оптимум (задача о назначениях): максимум
    суммарного соответствия по всем заявкам сразу. Это решает проблему
    жадного подхода, когда «локально лучший» выбор для одной заявки оставляет
    другую без исполнителя. Приоритет учитывается как множитель вклада заявки,
    поэтому при конфликте за ресурс выигрывает более важная заявка.

Каждое назначение сопровождается списком причин (требование «прозрачность
управления ресурсами»). Веса вынесены в конфиг — их можно настраивать
под политику школы, не трогая код.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .assignment_solver import solve_assignment
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
    partial: bool = False  # назначен с частичным покрытием навыков


def score_candidate(
    expert: Expert,
    request: Request,
    weights: Weights = Weights(),
    _max_capacity: int = 0,
    partial: bool = False,
) -> Candidate | None:
    """Оценить пару (эксперт, заявка). None — если кандидат не подходит.

    `_max_capacity` — максимальная ёмкость среди исполнителей пула; нужна для
    нормировки балансировки. Если 0, берётся ёмкость самого эксперта
    (одиночная оценка вне общего прогона).

    `partial=True` допускает частичное покрытие: эксперт берётся, даже если
    закрывает не все навыки (со штрафом к оценке), но только если он покрывает
    хотя бы один требуемый навык. Используется как fallback, когда точного
    исполнителя нет вовсе.
    """
    if _max_capacity <= 0:
        _max_capacity = expert.capacity
    if not expert.available:
        return None
    if expert.free_capacity <= 0:
        return None

    # детальный разбор покрытия
    matched: list[str] = []
    missing: list[str] = []
    any_hit = 0
    for skill, min_level in request.required_skills.items():
        have = expert.skills.get(skill, 0)
        if have >= min_level:
            matched.append(f"{skill} ({have}≥{min_level})")
            any_hit += 1
        elif partial and have > 0:
            matched.append(f"{skill} ({have}/{min_level}, частично)")
            missing.append(f"{skill} ({have}<{min_level})")
            any_hit += 1
        elif partial:
            missing.append(f"{skill} (отсутствует)")
        else:
            return None  # строгий режим: один промах — отказ

    if partial and any_hit == 0:
        return None  # частичное допустимо только при реальном пересечении

    # --- компетентность ---
    # Базис: фактический уровень владения требуемыми навыками относительно 5
    # (эксперт python=5 ценнее python=3 даже когда обоим хватает для задачи —
    # глубина важна). В частичном режиме недостающие навыки тянут оценку вниз
    # естественным образом, т.к. уровень там ниже требуемого.
    if request.required_skills:
        levels = [expert.skills.get(s, 0) for s in request.required_skills]
        competency = sum(levels) / (len(levels) * 5)
    else:
        competency = 0.5

    # --- балансировка загрузки ---
    # Берём АБСОЛЮТНУЮ свободную ёмкость (а не долю free/capacity): доля
    # системно поощряла бы исполнителей с маленькой ёмкостью (capacity=1,
    # load=0 -> 1.0), что искажает справедливость. Нормируем по максимально
    # возможной ёмкости в пуле, переданной снаружи.
    denom = max(_max_capacity, 1)
    load_balance = min(expert.free_capacity / denom, 1.0)

    # --- рейтинг ---
    rating = expert.rating / 5

    total_w = weights.competency + weights.load_balance + weights.rating
    raw = (
        weights.competency * competency
        + weights.load_balance * load_balance
        + weights.rating * rating
    ) / total_w
    # штраф за каждый недостающий навык (только в частичном режиме)
    penalty = min(len(missing) * 0.08, 0.6) if partial else 0.0
    score = round(raw * (1 - penalty) * 100, 1)

    reasons = [
        f"Покрывает навыки: {', '.join(matched)}",
        f"Загрузка {expert.current_load}/{expert.capacity} — "
        f"свободно {expert.free_capacity}",
        f"Рейтинг {expert.rating}/5",
        f"Роль: {expert.role}",
    ]
    if missing:
        reasons.append(f"⚠ Частичное покрытие: {', '.join(missing)}")
    return Candidate(
        expert=expert, score=score, reasons=reasons, partial=bool(missing)
    )


def assign_all(
    requests: list[Request],
    experts: list[Expert],
    weights: Weights = Weights(),
) -> AssignmentResult:
    """Глобально-оптимально распределить заявки по исполнителям.

    Строим матрицу «заявка × слот исполнителя» (эксперт с ёмкостью N даёт N
    слотов), где значение ячейки = соответствие, взвешенное на приоритет
    заявки. Венгерский алгоритм находит назначение с максимальной суммарной
    выгодой — это решает проблему жадного перебора, при котором локально
    лучший выбор оставлял другую заявку без исполнителя.
    """
    # рабочие копии, чтобы не мутировать вход (immutability)
    pool = {e.id: e.model_copy(deep=True) for e in experts}
    result = AssignmentResult(total=len(requests))

    if not requests:
        return result

    max_cap = max((e.capacity for e in pool.values()), default=0)

    # Детерминированный порядок заявок: приоритет ↓, срок ↑, id ↑.
    def req_key(r: Request) -> tuple[int, str, str]:
        deadline = r.deadline.isoformat() if r.deadline else "9999-12-31"
        return (-r.priority, deadline, r.id)

    ordered_reqs = sorted(requests, key=req_key)

    # «Слоты»: каждый свободный слот исполнителя — отдельный столбец.
    # slot_rank — какой это по счёту слот данного исполнителя (0,1,2,...).
    # Используется для мягкого штрафа за концентрацию нагрузки на одном
    # человеке: при прочих равных система разносит заявки по разным людям.
    slots: list[Expert] = []
    slot_rank: list[int] = []
    for e in sorted(pool.values(), key=lambda x: x.id):
        for k in range(e.free_capacity):
            slots.append(e)
            slot_rank.append(k)

    # Кандидаты по парам, чтобы построить матрицу выгод и переиспользовать
    # уже посчитанное объяснение.
    cand: dict[tuple[int, int], Candidate] = {}
    BIG = 1e6  # «запретная» стоимость для несовместимых пар

    cost: list[list[float]] = []
    for ri, req in enumerate(ordered_reqs):
        row: list[float] = []
        # множитель приоритета: чем важнее заявка, тем сильнее её вклад,
        # поэтому при конкуренции за слот выигрывает приоритетная.
        pmul = 1.0 + (max(req.priority, 1) - 1) * 0.5  # priority 1..5 -> 1.0..3.0
        # Сначала пробуем строгое покрытие. Если у заявки НЕТ ни одного
        # строгого кандидата — разрешаем частичное (со штрафом), чтобы не
        # оставлять заявку без исполнителя, когда есть «почти подходящий».
        has_strict = any(
            score_candidate(e, req, weights, _max_capacity=max_cap) is not None
            for e in pool.values()
        )
        use_partial = not has_strict
        for sj, expert in enumerate(slots):
            c = score_candidate(
                expert, req, weights, _max_capacity=max_cap, partial=use_partial
            )
            if c is None:
                row.append(BIG)
            else:
                cand[(ri, sj)] = c
                # solver минимизирует — переводим выгоду в стоимость.
                # Маленький штраф за каждый последующий слот того же эксперта
                # (slot_rank) разносит заявки по людям при равном качестве,
                # но не перебивает реальную разницу в score.
                spread_penalty = slot_rank[sj] * 0.01
                row.append(-c.score * pmul + spread_penalty)
        cost.append(row)

    assignment = solve_assignment(cost) if slots else [-1] * len(ordered_reqs)

    used_slot: set[int] = set()
    for ri, sj in enumerate(assignment):
        req = ordered_reqs[ri]
        chosen = cand.get((ri, sj)) if sj != -1 else None
        # отбрасываем «запретные» назначения (несовместимая пара) и дубли слотов
        if chosen is None or sj in used_slot:
            result.unassigned.append(
                Unassigned(
                    request_id=req.id,
                    request_title=req.title,
                    reason=_explain_no_match(req, pool),
                )
            )
            continue
        used_slot.add(sj)
        reasons = list(chosen.reasons)
        if req.priority >= 4:
            reasons.append(
                f"Заявка высокого приоритета ({req.priority}/5) — "
                "учтена с повышенным весом при распределении"
            )
        if chosen.partial:
            reasons.append(
                "Назначение с допуском частичного покрытия "
                "(точного исполнителя нет)"
            )
        result.assignments.append(
            Assignment(
                request_id=req.id,
                request_title=req.title,
                expert_id=chosen.expert.id,
                expert_name=chosen.expert.name,
                score=chosen.score,
                reasons=reasons,
                partial=chosen.partial,
            )
        )

    # стабильный порядок вывода: по приоритету (как ordered_reqs)
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
