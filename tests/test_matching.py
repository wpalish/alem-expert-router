"""Тесты ядра распределения: компетенции, загрузка, приоритеты, прозрачность.

Расширенные тесты покрывают:
- Контрпример жадной эвристики (глобальный оптимум через linear_sum_assignment)
- Дедлайны (сортировка и приоритет по сроку)
- Тай-брейк при равном score
- Проверку на отсутствие мутации входных данных
"""
from __future__ import annotations

from datetime import date

import pytest

from app.matching import Weights, assign_all, score_candidate
from app.models import Expert, Request


# ========================================================================== #
#  Оригинальные тесты (сохранены)
# ========================================================================== #

def test_disqualifies_when_skill_missing() -> None:
    """Эксперт без требуемого навыка не должен подходить."""
    e = Expert(id="e1", name="A", skills={"python": 5}, capacity=2)
    r = Request(id="r1", title="ML", required_skills={"ml": 3})
    assert score_candidate(e, r) is None


def test_disqualifies_when_no_free_capacity() -> None:
    e = Expert(id="e1", name="A", skills={"python": 5}, capacity=1, current_load=1)
    r = Request(id="r1", title="Py", required_skills={"python": 3})
    assert score_candidate(e, r) is None


def test_higher_competency_scores_higher() -> None:
    strong = Expert(id="e1", name="Strong", skills={"python": 5}, capacity=2)
    weak = Expert(id="e2", name="Weak", skills={"python": 3}, capacity=2)
    r = Request(id="r1", title="Py", required_skills={"python": 3})
    assert score_candidate(strong, r).score > score_candidate(weak, r).score


def test_priority_request_assigned_first() -> None:
    """Заявка с высоким приоритетом получает лучшего исполнителя первой."""
    expert = Expert(id="e1", name="Solo", skills={"python": 5}, capacity=1, rating=5)
    low = Request(id="low", title="low", required_skills={"python": 3}, priority=1)
    high = Request(id="high", title="high", required_skills={"python": 3}, priority=5)
    res = assign_all([low, high], [expert])
    assert res.matched == 1
    assert res.assignments[0].request_id == "high"
    assert res.unassigned[0].request_id == "low"


def test_load_balancing_across_experts() -> None:
    """Две заявки на двух равных экспертов -- по одной каждому, без перегруза."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=2, rating=4.5)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=2, rating=4.5)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3}, priority=3)
    r2 = Request(id="r2", title="t2", required_skills={"python": 3}, priority=3)
    res = assign_all([r1, r2], [e1, e2])
    assigned_experts = {a.expert_id for a in res.assignments}
    assert res.matched == 2
    assert assigned_experts == {"e1", "e2"}


def test_assignment_has_explanations() -> None:
    """Каждое назначение объяснимо (требование «прозрачность»)."""
    e = Expert(id="e1", name="A", skills={"python": 5, "ml": 4}, capacity=2, rating=4.8)
    r = Request(id="r1", title="t", required_skills={"python": 4, "ml": 3}, priority=5)
    res = assign_all([r], [e])
    a = res.assignments[0]
    assert a.reasons
    assert any("навык" in reason.lower() for reason in a.reasons)
    assert any("приоритет" in reason.lower() for reason in a.reasons)


def test_unassigned_reports_reason() -> None:
    e = Expert(id="e1", name="A", skills={"python": 5}, capacity=2)
    r = Request(id="r1", title="mobile", required_skills={"flutter": 4})
    res = assign_all([r], [e])
    assert res.matched == 0
    assert "flutter" in res.unassigned[0].reason.lower()


def test_seed_scenario_end_to_end() -> None:
    """Интеграция: демо-сценарий должен назначить большинство."""
    from app import seed

    res = assign_all(seed.REQUESTS, seed.EXPERTS)
    assert res.matched >= 4
    titles = {u.request_title for u in res.unassigned}
    assert "Мобильное приложение" in titles


# ========================================================================== #
#  НОВЫЕ ТЕСТЫ: граничные случаи из ревью
# ========================================================================== #

def test_greedy_counterexample_optimal_assignment() -> None:
    """П.1: Контрпример, где жадный алгоритм даёт субоптимальный результат.

    Заявка A: нужен только python.
    Заявка B: нужен python + ml.
    Эксперт X: python=5, ml=5.
    Эксперт Y: python=5 (ml нет).

    Жадный (если A первая) забрал бы X -- B остался бы без исполнителя.
    Глобальный оптимум: A->Y, B->X (обе закрыты).
    """
    req_a = Request(id="a", title="Python task", required_skills={"python": 3}, priority=3)
    req_b = Request(id="b", title="Python+ML task", required_skills={"python": 3, "ml": 3}, priority=3)
    exp_x = Expert(id="x", name="X_polyglot", skills={"python": 5, "ml": 5}, capacity=1, rating=4.5)
    exp_y = Expert(id="y", name="Y_python_only", skills={"python": 5}, capacity=1, rating=4.5)

    res = assign_all([req_a, req_b], [exp_x, exp_y])

    # Обе заявки должны быть назначены (глобальный оптимум)
    assert res.matched == 2, (
        f"Ожидались 2 назначения, получили {res.matched}. "
        f"Неназначенные: {[u.request_title for u in res.unassigned]}"
    )

    # B должна быть на X (единственный с ml), A на Y
    assignment_map = {a.request_id: a.expert_id for a in res.assignments}
    assert assignment_map["b"] == "x", "B (python+ml) должна быть на X (единственный с ml)"
    assert assignment_map["a"] == "y", "A (python only) должна быть на Y (чтобы X остался для B)"


def test_greedy_counterexample_with_high_priority_first() -> None:
    """Тот же контрпример, но A имеет более высокий приоритет.

    Даже при высоком приоритете A, глобальный алгоритм должен
    распределить оптимально: A->Y, B->X.
    """
    req_a = Request(id="a", title="Python task", required_skills={"python": 3}, priority=5)
    req_b = Request(id="b", title="Python+ML task", required_skills={"python": 3, "ml": 3}, priority=3)
    exp_x = Expert(id="x", name="X_polyglot", skills={"python": 5, "ml": 5}, capacity=1, rating=4.5)
    exp_y = Expert(id="y", name="Y_python_only", skills={"python": 5}, capacity=1, rating=4.5)

    res = assign_all([req_a, req_b], [exp_x, exp_y])

    assert res.matched == 2, (
        f"При приоритете A=5 обе заявки должны быть назначены. "
        f"Неназначенные: {[u.request_title for u in res.unassigned]}"
    )


def test_deadline_sorting_correct() -> None:
    """П.2: Дедлайны сортируются корректно (кортеж, не конкатенация строк).

    Если два эксперта и две заявки с равным приоритетом, но разными
    дедлайнами -- ближайший дедлайн должен обрабатываться в первую очередь.
    Проверяем что нет бага с id, влияющим на порядок дедлайнов.
    """
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=1)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=1)

    # Ближайший дедлайн -- r_early, но id начинается с "z" (проверка бага конкатенации)
    r_late = Request(
        id="a_late",
        title="Late deadline",
        required_skills={"python": 3},
        priority=3,
        deadline=date(2026, 12, 31),
    )
    r_early = Request(
        id="z_early",
        title="Early deadline",
        required_skills={"python": 3},
        priority=3,
        deadline=date(2026, 7, 1),
    )

    res = assign_all([r_late, r_early], [e1, e2])
    assert res.matched == 2

    # Заявка с ранним дедлайном (z_early) должна быть назначена первой
    # (получает лучшего эксперта -- e1, который идёт первым в пуле)
    assert res.assignments[0].request_id == "z_early", (
        "Заявка с ближайшим дедлайном должна быть первой"
    )


def test_deadline_none_goes_last() -> None:
    """П.2: Заявка без дедлайна обрабатывается после заявки с дедлайном."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=1)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=1)

    r_no_deadline = Request(
        id="r1", title="No deadline",
        required_skills={"python": 3}, priority=3,
    )
    r_with_deadline = Request(
        id="r2", title="Has deadline",
        required_skills={"python": 3}, priority=3,
        deadline=date(2026, 8, 1),
    )

    res = assign_all([r_no_deadline, r_with_deadline], [e1, e2])
    assert res.matched == 2
    # С дедлайном -- первая
    assert res.assignments[0].request_id == "r2"


def test_tiebreak_on_equal_score() -> None:
    """При равных score назначения всё равно происходят (детерминированно)."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=1, rating=5.0)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=1, rating=5.0)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3}, priority=3)
    r2 = Request(id="r2", title="t2", required_skills={"python": 3}, priority=3)

    res = assign_all([r1, r2], [e1, e2])

    assert res.matched == 2
    assigned = {a.expert_id for a in res.assignments}
    assert assigned == {"e1", "e2"}


def test_input_not_mutated() -> None:
    """Входные списки requests и experts не мутируются (immutability)."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=2)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=2)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3})
    r2 = Request(id="r2", title="t2", required_skills={"python": 3})

    original_load_e1 = e1.current_load
    original_load_e2 = e2.current_load

    assign_all([r1, r2], [e1, e2])

    assert e1.current_load == original_load_e1, "Эксперт e1 не должен быть изменён"
    assert e2.current_load == original_load_e2, "Эксперт e2 не должен быть изменён"


def test_empty_inputs() -> None:
    """Пустые входы не вызывают ошибок."""
    res_empty = assign_all([], [])
    assert res_empty.total == 0
    assert res_empty.matched == 0

    res_no_experts = assign_all(
        [Request(id="r1", title="t", required_skills={"python": 3})], []
    )
    assert res_no_experts.total == 1
    assert res_no_experts.matched == 0
    assert len(res_no_experts.unassigned) == 1


def test_capacity_limits_respected() -> None:
    """Эксперт не получает больше заявок, чем его ёмкость."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=1)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3})
    r2 = Request(id="r2", title="t2", required_skills={"python": 3})
    r3 = Request(id="r3", title="t3", required_skills={"python": 3})

    res = assign_all([r1, r2, r3], [e1])
    # Только 1 заявка может быть назначена (capacity=1)
    assert res.matched == 1
    assert len(res.unassigned) == 2


def test_weights_param_not_shared() -> None:
    """П.4: Изменение weights в одном вызове не влияет на другой."""
    w_custom = Weights(competency=0.9, load_balance=0.05, rating=0.05)
    e = Expert(id="e1", name="A", skills={"python": 5}, capacity=1, rating=3.0)
    r = Request(id="r1", title="t", required_skills={"python": 3})

    score_default = score_candidate(e, r)
    score_custom = score_candidate(e, r, w_custom)

    # С весом competency=0.9 score выше (эксперт покрывает навык на 100%)
    assert score_custom.score > score_default.score