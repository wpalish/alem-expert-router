"""Тесты ядра распределения: компетенции, загрузка, приоритеты, прозрачность."""
from __future__ import annotations

from app.matching import Weights, assign_all, score_candidate
from app.models import Expert, Request


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
    assert res.assignments[0].request_id == "high"  # высокий приоритет — первым
    assert res.unassigned[0].request_id == "low"


def test_load_balancing_across_experts() -> None:
    """Две заявки на двух равных экспертов — по одной каждому, без перегруза."""
    e1 = Expert(id="e1", name="A", skills={"python": 5}, capacity=2, rating=4.5)
    e2 = Expert(id="e2", name="B", skills={"python": 5}, capacity=2, rating=4.5)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3}, priority=3)
    r2 = Request(id="r2", title="t2", required_skills={"python": 3}, priority=3)
    res = assign_all([r1, r2], [e1, e2])
    assigned_experts = {a.expert_id for a in res.assignments}
    assert res.matched == 2
    assert assigned_experts == {"e1", "e2"}  # нагрузка распределена


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
    """Интеграция: демо-сценарий должен назначить большинство и честно
    оставить заявку без нужного навыка неназначенной."""
    from app import seed

    res = assign_all(seed.REQUESTS, seed.EXPERTS)
    assert res.matched >= 4
    titles = {u.request_title for u in res.unassigned}
    assert "Мобильное приложение" in titles  # нет flutter-навыка в пуле


# --------------------------------------------------------------------------- #
#  Регрессии после перехода на глобально-оптимальное распределение
# --------------------------------------------------------------------------- #
def test_global_optimum_beats_greedy() -> None:
    """Контрпример к жадному подходу.

    Заявке A нужен python; заявке B — python+ml. Эксперт X владеет обоими,
    Y — только python. Жадный отдал бы X заявке A (выше score) и оставил B
    без исполнителя. Оптимальное распределение закрывает ОБЕ: A->Y, B->X.
    """
    x = Expert(id="x", name="X", skills={"python": 5, "ml": 5}, capacity=1, rating=5)
    y = Expert(id="y", name="Y", skills={"python": 5}, capacity=1, rating=5)
    a = Request(id="a", title="py", required_skills={"python": 3}, priority=3)
    b = Request(id="b", title="py+ml", required_skills={"python": 3, "ml": 3}, priority=3)
    res = assign_all([a, b], [x, y])
    assert res.matched == 2  # жадный дал бы 1
    by_req = {r.request_id: r.expert_id for r in res.assignments}
    assert by_req["b"] == "x"  # единственный, кто закрывает ml
    assert by_req["a"] == "y"


def test_deadline_breaks_ties_correctly() -> None:
    """При равном приоритете ближний дедлайн обслуживается раньше и id не
    протекает в семантику срочности (баг конкатенации строк)."""
    from datetime import date

    expert = Expert(id="e", name="Solo", skills={"python": 5}, capacity=1, rating=5)
    # id специально подобран так, чтобы старая конкатенация дала неверный порядок
    soon = Request(id="zzz", title="soon", required_skills={"python": 3},
                   priority=3, deadline=date(2026, 1, 1))
    later = Request(id="aaa", title="later", required_skills={"python": 3},
                    priority=3, deadline=date(2026, 12, 31))
    res = assign_all([later, soon], [expert])
    assert res.matched == 1
    assert res.assignments[0].request_id == "zzz"  # ближний дедлайн победил


def test_capacity_used_as_multiple_slots() -> None:
    """Эксперт с ёмкостью 2 может вести две заявки одновременно."""
    e = Expert(id="e", name="Big", skills={"python": 5}, capacity=2, rating=5)
    r1 = Request(id="r1", title="t1", required_skills={"python": 3})
    r2 = Request(id="r2", title="t2", required_skills={"python": 3})
    res = assign_all([r1, r2], [e])
    assert res.matched == 2
    assert all(a.expert_id == "e" for a in res.assignments)


def test_partial_coverage_fallback() -> None:
    """Если строгого исполнителя нет, допускается частичное покрытие."""
    e = Expert(id="e", name="Half", skills={"python": 2}, capacity=1, rating=4)
    r = Request(id="r", title="needs py4", required_skills={"python": 4}, priority=3)
    res = assign_all([r], [e])
    assert res.matched == 1
    a = res.assignments[0]
    assert a.partial is True
    assert any("частичн" in x.lower() for x in a.reasons)


def test_no_overlap_stays_unassigned_even_in_partial() -> None:
    """Полное отсутствие пересечения навыков -> без исполнителя (не частично)."""
    e = Expert(id="e", name="NoMatch", skills={"python": 5}, capacity=1, rating=5)
    r = Request(id="r", title="mobile", required_skills={"flutter": 4})
    res = assign_all([r], [e])
    assert res.matched == 0
    assert "flutter" in res.unassigned[0].reason.lower()


def test_strict_preferred_over_partial() -> None:
    """Заявка со строгим кандидатом не уходит в частичный режим."""
    strict = Expert(id="s", name="Full", skills={"python": 5}, capacity=1, rating=4)
    res = assign_all(
        [Request(id="r", title="t", required_skills={"python": 4})], [strict]
    )
    assert res.assignments[0].partial is False
