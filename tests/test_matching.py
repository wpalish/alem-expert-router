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
