"""Юнит-тесты решателя задачи о назначениях (венгерский / scipy)."""
from __future__ import annotations

from app.assignment_solver import solve_assignment


def test_simple_optimum() -> None:
    # минимизация: оптимум — диагональ (0+0=0)
    cost = [[0.0, 9.0], [9.0, 0.0]]
    assert solve_assignment(cost) == [0, 1]


def test_picks_cheaper_crossing() -> None:
    cost = [[4.0, 1.0], [2.0, 3.0]]  # оптимум: row0->col1(1), row1->col0(2) = 3
    assert solve_assignment(cost) == [1, 0]


def test_rectangular_more_cols_than_rows() -> None:
    cost = [[5.0, 1.0, 9.0]]
    assert solve_assignment(cost) == [1]


def test_rectangular_more_rows_than_cols_leaves_unassigned() -> None:
    cost = [[1.0], [2.0]]
    res = solve_assignment(cost)
    assert sorted(res) == [-1, 0]  # один остаётся без столбца


def test_empty() -> None:
    assert solve_assignment([]) == []
