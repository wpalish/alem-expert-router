"""Решатель задачи о назначениях (assignment problem).

Жадный подбор «лучшего на каждую заявку по очереди» НЕ даёт глобального
оптимума: локально лучший выбор для одной заявки может оставить другую без
исполнителя. Здесь мы ищем глобально оптимальное распределение —
максимум суммарного соответствия по всем парам (заявка, исполнитель) сразу.

Реализован венгерский алгоритм (Кун–Манкрес) на чистом Python — без внешних
зависимостей. Эксперт с ёмкостью N представляется N «слотами», поэтому один
исполнитель может вести несколько заявок одновременно (в пределах ёмкости).

Если установлен SciPy — используется его проверенный
`linear_sum_assignment` (быстрее и надёжнее на больших матрицах).
"""
from __future__ import annotations

from typing import Callable, Optional

try:  # необязательная зависимость — ускоряет и валидирует решение
    from scipy.optimize import linear_sum_assignment as _scipy_lsa  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy опционален
    _HAS_SCIPY = False


def solve_assignment(
    cost: list[list[float]],
) -> list[int]:
    """Минимизировать суммарную стоимость назначения.

    `cost[i][j]` — стоимость назначения строки i на столбец j.
    Возвращает список длиной = числу строк: col = result[row], либо -1,
    если строка не назначена (когда столбцов меньше, чем строк).
    Матрица может быть прямоугольной.
    """
    n_rows = len(cost)
    if n_rows == 0:
        return []
    n_cols = len(cost[0]) if cost[0] else 0
    if n_cols == 0:
        return [-1] * n_rows

    if _HAS_SCIPY:
        rows, cols = _scipy_lsa(cost)
        out = [-1] * n_rows
        for r, c in zip(rows.tolist(), cols.tolist()):
            out[r] = c
        return out

    return _hungarian(cost)


def _hungarian(cost: list[list[float]]) -> list[int]:
    """Венгерский алгоритм O(n^3) для прямоугольной матрицы (минимизация).

    Дополняем матрицу до квадратной большими значениями (BIG), решаем,
    затем фиктивные назначения помечаем как -1.
    """
    n_rows = len(cost)
    n_cols = len(cost[0])
    n = max(n_rows, n_cols)

    flat = [v for row in cost for v in row]
    big = (max(flat) if flat else 0.0) + 1.0

    # квадратная матрица с паддингом
    a = [[big] * n for _ in range(n)]
    for i in range(n_rows):
        for j in range(n_cols):
            a[i][j] = cost[i][j]

    # Реализация Кёнига через потенциалы (1-индексация для классики).
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = строка, назначенная на столбец j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    # p[j] = строка для столбца j -> развернём в row -> col
    result = [-1] * n_rows
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= n_rows and (j - 1) < n_cols:
            result[i - 1] = j - 1
    return result
