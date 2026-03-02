from __future__ import annotations

from dataclasses import dataclass

from evaluation.metrics import Objectives


@dataclass(frozen=True)
class RankedIndex:
    index: int
    rank: int
    crowding: float


def select_survivors(objectives: list[Objectives], survivors: int) -> list[int]:
    if survivors <= 0:
        return []
    if survivors >= len(objectives):
        return list(range(len(objectives)))

    fronts = _fast_non_dominated_sort(objectives)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= survivors:
            selected.extend(front)
            continue

        remaining = survivors - len(selected)
        crowding = _crowding_distance(front, objectives)
        crowding_sorted = sorted(
            front, key=lambda i: crowding.get(i, 0.0), reverse=True
        )
        selected.extend(crowding_sorted[:remaining])
        break
    return selected


def _dominates(a: Objectives, b: Objectives) -> bool:
    a_vals = (a.task, a.novelty, a.stability, a.efficiency)
    b_vals = (b.task, b.novelty, b.stability, b.efficiency)
    ge = all(x >= y for x, y in zip(a_vals, b_vals))
    g = any(x > y for x, y in zip(a_vals, b_vals))
    return ge and g


def _fast_non_dominated_sort(objectives: list[Objectives]) -> list[list[int]]:
    n = len(objectives)
    dominates: list[list[int]] = [[] for _ in range(n)]
    dominated_count = [0] * n

    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _crowding_distance(front: list[int], objectives: list[Objectives]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {i: float("inf") for i in front}

    dist = {i: 0.0 for i in front}
    keys = ["task", "novelty", "stability", "efficiency"]
    for k in keys:
        sorted_idx = sorted(front, key=lambda i: getattr(objectives[i], k))
        dist[sorted_idx[0]] = float("inf")
        dist[sorted_idx[-1]] = float("inf")

        min_v = getattr(objectives[sorted_idx[0]], k)
        max_v = getattr(objectives[sorted_idx[-1]], k)
        span = max(1e-12, max_v - min_v)
        for j in range(1, len(sorted_idx) - 1):
            prev_v = getattr(objectives[sorted_idx[j - 1]], k)
            next_v = getattr(objectives[sorted_idx[j + 1]], k)
            dist[sorted_idx[j]] += (next_v - prev_v) / span
    return dist
