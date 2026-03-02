from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Objectives:
    task: float
    novelty: float
    stability: float
    efficiency: float

    def to_dict(self) -> dict[str, float]:
        return {
            "task": float(self.task),
            "novelty": float(self.novelty),
            "stability": float(self.stability),
            "efficiency": float(self.efficiency),
        }


def evaluate_population(
    episode_summaries: list[dict],
    k_novelty: int = 5,
) -> list[Objectives]:
    sequences = [s.get("action_sequence", []) for s in episode_summaries]
    tasks = [float(s.get("task", 0.0)) for s in episode_summaries]
    stabilities = [float(s.get("stability", 0.0)) for s in episode_summaries]
    efficiencies = [float(s.get("efficiency", 0.0)) for s in episode_summaries]

    novelty_scores = _novelty_scores(sequences, k=k_novelty)
    return [
        Objectives(task=t, novelty=n, stability=st, efficiency=eff)
        for t, n, st, eff in zip(tasks, novelty_scores, stabilities, efficiencies)
    ]


def _novelty_scores(sequences: list[list[str]], k: int) -> list[float]:
    if not sequences:
        return []

    n = len(sequences)
    dists: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _normalized_edit_distance(sequences[i], sequences[j])
            dists[i][j] = d
            dists[j][i] = d

    scores: list[float] = []
    kk = max(1, min(k, n - 1)) if n > 1 else 1
    for i in range(n):
        if n == 1:
            scores.append(0.0)
            continue
        sorted_d = sorted(dists[i][j] for j in range(n) if j != i)
        scores.append(sum(sorted_d[:kk]) / kk)
    return scores


def _normalized_edit_distance(a: list[str], b: list[str]) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0

    la = len(a)
    lb = len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    dist = prev[lb]
    return float(dist / max(la, lb))
