from __future__ import annotations

from dataclasses import dataclass

from agents.genome import Genome
from evaluation.metrics import Objectives


@dataclass(frozen=True)
class Offspring:
    genome: Genome
    parent_ids: tuple[str, ...]


def make_next_generation(
    parents: list[tuple[str, Genome, Objectives]],
    rng,
    population_size: int,
    elite_count: int,
    mutation_rate: float,
) -> tuple[list[tuple[str, Genome]], dict[str, tuple[str, ...]]]:
    if population_size <= 0:
        return [], {}

    elite_count = max(0, min(elite_count, population_size))
    parents_sorted = sorted(parents, key=lambda p: _scalar_score(p[2]), reverse=True)

    next_gen: list[tuple[str, Genome]] = []
    lineage: dict[str, tuple[str, ...]] = {}

    for agent_id, genome, _ in parents_sorted[:elite_count]:
        child_id = _new_id(rng)
        next_gen.append((child_id, genome))
        lineage[child_id] = (agent_id,)

    while len(next_gen) < population_size:
        p_id, p_genome, _ = _tournament(parents_sorted, rng)
        child_id = _new_id(rng)
        child_genome = p_genome.mutate(rng, mutation_rate)
        next_gen.append((child_id, child_genome))
        lineage[child_id] = (p_id,)

    return next_gen, lineage


def _scalar_score(obj: Objectives) -> float:
    return (
        1.0 * obj.task
        + 0.7 * obj.novelty
        + 0.6 * obj.stability
        + 0.4 * obj.efficiency
    )


def _tournament(
    parents_sorted: list[tuple[str, Genome, Objectives]],
    rng,
    k: int = 3,
) -> tuple[str, Genome, Objectives]:
    k = max(1, min(k, len(parents_sorted)))
    candidates = [rng.choice(parents_sorted) for _ in range(k)]
    return max(candidates, key=lambda p: _scalar_score(p[2]))


def _new_id(rng) -> str:
    return f"a{rng.getrandbits(48):012x}"
