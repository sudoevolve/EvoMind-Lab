from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    generations: int = 30
    population_size: int = 24
    survivors: int = 12
    elite_count: int = 3
    mutation_rate: float = 0.25

    env: str = "bandit"

    arms: int = 8
    horizon: int = 32
    allow_stop: bool = True

    initial_cash: float = 10_000.0
    start_price: float = 100.0
    mu: float = 0.0
    sigma: float = 0.02
    fee_bps: float = 2.0
    slippage_bps: float = 0.0
    data_path: str = ""
    genome_overrides: dict[str, Any] = field(default_factory=dict)

    output_dir: str = "logs/run"
