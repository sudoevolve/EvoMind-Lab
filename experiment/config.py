 from __future__ import annotations
 
 from dataclasses import dataclass
 
 
 @dataclass(frozen=True)
 class ExperimentConfig:
     seed: int = 0
     generations: int = 30
     population_size: int = 24
     survivors: int = 12
     elite_count: int = 3
     mutation_rate: float = 0.25
 
     arms: int = 8
     horizon: int = 32
     allow_stop: bool = True
 
     output_dir: str = "logs/run"
