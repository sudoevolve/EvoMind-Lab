from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Genome:
    prompt_template: str = "You are an agent. Think step by step."
    memory_write_strategy: str = "append"
    decision_strategy: str = "epsilon_greedy"
    biases: dict[str, float] = field(
        default_factory=lambda: {"exploration": 0.2, "risk_aversion": 0.0}
    )
    memory_short_term_limit: int = 32
    memory_long_term_limit: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_template": self.prompt_template,
            "memory_write_strategy": self.memory_write_strategy,
            "decision_strategy": self.decision_strategy,
            "biases": dict(self.biases),
            "memory_short_term_limit": self.memory_short_term_limit,
            "memory_long_term_limit": self.memory_long_term_limit,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Genome:
        return Genome(
            prompt_template=str(data.get("prompt_template", "You are an agent.")),
            memory_write_strategy=str(data.get("memory_write_strategy", "append")),
            decision_strategy=str(data.get("decision_strategy", "epsilon_greedy")),
            biases=dict(data.get("biases", {"exploration": 0.2, "risk_aversion": 0.0})),
            memory_short_term_limit=int(data.get("memory_short_term_limit", 32)),
            memory_long_term_limit=int(data.get("memory_long_term_limit", 64)),
        )

    def mutate(self, rng, mutation_rate: float) -> Genome:
        prompt_template = self.prompt_template
        memory_write_strategy = self.memory_write_strategy
        decision_strategy = self.decision_strategy
        biases = dict(self.biases)
        memory_short_term_limit = self.memory_short_term_limit
        memory_long_term_limit = self.memory_long_term_limit

        if rng.random() < mutation_rate:
            choices = [
                "You are an agent. Prefer novelty.",
                "You are an agent. Prefer stability.",
                "You are an agent. Prefer efficiency.",
                "You are an agent. Explore then exploit.",
            ]
            prompt_template = rng.choice(choices)

        if rng.random() < mutation_rate:
            memory_write_strategy = rng.choice(["append", "compress", "forgetful"])

        if rng.random() < mutation_rate:
            decision_strategy = rng.choice(
                ["epsilon_greedy", "softmax", "ucb1", "random"]
            )

        if rng.random() < mutation_rate:
            biases["exploration"] = float(
                min(1.0, max(0.0, biases.get("exploration", 0.2) + rng.gauss(0, 0.08)))
            )

        if rng.random() < mutation_rate:
            biases["risk_aversion"] = float(
                min(
                    1.0,
                    max(-1.0, biases.get("risk_aversion", 0.0) + rng.gauss(0, 0.08)),
                )
            )

        if rng.random() < mutation_rate:
            memory_short_term_limit = int(
                min(128, max(8, memory_short_term_limit + int(rng.gauss(0, 6))))
            )

        if rng.random() < mutation_rate:
            memory_long_term_limit = int(
                min(256, max(16, memory_long_term_limit + int(rng.gauss(0, 10))))
            )

        return Genome(
            prompt_template=prompt_template,
            memory_write_strategy=memory_write_strategy,
            decision_strategy=decision_strategy,
            biases=biases,
            memory_short_term_limit=memory_short_term_limit,
            memory_long_term_limit=memory_long_term_limit,
        )
