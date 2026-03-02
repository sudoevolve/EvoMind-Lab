from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


class Environment(Protocol):
    def reset(self) -> dict: ...

    def step(self, action: str) -> StepResult: ...
