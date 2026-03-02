from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import StepResult


@dataclass
class BanditEnv:
    rng: Any
    arms: int = 8
    horizon: int = 32
    noise_std: float = 0.05
    drift_std: float = 0.01
    allow_stop: bool = True

    def __post_init__(self) -> None:
        self._t = 0
        self._means: list[float] = []
        self._action_space = [str(i) for i in range(self.arms)]
        if self.allow_stop:
            self._action_space.append("stop")

    def reset(self) -> dict:
        self._t = 0
        self._means = [self.rng.random() for _ in range(self.arms)]
        return {"t": self._t, "action_space": self._action_space, "hint": "bandit"}

    def step(self, action: str) -> StepResult:
        if self.allow_stop and action == "stop":
            self._t += 1
            obs = {"t": self._t, "action_space": self._action_space, "hint": "bandit"}
            return StepResult(
                observation=obs, reward=0.0, done=True, info={"stopped": True}
            )

        a = int(action) if action.isdigit() else 0
        a = max(0, min(self.arms - 1, a))

        for i in range(self.arms):
            self._means[i] = float(
                min(1.0, max(0.0, self._means[i] + self.rng.gauss(0, self.drift_std)))
            )

        mean = self._means[a]
        reward = float(mean + self.rng.gauss(0, self.noise_std))
        reward = float(min(1.0, max(0.0, reward)))

        self._t += 1
        done = self._t >= self.horizon
        obs = {"t": self._t, "action_space": self._action_space, "hint": "bandit"}
        info = {"mean": mean}
        return StepResult(observation=obs, reward=reward, done=done, info=info)
