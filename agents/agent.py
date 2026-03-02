from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any
from urllib import request

from .genome import Genome
from .memory import Memory


@dataclass
class Agent:
    agent_id: str
    genome: Genome
    rng: Any
    memory: Memory = field(default_factory=Memory)
    behavior_history: list[dict[str, Any]] = field(default_factory=list)
    objectives: dict[str, float] = field(default_factory=dict)
    action_value: dict[str, float] = field(default_factory=dict)
    action_count: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def create(agent_id: str, genome: Genome, rng) -> Agent:
        memory = Memory(
            short_term_limit=genome.memory_short_term_limit,
            long_term_limit=genome.memory_long_term_limit,
            write_strategy=genome.memory_write_strategy,
        )
        return Agent(agent_id=agent_id, genome=genome, rng=rng, memory=memory)

    def reset_episode(self) -> None:
        self.behavior_history.clear()
        self.objectives.clear()
        self.action_value.clear()
        self.action_count.clear()
        self.memory.short_term.clear()

    def act(self, observation: dict[str, Any]) -> str:
        action_space = list(map(str, observation.get("action_space", [])))
        if not action_space:
            return ""

        strategy = os.getenv("EVOMIND_FORCE_STRATEGY") or self.genome.decision_strategy
        if strategy == "ollama":
            return self._act_ollama(observation=observation, action_space=action_space)

        if strategy == "rule":
            return self._act_rule(observation=observation, action_space=action_space)

        if strategy == "random":
            return self.rng.choice(action_space)

        if strategy == "softmax":
            return self._act_softmax(action_space)

        if strategy == "ucb1":
            return self._act_ucb1(action_space)

        return self._act_epsilon_greedy(action_space)

    def observe_transition(
        self,
        observation: dict[str, Any],
        action: str,
        reward: float,
        next_observation: dict[str, Any],
        done: bool,
    ) -> None:
        self._update_estimates(action, reward)
        self.behavior_history.append(
            {
                "t": int(observation.get("t", len(self.behavior_history))),
                "action": action,
                "reward": float(reward),
                "done": bool(done),
            }
        )
        self.memory.observe(f"a={action} r={reward:.3f}")
        if done:
            self.memory.consolidate(self.rng)

    def _update_estimates(self, action: str, reward: float) -> None:
        n = self.action_count.get(action, 0) + 1
        self.action_count[action] = n
        old = self.action_value.get(action, 0.0)
        self.action_value[action] = old + (reward - old) / n

    def _act_epsilon_greedy(self, action_space: list[str]) -> str:
        eps = float(self.genome.biases.get("exploration", 0.2))
        if self.rng.random() < eps:
            return self.rng.choice(action_space)

        best_action = action_space[0]
        best_value = -float("inf")
        for a in action_space:
            v = self.action_value.get(a, 0.0)
            if v > best_value:
                best_value = v
                best_action = a
        return best_action

    def _act_softmax(self, action_space: list[str]) -> str:
        temp = 0.4 + 1.6 * float(self.genome.biases.get("exploration", 0.2))
        values = [self.action_value.get(a, 0.0) for a in action_space]
        m = max(values)
        exps = [math.exp((v - m) / max(1e-6, temp)) for v in values]
        s = sum(exps)
        r = self.rng.random() * s
        acc = 0.0
        for a, e in zip(action_space, exps):
            acc += e
            if acc >= r:
                return a
        return action_space[-1]

    def _act_ucb1(self, action_space: list[str]) -> str:
        total = sum(self.action_count.get(a, 0) for a in action_space) + 1
        c = 0.5 + 2.0 * float(self.genome.biases.get("exploration", 0.2))

        best_action = action_space[0]
        best_score = -float("inf")
        for a in action_space:
            n = self.action_count.get(a, 0)
            if n == 0:
                return a
            avg = self.action_value.get(a, 0.0)
            score = avg + c * math.sqrt(math.log(total) / n)
            if score > best_score:
                best_score = score
                best_action = a
        return best_action

    def _act_ollama(self, observation: dict[str, Any], action_space: list[str]) -> str:
        model = os.getenv("EVOMIND_OLLAMA_MODEL") or "qwen3:8b"
        base_url = os.getenv("EVOMIND_OLLAMA_URL") or "http://localhost:11434"
        timeout_s = float(os.getenv("EVOMIND_OLLAMA_TIMEOUT") or "10")

        exploration = float(self.genome.biases.get("exploration", 0.2))
        temperature = float(min(1.2, max(0.0, 0.2 + 0.8 * exploration)))

        t = observation.get("t", len(self.behavior_history))
        stats_lines: list[str] = []
        for a in action_space:
            v = float(self.action_value.get(a, 0.0))
            n = int(self.action_count.get(a, 0))
            stats_lines.append(f"{a}\tvalue={v:.4f}\tcount={n}")
        recent = self.memory.short_term[-6:]
        long_term = self.memory.long_term[-6:]

        user = "\n".join(
            [
                f"t={t}",
                f"Allowed actions: {', '.join(action_space)}",
                "Return exactly one allowed action. No explanation.",
                "Estimates:",
                *stats_lines,
                "Recent memory:",
                *([*recent] if recent else ["(none)"]),
                "Long-term memory:",
                *([*long_term] if long_term else ["(none)"]),
            ]
        )

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.genome.prompt_template},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": 32},
        }

        try:
            req = request.Request(
                url=f"{base_url.rstrip('/')}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return self.rng.choice(action_space)

        content = ""
        if isinstance(data, dict):
            msg = data.get("message")
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
            else:
                content = str(data.get("response", ""))

        stripped = content.strip()
        if not stripped:
            return self.rng.choice(action_space)
        text = stripped.splitlines()[0].strip().strip("\"'`")
        if text in action_space:
            return text

        for a in action_space:
            if a == text:
                return a
        for a in action_space:
            if a in text:
                return a

        return self.rng.choice(action_space)

    def _act_rule(self, observation: dict[str, Any], action_space: list[str]) -> str:
        closes = observation.get("close_history", [])
        if not isinstance(closes, list) or len(closes) < 4:
            return self.rng.choice(action_space)

        template = str(getattr(self.genome, "trading_template", "ma_cross"))
        params = getattr(self.genome, "trading_params", {})
        if not isinstance(params, dict):
            params = {}

        desired = "target_0"
        if template == "breakout":
            lookback = int(float(params.get("lookback", 48)))
            threshold = float(params.get("threshold", 0.002))
            if len(closes) >= lookback + 2:
                prev = closes[-lookback - 1 : -1]
                prev_max = max(prev) if prev else closes[-2]
                if float(closes[-1]) > float(prev_max) * (1.0 + threshold):
                    desired = "target_100"
        elif template == "mean_reversion":
            z_window = int(float(params.get("z_window", 48)))
            z_entry = float(params.get("z_entry", 1.0))
            if len(closes) >= z_window:
                window = [float(x) for x in closes[-z_window:]]
                mu = sum(window) / max(1, len(window))
                var = sum((x - mu) ** 2 for x in window) / max(1, len(window))
                sd = var ** 0.5
                z = 0.0 if sd <= 1e-12 else (float(closes[-1]) - mu) / sd
                if z <= -z_entry:
                    desired = "target_100"
                elif z >= z_entry:
                    desired = "target_0"
                else:
                    desired = "target_50"
        else:
            fast = int(float(params.get("fast", 12)))
            slow = int(float(params.get("slow", 48)))
            if slow <= fast:
                slow = fast + 4
            if len(closes) >= slow:
                fast_ma = sum(float(x) for x in closes[-fast:]) / max(1, fast)
                slow_ma = sum(float(x) for x in closes[-slow:]) / max(1, slow)
                desired = "target_100" if fast_ma > slow_ma else "target_0"

        risk = float(self.genome.biases.get("risk_aversion", 0.0))
        risk = float(min(1.0, max(0.0, risk)))
        max_frac = 1.0 - 0.75 * risk
        desired_frac = 0.0
        if desired.startswith("target_"):
            try:
                desired_frac = int(desired.split("_", 1)[1]) / 100.0
            except Exception:
                desired_frac = 0.0
        clipped = min(desired_frac, max_frac)
        if clipped <= 0.125:
            desired = "target_0"
        elif clipped <= 0.375:
            desired = "target_25"
        elif clipped <= 0.625:
            desired = "target_50"
        elif clipped <= 0.875:
            desired = "target_75"
        else:
            desired = "target_100"

        if desired in action_space:
            return desired
        for a in action_space:
            if a.startswith("target_"):
                return a
        return self.rng.choice(action_space)
