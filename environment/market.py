from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import StepResult


@dataclass
class MarketEnv:
    rng: Any
    horizon: int = 256
    initial_cash: float = 10_000.0
    start_price: float = 100.0
    mu: float = 0.0
    sigma: float = 0.02
    fee_bps: float = 2.0
    allow_stop: bool = True

    def __post_init__(self) -> None:
        self._t = 0
        self._price = float(self.start_price)
        self._cash = float(self.initial_cash)
        self._position = 0.0
        self._peak_equity = float(self.initial_cash)
        self._max_drawdown = 0.0
        self._last_return = 0.0
        self._action_space = [
            "target_0",
            "target_25",
            "target_50",
            "target_75",
            "target_100",
        ]
        if self.allow_stop:
            self._action_space.append("stop")

    @property
    def price(self) -> float:
        return float(self._price)

    @property
    def cash(self) -> float:
        return float(self._cash)

    @property
    def position(self) -> float:
        return float(self._position)

    @property
    def equity(self) -> float:
        return float(self._cash + self._position * self._price)

    @property
    def max_drawdown(self) -> float:
        return float(self._max_drawdown)

    def reset(self) -> dict:
        self._t = 0
        self._price = float(self.start_price)
        self._cash = float(self.initial_cash)
        self._position = 0.0
        self._peak_equity = float(self.initial_cash)
        self._max_drawdown = 0.0
        self._last_return = 0.0
        return self._observation()

    def step(self, action: str) -> StepResult:
        if self.allow_stop and action == "stop":
            self._t += 1
            obs = self._observation()
            info = {
                "price": self.price,
                "cash": self.cash,
                "position": self.position,
                "equity": self.equity,
                "trade_value": 0.0,
                "fee": 0.0,
                "max_drawdown": self.max_drawdown,
            }
            return StepResult(observation=obs, reward=0.0, done=True, info=info)

        equity_before = self.equity

        frac = _parse_target_fraction(action)
        target_value = equity_before * frac
        current_value = self._position * self._price
        delta_value = target_value - current_value

        trade_value = float(abs(delta_value))
        fee = float(trade_value * (self.fee_bps / 10_000.0))

        if delta_value > 0:
            spend = delta_value + fee
            if spend > self._cash:
                spend = self._cash
                if spend <= 0:
                    delta_value = 0.0
                    trade_value = 0.0
                    fee = 0.0
                else:
                    trade_value = float(spend / (1.0 + self.fee_bps / 10_000.0))
                    fee = float(spend - trade_value)
                    delta_value = trade_value
            self._cash -= float(delta_value + fee)
            self._position += float(delta_value / self._price) if self._price > 0 else 0.0
        elif delta_value < 0:
            sell_value = float(min(-delta_value, current_value))
            trade_value = float(sell_value)
            fee = float(trade_value * (self.fee_bps / 10_000.0))
            self._cash += float(trade_value - fee)
            self._position -= float(trade_value / self._price) if self._price > 0 else 0.0
        else:
            trade_value = 0.0
            fee = 0.0

        self._price = float(self._price * _price_step(self.rng, self.mu, self.sigma))
        if self._price <= 1e-9:
            self._price = 1e-9

        equity_after = self.equity
        reward = float((equity_after - equity_before) / max(1e-9, self.initial_cash))
        self._last_return = float((equity_after / max(1e-9, equity_before)) - 1.0)

        self._peak_equity = float(max(self._peak_equity, equity_after))
        dd = 1.0 - (equity_after / max(1e-9, self._peak_equity))
        self._max_drawdown = float(max(self._max_drawdown, dd))

        self._t += 1
        done = self._t >= self.horizon
        obs = self._observation()
        info = {
            "price": self.price,
            "cash": self.cash,
            "position": self.position,
            "equity": self.equity,
            "trade_value": float(trade_value),
            "fee": float(fee),
            "max_drawdown": self.max_drawdown,
        }
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _observation(self) -> dict:
        return {
            "t": int(self._t),
            "hint": "market",
            "action_space": list(self._action_space),
            "price": float(self.price),
            "cash": float(self.cash),
            "position": float(self.position),
            "equity": float(self.equity),
            "last_return": float(self._last_return),
            "max_drawdown": float(self.max_drawdown),
            "initial_cash": float(self.initial_cash),
        }


def _parse_target_fraction(action: str) -> float:
    if action.startswith("target_"):
        try:
            p = int(action.split("_", 1)[1])
        except Exception:
            p = 0
        return float(min(1.0, max(0.0, p / 100.0)))
    return 0.0


def _price_step(rng: Any, mu: float, sigma: float) -> float:
    z = float(rng.gauss(0.0, 1.0))
    log_r = (mu - 0.5 * sigma * sigma) + sigma * z
    return float(pow(2.718281828459045, log_r))
