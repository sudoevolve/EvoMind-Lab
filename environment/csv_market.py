from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import StepResult


@dataclass
class CsvMarketEnv:
    rng: Any
    data_path: str
    horizon: int = 720
    initial_cash: float = 10_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 0.0
    history_window: int = 256
    allow_stop: bool = True

    def __post_init__(self) -> None:
        self._rows = _read_ohlcv(self.data_path)
        if len(self._rows) < 2:
            raise ValueError("data_path must contain at least 2 rows")
        self._action_space = [
            "target_0",
            "target_25",
            "target_50",
            "target_75",
            "target_100",
        ]
        if self.allow_stop:
            self._action_space.append("stop")
        self._t = 0
        self._i = 0
        self._cash = float(self.initial_cash)
        self._position = 0.0
        self._peak_equity = float(self.initial_cash)
        self._max_drawdown = 0.0
        self._last_return = 0.0

    def reset(self) -> dict:
        self._t = 0
        self._i = 0
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
            info = self._info(trade_value=0.0, fee=0.0)
            return StepResult(observation=obs, reward=0.0, done=True, info=info)

        price = self._price()
        equity_before = self._equity(price)

        frac = _parse_target_fraction(action)
        target_value = equity_before * frac
        current_value = self._position * price
        delta_value = target_value - current_value

        trade_value = float(abs(delta_value))
        cost_bps = float(self.fee_bps + self.slippage_bps)
        fee = float(trade_value * (cost_bps / 10_000.0))

        if delta_value > 0:
            spend = delta_value + fee
            if spend > self._cash:
                spend = self._cash
                if spend <= 0:
                    delta_value = 0.0
                    trade_value = 0.0
                    fee = 0.0
                else:
                    trade_value = float(spend / (1.0 + cost_bps / 10_000.0))
                    fee = float(spend - trade_value)
                    delta_value = trade_value
            self._cash -= float(delta_value + fee)
            self._position += float(delta_value / price) if price > 0 else 0.0
        elif delta_value < 0:
            sell_value = float(min(-delta_value, current_value))
            trade_value = float(sell_value)
            fee = float(trade_value * (cost_bps / 10_000.0))
            self._cash += float(trade_value - fee)
            self._position -= float(trade_value / price) if price > 0 else 0.0
        else:
            trade_value = 0.0
            fee = 0.0

        self._advance()
        next_price = self._price()
        equity_after = self._equity(next_price)
        reward = float((equity_after - equity_before) / max(1e-9, self.initial_cash))
        self._last_return = float((equity_after / max(1e-9, equity_before)) - 1.0)

        self._peak_equity = float(max(self._peak_equity, equity_after))
        dd = 1.0 - (equity_after / max(1e-9, self._peak_equity))
        self._max_drawdown = float(max(self._max_drawdown, dd))

        self._t += 1
        done = self._t >= self.horizon or self._i >= (len(self._rows) - 1)
        obs = self._observation()
        info = self._info(trade_value=trade_value, fee=fee)
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _advance(self) -> None:
        self._i = min(self._i + 1, len(self._rows) - 1)

    def _price(self) -> float:
        return float(self._rows[self._i]["close"])

    def _equity(self, price: float) -> float:
        return float(self._cash + self._position * price)

    def _info(self, trade_value: float, fee: float) -> dict:
        price = self._price()
        return {
            "price": float(price),
            "cash": float(self._cash),
            "position": float(self._position),
            "equity": float(self._equity(price)),
            "trade_value": float(trade_value),
            "fee": float(fee),
            "max_drawdown": float(self._max_drawdown),
        }

    def _observation(self) -> dict:
        start = max(0, self._i - self.history_window + 1)
        closes = [float(r["close"]) for r in self._rows[start : self._i + 1]]
        price = self._price()
        return {
            "t": int(self._t),
            "hint": "market",
            "action_space": list(self._action_space),
            "price": float(price),
            "cash": float(self._cash),
            "position": float(self._position),
            "equity": float(self._equity(price)),
            "last_return": float(self._last_return),
            "max_drawdown": float(self._max_drawdown),
            "initial_cash": float(self.initial_cash),
            "close_history": closes,
        }


def _parse_target_fraction(action: str) -> float:
    if action.startswith("target_"):
        try:
            p = int(action.split("_", 1)[1])
        except Exception:
            p = 0
        return float(min(1.0, max(0.0, p / 100.0)))
    return 0.0


def _read_ohlcv(data_path: str) -> list[dict[str, float]]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows: list[dict[str, float]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        first = f.readline()
        if not first:
            return rows
        f.seek(0)

        header_like = ("close" in first.lower()) or any(c.isalpha() for c in first)
        if header_like:
            reader = csv.DictReader(f)
            for r in reader:
                close = _as_float(r.get("close"))
                if close is None:
                    continue
                rows.append({"close": float(close)})
            return rows

        reader2 = csv.reader(f)
        for r in reader2:
            if not r:
                continue
            close = _as_float(r[4]) if len(r) > 4 else None
            if close is None:
                continue
            rows.append({"close": float(close)})
    return rows


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None
