from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
root_s = str(root)
if root_s not in sys.path:
    sys.path.insert(0, root_s)

from experiment.config import ExperimentConfig
from experiment.core import run


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evomind-lab")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--survivors", type=int, default=12)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--mutation", type=float, default=0.25)
    p.add_argument("--env", type=str, default="bandit", choices=["bandit", "market"])
    p.add_argument("--arms", type=int, default=8)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--start-price", type=float, default=100.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--fee-bps", type=float, default=2.0)
    p.add_argument("--out", type=str, default="logs/run")
    args = p.parse_args(argv)

    cfg = ExperimentConfig(
        seed=args.seed,
        generations=args.generations,
        population_size=args.population,
        survivors=args.survivors,
        elite_count=args.elite,
        mutation_rate=args.mutation,
        env=args.env,
        arms=args.arms,
        horizon=args.horizon,
        allow_stop=not args.no_stop,
        initial_cash=args.initial_cash,
        start_price=args.start_price,
        mu=args.mu,
        sigma=args.sigma,
        fee_bps=args.fee_bps,
        output_dir=args.out,
    )
    run_dir = run(cfg)
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
