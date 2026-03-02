from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
root_s = str(root)
if root_s not in sys.path:
    sys.path.insert(0, root_s)

from dataclasses import asdict

from experiment.config import ExperimentConfig
from experiment.core import run
from experiment.project import load_project_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evomind-lab", argument_default=argparse.SUPPRESS)
    p.add_argument("--project", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--population", type=int)
    p.add_argument("--survivors", type=int)
    p.add_argument("--elite", type=int)
    p.add_argument("--mutation", type=float)
    p.add_argument("--env", type=str, choices=["bandit", "market", "csv_market"])
    p.add_argument("--arms", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--initial-cash", type=float)
    p.add_argument("--start-price", type=float)
    p.add_argument("--mu", type=float)
    p.add_argument("--sigma", type=float)
    p.add_argument("--fee-bps", type=float)
    p.add_argument("--slippage-bps", type=float)
    p.add_argument("--data", type=str)
    p.add_argument("--out", type=str)
    args = p.parse_args(argv)

    cli = vars(args)
    project_cfg = {}
    if "project" in cli and cli["project"]:
        project_cfg = load_project_config(cli["project"])

    base = asdict(ExperimentConfig())
    base.update(dict(project_cfg.get("experiment", {})))

    if "seed" in cli:
        base["seed"] = int(cli["seed"])
    if "generations" in cli:
        base["generations"] = int(cli["generations"])
    if "population" in cli:
        base["population_size"] = int(cli["population"])
    if "survivors" in cli:
        base["survivors"] = int(cli["survivors"])
    if "elite" in cli:
        base["elite_count"] = int(cli["elite"])
    if "mutation" in cli:
        base["mutation_rate"] = float(cli["mutation"])
    if "env" in cli:
        base["env"] = str(cli["env"])
    if "arms" in cli:
        base["arms"] = int(cli["arms"])
    if "horizon" in cli:
        base["horizon"] = int(cli["horizon"])
    if "no_stop" in cli:
        base["allow_stop"] = not bool(cli["no_stop"])
    if "initial_cash" in cli:
        base["initial_cash"] = float(cli["initial_cash"])
    if "start_price" in cli:
        base["start_price"] = float(cli["start_price"])
    if "mu" in cli:
        base["mu"] = float(cli["mu"])
    if "sigma" in cli:
        base["sigma"] = float(cli["sigma"])
    if "fee_bps" in cli:
        base["fee_bps"] = float(cli["fee_bps"])
    if "slippage_bps" in cli:
        base["slippage_bps"] = float(cli["slippage_bps"])
    if "data" in cli:
        base["data_path"] = str(cli["data"])
    if "out" in cli:
        base["output_dir"] = str(cli["out"])

    cfg = ExperimentConfig(**base)
    run_dir = run(cfg)
    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
