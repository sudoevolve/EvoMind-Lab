 from __future__ import annotations
 
 import argparse
 
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
     p.add_argument("--arms", type=int, default=8)
     p.add_argument("--horizon", type=int, default=32)
     p.add_argument("--no-stop", action="store_true")
     p.add_argument("--out", type=str, default="logs/run")
     args = p.parse_args(argv)
 
     cfg = ExperimentConfig(
         seed=args.seed,
         generations=args.generations,
         population_size=args.population,
         survivors=args.survivors,
         elite_count=args.elite,
         mutation_rate=args.mutation,
         arms=args.arms,
         horizon=args.horizon,
         allow_stop=not args.no_stop,
         output_dir=args.out,
     )
     run_dir = run(cfg)
     print(str(run_dir))
     return 0
 
 
 if __name__ == "__main__":
     raise SystemExit(main())
