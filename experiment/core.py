from __future__ import annotations

import random
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from agents.agent import Agent
from agents.genome import Genome
from environment.bandit import BanditEnv
from environment.csv_market import CsvMarketEnv
from environment.market import MarketEnv
from evaluation.metrics import evaluate_population
from evolution.reproduction import make_next_generation
from evolution.selection import select_survivors
from experiment.config import ExperimentConfig
from experiment.io import append_jsonl, ensure_dir, write_json
from visualization.ascii import ascii_sparkline


def run(cfg: ExperimentConfig) -> Path:
    rng = random.Random(cfg.seed)
    run_dir = _make_run_dir(cfg.output_dir)
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", asdict(cfg))

    population = _init_population(cfg, rng)
    history_mean_task: list[float] = []
    history_mean_novelty: list[float] = []
    if not population or cfg.generations <= 0:
        return run_dir

    for gen in range(cfg.generations):
        episode_summaries: list[dict] = []
        for agent in population:
            env_rng = random.Random(rng.getrandbits(64))
            if cfg.env == "csv_market":
                env = CsvMarketEnv(
                    rng=env_rng,
                    data_path=cfg.data_path,
                    horizon=cfg.horizon,
                    initial_cash=cfg.initial_cash,
                    fee_bps=cfg.fee_bps,
                    slippage_bps=cfg.slippage_bps,
                    allow_stop=cfg.allow_stop,
                )
            elif cfg.env == "market":
                env = MarketEnv(
                    rng=env_rng,
                    horizon=cfg.horizon,
                    initial_cash=cfg.initial_cash,
                    start_price=cfg.start_price,
                    mu=cfg.mu,
                    sigma=cfg.sigma,
                    fee_bps=cfg.fee_bps,
                    allow_stop=cfg.allow_stop,
                )
            else:
                env = BanditEnv(
                    rng=env_rng,
                    arms=cfg.arms,
                    horizon=cfg.horizon,
                    allow_stop=cfg.allow_stop,
                )
            summary = _run_episode(agent, env)
            episode_summaries.append(summary)

        objectives = evaluate_population(episode_summaries)
        for agent, obj in zip(population, objectives):
            agent.objectives = obj.to_dict()

        mean_task = statistics.fmean(o.task for o in objectives) if objectives else 0.0
        mean_novelty = statistics.fmean(o.novelty for o in objectives) if objectives else 0.0
        history_mean_task.append(mean_task)
        history_mean_novelty.append(mean_novelty)

        survivor_idx = select_survivors(objectives, survivors=cfg.survivors)
        if not survivor_idx and population:
            survivor_idx = [0]

        parents = [
            (population[i].agent_id, population[i].genome, objectives[i])
            for i in survivor_idx
        ]
        next_pairs, lineage = make_next_generation(
            parents=parents,
            rng=rng,
            population_size=cfg.population_size,
            elite_count=cfg.elite_count,
            mutation_rate=cfg.mutation_rate,
        )

        gen_record = {
            "generation": gen,
            "population_size": cfg.population_size,
            "mean": {"task": mean_task, "novelty": mean_novelty},
            "spark": {
                "task": ascii_sparkline(history_mean_task[-40:]),
                "novelty": ascii_sparkline(history_mean_novelty[-40:]),
            },
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "genome": a.genome.to_dict(),
                    "objectives": a.objectives,
                    "episode": s,
                }
                for a, s in zip(population, episode_summaries)
            ],
            "lineage_next": {k: list(v) for k, v in lineage.items()},
        }
        append_jsonl(run_dir / "generations.jsonl", gen_record)

        population = [
            Agent.create(
                agent_id=child_id,
                genome=genome,
                rng=random.Random(rng.getrandbits(64)),
            )
            for child_id, genome in next_pairs
        ]

    return run_dir


def _init_population(cfg: ExperimentConfig, rng: random.Random) -> list[Agent]:
    population: list[Agent] = []
    base = Genome()
    if cfg.genome_overrides:
        base_data = base.to_dict()
        base_data.update(dict(cfg.genome_overrides))
        base = Genome.from_dict(base_data)

    for _ in range(cfg.population_size):
        agent_id = f"a{rng.getrandbits(48):012x}"
        genome = base.mutate(rng, mutation_rate=1.0)
        population.append(
            Agent.create(
                agent_id=agent_id,
                genome=genome,
                rng=random.Random(rng.getrandbits(64)),
            )
        )
    return population


def _run_episode(agent: Agent, env) -> dict:
    agent.reset_episode()
    obs = env.reset()
    total_reward = 0.0
    rewards: list[float] = []
    actions: list[str] = []
    total_fee = 0.0
    total_trade_value = 0.0
    final_equity = None
    max_drawdown = None

    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        agent.observe_transition(
            obs, action, result.reward, result.observation, result.done
        )
        obs = result.observation
        done = result.done
        total_reward += result.reward
        rewards.append(result.reward)
        actions.append(action)
        info = result.info or {}
        total_fee += float(info.get("fee", 0.0))
        total_trade_value += float(info.get("trade_value", 0.0))
        if "equity" in info:
            final_equity = float(info.get("equity"))
        if "max_drawdown" in info:
            max_drawdown = float(info.get("max_drawdown"))

    stability = _stability_score(rewards)
    if obs.get("hint") == "market":
        init_cash = float(obs.get("initial_cash", 0.0))
        fee_ratio = total_fee / max(1e-9, init_cash)
        turnover_ratio = total_trade_value / max(1e-9, init_cash)
        efficiency = float(min(1.0, max(0.0, 1.0 - (0.2 * turnover_ratio + 6.0 * fee_ratio))))
        task = float(((final_equity if final_equity is not None else init_cash) / max(1e-9, init_cash)) - 1.0)
        return {
            "task": float(task),
            "stability": float(stability),
            "efficiency": float(efficiency),
            "action_sequence": actions,
            "final_equity": float(final_equity) if final_equity is not None else float(init_cash),
            "total_fee": float(total_fee),
            "turnover": float(total_trade_value),
            "max_drawdown": float(max_drawdown) if max_drawdown is not None else 0.0,
        }

    efficiency = _efficiency_score(steps=len(rewards), horizon=env.horizon)
    return {
        "task": float(total_reward),
        "stability": float(stability),
        "efficiency": float(efficiency),
        "action_sequence": actions,
    }


def _stability_score(rewards: list[float]) -> float:
    if len(rewards) <= 1:
        return 1.0
    std = statistics.pstdev(rewards)
    return float(min(1.0, max(0.0, 1.0 - 2.0 * std)))


def _efficiency_score(steps: int, horizon: int) -> float:
    if horizon <= 0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - (steps / horizon))))


def _make_run_dir(base: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(base) / stamp
