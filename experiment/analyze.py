from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentRow:
    agent_id: str
    generation: int
    genome: dict[str, Any]
    objectives: dict[str, float]
    episode: dict[str, Any]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evomind-analyze")
    p.add_argument("--run", type=str, required=True, help="run dir or generations.jsonl")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--per-gen", action="store_true")
    args = p.parse_args(argv)

    run_path = Path(args.run)
    run_dir, gen_path = _resolve_run_paths(run_path)
    cfg = _read_json(run_dir / "config.json") if run_dir else {}
    rows = _read_generations(gen_path)

    report = build_report(cfg=cfg, rows=rows, top=int(args.top), per_gen=bool(args.per_gen))
    print(report)
    out_path = (run_dir or gen_path.parent) / "analysis.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"已写入: {out_path}")
    return 0


def build_report(cfg: dict[str, Any], rows: list[AgentRow], top: int, per_gen: bool) -> str:
    if not rows:
        return "没有读到任何 generations.jsonl 记录。"

    env = str(cfg.get("env") or _guess_env(rows))
    initial_cash = float(cfg.get("initial_cash") or _guess_initial_cash(rows) or 10_000.0)
    gens = sorted({r.generation for r in rows})
    by_gen: dict[int, list[AgentRow]] = {g: [] for g in gens}
    for r in rows:
        by_gen.setdefault(r.generation, []).append(r)

    lines: list[str] = []
    lines.append("EvoMind-Lab 运行结果分析")
    lines.append(f"- 环境: {env}")
    lines.append(f"- 代数: {len(gens)} ({gens[0]}..{gens[-1]})")
    if env in {"market", "csv_market"}:
        lines.append(f"- 初始资金: {initial_cash:.2f}")
    lines.append("")

    lines.extend(_metric_legend(env=env))
    lines.append("")

    last_gen = gens[-1]
    last = by_gen.get(last_gen, [])
    ranked = sorted(last, key=lambda r: _score(r, env=env, initial_cash=initial_cash), reverse=True)

    lines.append(f"本次最推荐（按综合评分，来自第 {last_gen} 代）")
    lines.extend(_format_top(ranked[: max(1, top)], env=env, initial_cash=initial_cash))
    lines.append("")

    lines.append(f"第 {last_gen} 代概览（全体）")
    lines.extend(_format_summary(last, env=env, initial_cash=initial_cash))
    lines.append("")

    if per_gen:
        lines.append("每代趋势（均值）")
        for g in gens:
            lines.extend(_format_generation(g, by_gen.get(g, []), env=env, initial_cash=initial_cash))
        lines.append("")

    lines.append("改进建议（针对当前任务：币圈回测筛选）")
    lines.extend(_improvement_suggestions(last, env=env, initial_cash=initial_cash))

    return "\n".join(lines).rstrip() + "\n"


def _resolve_run_paths(run: Path) -> tuple[Path | None, Path]:
    if run.is_file() and run.name.endswith(".jsonl"):
        return run.parent, run
    if run.is_dir():
        gen = run / "generations.jsonl"
        if gen.exists():
            return run, gen
    raise FileNotFoundError(str(run))


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def _read_generations(path: Path) -> list[AgentRow]:
    rows: list[AgentRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        g = json.loads(s)
        gen = int(g.get("generation", 0))
        for a in g.get("agents", []) or []:
            rows.append(
                AgentRow(
                    agent_id=str(a.get("agent_id", "")),
                    generation=gen,
                    genome=dict(a.get("genome", {}) or {}),
                    objectives=dict(a.get("objectives", {}) or {}),
                    episode=dict(a.get("episode", {}) or {}),
                )
            )
    return rows


def _guess_env(rows: list[AgentRow]) -> str:
    for r in rows:
        if "final_equity" in r.episode:
            return "market"
    return "bandit"


def _guess_initial_cash(rows: list[AgentRow]) -> float | None:
    for r in rows:
        v = r.episode.get("final_equity")
        if v is not None:
            try:
                eq = float(v)
            except Exception:
                continue
            if eq > 0:
                return 10_000.0
    return None


def _metric_legend(env: str) -> list[str]:
    lines = ["字段说明（你只需要记住：收益↑、回撤↓、费用↓、换手↓、稳定↑）"]
    lines.append("- task: 主要目标。在币圈回测里=收益率（0.01≈+1%）。")
    lines.append("- novelty: 行为新颖度。越高=和别人动作序列差异越大。")
    lines.append("- stability: 回报波动的反向指标。越高=每步 reward 更平稳。")
    lines.append("- efficiency: 越高=越少步数/越少无谓动作（含早停影响）。")
    if env in {"market", "csv_market"}:
        lines.append("- final_equity: 最终资金。盈亏=final_equity-初始资金。")
        lines.append("- total_fee: 本次回测支付的成本（手续费+滑点）。越小越好。")
        lines.append("- turnover: 换手（交易金额总和）。越大越容易过拟合、越吃成本。")
        lines.append("- max_drawdown: 最大回撤（0.1≈10%）。越小越好。")
    return lines


def _score(r: AgentRow, env: str, initial_cash: float) -> float:
    task = float(r.objectives.get("task", 0.0))
    stability = float(r.objectives.get("stability", 0.0))
    efficiency = float(r.objectives.get("efficiency", 0.0))

    if env not in {"market", "csv_market"}:
        return float(task + 0.1 * stability + 0.05 * efficiency)

    dd = float(r.episode.get("max_drawdown", 0.0) or 0.0)
    fee = float(r.episode.get("total_fee", 0.0) or 0.0)
    turn = float(r.episode.get("turnover", 0.0) or 0.0)
    fee_ratio = fee / max(1e-9, initial_cash)
    turn_ratio = turn / max(1e-9, initial_cash)

    score = float(task + 0.01 * stability + 0.005 * efficiency - 0.8 * dd - 2.0 * fee_ratio - 0.05 * turn_ratio)
    if turn_ratio == 0.0:
        score -= 0.2
    return score


def _format_top(rows: list[AgentRow], env: str, initial_cash: float) -> list[str]:
    lines: list[str] = []
    for i, r in enumerate(rows, 1):
        lines.append(_format_agent_line(i, r, env=env, initial_cash=initial_cash))
        lines.extend(_format_agent_insight(r, env=env, initial_cash=initial_cash))
    return lines


def _format_agent_line(rank: int, r: AgentRow, env: str, initial_cash: float) -> str:
    task = float(r.objectives.get("task", 0.0))
    novelty = float(r.objectives.get("novelty", 0.0))
    stability = float(r.objectives.get("stability", 0.0))
    efficiency = float(r.objectives.get("efficiency", 0.0))
    strat = r.genome.get("decision_strategy")
    tmpl = r.genome.get("trading_template")
    parts = [f"#{rank} {r.agent_id} 评分={_score(r, env=env, initial_cash=initial_cash):.4f}"]
    parts.append(f"task={task:.4f} novelty={novelty:.3f} stability={stability:.3f} efficiency={efficiency:.3f}")
    if strat or tmpl:
        parts.append(f"策略={strat}/{tmpl}")
    if env in {"market", "csv_market"}:
        eq = float(r.episode.get("final_equity", 0.0) or 0.0)
        pnl = eq - initial_cash
        dd = float(r.episode.get("max_drawdown", 0.0) or 0.0)
        fee = float(r.episode.get("total_fee", 0.0) or 0.0)
        turn = float(r.episode.get("turnover", 0.0) or 0.0)
        parts.append(f"盈亏={pnl:+.2f} 资金={eq:.2f} 回撤={dd:.4f} 费用={fee:.2f} 换手={turn/ max(1e-9, initial_cash):.2f}x")
    return " - " + " | ".join(parts)


def _format_agent_insight(r: AgentRow, env: str, initial_cash: float) -> list[str]:
    if env not in {"market", "csv_market"}:
        return []

    lines: list[str] = []
    eq = float(r.episode.get("final_equity", 0.0) or 0.0)
    pnl = eq - initial_cash
    dd = float(r.episode.get("max_drawdown", 0.0) or 0.0)
    fee = float(r.episode.get("total_fee", 0.0) or 0.0)
    turn = float(r.episode.get("turnover", 0.0) or 0.0)
    fee_ratio = fee / max(1e-9, initial_cash)
    turn_ratio = turn / max(1e-9, initial_cash)

    flags: list[str] = []
    if turn_ratio == 0.0:
        flags.append("未交易/全程空仓")
    if pnl < 0:
        flags.append("亏损")
    if dd >= 0.02:
        flags.append("回撤偏大")
    if fee_ratio >= 0.002:
        flags.append("费用偏高")
    if turn_ratio >= 5.0:
        flags.append("换手过高")
    if not flags:
        flags.append("结构正常")

    seq = r.episode.get("action_sequence", []) or []
    if isinstance(seq, list) and seq:
        preview = " ".join(map(str, seq[:10])) + (" ..." if len(seq) > 10 else "")
        lines.append(f"   - 动作: {preview}")
    lines.append(f"   - 诊断: {', '.join(flags)}")
    return lines


def _format_summary(rows: list[AgentRow], env: str, initial_cash: float) -> list[str]:
    if not rows:
        return ["- (空)"]

    tasks = [float(r.objectives.get("task", 0.0)) for r in rows]
    novs = [float(r.objectives.get("novelty", 0.0)) for r in rows]
    stabs = [float(r.objectives.get("stability", 0.0)) for r in rows]
    effs = [float(r.objectives.get("efficiency", 0.0)) for r in rows]
    scores = [_score(r, env=env, initial_cash=initial_cash) for r in rows]

    lines = [
        f"- task: 均值 {sum(tasks)/len(tasks):.4f} | 最好 {max(tasks):.4f} | 最差 {min(tasks):.4f}",
        f"- novelty: 均值 {sum(novs)/len(novs):.3f}",
        f"- stability: 均值 {sum(stabs)/len(stabs):.3f}",
        f"- efficiency: 均值 {sum(effs)/len(effs):.3f}",
        f"- 综合评分: 均值 {sum(scores)/len(scores):.4f} | 最好 {max(scores):.4f}",
    ]

    if env in {"market", "csv_market"}:
        eqs = [float(r.episode.get("final_equity", initial_cash) or initial_cash) for r in rows]
        pnls = [eq - initial_cash for eq in eqs]
        dds = [float(r.episode.get("max_drawdown", 0.0) or 0.0) for r in rows]
        fees = [float(r.episode.get("total_fee", 0.0) or 0.0) for r in rows]
        turns = [float(r.episode.get("turnover", 0.0) or 0.0) / max(1e-9, initial_cash) for r in rows]
        lines.append(f"- 盈亏: 均值 {sum(pnls)/len(pnls):+.2f} | 最好 {max(pnls):+.2f} | 最差 {min(pnls):+.2f}")
        lines.append(f"- 回撤: 均值 {sum(dds)/len(dds):.4f} | 最差 {max(dds):.4f}")
        lines.append(f"- 费用: 均值 {sum(fees)/len(fees):.2f} | 最差 {max(fees):.2f}")
        lines.append(f"- 换手: 均值 {sum(turns)/len(turns):.2f}x | 最差 {max(turns):.2f}x")

    return lines


def _format_generation(g: int, rows: list[AgentRow], env: str, initial_cash: float) -> list[str]:
    if not rows:
        return [f"- 第 {g} 代: (空)"]
    tasks = [float(r.objectives.get("task", 0.0)) for r in rows]
    scores = [_score(r, env=env, initial_cash=initial_cash) for r in rows]
    best = max(rows, key=lambda r: _score(r, env=env, initial_cash=initial_cash))
    return [
        f"- 第 {g} 代: mean_task={sum(tasks)/len(tasks):.4f} best_task={max(tasks):.4f} best={best.agent_id} best_score={max(scores):.4f}"
    ]


def _improvement_suggestions(rows: list[AgentRow], env: str, initial_cash: float) -> list[str]:
    lines: list[str] = []
    if env not in {"market", "csv_market"}:
        lines.append("- 先把 env 切到 csv_market，用真实 K 线回放再评价。")
        return lines

    if not rows:
        return ["- (无数据)"]

    fees = [float(r.episode.get("total_fee", 0.0) or 0.0) / max(1e-9, initial_cash) for r in rows]
    turns = [float(r.episode.get("turnover", 0.0) or 0.0) / max(1e-9, initial_cash) for r in rows]
    dds = [float(r.episode.get("max_drawdown", 0.0) or 0.0) for r in rows]
    tasks = [float(r.objectives.get("task", 0.0)) for r in rows]

    lines.append(f"- 平均收益率约 {100.0 * (sum(tasks)/len(tasks)):.2f}%（注意：样本很短时不稳定）。")
    if sum(turns) / len(turns) > 3.0:
        lines.append("- 换手偏高：建议把目标仓位档位从 5 档降到 3 档，或加“最小调仓间隔”。")
    if sum(fees) / len(fees) > 0.001:
        lines.append("- 成本敏感：建议提高 fee_bps/slippage_bps 做压力测试，淘汰靠频繁交易赚钱的个体。")
    if max(dds) > 0.02:
        lines.append("- 部分回撤偏大：建议把 max_drawdown 纳入选择目标（作为惩罚项/新 objective）。")

    lines.append("- 进化还太短：建议 generations≥50，且用更长 CSV（至少几千根K线）再看趋势。")
    lines.append("- 目前 novelty 只是“动作序列差异”：后续可换成“收益曲线形状差异 + 行为差异”结合。")
    lines.append("- 想更像真实：加入多段 walk-forward（训练段选，测试段验），把测试段得分作为 task。")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())

