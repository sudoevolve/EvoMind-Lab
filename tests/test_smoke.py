import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from experiment.config import ExperimentConfig
from experiment.core import run
from experiment import analyze


class SmokeTest(unittest.TestCase):
    def test_run_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = ExperimentConfig(
                seed=1,
                generations=2,
                population_size=8,
                survivors=4,
                elite_count=1,
                mutation_rate=0.3,
                arms=6,
                horizon=12,
                allow_stop=True,
                output_dir=d,
            )
            run_dir = run(cfg)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "generations.jsonl").exists())

    def test_run_market_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cfg = ExperimentConfig(
                seed=2,
                generations=2,
                population_size=8,
                survivors=4,
                elite_count=1,
                mutation_rate=0.3,
                env="market",
                horizon=32,
                allow_stop=True,
                initial_cash=10_000.0,
                start_price=100.0,
                mu=0.0,
                sigma=0.02,
                fee_bps=2.0,
                output_dir=d,
            )
            run_dir = run(cfg)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "generations.jsonl").exists())

    def test_run_csv_market_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ohlcv.csv"
            p.write_text(
                "ts,open,high,low,close,volume\n"
                "0,1,1,1,100,0\n"
                "1,1,1,1,101,0\n"
                "2,1,1,1,99,0\n"
                "3,1,1,1,102,0\n"
                "4,1,1,1,103,0\n",
                encoding="utf-8",
            )

            cfg = ExperimentConfig(
                seed=3,
                generations=2,
                population_size=8,
                survivors=4,
                elite_count=1,
                mutation_rate=0.3,
                env="csv_market",
                horizon=4,
                allow_stop=True,
                initial_cash=10_000.0,
                fee_bps=2.0,
                slippage_bps=1.0,
                data_path=str(p),
                genome_overrides={"decision_strategy": "rule", "trading_template": "ma_cross"},
                output_dir=d,
            )
            run_dir = run(cfg)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "generations.jsonl").exists())

    def test_run_csv_market_no_header_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "kline.csv"
            p.write_text(
                "0,1,1,1,100,0,0,0,0,0,0,0\n"
                "1,1,1,1,101,0,0,0,0,0,0,0\n"
                "2,1,1,1,99,0,0,0,0,0,0,0\n"
                "3,1,1,1,102,0,0,0,0,0,0,0\n"
                "4,1,1,1,103,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )

            cfg = ExperimentConfig(
                seed=33,
                generations=2,
                population_size=8,
                survivors=4,
                elite_count=1,
                mutation_rate=0.3,
                env="csv_market",
                horizon=4,
                allow_stop=True,
                initial_cash=10_000.0,
                fee_bps=2.0,
                slippage_bps=1.0,
                data_path=str(p),
                genome_overrides={"decision_strategy": "rule", "trading_template": "ma_cross"},
                output_dir=d,
            )
            run_dir = run(cfg)
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "generations.jsonl").exists())

    def test_analyze_report_runs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ohlcv.csv"
            p.write_text(
                "ts,open,high,low,close,volume\n"
                "0,1,1,1,100,0\n"
                "1,1,1,1,101,0\n"
                "2,1,1,1,99,0\n"
                "3,1,1,1,102,0\n"
                "4,1,1,1,103,0\n",
                encoding="utf-8",
            )

            cfg = ExperimentConfig(
                seed=4,
                generations=2,
                population_size=8,
                survivors=4,
                elite_count=1,
                mutation_rate=0.3,
                env="csv_market",
                horizon=4,
                allow_stop=True,
                initial_cash=10_000.0,
                fee_bps=2.0,
                slippage_bps=1.0,
                data_path=str(p),
                genome_overrides={"decision_strategy": "rule", "trading_template": "ma_cross"},
                output_dir=d,
            )
            run_dir = run(cfg)

            buf = StringIO()
            with redirect_stdout(buf):
                rc = analyze.main(["--run", str(run_dir), "--top", "3", "--per-gen"])
            out = buf.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("运行结果分析", out)
            self.assertIn("字段说明", out)
            self.assertTrue((run_dir / "analysis.txt").exists())


if __name__ == "__main__":
    unittest.main()
