 import tempfile
 import unittest
 
 from experiment.config import ExperimentConfig
 from experiment.core import run
 
 
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
 
 
 if __name__ == "__main__":
     unittest.main()
