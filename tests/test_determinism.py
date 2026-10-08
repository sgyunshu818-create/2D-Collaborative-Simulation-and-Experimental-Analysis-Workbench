"""Every built-in scene must keep recording the same run.

The workbench lets a user change one parameter and compare two runs, which only
means something if identical input produces an identical saved record. These
checks pin that property across the built-in scenes and guard it against later
changes to the movement and observation core.
"""

import unittest
from pathlib import Path

from sim_app.scene import load_scene
from sim_app.simulation import Simulation
from tools.scene_fingerprints import (
    GOLDEN_PATH,
    advance_to_horizon,
    compare,
    describe,
    load_golden,
    measure,
    measure_all,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

# A ground, an obstacle and a rules scene between them exercise navigation,
# collision handling and the observation game.
REPEATED_SCENES = ("basic_scene.json", "obstacle_scene.json", "sharing_scene.json")


class SceneFingerprintTests(unittest.TestCase):
    def run_to_horizon(self, path: Path) -> Simulation:
        sim = Simulation(load_scene(path))
        sim.start()
        advance_to_horizon(sim)
        return sim

    def test_every_scene_matches_the_golden_fingerprint(self):
        self.assertTrue(
            GOLDEN_PATH.is_file(),
            f"missing {GOLDEN_PATH.name}; create it with: python tools/scene_fingerprints.py --update",
        )
        differences = compare(measure_all(), load_golden())
        self.assertEqual(
            differences,
            [],
            "recorded behaviour changed for:\n  " + "\n  ".join(differences) +
            "\nReview each change; if it is intended, run: python tools/scene_fingerprints.py --update",
        )

    def test_repeated_runs_of_the_same_scene_are_identical(self):
        for name in REPEATED_SCENES:
            with self.subTest(scene=name):
                path = CONFIG_DIR / name
                first, second = measure(path), measure(path)
                self.assertEqual(first, second, f"{name} is not reproducible")
                self.assertEqual(first["state"], "FINISHED", f"{name} should end within the horizon")

    def test_reset_reproduces_a_fresh_run(self):
        # Anything a finished run leaves behind - tag cooldowns, scores, contacts,
        # trails - must not change the next run of the same scene.
        path = CONFIG_DIR / "sharing_scene.json"
        fresh = describe(self.run_to_horizon(path), path)

        reused = self.run_to_horizon(path)
        self.assertTrue(reused.reset(), "a finished run should report a state change")
        self.assertEqual(reused.snapshot()["units"][0]["waypoint_index"], 0)

        reused.start()
        advance_to_horizon(reused)
        self.assertEqual(describe(reused, path), fresh, "reset left stale state behind")


if __name__ == "__main__":
    unittest.main()
