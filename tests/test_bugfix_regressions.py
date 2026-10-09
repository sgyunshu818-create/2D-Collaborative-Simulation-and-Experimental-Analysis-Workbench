"""Regression coverage for boundary recordings and failed golden updates."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_app.recording import export_run
from sim_app.replay import load_replay
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation
from tests.test_replay import game_config
from tools import scene_fingerprints as gate


class BoundaryRecordingTests(unittest.TestCase):
    def test_contacts_at_world_edges_round_trip_without_precision_loss(self):
        for x, y in ((899.999, 300), (400, 599.999), (899.999, 599.999)):
            with self.subTest(x=x, y=y), tempfile.TemporaryDirectory() as directory:
                config = game_config()
                config["rules"]["tag_range"] = 0.001
                config["units"] = [
                    {"id": "observer", "team": "red", "type": "air",
                     "x": x - 1, "y": y - 1, "sensor_range": 10},
                    {"id": "target", "team": "blue", "type": "air",
                     "x": x, "y": y, "sensor_range": 10},
                ]
                sim = Simulation(scene_from_data(config))
                sim.start()
                sim.advance(config["fixed_dt"])
                contact = sim.snapshot()["units"][0]["contacts"][0]
                self.assertEqual((contact["x"], contact["y"]), (x, y))
                replay = load_replay(export_run(sim, directory)["run"])
                replay.seek(replay.count - 1)
                restored = replay.units[0].contacts["target"].position
                self.assertEqual((restored.x, restored.y), (x, y))


class FingerprintFailureTests(unittest.TestCase):
    def test_scene_error_fails_both_modes_and_preserves_golden(self):
        for args in ([], ["--update"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                valid = directory / "valid.json"
                invalid = directory / "invalid.json"
                import json
                valid.write_text(json.dumps(game_config()), encoding="utf-8")
                invalid.write_text("{", encoding="utf-8")
                golden = directory / "golden.json"
                before = b'{"horizon_seconds":60,"scenes":{}}'
                golden.write_bytes(before)
                stderr, stdout = io.StringIO(), io.StringIO()
                with patch.object(gate, "GOLDEN_PATH", golden), \
                     patch.object(gate, "scene_files", return_value=[valid, invalid]), \
                     contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    status = gate.main(args)
                self.assertNotEqual(status, 0)
                self.assertIn("invalid.json: SCENE_ERROR", stderr.getvalue())
                self.assertNotIn("GOLDEN_UPDATED", stdout.getvalue())
                self.assertEqual(golden.read_bytes(), before)
