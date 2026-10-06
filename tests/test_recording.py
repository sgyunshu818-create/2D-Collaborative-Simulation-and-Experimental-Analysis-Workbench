"""Exports preserve configuration and recorded facts, including incomplete runs."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from sim_app.models import BehaviorState, RunState
from sim_app.recording import export_run
from sim_app.scene import load_scene
from sim_app.simulation import Simulation
from tests.test_motion import mission_config, scene_from_config
from tests.test_scene import valid_config


class RecordingTests(unittest.TestCase):
    def export(self, simulation, directory):
        paths = export_run(simulation, directory)
        self.assertEqual(set(paths), {"run", "events", "states"})
        for path in paths.values():
            self.assertIsInstance(path, Path)
            self.assertTrue(path.is_file())
        payload = json.loads(paths["run"].read_text(encoding="utf-8"))
        with paths["events"].open(encoding="utf-8-sig", newline="") as stream:
            events = list(csv.DictReader(stream))
        with paths["states"].open(encoding="utf-8-sig", newline="") as stream:
            states = list(csv.DictReader(stream))
        return paths, payload, events, states

    def test_completed_export_round_trips_scene_and_agrees_with_both_csv_files(self):
        config = mission_config()
        config["name"] = "第二阶段导出核对"
        simulation = Simulation(scene_from_config(config))
        simulation.start()
        simulation.advance(100)
        self.assertTrue(simulation.finished)
        with tempfile.TemporaryDirectory() as directory:
            paths, payload, events, states = self.export(simulation, Path(directory) / "run")
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["result"]["state"], RunState.FINISHED.value)
            self.assertEqual(payload["result"]["step"], simulation.step_count)
            self.assertEqual(payload["result"]["time"], simulation.sim_time)
            self.assertEqual(payload["events"], simulation.events)
            self.assertEqual(payload["snapshots"][-1], payload["result"])
            restored = Path(directory) / "restored_scene.json"
            restored.write_text(json.dumps(payload["scene"], ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_scene(restored), simulation.scene)
            expected_events = [{key: "" if value is None else str(value)
                                for key, value in event.items()}
                               for event in payload["events"]]
            self.assertEqual(events, expected_events)
            expected_states = [
                {key: str(value) for key, value in
                 {"step": snapshot["step"], "time": snapshot["time"],
                  "state": snapshot["state"], **unit}.items()}
                for snapshot in payload["snapshots"] for unit in snapshot["units"]
            ]
            self.assertEqual(states, expected_states)
            self.assertTrue(paths["events"].read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertTrue(paths["states"].read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_ready_static_scene_can_be_exported_without_starting(self):
        simulation = Simulation(scene_from_config(valid_config()))
        with tempfile.TemporaryDirectory() as directory:
            _, payload, events, states = self.export(simulation, directory)
        self.assertEqual(payload["result"]["state"], RunState.READY.value)
        self.assertEqual(payload["result"]["step"], 0)
        self.assertEqual(len(payload["snapshots"]), 1)
        self.assertEqual(events, [])
        self.assertEqual(len(states), 4)
        self.assertTrue(all(row["behavior"] == BehaviorState.IDLE.value for row in states))

    def test_running_and_paused_exports_preserve_current_state_without_mutation(self):
        for pause in (False, True):
            simulation = Simulation(scene_from_config(mission_config()))
            simulation.start()
            simulation.advance(0.375)
            if pause:
                simulation.pause()
            before = copy.deepcopy((simulation.events, simulation.snapshots, simulation.units))
            clock = (simulation.state, simulation.step_count, simulation.sim_time)
            with self.subTest(paused=pause), tempfile.TemporaryDirectory() as directory:
                _, payload, _, _ = self.export(simulation, directory)
                self.assertEqual(payload["result"]["state"], clock[0].value)
                self.assertEqual(payload["snapshots"][-1], payload["result"])
                self.assertEqual(payload["result"]["step"], 3)
                self.assertFalse(simulation.finished)
                self.assertEqual((simulation.events, simulation.snapshots, simulation.units), before)
                self.assertEqual((simulation.state, simulation.step_count, simulation.sim_time), clock)
                second = export_run(simulation, directory)
                self.assertEqual(json.loads(second["run"].read_text(encoding="utf-8")), payload)

    def test_source_scene_path_is_recorded_and_export_can_overwrite_earlier_run(self):
        simulation = Simulation(scene_from_config(mission_config()))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            source = Path(directory) / "source.json"
            first = export_run(simulation, output, source)
            simulation.start()
            simulation.advance(0.125)
            paths = export_run(simulation, output, source)
            self.assertEqual(paths, first)
            payload = json.loads(paths["run"].read_text(encoding="utf-8"))
            self.assertEqual(payload["source_scene"], str(source.resolve()))
            self.assertEqual(payload["result"]["step"], 1)

    def test_blocked_export_keeps_failure_reason_and_original_position(self):
        config = valid_config()
        config["obstacles"] = [{"id": "wall", "x": 400, "y": 0,
                                "width": 40, "height": 600}]
        config["units"] = [config["units"][0]]
        config["units"][0].update(speed=80, waypoints=[{"x": 800, "y": 150}])
        simulation = Simulation(scene_from_config(config))
        simulation.start()
        self.assertTrue(simulation.finished)
        with tempfile.TemporaryDirectory() as directory:
            _, payload, events, states = self.export(simulation, directory)
        result = payload["result"]["units"][0]
        self.assertEqual(result["behavior"], BehaviorState.BLOCKED.value)
        self.assertEqual(result["reason"], simulation.units[0].reason)
        self.assertTrue(result["reason"])
        self.assertEqual((result["x"], result["y"]), (80, 150))
        self.assertTrue(events)
        self.assertEqual(states[-1]["reason"], result["reason"])


if __name__ == "__main__":
    unittest.main()
