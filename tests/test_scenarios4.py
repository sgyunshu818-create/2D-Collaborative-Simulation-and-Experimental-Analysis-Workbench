"""Fourth-stage presets and their headless batch-validation boundaries."""

import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sim_app.models import BehaviorState, RunState, Team, UnitType
from sim_app.navigation import plan_path, segment_clear
from sim_app.recording import export_run
from sim_app.scene import load_scene
from sim_app.simulation import Simulation
from tools.validate_scenarios import (
    SCENARIO_FILES, advance_to_end, audit_record, geometry_valid, knowledge_metrics,
    main, validate_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"


def completed(filename):
    sim = Simulation(load_scene(CONFIGS / filename))
    advance_to_end(sim, 7200)
    return sim


class Stage4SceneTests(unittest.TestCase):
    def test_three_business_presets_are_complete_and_finish_without_blocked_units(self):
        expected = {
            "basic_scene.json": ("stage4_basic", 1081, {Team.RED: 2, Team.BLUE: 2}, "missions_complete"),
            "obstacle_scene.json": ("stage4_obstacle", 1035, {Team.RED: 0, Team.BLUE: 0}, "missions_complete"),
            "sharing_scene.json": ("stage4_sharing", 422, {Team.RED: 3, Team.BLUE: 3}, "score_limit"),
        }
        for filename, (name, steps, scores, reason) in expected.items():
            with self.subTest(filename=filename):
                sim = completed(filename)
                self.assertEqual(sim.scene.name, name)
                self.assertEqual(len(sim.scene.units), 4)
                self.assertEqual({(unit.team, unit.unit_type) for unit in sim.units},
                                 {(team, kind) for team in Team for kind in UnitType})
                self.assertEqual((sim.state, sim.step_count, sim.scores, sim.finish_reason),
                                 (RunState.FINISHED, steps, scores, reason))
                self.assertTrue(all(unit.behavior == BehaviorState.STOPPED for unit in sim.units))
                self.assertTrue(geometry_valid(sim))
                self.assertTrue(any(event["kind"] == "object_discovered" for event in sim.events))
                if filename != "sharing_scene.json":
                    self.assertTrue(all(unit.return_home and unit.position == sim.scene.return_points[unit.team]
                                        for unit in sim.units))
                    self.assertEqual(sim.scene.rules.score_limit, 99)

    def test_obstacle_preset_is_same_task_and_speed_with_ground_detours(self):
        sim = completed("obstacle_scene.json")
        for team in Team:
            ground, air = [unit for unit in sim.units if unit.team == team]
            ground_spec, air_spec = [unit for unit in sim.scene.units if unit.team == team]
            self.assertEqual((ground_spec.position, ground_spec.speed, ground_spec.waypoints),
                             (air_spec.position, air_spec.speed, air_spec.waypoints))
            ground_path = plan_path(sim.scene, ground_spec.position, ground_spec.waypoints[0], UnitType.GROUND)
            air_path = plan_path(sim.scene, air_spec.position, air_spec.waypoints[0], UnitType.AIR)
            self.assertGreater(len(ground_path), 1)
            self.assertEqual(air_path, (air_spec.waypoints[0],))
            self.assertFalse(segment_clear(sim.scene, ground_spec.position, ground_spec.waypoints[0], UnitType.GROUND))
            self.assertGreater(ground.distance_travelled, air.distance_travelled)
            self.assertEqual(ground.position, air.position)

    def test_only_sharing_flag_changes_paired_knowledge_and_sources(self):
        on = completed("sharing_scene.json")
        off_scene = replace(on.scene, rules=replace(on.scene.rules, sharing_enabled=False))
        off = Simulation(off_scene)
        advance_to_end(off, 7200)
        self.assertEqual(off.scene, replace(on.scene, rules=replace(on.scene.rules, sharing_enabled=False)))
        on_metrics = knowledge_metrics(on.snapshots, on.events, on.scene.fixed_dt)
        off_metrics = knowledge_metrics(off.snapshots, off.events, off.scene.fixed_dt)
        self.assertEqual((on_metrics["shared_receiver_count"], on_metrics["unique_shared_receiver_target_pairs"],
                          on_metrics["logged_info_shared_transitions"], on_metrics["fresh_shared_contact_samples"]),
                         (2, 4, 4, 712))
        self.assertEqual(off_metrics["shared_contact_samples"], 0)
        self.assertEqual(off_metrics["logged_info_shared_transitions"], 0)
        self.assertEqual(on_metrics["shared_sources_by_receiver"],
                         {"blue_ground_01": ["blue_air_01"], "red_ground_01": ["red_air_01"]})
        self.assertEqual((off.step_count, off.scores, off.winner),
                         (446, {Team.RED: 2, Team.BLUE: 3}, "blue"))

    def test_blocked_helper_is_known_failure_outside_three_business_scenes(self):
        self.assertNotIn("blocked_scene.json", SCENARIO_FILES)
        sim = completed("blocked_scene.json")
        self.assertEqual((sim.scene.name, sim.step_count), ("stage4_blocked", 536))
        ground = [unit for unit in sim.units if unit.unit_type == UnitType.GROUND]
        air = [unit for unit in sim.units if unit.unit_type == UnitType.AIR]
        self.assertTrue(all(unit.behavior == BehaviorState.BLOCKED and unit.reason
                            and unit.distance_travelled == 0 for unit in ground))
        self.assertTrue(all(unit.behavior == BehaviorState.STOPPED
                            and unit.position == sim.scene.return_points[unit.team] for unit in air))
        self.assertEqual(sum(event["kind"] == "path_blocked" for event in sim.events), 2)


class Stage4ValidationTests(unittest.TestCase):
    def test_knowledge_metrics_distinguish_memory_samples_from_transition_logs(self):
        contact = {"target_id": "target", "x": 20, "y": 20, "observed_step": 1,
                   "source_id": "observer", "shared": True}
        snapshots = [{"step": step, "units": [{"id": "receiver", "contacts": [copy.deepcopy(contact)]}]}
                     for step in (1, 1, 2)]
        metrics = knowledge_metrics(snapshots, [{"kind": "info_shared"}], 0.125)
        self.assertEqual(metrics["logged_info_shared_transitions"], 1)
        self.assertEqual(metrics["shared_contact_samples"], 2)
        self.assertEqual(metrics["fresh_shared_contact_samples"], 1)
        self.assertEqual(metrics["shared_receiver_count"], 1)
        self.assertEqual(metrics["shared_knowledge_unit_seconds"], 0.25)

    def test_batch_records_every_scene_reset_replay_and_comparison_independently(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = validate_batch(directory, repeat=1)
            root = Path(report["directory"])
            self.assertEqual(report["status"], "PASS")
            self.assertEqual((report["completed_business_records"], report["extra_record_count"],
                              report["saved_record_count"], report["reset_rerun_count"]), (3, 2, 5, 5))
            self.assertEqual(len(report["records"]), 5)
            self.assertEqual(report["sharing_comparison"]["status"], "PASS")
            self.assertTrue(all(report["sharing_comparison"]["checks"].values()))
            self.assertEqual(report["sharing_comparison"]["matched_logical_step"], 422)
            self.assertEqual(sum(row["replay_checked_frames"] for row in report["records"]), 3530)
            self.assertTrue(all(all(row["checks"].values()) for row in report["records"]))
            self.assertTrue((root / "validation.json").is_file())
            with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as source:
                self.assertEqual(len(list(csv.DictReader(source))), 5)
            for row in report["records"]:
                self.assertEqual(len(row["source_scene_sha256"]), 64)
                self.assertEqual(len(row["effective_scene_sha256"]), 64)
                self.assertTrue(Path(row["run_json"]).is_file())
                self.assertTrue(row["reset_reproduced"])
                self.assertIn("not GUI FPS", report["environment"]["measurement"])
            self.assertEqual(report["records"][-1]["status"], "EXPECTED_BLOCKED")

    def test_step_limit_returns_nonzero_and_preserves_incomplete_facts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--output-dir", directory, "--repeat", "1", "--max-steps", "1"]), 1)
            batch = next(Path(directory).glob("batch_*"))
            report = json.loads((batch / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["completed_business_records"], 0)
            self.assertTrue(report["failures"])
            payload = json.loads(Path(report["records"][0]["run_json"]).read_text(encoding="utf-8"))
            self.assertEqual((payload["result"]["state"], payload["result"]["step"]), ("RUNNING", 1))
            self.assertIn("finished_within_limit", report["records"][0]["errors"])

    def test_missing_scene_is_reported_and_never_counted_as_a_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            for filename in (*SCENARIO_FILES, "blocked_scene.json"):
                if filename != "basic_scene.json":
                    (config_dir / filename).write_bytes((CONFIGS / filename).read_bytes())
            report = validate_batch(root / "outputs", repeat=1, max_steps=1, config_dir=config_dir)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["business_run_count"], 2)
            self.assertTrue(any("source_scene" in item and "basic_scene.json" in item["source_scene"]
                                for item in report["failures"]))
            self.assertTrue(any("business_records" in item for item in report["failures"]))

    def test_record_audit_detects_csv_value_corruption_not_only_row_counts(self):
        sim = Simulation(load_scene(CONFIGS / "basic_scene.json"))
        sim.start()
        sim.advance(sim.scene.fixed_dt)
        with tempfile.TemporaryDirectory() as directory:
            paths = export_run(sim, directory)
            with paths["states"].open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            rows[0]["x"] = "222"
            with paths["states"].open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "state CSV field x"):
                audit_record(sim, paths)

    def test_repeat_and_max_steps_reject_nonpositive_values(self):
        for kwargs in ({"repeat": 0}, {"repeat": True}, {"max_steps": -1}, {"max_steps": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_batch("unused-output", **kwargs)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--repeat", "0"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
