"""Benchmark statistics and failure handling; dummy runs are mechanisms only."""

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from sim_app.models import BehaviorState, RunState, Team, UnitType
from sim_app.navigation import point_clear
from sim_app.replay import load_replay
from sim_app.simulation import Simulation
from tools.benchmark_simulation import (
    FrameStats, canonical_json, check_parameters, make_workload, percentile,
    run_benchmark, write_full_record,
)


class PerformanceStatisticsTests(unittest.TestCase):
    def test_parameters_reject_boolean_nonfinite_nonpositive_and_invalid_unit_counts(self):
        for name in ("duration", "warmup"):
            for value in (True, None, "5", 0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    check_parameters(**{name: value})
        for value in (True, 0, -1, 60.0, float("inf"), "60"):
            with self.subTest(target_fps=value), self.assertRaises(ValueError):
                check_parameters(target_fps=value)
        for value in (True, 0, 5, 4.0, "4"):
            with self.subTest(units=value), self.assertRaises(ValueError):
                check_parameters(units=value)
        self.assertEqual(check_parameters(8, 60, 5, 60),
                         {"units": 8, "duration_seconds": 60.0, "warmup_seconds": 5.0, "target_fps": 60})

    def test_percentiles_use_documented_interpolation_not_nearest_rank(self):
        data = [40, 10, 30, 20]
        self.assertEqual(percentile(data, 0.5), 25)
        self.assertAlmostEqual(percentile(data, 0.95), 38.5)
        self.assertAlmostEqual(percentile(data, 0.99), 39.7)
        self.assertEqual(percentile([16], 0.99), 16)
        self.assertIsNone(percentile([], 0.95))

    def test_overall_rate_and_exact_second_boundary_preserve_all_frames(self):
        origin = 10000000000
        stats = FrameStats(origin)
        previous = origin
        for elapsed in (1000000000, 2000000000, 2500000000):
            end = origin + elapsed
            row = {"frame_end_monotonic_ns": end, "flip_interval_ns": end - previous,
                   "window_active": True, "keyboard_focus": False, "logic_steps": 1,
                   **{name: 1.0 for name in stats.parts}}
            stats.add(row, ["moving"])
            previous = end
        summary, seconds = stats.finish(previous, True)
        self.assertEqual(summary["total_frames"], 3)
        self.assertAlmostEqual(summary["gui_average_fps"], 3 / 2.5)
        self.assertEqual([row["frames"] for row in seconds], [1, 1, 1])
        self.assertEqual([row["actual_seconds"] for row in seconds], [1, 1, 0.5])
        self.assertEqual(summary["worst_complete_one_second_gui_fps"], 1)
        self.assertEqual(summary["complete_one_second_window_count"], 2)
        self.assertEqual(summary["active_frame_fraction"], 1)
        self.assertEqual(summary["keyboard_focus_frame_fraction"], 0)
        self.assertEqual(summary["units_with_observed_position_changes"], ["moving"])
        mechanism, mechanism_seconds = stats.finish(previous, False)
        self.assertIsNone(mechanism["gui_average_fps"])
        self.assertTrue(all(row["gui_fps"] is None for row in mechanism_seconds))

    def test_slow_frame_threshold_is_strictly_above_33_33_milliseconds(self):
        stats = FrameStats(0)
        previous = 0
        for interval in (33330000, 33330001):
            end = previous + interval
            stats.add({"frame_end_monotonic_ns": end, "flip_interval_ns": interval,
                       "window_active": True, "keyboard_focus": True, "logic_steps": 1,
                       **{name: 0.0 for name in stats.parts}})
            previous = end
        summary, seconds = stats.finish(previous, True)
        self.assertEqual(summary["slow_frames_over_33_33_ms"], 1)
        self.assertIsNone(summary["worst_complete_one_second_gui_fps"])
        self.assertFalse(seconds[0]["complete_one_second"])


class PerformanceWorkloadTests(unittest.TestCase):
    def test_4_8_12_unit_fixtures_preserve_static_geometry_and_sustain_long_routes(self):
        for count in (4, 8, 12):
            with self.subTest(count=count):
                scene, data, metadata = make_workload(count, 600, 5)
                self.assertEqual(scene.name, f"stage5_benchmark_{count}_units")
                self.assertEqual(len(scene.units), count)
                self.assertEqual(len({unit.id for unit in scene.units}), count)
                self.assertEqual(len(scene.obstacles), 2)
                self.assertEqual(scene.fixed_dt, 1 / 60)
                self.assertGreater(scene.rules.time_limit, 605)
                self.assertGreater(scene.rules.score_limit, (count // 2) ** 2)
                for unit in scene.units:
                    self.assertEqual(unit.speed, 90)
                    self.assertFalse(unit.return_home)
                    self.assertEqual(unit.sensor_range, 160 if unit.unit_type == UnitType.GROUND else 240)
                    self.assertGreater(len(unit.waypoints), 100)
                    lower_bound = 0
                    previous = unit.position
                    for point in unit.waypoints:
                        self.assertTrue(point_clear(scene, point, unit.unit_type))
                        lower_bound += ((point.x - previous.x) ** 2 + (point.y - previous.y) ** 2) ** 0.5 / unit.speed
                        previous = point
                    self.assertGreater(lower_bound, metadata["minimum_route_horizon_seconds"])
                self.assertEqual(len(metadata["source_scene_sha256"]), 64)
                self.assertEqual(metadata["effective_scene_sha256"], hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest())
                sim = Simulation(scene)
                sim.start()
                before = [unit.position for unit in sim.units]
                sim.advance(0.5)
                self.assertEqual(sim.state, RunState.RUNNING)
                self.assertNotEqual([unit.position for unit in sim.units], before)
                self.assertTrue(all(unit.behavior == BehaviorState.NAVIGATING for unit in sim.units))

    def test_streamed_full_record_round_trips_without_dropping_or_mutating_snapshots(self):
        scene, data, _ = make_workload(4, 0.1, 0.05)
        sim = Simulation(scene)
        sim.start()
        sim.advance(0.2)
        original = copy.deepcopy((sim.snapshot(), sim.events, sim.snapshots))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = root / "scene.json"
            scene_path.write_text(json.dumps(data), encoding="utf-8")
            result = write_full_record(sim, data, scene_path, root / "recording")
            payload = json.loads(Path(result["run_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["snapshots"], sim.snapshots)
            self.assertEqual(payload["events"], sim.events)
            self.assertEqual(payload["result"], sim.snapshot())
            expected_digest = hashlib.sha256("".join(canonical_json(frame) + "\n" for frame in sim.snapshots).encode("utf-8")).hexdigest()
            self.assertEqual(result["snapshot_content_sha256"], expected_digest)
            replay = load_replay(result["run_json"])
            replay.seek(replay.count - 1)
            self.assertEqual(replay.result_snapshot, sim.snapshot())
            self.assertEqual(replay.events, sim.events)
            with Path(result["snapshot_index_csv"]).open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), len(sim.snapshots))
            self.assertEqual(int(rows[-1]["event_count"]), len(sim.events))
        self.assertEqual((sim.snapshot(), sim.events, sim.snapshots), original)


class PerformanceMechanismTests(unittest.TestCase):
    def dummy_run(self, directory, **kwargs):
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_benchmark(duration=0.07, warmup=0.03, output_dir=directory, **kwargs)

    def test_dummy_display_rejected_as_true_gui_even_when_native_api_is_present(self):
        import pygame
        with tempfile.TemporaryDirectory() as directory, patch("pygame.display.get_wm_info", return_value={"window": 1}):
            code, report = self.dummy_run(directory)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "INVALID_DISPLAY")
            self.assertFalse(report["real_window_measurement"])
            self.assertIsNone(report["performance_goal"]["met"])
            self.assertEqual(report["environment"]["display_driver"], "dummy")
            self.assertTrue(Path(report["recording"]["run_json"]).is_file())

    def test_explicit_dummy_mechanism_has_no_gui_fps_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.dummy_run(directory, allow_non_window=True)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "MECHANISM_ONLY")
            self.assertFalse(report["real_window_measurement"])
            phase = report["phases"]["measurement"]
            self.assertGreater(phase["total_frames"], 0)
            self.assertIsNone(phase["gui_average_fps"])
            self.assertIsNone(report["performance_goal"]["met"])
            self.assertTrue(phase["completed_duration"])
            self.assertTrue(phase["continuous_running"])
            self.assertEqual(report["terminal"]["state"], "RUNNING")
            self.assertFalse(report["blocked_unit_ids"])
            root = Path(report["directory"])
            with (root / "frames.csv").open(encoding="utf-8-sig", newline="") as source:
                frames = list(csv.DictReader(source))
            self.assertEqual(len(frames), phase["total_frames"])
            self.assertTrue(all(row["phase"] == "measurement" for row in frames))
            self.assertTrue(all(int(row["frame_end_monotonic_ns"]) > int(row["frame_start_monotonic_ns"]) for row in frames))
            self.assertAlmostEqual(sum(int(row["flip_interval_ns"]) for row in frames) / 1e9, phase["actual_wall_seconds"])
            self.assertTrue((root / "after_warmup.png").is_file())
            self.assertTrue((root / "finished.png").is_file())
            self.assertEqual(len(list(root.glob("*.png"))), 2)

    def test_early_game_finish_is_partial_nonzero_and_keeps_actual_finished_record(self):
        scene, data, metadata = make_workload(4, 0.07, 0.03)
        scene = replace(scene, rules=replace(scene.rules, time_limit=0.01))
        data["rules"]["time_limit"] = 0.01
        metadata["effective_scene_sha256"] = hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory, patch("tools.benchmark_simulation.make_workload", return_value=(scene, data, metadata)):
            code, report = self.dummy_run(directory, allow_non_window=True)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "PARTIAL")
            self.assertTrue(report["errors"])
            self.assertEqual(report["terminal"]["state"], "FINISHED")
            payload = json.loads(Path(report["recording"]["run_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["result"], report["terminal"])
            self.assertEqual(payload["result"]["finish_reason"], "time_limit")

    def test_window_close_saves_partial_state_and_never_claims_completed_duration(self):
        import pygame
        event = pygame.event.Event(pygame.QUIT)
        with tempfile.TemporaryDirectory() as directory, patch("pygame.event.get", return_value=[event]):
            code, report = self.dummy_run(directory, allow_non_window=True)
            self.assertEqual(code, 1)
            self.assertIn(report["status"], ("ERROR", "PARTIAL"))
            self.assertFalse(report["phases"]["warmup"]["completed_duration"])
            self.assertEqual(report["phases"]["warmup"]["total_frames"], 0)
            self.assertTrue(Path(report["recording"]["run_json"]).is_file())
            self.assertTrue(report["errors"])

    def test_record_export_failure_is_nonzero_and_keeps_frame_measurements(self):
        with tempfile.TemporaryDirectory() as directory, patch("tools.benchmark_simulation.write_full_record", side_effect=OSError("test blocked output")):
            code, report = self.dummy_run(directory, allow_non_window=True)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "ERROR")
            self.assertTrue(Path(report["directory"], "frames.csv").is_file())
            self.assertTrue(any("recording export" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
