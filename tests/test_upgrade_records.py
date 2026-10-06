"""Behavioral regressions for reproducible history and faithful analysis/replay."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_app import run_history
from sim_app.analysis import compare_runs, export_comparison, export_metrics, summarize_run
from sim_app.models import Point, RunState
from sim_app.recording import export_run
from sim_app.replay import load_replay
from sim_app.simulation import Simulation
from tests.test_motion import mission_config, scene_from_config
from tests.test_replay import game_config
from tests.test_scene import valid_config


class UpgradeRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        run_history._CACHE.clear()

    def export(self, simulation, name="run"):
        path = export_run(simulation, self.directory / name)["run"]
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_metadata_tracks_actual_start_finish_reset_and_stable_repeat_exports(self):
        simulation = Simulation(scene_from_config(mission_config()))
        before = copy.deepcopy(simulation.snapshots)
        _, ready = self.export(simulation)
        self.assertIsNone(ready["metadata"]["started_at"])
        self.assertIsNone(ready["metadata"]["ended_at"])
        self.assertEqual(ready["metadata"]["save_status"], "not_started")
        self.assertEqual(simulation.snapshots, before)
        simulation.start()
        simulation.advance(0.375)
        simulation.pause()
        _, paused = self.export(simulation)
        _, repeat = self.export(simulation)
        self.assertEqual(paused, repeat)
        self.assertEqual(paused["metadata"]["run_id"], ready["metadata"]["run_id"])
        self.assertTrue(paused["metadata"]["started_at"])
        self.assertIsNone(paused["metadata"]["ended_at"])
        self.assertEqual(paused["metadata"]["save_status"], "interrupted")
        simulation.resume()
        simulation.advance(100)
        _, complete = self.export(simulation)
        self.assertTrue(complete["metadata"]["ended_at"])
        self.assertEqual(complete["metadata"]["started_at"], paused["metadata"]["started_at"])
        self.assertEqual(complete["metadata"]["save_status"], "completed")
        self.assertEqual(complete["metadata"]["recording_format_version"], 2)
        self.assertEqual(complete["snapshots"][-1], complete["result"])
        simulation.reset()
        self.assertNotEqual(simulation.metadata["run_id"], complete["metadata"]["run_id"])
        self.assertIsNone(simulation.metadata["started_at"])

    def test_metadata_configuration_hash_is_canonical_and_environment_is_recorded(self):
        simulation = Simulation(scene_from_config(mission_config()))
        _, first = self.export(simulation, "first")
        _, second = self.export(Simulation(scene_from_config(mission_config())), "second")
        canonical = json.dumps(first["scene"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertEqual(first["metadata"]["config_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(first["metadata"]["config_sha256"], second["metadata"]["config_sha256"])
        self.assertNotEqual(first["metadata"]["run_id"], second["metadata"]["run_id"])
        self.assertEqual(len(first["metadata"]["source_sha256"]), 64)
        self.assertIn("python", first["metadata"]["dependencies"])
        self.assertIn("pygame", first["metadata"]["dependencies"])
        self.assertIn("platform", first["metadata"]["environment"])

    def test_interval_speed_is_independent_of_player_state_and_same_time_controls(self):
        simulation = Simulation(scene_from_config(mission_config()))
        simulation.start()
        simulation.advance(1)
        simulation.pause()
        simulation.resume()
        simulation.pause()
        path, payload = self.export(simulation)
        replay = load_replay(path)
        self.assertTrue(all(value is None for value in replay.recorded_speeds.values()))
        self.assertTrue(replay.seek_time(1))
        expected = {}
        frame = payload["snapshots"][-1]
        previous = next(row for row in reversed(payload["snapshots"]) if row["time"] < frame["time"])
        previous_units = {row["id"]: row for row in previous["units"]}
        for row in frame["units"]:
            expected[row["id"]] = (row["distance_travelled"] - previous_units[row["id"]]["distance_travelled"]) / (frame["time"] - previous["time"])
        self.assertEqual(replay.recorded_speeds, expected)
        replay.seek(next(n for n, row in enumerate(payload["snapshots"]) if row["time"] == 1))
        self.assertEqual(replay.recorded_speeds, expected)
        self.assertTrue(replay.resume())
        self.assertEqual(replay.recorded_speeds, expected)
        self.assertTrue(replay.pause())
        self.assertEqual(replay.recorded_speeds, expected)
        replay.seek(0)
        self.assertTrue(all(value is None for value in replay.recorded_speeds.values()))

    def test_event_seek_uses_first_exposing_frame_for_v2_and_v3_controls(self):
        for config in (mission_config(), game_config()):
            simulation = Simulation(scene_from_config(config))
            simulation.start()
            simulation.pause()
            simulation.resume()
            simulation.advance(0.125)
            simulation.pause()
            path, payload = self.export(simulation)
            replay = load_replay(path)
            for event_index in range(len(payload["events"])):
                self.assertTrue(replay.seek_event(event_index))
                self.assertGreater(len(replay.events), event_index)
                target = replay.index
                if target:
                    replay.seek(target - 1)
                    self.assertLessEqual(len(replay.events), event_index)
            for value in (-1, True, 1.5, len(payload["events"])):
                self.assertFalse(replay.seek_event(value))
            for value in (-1, True, float("nan"), float("inf"), "1"):
                self.assertFalse(replay.seek_time(value))
            self.assertTrue(replay.seek_time(10**6))
            self.assertEqual(replay.index, replay.count - 1)

    def test_bounded_trail_seek_discards_future_and_fast_forward_applies_only_target(self):
        config = game_config()
        config["fixed_dt"] = 0.01
        config["rules"].update(time_limit=8, score_limit=100000, tag_range=1)
        config["units"][0].update(speed=20, waypoints=[{"x": 800, "y": 200}])
        simulation = Simulation(scene_from_config(config))
        simulation.start()
        simulation.advance(8)
        path, payload = self.export(simulation)
        replay = load_replay(path)
        replay.seek(replay.count - 1)
        self.assertEqual(len(replay.units[0].trail), replay.TRAIL_LIMIT)
        middle = 101
        replay.seek(middle)
        expected = []
        for frame in payload["snapshots"][:middle + 1]:
            saved = frame["units"][0]
            point = Point(saved["x"], saved["y"])
            if not expected or point != expected[-1]:
                expected.append(point)
        self.assertEqual(replay.units[0].trail, expected[-replay.TRAIL_LIMIT:])
        self.assertEqual(replay.units[0].trail[-1], replay.units[0].position)
        previous_tail = list(replay.units[0].trail)
        replay.next()
        replay.previous()
        self.assertEqual(replay.units[0].trail, previous_tail)
        replay.reset()
        replay.start()
        with patch.object(replay, "_apply", wraps=replay._apply) as apply:
            replay.advance(8)
            self.assertEqual(apply.call_count, 1)
        self.assertEqual(replay.index, replay.count - 1)
        self.assertLessEqual(len(replay.units[0].trail), replay.TRAIL_LIMIT)

    def test_early_round_stop_is_incomplete_and_units_without_tasks_are_separate(self):
        config = game_config()
        config["rules"].update(time_limit=0.25, score_limit=100000, tag_range=1)
        config["units"][0].update(speed=10, waypoints=[{"x": 800, "y": 200}])
        simulation = Simulation(scene_from_config(config))
        simulation.start()
        simulation.advance(10)
        _, payload = self.export(simulation)
        summary = summarize_run(payload)
        self.assertEqual(summary["state"], "FINISHED")
        self.assertEqual(summary["finish_reason"], "time_limit")
        self.assertEqual(summary["completed_count"], 0)
        self.assertEqual(summary["incomplete_count"], 1)
        self.assertEqual(summary["unassigned_count"], 3)
        self.assertEqual(summary["per_unit"][0]["behavior"], "STOPPED")
        self.assertEqual(summary["data_quality"]["status"], "complete")

    def test_completed_blocked_and_interrupted_runs_have_distinct_result_counts(self):
        complete = Simulation(scene_from_config(mission_config()))
        complete.start()
        complete.advance(100)
        _, payload = self.export(complete, "complete")
        summary = summarize_run(payload)
        self.assertGreater(summary["completed_count"], 0)
        self.assertEqual(summary["incomplete_count"], 0)
        config = valid_config()
        config["obstacles"] = [{"id": "wall", "x": 400, "y": 0, "width": 40, "height": 600}]
        config["units"] = [config["units"][0]]
        config["units"][0].update(speed=80, waypoints=[{"x": 800, "y": 150}])
        blocked = Simulation(scene_from_config(config))
        blocked.start()
        _, payload = self.export(blocked, "blocked")
        summary = summarize_run(payload)
        self.assertEqual((summary["completed_count"], summary["blocked_count"], summary["incomplete_count"]), (0, 1, 0))
        self.assertTrue(summary["per_unit"][0]["reason"])
        paused = Simulation(scene_from_config(mission_config()))
        paused.start()
        paused.advance(0.375)
        paused.pause()
        _, payload = self.export(paused, "paused")
        summary = summarize_run(payload)
        self.assertEqual(summary["finish_reason"], "interrupted")
        self.assertGreater(summary["incomplete_count"], 0)
        self.assertFalse(summary["data_quality"]["run_complete"])

    def test_legacy_metadata_and_unrecorded_contacts_remain_unknown(self):
        simulation = Simulation(scene_from_config(mission_config()))
        simulation.start()
        simulation.advance(1)
        path, payload = self.export(simulation)
        payload.pop("metadata")
        path.write_text(json.dumps(payload), encoding="utf-8")
        replay = load_replay(path)
        self.assertEqual(replay.metadata, {})
        summary = summarize_run(payload)
        self.assertTrue(summary["data_quality"]["missing_metadata"])
        self.assertTrue(all(point["value"] is None for point in summary["series"]["score"]))
        self.assertTrue(all(point["value"] is None for point in summary["series"]["contact"]))
        self.assertEqual(summary["distance_total"], sum(row["distance_travelled"] for row in payload["result"]["units"]))

    def test_comparison_aligns_unit_ids_ignores_metadata_and_retains_each_time_extent(self):
        simulation = Simulation(scene_from_config(mission_config()))
        simulation.start()
        simulation.advance(1)
        simulation.pause()
        _, left = self.export(simulation, "left")
        simulation.resume()
        simulation.advance(1)
        simulation.pause()
        _, right = self.export(simulation, "right")
        right["scene"]["units"].reverse()
        right["metadata"]["run_id"] = "a-different-id"
        comparison = compare_runs(left, right)
        self.assertEqual(comparison["input_differences"], [])
        self.assertEqual(comparison["metric_differences"]["duration_s"]["delta"], 1)
        self.assertEqual(comparison["series"]["left"]["time"][-1]["time"], 1)
        self.assertEqual(comparison["series"]["right"]["time"][-1]["time"], 2)
        self.assertEqual([point["time"] for point in comparison["aligned_series"]], [0, 1])
        changed_id = right["scene"]["units"][0]["id"]
        right["scene"]["units"][0]["speed"] = 123
        comparison = compare_runs(left, right)
        self.assertEqual(len(comparison["input_differences"]), 1)
        self.assertEqual(comparison["input_differences"][0]["path"], f"scene.units[{changed_id}].speed")

    def test_metrics_and_comparison_csv_match_shared_metrics_and_have_source_paths(self):
        simulation = Simulation(scene_from_config(game_config()))
        simulation.start()
        simulation.advance(1)
        source, payload = self.export(simulation)
        summary = summarize_run(payload)
        path = export_metrics(payload, self.directory / "metrics.csv", source_path=source)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(row for row in rows if row["section"] == "summary" and row["metric"] == "distance_total")
        self.assertEqual(float(row["value"]), summary["distance_total"])
        self.assertEqual(row["source_path"], str(source))
        series_rows = [row for row in rows if row["section"] == "series" and row["metric"] == "score"]
        self.assertEqual([float(row["value"]) for row in series_rows], [point["value"] for point in summary["series"]["score"]])
        path = export_comparison(payload, payload, self.directory / "comparison.csv", left_source_path=source, right_source_path=source)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row = next(row for row in rows if row["section"] == "summary" and row["metric"] == "duration_s")
        self.assertEqual(float(row["delta"]), 0)
        self.assertEqual(row["left_source_path"], str(source))

    def test_history_persists_with_bad_records_and_notes_do_not_modify_facts(self):
        simulation = Simulation(scene_from_config(mission_config()))
        simulation.start()
        simulation.advance(1)
        source, payload = self.export(simulation, "good")
        before = source.read_bytes()
        bad = self.directory / "bad" / "run.json"
        bad.parent.mkdir()
        bad.write_text("{broken", encoding="utf-8")
        rows = run_history.list_runs(self.directory)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["status"] == "ERROR" for row in rows), 1)
        self.assertTrue(next(row for row in rows if row["status"] == "ERROR")["error"])
        with patch("sim_app.run_history.load_run", side_effect=AssertionError("unchanged history must use cache")):
            self.assertEqual(run_history.list_runs(self.directory), rows)
            note = run_history.save_note(source, "修改输入前的对照")
            refreshed = run_history.list_runs(self.directory)
        self.assertTrue(note.is_file())
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(next(row for row in refreshed if row["status"] != "ERROR")["note"], "修改输入前的对照")
        run_history._CACHE.clear()
        self.assertEqual(run_history.list_runs(self.directory), refreshed)
        self.assertEqual(run_history.load_run(source), payload)

    def test_bad_metadata_date_is_isolated_and_legacy_file_time_is_labeled(self):
        source, payload = self.export(Simulation(scene_from_config(valid_config())), "legacy")
        payload.pop("metadata")
        source.write_text(json.dumps(payload), encoding="utf-8")
        bad, malformed = self.export(Simulation(scene_from_config(valid_config())), "bad_metadata")
        malformed["metadata"]["started_at"] = 42
        bad.write_text(json.dumps(malformed), encoding="utf-8")
        rows = run_history.list_runs(self.directory)
        self.assertEqual(len(rows), 2)
        error = next(row for row in rows if row["status"] == "ERROR")
        self.assertIn("metadata.started_at", error["error"])
        legacy = next(row for row in rows if row["status"] != "ERROR")
        self.assertEqual(legacy["time_source"], "file_mtime")
        self.assertIn("非实际开始时间", legacy["time_label"])
        self.assertEqual(legacy["run_id"], "旧记录 / 未记录")

    def test_corrupt_note_is_isolated_from_valid_run(self):
        source, _ = self.export(Simulation(scene_from_config(valid_config())))
        note = run_history.save_note(source, "note")
        note.write_text("{broken", encoding="utf-8")
        row = run_history.list_runs(self.directory)[0]
        self.assertEqual(row["state"], "READY")
        self.assertIsNone(row["error"])
        self.assertTrue(row["note_error"])


if __name__ == "__main__":
    unittest.main()
