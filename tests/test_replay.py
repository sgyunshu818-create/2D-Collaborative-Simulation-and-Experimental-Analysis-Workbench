"""Recorded facts are validated and replayed without executing game logic."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sim_app.models import Point, RunState
from sim_app.recording import export_run
from sim_app.replay import ReplayError, load_replay
from sim_app.simulation import Simulation
from tests.test_motion import mission_config, scene_from_config
from tests.test_scene import valid_config


def game_config():
    config = valid_config()
    config["obstacles"] = []
    config["rules"] = {"sharing_enabled": True, "tag_range": 90,
                       "tag_cooldown": 0.125, "score_limit": 9,
                       "time_limit": 0.5, "contact_ttl": 0.25}
    for unit, (x, y) in zip(config["units"], ((100, 200), (110, 210), (160, 200), (170, 210))):
        unit.update(x=x, y=y, sensor_range=200)
    return config


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def save(self, config, elapsed=100, pause=False):
        simulation = Simulation(scene_from_config(config))
        simulation.start()
        simulation.advance(elapsed)
        if pause:
            simulation.pause()
        path = export_run(simulation, self.directory / "record")["run"]
        return simulation, path, json.loads(path.read_text(encoding="utf-8"))

    def rewrite(self, payload):
        path = self.directory / "edited.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def assert_frame(self, replay, frame):
        self.assertEqual(replay.current_snapshot, frame)
        self.assertEqual((replay.step_count, replay.sim_time, replay.source_state),
                         (frame["step"], frame["time"], RunState(frame["state"])))
        for unit, recorded in zip(replay.units, frame["units"]):
            self.assertEqual(unit.position, Point(recorded["x"], recorded["y"]))
            self.assertEqual(unit.behavior.value, recorded["behavior"])
            self.assertEqual(unit.waypoint_index, recorded["waypoint_index"])
            self.assertEqual(unit.distance_travelled, recorded["distance_travelled"])
            self.assertEqual(unit.reason, recorded["reason"])
            self.assertEqual(unit.path, ())
            if "contacts" in recorded:
                self.assertEqual(set(unit.contacts), {c["target_id"] for c in recorded["contacts"]})
                for contact in recorded["contacts"]:
                    restored = unit.contacts[contact["target_id"]]
                    self.assertEqual(restored.position, Point(contact["x"], contact["y"]))
                    self.assertEqual(restored.observed_step, contact["observed_step"])
                    self.assertEqual(restored.source_id, contact["source_id"])
                    self.assertEqual(restored.shared, contact["shared"])
                self.assertEqual(unit.tagged_targets, set(recorded["tagged_targets"]))
                self.assertEqual(unit.tag_count, recorded["tag_count"])

    def test_v2_roundtrip_preserves_every_sparse_frame_and_final_events(self):
        simulation, path, payload = self.save(mission_config())
        replay = load_replay(path)
        self.assertEqual(replay.format_version, 2)
        self.assertEqual(replay.scene, simulation.scene)
        self.assertIsNot(replay.scene, simulation.scene)
        self.assertEqual(replay.count, len(payload["snapshots"]))
        for index, frame in enumerate(payload["snapshots"]):
            self.assertTrue(replay.seek(index))
            self.assert_frame(replay, frame)
        self.assertEqual(replay.events, payload["events"])
        self.assertEqual(replay.result_snapshot, payload["result"])

    def test_same_step_pause_resume_frames_and_event_boundaries_are_preserved(self):
        simulation = Simulation(scene_from_config(mission_config()))
        expected = [(copy.deepcopy(simulation.snapshots[-1]), 0)]
        for operation in (simulation.start, simulation.pause, simulation.resume,
                          simulation.pause, simulation.resume):
            operation()
            expected.append((copy.deepcopy(simulation.snapshots[-1]), len(simulation.events)))
        path = export_run(simulation, self.directory)["run"]
        replay = load_replay(path)
        self.assertEqual(replay.count, len(expected))
        for index, (frame, count) in enumerate(expected):
            replay.seek(index)
            self.assert_frame(replay, frame)
            self.assertEqual(replay.events, simulation.events[:count])

    def test_optional_sensor_setting_in_v2_scene_is_not_lost_on_export(self):
        config = valid_config()
        config["units"][1]["sensor_range"] = 321
        simulation, path, payload = self.save(config, elapsed=0.125)
        self.assertEqual(payload["format_version"], 2)
        replay = load_replay(path)
        self.assertEqual(replay.scene, simulation.scene)
        self.assertEqual(replay.units[1].sensor_range, 321)

    def test_v3_roundtrip_restores_every_tick_contacts_scores_and_source_state(self):
        simulation, path, payload = self.save(game_config())
        self.assertTrue(simulation.finished)
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(payload["scene"]["rules"]["time_limit"], 0.5)
        self.assertEqual(payload["scene"]["units"][0]["sensor_range"], 200)
        replay = load_replay(path)
        self.assertEqual(replay.scene, simulation.scene)
        self.assertEqual({frame["step"] for frame in replay.snapshots},
                         set(range(simulation.step_count + 1)))
        for index, frame in enumerate(payload["snapshots"]):
            replay.seek(index)
            self.assert_frame(replay, frame)
            self.assertEqual(replay.scores, frame["scores"])
            self.assertEqual(replay.events, payload["events"][:frame["event_count"]])
            self.assertEqual(replay.sharing_enabled, frame["sharing_enabled"])
        self.assertEqual(replay.finish_reason, payload["result"]["finish_reason"])
        self.assertEqual(replay.winner, payload["result"]["winner"])
        self.assertEqual(replay.events, simulation.events)

    def test_v3_csv_keeps_extra_event_fields_and_contact_json(self):
        simulation, path, payload = self.save(game_config())
        with path.with_name("events.csv").open(encoding="utf-8-sig", newline="") as stream:
            events = list(csv.DictReader(stream))
        self.assertEqual(len(events), len(simulation.events))
        for original, row in zip(simulation.events, events):
            extras = {key: value for key, value in original.items()
                      if key not in ("step", "time", "kind", "unit_id", "message")}
            self.assertEqual(json.loads(row["details"]), extras)
        with path.with_name("states.csv").open(encoding="utf-8-sig", newline="") as stream:
            states = list(csv.DictReader(stream))
        expected = [(frame, unit) for frame in payload["snapshots"] for unit in frame["units"]]
        self.assertEqual(len(states), len(expected))
        for row, (frame, unit) in zip(states, expected):
            self.assertEqual(json.loads(row["contacts"]), unit["contacts"])
            self.assertEqual(json.loads(row["tagged_targets"]), unit["tagged_targets"])
            self.assertEqual(int(row["score_red"]), frame["scores"]["red"])
            self.assertEqual(int(row["score_blue"]), frame["scores"]["blue"])
            self.assertEqual(row["finish_reason"], frame["finish_reason"])

    def test_export_never_mutates_game_records_or_units(self):
        simulation = Simulation(scene_from_config(game_config()))
        simulation.start()
        simulation.advance(0.25)
        before = copy.deepcopy((simulation.units, simulation.events, simulation.snapshots, simulation.snapshot()))
        export_run(simulation, self.directory)
        self.assertEqual((simulation.units, simulation.events, simulation.snapshots, simulation.snapshot()), before)

    def test_seek_pause_reset_and_large_advance_preserve_saved_partial_result(self):
        _, path, payload = self.save(mission_config(), elapsed=2.375, pause=True)
        replay = load_replay(path)
        replay.start()
        replay.advance(0.25)
        replay.pause()
        before = (replay.current_snapshot, replay.index)
        self.assertEqual(replay.advance(999), 0)
        self.assertEqual((replay.current_snapshot, replay.index), before)
        replay.resume()
        replay.advance(0.75)
        self.assertEqual(replay.sim_time, 1)
        replay.seek(0)
        self.assertEqual(replay.state, RunState.PAUSED)
        replay.resume()
        replay.advance(float.fromhex("0x1.fffffffffffffp+1023"))
        self.assertTrue(replay.finished)
        self.assertEqual(replay.source_state, RunState.PAUSED)
        self.assertEqual(replay.current_snapshot, payload["result"])
        replay.reset()
        self.assertEqual((replay.index, replay.state), (0, RunState.READY))
        self.assertEqual(replay.events, [])
        self.assertFalse(replay.seek(-1))
        self.assertFalse(replay.seek(replay.count))
        self.assertFalse(replay.seek(True))
        self.assertTrue(replay.next())
        self.assertTrue(replay.previous())

    def test_single_ready_frame_can_play_to_end_without_becoming_original_finished(self):
        simulation = Simulation(scene_from_config(valid_config()))
        replay = load_replay(export_run(simulation, self.directory)["run"])
        self.assertEqual(replay.count, 1)
        self.assertTrue(replay.start())
        self.assertTrue(replay.finished)
        self.assertEqual(replay.source_state, RunState.READY)

    def test_replay_does_not_execute_simulation_or_navigation(self):
        _, path, payload = self.save(game_config())
        with patch("sim_app.simulation.Simulation", side_effect=AssertionError("must not create a simulation")), \
             patch("sim_app.simulation.Simulation.advance", side_effect=AssertionError("must not simulate")), \
             patch("sim_app.navigation.plan_path", side_effect=AssertionError("must not plan routes")):
            replay = load_replay(path)
            replay.start()
            replay.advance(100)
            self.assertEqual(replay.current_snapshot, payload["result"])

    def test_returned_frames_events_and_unit_contacts_are_independent_copies(self):
        _, path, payload = self.save(game_config())
        first = load_replay(path)
        second = load_replay(path)
        first.seek(first.count - 1)
        original = first.current_snapshot
        first.units[0].position = Point(0, 0)
        first.units[0].contacts.clear()
        first.scores["red"] = 1000
        first.current_snapshot["units"][0]["x"] = 0
        first.snapshots[-1]["scores"]["red"] = 1000
        first.events.clear()
        first.seek(first.index)
        self.assert_frame(first, original)
        self.assertEqual(first.scores, original["scores"])
        self.assertEqual(second.index, 0)
        self.assertEqual(payload["result"], first.result_snapshot)

    def test_non_finite_or_invalid_elapsed_time_is_rejected_in_any_playback_state(self):
        _, path, _ = self.save(mission_config(), elapsed=2)
        replay = load_replay(path)
        for state in RunState:
            replay.state = state
            for value in (True, None, "1", -1, float("nan"), float("inf"), 10 ** 1000):
                with self.subTest(state=state, value=value):
                    with self.assertRaises(ValueError):
                        replay.advance(value)

    def test_bad_json_encoding_missing_file_and_unknown_versions_have_filename(self):
        path = self.directory / "broken.json"
        for contents in (b"{", b"\xff\xfe", b"[]", b'{"format_version": 99}',
                         b'{"format_version": true}', b'{"format_version": 3, "scene": NaN}'):
            path.write_bytes(contents)
            with self.subTest(contents=contents):
                with self.assertRaises(ReplayError) as caught:
                    load_replay(path)
                self.assertIn(str(path), str(caught.exception))
        with self.assertRaises(ReplayError):
            load_replay(self.directory / "missing.json")

    def test_bad_record_frames_are_rejected_before_playback(self):
        _, _, original = self.save(mission_config(), elapsed=2.375)
        mutations = [
            lambda p: p.update(snapshots=[]),
            lambda p: p["snapshots"][0].update(state=[]),
            lambda p: p["snapshots"][0].update(time=-1),
            lambda p: p["snapshots"][0].update(step=True),
            lambda p: p["snapshots"][0]["units"][0].update(id="stranger"),
            lambda p: p["snapshots"][0]["units"][0].update(x=-1),
            lambda p: p["snapshots"][0]["units"][0].update(y=600),
            lambda p: p["snapshots"][0]["units"][0].update(x=float("inf")),
            lambda p: p["snapshots"][0]["units"][0].update(behavior=[]),
            lambda p: p["snapshots"][0]["units"][0].update(waypoint_index=99),
            lambda p: p["snapshots"][0]["units"].pop(),
            lambda p: p["snapshots"].reverse(),
            lambda p: p["result"].update(state="READY"),
            lambda p: p["events"][0].update(unit_id="unknown"),
            lambda p: p["events"][0].update(time=0.1),
            lambda p: p["scene"].update(world=None),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.subTest(mutation=index):
                with self.assertRaises(ReplayError) as caught:
                    load_replay(self.rewrite(changed))
                self.assertIn("edited.json", str(caught.exception))

    def test_bad_game_scores_contacts_and_event_counts_are_rejected(self):
        _, _, original = self.save(game_config())
        contact_frame = next(index for index, frame in enumerate(original["snapshots"])
                             if frame["units"][0]["contacts"])
        mutations = [
            lambda p: p["snapshots"][0].update(scores={"red": True, "blue": 0}),
            lambda p: p["snapshots"][0].update(scores={"red": -1, "blue": 0}),
            lambda p: p["snapshots"][0].update(scores={"red": 0}),
            lambda p: p["snapshots"][0].update(sharing_enabled=1),
            lambda p: p["snapshots"][0].update(event_count=10000),
            lambda p: p["snapshots"][-1].update(event_count=0),
            lambda p: p["snapshots"][contact_frame]["units"][0]["contacts"][0].update(observed_step=1000),
            lambda p: p["snapshots"][contact_frame]["units"][0]["contacts"][0].update(target_id="unknown"),
            lambda p: p["snapshots"][contact_frame]["units"][0]["contacts"][0].update(shared="true"),
            lambda p: p["snapshots"][0]["units"][0].update(sensor_range=0),
            lambda p: p["events"].reverse(),
            lambda p: p.update(format_version=2),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(original)
            mutate(changed)
            with self.subTest(mutation=index):
                with self.assertRaises(ReplayError):
                    load_replay(self.rewrite(changed))


if __name__ == "__main__":
    unittest.main()
