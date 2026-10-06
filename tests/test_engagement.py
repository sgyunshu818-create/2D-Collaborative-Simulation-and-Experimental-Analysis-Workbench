"""Knowledge boundaries and deterministic rules of the fictional tag game."""

import copy
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from sim_app.engagement import (
    TagCandidate, adjudicate_tags, duration_steps, line_of_sight,
    observe_opponents, refresh_contacts, tag_candidates,
)
from sim_app.models import BehaviorState, Contact, GameRules, Point, RunState, Team, Unit, UnitType
from sim_app.navigation import segment_clear
from sim_app.scene import SceneConfigError, load_scene, scene_from_data
from sim_app.simulation import Simulation
from tests.test_scene import valid_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def game_config():
    config = valid_config()
    config["obstacles"] = []
    config["rules"] = {
        "score_limit": 99, "time_limit": 5,
        "tag_cooldown": 0.3, "contact_ttl": 0.25,
    }
    for item, x in zip(config["units"], (100, 120, 150, 170)):
        item.update(x=x, y=100, sensor_range=300)
    return config


def game_sim(config=None):
    return Simulation(scene_from_data(game_config() if config is None else config))


class GameSceneTests(unittest.TestCase):
    def test_optional_rules_and_sensor_defaults_preserve_legacy(self):
        scene = scene_from_data(valid_config())
        self.assertIsNone(scene.rules)
        self.assertEqual([unit.sensor_range for unit in scene.units], [160, 240, 160, 240])
        config = valid_config()
        config["rules"] = {}
        scene = scene_from_data(config)
        self.assertEqual(scene.rules, GameRules())
        with self.assertRaises(FrozenInstanceError):
            scene.rules.score_limit = 2

    def test_rule_containers_and_boolean_are_strict(self):
        for value in (None, [], True, 1, "game"):
            config = game_config()
            config["rules"] = value
            with self.subTest(value=value), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config, "game-input.json")
            self.assertIn("game-input.json", str(caught.exception))
            self.assertIn("rules", str(caught.exception))
        for value in (0, 1, None, "true"):
            config = game_config()
            config["rules"]["sharing_enabled"] = value
            with self.subTest(sharing=value), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config)
            self.assertIn("rules.sharing_enabled", str(caught.exception))

    def test_ranges_and_durations_require_positive_finite_numbers(self):
        for field in ("tag_range", "tag_cooldown", "time_limit", "contact_ttl", "sensor_range"):
            for value in (0, -1, True, None, "90", float("nan"), float("inf")):
                config = game_config()
                if field == "sensor_range":
                    config["units"][0][field] = value
                    expected = "units[0].sensor_range"
                else:
                    config["rules"][field] = value
                    expected = f"rules.{field}"
                with self.subTest(field=field, value=value), self.assertRaises(SceneConfigError) as caught:
                    scene_from_data(config)
                self.assertIn(expected, str(caught.exception))

    def test_score_limit_is_strict_integer_and_unknown_rules_fail(self):
        for value in (0, -1, True, 3.0, "3", None):
            config = game_config()
            config["rules"]["score_limit"] = value
            with self.subTest(value=value), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config)
            self.assertIn("rules.score_limit", str(caught.exception))
        config = game_config()
        config["rules"]["unknown"] = 1
        with self.assertRaisesRegex(SceneConfigError, "rules.unknown"):
            scene_from_data(config)

    def test_game_requires_both_teams_but_legacy_can_use_one(self):
        config = game_config()
        config["units"] = config["units"][:2]
        with self.assertRaisesRegex(SceneConfigError, "each team"):
            scene_from_data(config)
        del config["rules"]
        self.assertEqual(len(scene_from_data(config).units), 2)


class ContactTests(unittest.TestCase):
    def test_sensor_boundary_and_independent_frozen_observation(self):
        sim = game_sim()
        red, _, blue, _ = sim.units
        red.sensor_range = 50
        contacts = observe_opponents(sim.scene, red, sim.units, 7)
        self.assertEqual(set(contacts), {blue.id})
        contact = contacts[blue.id]
        self.assertEqual((contact.position, contact.observed_step, contact.source_id, contact.shared),
                         (Point(150, 100), 7, red.id, False))
        self.assertIsNot(contact.position, blue.position)
        blue.position = Point(500, 500)
        self.assertEqual(contact.position, Point(150, 100))
        with self.assertRaises(FrozenInstanceError):
            contact.shared = True
        self.assertEqual(observe_opponents(sim.scene, red, sim.units, 8), {})

    def test_ground_obstacles_occlude_but_air_ignores_them(self):
        config = game_config()
        config["obstacles"] = [{"id": "wall", "x": 130, "y": 80, "width": 10, "height": 40}]
        sim = game_sim(config)
        red_ground, red_air, blue_ground, _ = sim.units
        self.assertNotIn(blue_ground.id, observe_opponents(sim.scene, red_ground, sim.units, 1))
        self.assertIn(blue_ground.id, observe_opponents(sim.scene, red_air, sim.units, 1))
        self.assertFalse(line_of_sight(sim.scene, Point(100, 80), Point(150, 80), UnitType.GROUND))

    def test_sight_uses_original_rectangle_without_navigation_clearance(self):
        config = game_config()
        config["obstacles"] = [{"id": "wall", "x": 130, "y": 80, "width": 10, "height": 40}]
        scene = scene_from_data(config)
        start, end = Point(100, 70), Point(150, 70)
        self.assertTrue(line_of_sight(scene, start, end, UnitType.GROUND))
        self.assertFalse(segment_clear(scene, start, end, UnitType.GROUND))

    def test_sharing_uses_current_direct_sight_and_does_not_cross_teams(self):
        sim = game_sim()
        red_ground, red_air, blue_ground, blue_air = sim.units
        red_ground.sensor_range = 1
        red_air.sensor_range = 300
        blue_ground.sensor_range = blue_air.sensor_range = 1
        events = refresh_contacts(sim.scene, sim.units, 1, True)
        contact = red_ground.contacts[blue_ground.id]
        self.assertTrue(contact.shared)
        self.assertEqual(contact.source_id, red_air.id)
        self.assertEqual(contact.observed_step, 1)
        self.assertFalse(blue_ground.contacts)
        self.assertFalse(blue_air.contacts)
        self.assertTrue(any(event.kind == "info_shared" for event in events))
        self.assertEqual(refresh_contacts(sim.scene, sim.units, 2, True), [])

    def test_shared_memory_cannot_relay_and_expires_at_ttl(self):
        sim = game_sim()
        red_ground, red_air, _, _ = sim.units
        red_ground.sensor_range = 1
        refresh_contacts(sim.scene, sim.units, 1, True)
        original = dict(red_ground.contacts)
        red_air.sensor_range = 1
        refresh_contacts(sim.scene, sim.units, 2, True)
        self.assertEqual(red_ground.contacts, original)
        events = refresh_contacts(sim.scene, sim.units, 3, True)
        self.assertFalse(red_ground.contacts)
        self.assertFalse(red_air.contacts)
        self.assertEqual(sum(event.kind == "contact_lost" and event.unit_id == red_ground.id for event in events), 2)

    def test_direct_sight_has_priority_over_teammate_broadcast(self):
        sim = game_sim()
        refresh_contacts(sim.scene, sim.units, 1, True)
        for unit in sim.units:
            self.assertTrue(unit.contacts)
            self.assertTrue(all(not contact.shared and contact.source_id == unit.id
                                for contact in unit.contacts.values()))

    def test_sharing_toggle_clears_only_shared_contacts_and_reset_restores_default(self):
        sim = game_sim()
        sim.units[0].sensor_range = 1
        sim.start()
        sim.advance(sim.scene.fixed_dt)
        self.assertTrue(sim.units[0].contacts)
        direct = dict(sim.units[1].contacts)
        sim.pause()
        self.assertTrue(sim.set_sharing(False))
        self.assertFalse(sim.units[0].contacts)
        self.assertEqual(sim.units[1].contacts, direct)
        self.assertFalse(sim.set_sharing(False))
        self.assertEqual(sim.snapshots[-1]["sharing_enabled"], False)
        self.assertTrue(sim.reset())
        self.assertTrue(sim.sharing_enabled)
        self.assertFalse(any(unit.contacts or unit.tagged_targets or unit.tag_count for unit in sim.units))
        self.assertEqual(sim.scores, {Team.RED: 0, Team.BLUE: 0})
        self.assertEqual((sim.finish_reason, sim.winner), ("", ""))
        self.assertEqual(sim.events, [])
        self.assertEqual(sim.snapshots, [sim.snapshot()])


class TagTests(unittest.TestCase):
    def test_candidates_use_only_known_contacts_and_remembered_positions(self):
        sim = game_sim()
        red, _, blue, _ = sim.units
        self.assertEqual(tag_candidates([red], 1, sim.scene.fixed_dt, sim.scene.rules, {}), ())
        red.contacts[blue.id] = Contact(blue.id, Point(110, 100), 1, red.id)
        blue.position = Point(600, 500)
        candidates = tag_candidates([red], 2, sim.scene.fixed_dt, sim.scene.rules, {})
        self.assertEqual(candidates, (TagCandidate(red.id, blue.id),))
        self.assertEqual(adjudicate_tags(sim.scene, sim.units, candidates), ())
        red.contacts[blue.id] = Contact(blue.id, Point(600, 500), 1, red.id)
        blue.position = Point(110, 100)
        self.assertEqual(tag_candidates([red], 2, sim.scene.fixed_dt, sim.scene.rules, {}), ())

    def test_referee_rejects_ground_occlusion_unknown_and_friendly_targets(self):
        config = game_config()
        config["obstacles"] = [{"id": "wall", "x": 130, "y": 80, "width": 10, "height": 40}]
        sim = game_sim(config)
        red_ground, red_air, blue_ground, _ = sim.units
        candidates = (TagCandidate(red_ground.id, blue_ground.id),
                      TagCandidate(red_air.id, blue_ground.id),
                      TagCandidate("unknown", blue_ground.id),
                      TagCandidate(blue_ground.id, "unknown"),
                      TagCandidate(blue_ground.id, sim.units[3].id))
        self.assertEqual(adjudicate_tags(sim.scene, sim.units, candidates),
                         (TagCandidate(red_air.id, blue_ground.id),))

    def test_cooldown_rounds_to_whole_steps_and_each_target_scores_once(self):
        sim = game_sim()
        self.assertEqual(duration_steps(0.3, 0.125), 3)
        self.assertEqual(duration_steps(1, 1 / 60), 60)
        self.assertGreater(duration_steps(1e308, 0.125), 10 ** 308)
        sim.start()
        sim.advance(0.125)
        self.assertEqual([unit.tag_count for unit in sim.units], [1, 1, 1, 1])
        self.assertEqual(sim.scores, {Team.RED: 2, Team.BLUE: 2})
        sim.advance(0.25)
        self.assertEqual([unit.tag_count for unit in sim.units], [1, 1, 1, 1])
        sim.advance(0.125)
        self.assertEqual([unit.tag_count for unit in sim.units], [2, 2, 2, 2])
        sim.advance(0.5)
        self.assertEqual([unit.tag_count for unit in sim.units], [2, 2, 2, 2])
        self.assertEqual(sum(event["kind"] == "virtual_tag" for event in sim.events), 8)

    def test_score_limit_settles_both_teams_before_deciding_tie(self):
        config = game_config()
        config["units"] = [config["units"][0], config["units"][2]]
        config["rules"]["score_limit"] = 1
        sim = game_sim(config)
        sim.start()
        sim.advance(10)
        self.assertEqual(sim.step_count, 1)
        self.assertEqual(sim.scores, {Team.RED: 1, Team.BLUE: 1})
        self.assertEqual((sim.winner, sim.finish_reason), ("draw", "score_limit"))
        self.assertTrue(all(unit.behavior is BehaviorState.STOPPED for unit in sim.units))
        self.assertEqual(sum(event["kind"] == "virtual_tag" for event in sim.events), 2)
        self.assertFalse(sim.set_sharing(False))
        self.assertFalse(sim.start() or sim.pause() or sim.resume())
        final = sim.snapshot()
        self.assertEqual(sim.advance(100), 0)
        self.assertEqual(sim.snapshot(), final)

    def test_unit_input_order_cannot_change_same_step_scores_or_events(self):
        config = game_config()
        config["rules"]["score_limit"] = 2
        first = game_sim(config)
        config["units"].reverse()
        second = game_sim(config)
        for sim in (first, second):
            sim.start()
            sim.advance(1)
        self.assertEqual((first.scores, first.winner, first.step_count),
                         (second.scores, second.winner, second.step_count))
        tags = lambda sim: [event for event in sim.events if event["kind"] == "virtual_tag"]
        self.assertEqual(tags(first), tags(second))

    def test_virtual_tag_does_not_destroy_units_or_change_preset_navigation(self):
        config = game_config()
        config["units"][0].update(speed=10, waypoints=[{"x": 300, "y": 100}])
        sim = game_sim(config)
        sim.start()
        sim.advance(0.125)
        unit = sim.units[0]
        self.assertGreater(unit.tag_count, 0)
        self.assertEqual(unit.behavior, BehaviorState.NAVIGATING)
        self.assertEqual(unit.position, Point(101.25, 100))
        self.assertEqual(unit.target, Point(300, 100))
        self.assertEqual(len(sim.units), 4)


class RoundTests(unittest.TestCase):
    def test_end_priority_is_score_then_time_then_mission_completion(self):
        config = game_config()
        config["units"] = [config["units"][0], config["units"][2]]
        config["units"][0].update(speed=1000, waypoints=[{"x": 110, "y": 100}])
        config["rules"].update(score_limit=1, time_limit=0.125)
        sim = game_sim(config)
        sim.start()
        sim.advance(1)
        self.assertEqual(sim.finish_reason, "score_limit")
        config["rules"]["score_limit"] = 99
        sim = game_sim(config)
        sim.start()
        sim.advance(1)
        self.assertEqual(sim.finish_reason, "time_limit")

    def test_no_routes_starts_then_time_limit_finishes_at_whole_step(self):
        config = game_config()
        config["rules"]["time_limit"] = 0.3
        sim = game_sim(config)
        self.assertFalse(sim.has_missions)
        self.assertTrue(sim.start())
        self.assertEqual(sim.state, RunState.RUNNING)
        sim.advance(10)
        self.assertEqual((sim.step_count, sim.sim_time), (3, 0.375))
        self.assertEqual((sim.finish_reason, sim.winner), ("time_limit", "draw"))

    def test_mission_completion_finishes_and_retains_blocked_reason(self):
        config = game_config()
        config["units"] = [config["units"][0], config["units"][2]]
        config["units"][0].update(speed=1000, waypoints=[{"x": 110, "y": 100}])
        sim = game_sim(config)
        sim.start()
        sim.advance(1)
        self.assertEqual(sim.finish_reason, "missions_complete")
        self.assertTrue(all(unit.behavior is BehaviorState.STOPPED for unit in sim.units))
        config["obstacles"] = [{"id": "wall", "x": 130, "y": 0, "width": 10, "height": 600}]
        config["units"][0]["waypoints"] = [{"x": 250, "y": 100}]
        sim = game_sim(config)
        sim.start()
        self.assertEqual(sim.finish_reason, "missions_complete")
        self.assertEqual(sim.units[0].behavior, BehaviorState.BLOCKED)
        self.assertTrue(sim.units[0].reason)
        self.assertEqual(sim.units[1].behavior, BehaviorState.STOPPED)

    def test_pause_freezes_scores_knowledge_and_records_control_events(self):
        sim = game_sim()
        sim.start()
        sim.advance(0.125)
        sim.pause()
        paused = sim.snapshot()
        self.assertEqual(sim.advance(100), 0)
        self.assertEqual(sim.snapshot(), paused)
        sim.resume()
        sim.advance(0.125)
        self.assertEqual(sim.step_count, 2)
        self.assertEqual([snap["state"] for snap in sim.snapshots if snap["step"] == 1],
                         ["RUNNING", "PAUSED", "RUNNING"])
        counts = [snap["event_count"] for snap in sim.snapshots]
        self.assertEqual(counts, sorted(counts))

    def test_snapshots_are_detached_and_record_every_game_logic_step(self):
        sim = game_sim()
        sim.start()
        sim.advance(0.5)
        self.assertEqual([snap["step"] for snap in sim.snapshots], [0, 0, 1, 2, 3, 4])
        external = sim.snapshot()
        external["scores"]["red"] = 200
        external["units"][0]["contacts"][0]["x"] = -200
        external["units"][0]["tagged_targets"].append("fake")
        self.assertNotEqual(external, sim.snapshot())
        self.assertNotEqual(sim.scores[Team.RED], 200)
        self.assertNotIn("fake", sim.units[0].tagged_targets)
        self.assertEqual(sim.snapshots[-1], sim.snapshot())

    def test_equal_fixed_steps_ignore_render_frame_partition(self):
        scene = scene_from_data(game_config())
        runs = []
        for frames in ([1], [0.125] * 8, [0.03125] * 32, [0.1, 0.025] * 8):
            sim = Simulation(scene)
            sim.start()
            for frame in frames:
                sim.advance(frame)
            runs.append(sim)
        for sim in runs[1:]:
            self.assertEqual(sim.snapshot(), runs[0].snapshot())
            self.assertEqual(sim.events, runs[0].events)
            self.assertEqual(sim.snapshots, runs[0].snapshots)

    def test_default_game_has_discoveries_shares_tags_and_reproducible_reset(self):
        scene = load_scene(PROJECT_ROOT / "configs" / "interaction_scene.json")
        sim = Simulation(scene)
        sim.start()
        sim.advance(100)
        self.assertEqual((sim.step_count, sim.sim_time), (422, 422 / 60))
        self.assertEqual((sim.scores, sim.winner, sim.finish_reason),
                         ({Team.RED: 3, Team.BLUE: 3}, "draw", "score_limit"))
        kinds = {event["kind"] for event in sim.events}
        self.assertTrue({"object_discovered", "info_shared", "virtual_tag", "round_finished"} <= kinds)
        first = copy.deepcopy((sim.snapshot(), sim.events, sim.snapshots))
        sim.reset()
        sim.start()
        sim.advance(100)
        self.assertEqual((sim.snapshot(), sim.events, sim.snapshots), first)
        sim.reset()
        sim.set_sharing(False)
        sim.start()
        sim.advance(100)
        self.assertEqual((sim.step_count, sim.scores, sim.winner),
                         (446, {Team.RED: 2, Team.BLUE: 3}, "blue"))
        self.assertNotIn("info_shared", {event["kind"] for event in sim.events})

    def test_stage_two_result_and_snapshot_fields_remain_identical(self):
        sim = Simulation(load_scene(PROJECT_ROOT / "configs" / "navigation_scene.json"))
        sim.start()
        sim.advance(100)
        self.assertEqual((sim.step_count, sim.sim_time, len(sim.events), len(sim.snapshots)),
                         (1038, 17.3, 30, 20))
        self.assertEqual(set(sim.snapshot()), {"step", "time", "state", "units"})
        self.assertNotIn("sensor_range", sim.snapshot()["units"][0])
        self.assertFalse(sim.set_sharing(True))
        with self.assertRaisesRegex(ValueError, "boolean"):
            game_sim().set_sharing(1)


if __name__ == "__main__":
    unittest.main()
