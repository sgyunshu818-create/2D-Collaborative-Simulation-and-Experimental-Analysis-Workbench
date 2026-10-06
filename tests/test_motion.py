"""Mission behavior, validation and fixed-step reproducibility without a window."""

import json
import tempfile
import unittest
from pathlib import Path

from sim_app.models import BehaviorState, Point, RunState, Team, UnitType
from sim_app.navigation import segment_clear
from sim_app.scene import SceneConfigError, load_scene
from sim_app.simulation import Simulation
from tests.test_scene import valid_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mission_config():
    config = valid_config()
    for unit in config["units"]:
        unit.update(speed=80, waypoints=[{"x": 450, "y": 300}], return_home=True)
    return config


def scene_from_config(config):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scene.json"
        path.write_text(json.dumps(config, allow_nan=True), encoding="utf-8")
        return load_scene(path)


class MotionConfigTests(unittest.TestCase):
    def test_missing_motion_fields_keep_original_static_scene(self):
        scene = scene_from_config(valid_config())
        self.assertTrue(all(unit.speed == 0 for unit in scene.units))
        self.assertTrue(all(unit.waypoints == () and not unit.return_home for unit in scene.units))
        self.assertFalse(Simulation(scene).has_missions)

    def test_route_without_explicit_speed_gets_type_default(self):
        config = valid_config()
        for unit in config["units"]:
            unit.update(waypoints=[{"x": 450, "y": 300}], return_home=True)
        scene = scene_from_config(config)
        for unit in scene.units:
            self.assertEqual(unit.speed, 90 if unit.unit_type == UnitType.GROUND else 140)
            self.assertEqual(unit.waypoints, (Point(450, 300),))
            self.assertTrue(unit.return_home)

    def test_multiple_waypoint_objects_preserve_order(self):
        config = mission_config()
        config["units"][0]["waypoints"] = [{"x": 440, "y": 300}, {"x": 500, "y": 300}]
        scene = scene_from_config(config)
        self.assertEqual(scene.units[0].waypoints, (Point(440, 300), Point(500, 300)))

    def test_invalid_motion_numbers_and_return_flag_have_precise_errors(self):
        for field, values in (
            ("speed", (0, -1, True, "80", None, float("nan"), float("inf"))),
            ("return_home", (0, 1, "true", None, [])),
        ):
            for value in values:
                config = mission_config()
                config["units"][0][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(SceneConfigError) as caught:
                        scene_from_config(config)
                    self.assertIn(f"units[0].{field}", str(caught.exception))

    def test_waypoint_shape_coordinates_and_bounds_are_validated(self):
        values = (None, {}, "450,300", [1], [[1]], [[1, 2]], [[1, 2, 3]],
                  [{"x": True, "y": 300}], [{"x": 450, "y": float("nan")}],
                  [{"x": 900, "y": 300}], [{"x": 450, "y": -1}],
                  [{"x": 450}], [["450", 300]])
        for value in values:
            config = mission_config()
            config["units"][0]["waypoints"] = value
            with self.subTest(waypoints=value):
                with self.assertRaises(SceneConfigError) as caught:
                    scene_from_config(config)
                self.assertIn("units[0].waypoints", str(caught.exception))


class MotionTests(unittest.TestCase):
    def finish(self, simulation, limit=5000):
        simulation.start()
        for _ in range(limit):
            if simulation.finished:
                return
            simulation.advance(simulation.scene.fixed_dt)
        self.fail("Mission failed to finish within its logical-step limit")

    def test_motion_is_bounded_by_speed_and_stops_exactly_at_goal(self):
        config = valid_config()
        config["obstacles"] = []
        config["units"] = [config["units"][0]]
        config["units"][0].update(speed=80, waypoints=[{"x": 101, "y": 150}])
        simulation = Simulation(scene_from_config(config))
        self.assertTrue(simulation.has_missions)
        simulation.start()
        unit = simulation.units[0]
        self.assertEqual(unit.behavior, BehaviorState.NAVIGATING)
        simulation.advance(0.125)
        self.assertEqual(unit.position, Point(90, 150))
        simulation.advance(0.125)
        self.assertEqual(unit.position, Point(100, 150))
        simulation.advance(0.125)
        self.assertEqual(unit.position, Point(101, 150))
        self.assertAlmostEqual(unit.distance_travelled, 21)
        self.assertEqual(unit.behavior, BehaviorState.STOPPED)
        self.assertEqual(simulation.state, RunState.FINISHED)
        self.assertTrue(simulation.finished)
        self.assertEqual(simulation.advance(100), 0)
        self.assertEqual((simulation.step_count, simulation.sim_time), (3, 0.375))

    def test_high_speed_step_follows_all_safe_segments_without_overshoot(self):
        config = valid_config()
        config["units"] = [config["units"][0]]
        config["units"][0].update(x=80, y=210, speed=10000,
                                  waypoints=[{"x": 500, "y": 210}])
        simulation = Simulation(scene_from_config(config))
        self.finish(simulation)
        unit = simulation.units[0]
        self.assertEqual(unit.position, Point(500, 210))
        self.assertEqual(simulation.step_count, 1)
        self.assertGreater(unit.distance_travelled, 420)
        self.assertGreaterEqual(len(unit.trail), 4)
        for start, goal in zip(unit.trail, unit.trail[1:]):
            self.assertTrue(segment_clear(simulation.scene, start, goal, UnitType.GROUND),
                            (start, goal))

    def test_pause_freezes_motion_and_resume_discards_paused_time(self):
        simulation = Simulation(scene_from_config(mission_config()))
        simulation.start()
        simulation.advance(0.3125)
        simulation.pause()
        positions = [unit.position for unit in simulation.units]
        trail = [tuple(unit.trail) for unit in simulation.units]
        self.assertEqual(simulation.advance(100), 0)
        self.assertEqual([unit.position for unit in simulation.units], positions)
        self.assertEqual([tuple(unit.trail) for unit in simulation.units], trail)
        self.assertEqual(simulation.step_count, 2)
        simulation.resume()
        # The valid pre-pause fractional step is kept; the 100 paused seconds
        # do not enter the accumulator or create a catch-up burst.
        self.assertEqual(simulation.advance(0.0625), 1)
        self.assertEqual(simulation.step_count, 3)
        self.assertEqual(simulation.advance(0.0625), 0)
        self.assertEqual(simulation.step_count, 3)

    def test_equal_fixed_steps_yield_equal_motion_events_and_snapshots(self):
        scene = scene_from_config(mission_config())
        runs = []
        for frame_times in ([2.0], [0.125] * 16, [0.03125] * 64,
                            [0.1, 0.025] * 16):
            simulation = Simulation(scene)
            simulation.start()
            for frame_dt in frame_times:
                simulation.advance(frame_dt)
            self.assertEqual(simulation.step_count, 16)
            runs.append(simulation)
        expected = runs[0]
        for simulation in runs[1:]:
            self.assertEqual(simulation.events, expected.events)
            self.assertEqual(simulation.snapshots, expected.snapshots)
            self.assertEqual(
                [(u.position, u.behavior, u.waypoint_index, u.distance_travelled) for u in simulation.units],
                [(u.position, u.behavior, u.waypoint_index, u.distance_travelled) for u in expected.units],
            )

    def test_reset_discards_run_records_trails_and_mutated_units(self):
        scene = scene_from_config(mission_config())
        simulation = Simulation(scene)
        simulation.start()
        simulation.advance(2.0)
        previous = simulation.units[0]
        self.assertGreater(len(previous.trail), 1)
        self.assertTrue(simulation.events)
        simulation.reset()
        self.assertEqual(simulation.state, RunState.READY)
        self.assertEqual((simulation.step_count, simulation.sim_time), (0, 0))
        self.assertEqual(simulation.events, [])
        self.assertEqual(len(simulation.snapshots), 1)
        self.assertEqual(simulation.snapshots[0]["step"], 0)
        fresh = Simulation(scene)
        self.assertEqual(simulation.snapshots, fresh.snapshots)
        self.assertIsNot(simulation.units[0], previous)
        for unit, spec in zip(simulation.units, scene.units):
            self.assertEqual(unit.position, spec.position)
            self.assertEqual(unit.behavior, BehaviorState.IDLE)
            self.assertEqual(unit.distance_travelled, 0)
            self.assertEqual(list(unit.trail), [spec.position])
        simulation.start()
        fresh.start()
        simulation.advance(1.0)
        fresh.advance(1.0)
        self.assertEqual(simulation.snapshots, fresh.snapshots)
        self.assertEqual(simulation.events, fresh.events)

    def test_waypoints_including_initial_point_finish_and_return_home(self):
        config = valid_config()
        config["obstacles"] = []
        config["units"] = [config["units"][0]]
        config["units"][0].update(speed=1000,
                                  waypoints=[{"x": 80, "y": 150}, {"x": 200, "y": 150},
                                             {"x": 200, "y": 150}],
                                  return_home=True)
        simulation = Simulation(scene_from_config(config))
        self.finish(simulation)
        unit = simulation.units[0]
        self.assertEqual(unit.position, simulation.scene.return_points[Team.RED])
        self.assertEqual(unit.waypoint_index, 3)
        self.assertEqual(unit.behavior, BehaviorState.STOPPED)
        self.assertGreater(unit.distance_travelled, 120)
        self.assertTrue(any(event["unit_id"] == unit.id for event in simulation.events))

    def test_static_units_do_not_prevent_mission_completion(self):
        config = valid_config()
        config["obstacles"] = []
        config["units"][0].update(speed=1000, waypoints=[{"x": 200, "y": 150}])
        simulation = Simulation(scene_from_config(config))
        self.finish(simulation)
        self.assertEqual(simulation.units[0].behavior, BehaviorState.STOPPED)
        for unit, spec in zip(simulation.units[1:], simulation.scene.units[1:]):
            self.assertEqual(unit.position, spec.position)
            self.assertEqual(unit.behavior, BehaviorState.IDLE)
        self.assertEqual(simulation.state, RunState.FINISHED)

    def test_return_only_mission_uses_team_return_point(self):
        config = valid_config()
        config["units"] = [config["units"][1]]
        config["units"][0].update(return_home=True)
        simulation = Simulation(scene_from_config(config))
        self.assertTrue(simulation.has_missions)
        simulation.start()
        self.assertEqual(simulation.units[0].behavior, BehaviorState.RETURNING)
        self.finish(simulation)
        self.assertEqual(simulation.units[0].position, simulation.scene.return_points[Team.RED])
        self.assertEqual(simulation.units[0].behavior, BehaviorState.STOPPED)

    def test_finished_controls_preserve_final_result_until_reset(self):
        config = valid_config()
        config["obstacles"] = []
        config["units"] = [config["units"][0]]
        config["units"][0].update(speed=1000, waypoints=[{"x": 200, "y": 150}])
        simulation = Simulation(scene_from_config(config))
        self.finish(simulation)
        final = (simulation.step_count, simulation.sim_time, simulation.units[0].position)
        for operation in ("start", "pause", "resume"):
            with self.subTest(operation=operation):
                self.assertFalse(getattr(simulation, operation)())
                self.assertEqual(simulation.state, RunState.FINISHED)
                self.assertEqual((simulation.step_count, simulation.sim_time,
                                  simulation.units[0].position), final)
        self.assertTrue(simulation.reset())
        self.assertTrue(simulation.start())
        self.assertEqual(simulation.state, RunState.RUNNING)

    def test_unreachable_ground_finishes_blocked_while_air_reaches_goal(self):
        config = valid_config()
        config["obstacles"] = [{"id": "wall", "x": 400, "y": 0, "width": 40, "height": 600}]
        config["units"] = config["units"][:2]
        for unit in config["units"]:
            unit.update(speed=1000, waypoints=[{"x": 800, "y": 150}])
        simulation = Simulation(scene_from_config(config))
        self.finish(simulation)
        ground, air = simulation.units
        self.assertEqual(ground.behavior, BehaviorState.BLOCKED)
        self.assertTrue(ground.reason)
        self.assertEqual(ground.position, simulation.scene.units[0].position)
        self.assertEqual(air.behavior, BehaviorState.STOPPED)
        self.assertEqual(air.position, Point(800, 150))
        self.assertEqual(simulation.state, RunState.FINISHED)
        self.assertTrue(any(event["unit_id"] == ground.id for event in simulation.events))

    def test_navigation_demo_four_units_complete_routes_and_return_home(self):
        scene = load_scene(PROJECT_ROOT / "configs" / "navigation_scene.json")
        simulation = Simulation(scene)
        self.assertEqual(len(simulation.units), 4)
        self.finish(simulation, limit=12000)
        for unit, spec in zip(simulation.units, scene.units):
            self.assertEqual(unit.position, scene.return_points[unit.team])
            self.assertEqual(unit.behavior, BehaviorState.STOPPED)
            self.assertEqual(unit.waypoint_index, len(spec.waypoints))
            self.assertGreater(unit.distance_travelled, 0)
            for start, goal in zip(unit.trail, unit.trail[1:]):
                self.assertTrue(segment_clear(scene, start, goal, unit.unit_type))
        self.assertEqual(simulation.state, RunState.FINISHED)
        self.assertEqual(simulation.snapshots[-1]["state"], RunState.FINISHED.value)


if __name__ == "__main__":
    unittest.main()
