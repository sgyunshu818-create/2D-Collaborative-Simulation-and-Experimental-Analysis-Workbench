"""Scene loading checks without a display or optional third-party dependency."""

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from sim_app.models import Team, UnitType
from sim_app.scene import SceneConfigError, load_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_config():
    """A small, complete configuration with both teams and both unit types."""
    return {
        "name": "test_scene",
        "world": {"width": 900, "height": 600},
        "fixed_dt": 0.125,
        "spawn_points": {
            "red": {"x": 80, "y": 120},
            "blue": {"x": 800, "y": 460},
        },
        "return_points": {
            "red": {"x": 80, "y": 120},
            "blue": {"x": 800, "y": 460},
        },
        "obstacles": [
            {"id": "obstacle_01", "x": 300, "y": 180, "width": 100, "height": 80},
            {"id": "obstacle_02", "x": 550, "y": 350, "width": 120, "height": 70},
        ],
        "units": [
            {"id": "red_ground_01", "team": "red", "type": "ground", "x": 80, "y": 150},
            {"id": "red_air_01", "team": "red", "type": "air", "x": 130, "y": 100},
            {"id": "blue_ground_01", "team": "blue", "type": "ground", "x": 800, "y": 450},
            {"id": "blue_air_01", "team": "blue", "type": "air", "x": 750, "y": 500},
        ],
    }


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "scene.json"

    def load(self, config):
        self.path.write_text(json.dumps(config, allow_nan=True), encoding="utf-8")
        return load_scene(self.path)

    def assert_invalid(self, config, field):
        with self.assertRaises(SceneConfigError) as caught:
            self.load(config)
        self.assertIn(str(self.path), str(caught.exception))
        self.assertIn(field, str(caught.exception))

    def test_default_config_has_each_team_and_unit_type(self):
        scene = load_scene(PROJECT_ROOT / "configs" / "default_scene.json")
        self.assertEqual((scene.width, scene.height), (900, 600))
        self.assertEqual(len(scene.units), 4)
        self.assertEqual(len(scene.obstacles), 2)
        self.assertEqual(
            {(unit.team, unit.unit_type) for unit in scene.units},
            {(team, unit_type) for team in Team for unit_type in UnitType},
        )
        self.assertEqual(set(scene.spawn_points), set(Team))
        self.assertEqual(set(scene.return_points), set(Team))
        self.assertAlmostEqual(scene.fixed_dt, 1 / 60)

    def test_scene_initial_data_is_frozen(self):
        scene = self.load(valid_config())
        for instance, attribute, value in (
            (scene, "width", 100),
            (scene.units[0], "id", "changed"),
            (scene.units[0].position, "x", 0),
            (scene.obstacles[0], "width", 1),
        ):
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, attribute, value)

    def test_malformed_json_reports_the_file(self):
        self.path.write_text('{"name": "broken",}', encoding="utf-8")
        with self.assertRaises(SceneConfigError) as caught:
            load_scene(self.path)
        self.assertIn(str(self.path), str(caught.exception))

    def test_missing_file_reports_the_file(self):
        with self.assertRaises(SceneConfigError) as caught:
            load_scene(self.path)
        self.assertIn(str(self.path), str(caught.exception))

    def test_required_fields_have_precise_errors(self):
        for field in valid_config():
            config = valid_config()
            del config[field]
            with self.subTest(field=field):
                self.assert_invalid(config, field)
        cases = (
            ("world", "width", "world.width"),
            ("spawn_points", "red", "spawn_points.red"),
            ("return_points", "blue", "return_points.blue"),
        )
        for container, key, field in cases:
            config = valid_config()
            del config[container][key]
            with self.subTest(field=field):
                self.assert_invalid(config, field)
        for collection, key, field in (
            ("units", "team", "units[0].team"),
            ("units", "x", "units[0].x"),
            ("obstacles", "width", "obstacles[0].width"),
        ):
            config = valid_config()
            del config[collection][0][key]
            with self.subTest(field=field):
                self.assert_invalid(config, field)

    def test_wrong_container_shapes_are_rejected(self):
        for field, value in (
            ("world", []),
            ("spawn_points", []),
            ("return_points", None),
            ("obstacles", {}),
            ("units", {}),
        ):
            config = valid_config()
            config[field] = value
            with self.subTest(field=field):
                self.assert_invalid(config, field)
        self.assert_invalid([], "$")

    def test_duplicate_unit_and_obstacle_ids_are_rejected(self):
        for collection in ("units", "obstacles"):
            config = valid_config()
            config[collection][1]["id"] = config[collection][0]["id"]
            with self.subTest(collection=collection):
                self.assert_invalid(config, f"{collection}[1].id")

    def test_invalid_team_and_type_report_the_unit_field(self):
        for key, value in (("team", "green"), ("type", "boat"), ("team", 1), ("type", None)):
            config = valid_config()
            config["units"][2][key] = value
            with self.subTest(key=key, value=value):
                self.assert_invalid(config, f"units[2].{key}")

    def test_map_size_step_and_obstacle_size_must_be_positive(self):
        for value in (0, -1):
            for key in ("width", "height"):
                config = valid_config()
                config["world"][key] = value
                with self.subTest(field=f"world.{key}", value=value):
                    self.assert_invalid(config, f"world.{key}")
                config = valid_config()
                config["obstacles"][0][key] = value
                with self.subTest(field=f"obstacles[0].{key}", value=value):
                    self.assert_invalid(config, f"obstacles[0].{key}")
            config = valid_config()
            config["fixed_dt"] = value
            with self.subTest(field="fixed_dt", value=value):
                self.assert_invalid(config, "fixed_dt")

    def test_numeric_fields_reject_booleans_strings_and_non_finite_values(self):
        targets = (
            (("world", "width"), "world.width"),
            (("world", "height"), "world.height"),
            (("fixed_dt",), "fixed_dt"),
            (("spawn_points", "red", "x"), "spawn_points.red.x"),
            (("return_points", "blue", "y"), "return_points.blue.y"),
            (("units", 0, "x"), "units[0].x"),
            (("units", 0, "y"), "units[0].y"),
            (("obstacles", 0, "x"), "obstacles[0].x"),
            (("obstacles", 0, "y"), "obstacles[0].y"),
            (("obstacles", 0, "width"), "obstacles[0].width"),
            (("obstacles", 0, "height"), "obstacles[0].height"),
        )
        for value in (True, False, "10", float("nan"), float("inf"), -float("inf")):
            for path, field in targets:
                config = valid_config()
                current = config
                for part in path[:-1]:
                    current = current[part]
                current[path[-1]] = value
                with self.subTest(field=field, value=value):
                    self.assert_invalid(config, field)

    def test_points_use_half_open_map_bounds(self):
        for collection, key in (("spawn_points", "red"), ("return_points", "blue"), ("units", 0)):
            for axis, outside_values in (("x", (-1, 900)), ("y", (-1, 600))):
                for value in outside_values:
                    config = valid_config()
                    config[collection][key][axis] = value
                    field = f"units[0].{axis}" if collection == "units" else f"{collection}.{key}.{axis}"
                    with self.subTest(field=field, value=value):
                        self.assert_invalid(config, field)
        config = valid_config()
        config["units"][0].update(x=0, y=0)
        config["units"][1].update(x=899.999, y=599.999)
        scene = self.load(config)
        self.assertEqual(scene.units[0].position.x, 0)
        self.assertAlmostEqual(scene.units[1].position.y, 599.999)

    def test_obstacles_can_touch_edges_but_cannot_cross_them(self):
        config = valid_config()
        config["obstacles"][0].update(x=0, y=0, width=900, height=600)
        scene = self.load(config)
        self.assertEqual((scene.obstacles[0].width, scene.obstacles[0].height), (900, 600))
        for updates, field in (
            ({"x": -1}, "obstacles[0].x"),
            ({"y": -1}, "obstacles[0].y"),
            ({"x": 801}, "obstacles[0]"),
            ({"y": 521}, "obstacles[0]"),
        ):
            invalid = copy.deepcopy(valid_config())
            invalid["obstacles"][0].update(updates)
            with self.subTest(updates=updates):
                self.assert_invalid(invalid, field)


if __name__ == "__main__":
    unittest.main()
