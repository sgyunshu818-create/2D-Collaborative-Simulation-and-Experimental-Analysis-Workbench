"""Equipment identity persists independently of existing movement and game rules."""

import copy
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sim_app.equipment import equipment_choices, equipment_label, equipment_profile, profile_key
from sim_app.geography import scene_world
from sim_app.models import EquipmentType, Unit, UnitType
from sim_app.recording import export_run
from sim_app.replay import ReplayError, load_replay
from sim_app.scene import SceneConfigError, scene_from_data
from sim_app.scene_document import SceneDocument, TEMPLATES
from sim_app.simulation import Simulation
from tests.test_scene import valid_config


def equipment_config(*, game=False):
    config = valid_config()
    config["world"]["terrain"] = "detailed_virtual"
    config["obstacles"] = []
    for unit, equipment in zip(config["units"], ("tank", "airplane", "armored_car", "drone")):
        unit.update(equipment=equipment, speed=73.5, sensor_range=321.5,
                    waypoints=[{"x": unit["x"] + 20, "y": unit["y"]}])
    if game:
        config["rules"] = {"time_limit": .5, "score_limit": 100}
    return config


class EquipmentTests(unittest.TestCase):
    def test_all_choices_have_labels_and_preserve_existing_category_defaults(self):
        self.assertEqual({profile.equipment_type for profile in equipment_choices()}, set(EquipmentType))
        self.assertEqual({profile.key for profile in equipment_choices("air")}, {"drone", "airplane"})
        self.assertEqual({profile.key for profile in equipment_choices(UnitType.GROUND)}, {"armored_car", "tank"})
        for profile in equipment_choices():
            with self.subTest(equipment=profile.key):
                self.assertEqual(equipment_label(profile.key), profile.label)
                self.assertEqual(profile_key(profile.equipment_type), profile.key)
                self.assertEqual(equipment_profile(profile), profile)
                self.assertEqual(profile.default_speed, 140 if profile.unit_type is UnitType.AIR else 90)
                self.assertEqual(profile.default_sensor_range, 240 if profile.unit_type is UnitType.AIR else 160)

    def test_legacy_scenes_keep_optional_fields_and_original_snapshot_shape(self):
        scene = scene_from_data(valid_config())
        simulation = Simulation(scene)
        self.assertIsNone(scene.terrain)
        for spec, unit, row in zip(scene.units, simulation.units, simulation.snapshot()["units"]):
            with self.subTest(unit=spec.id):
                self.assertIsNone(spec.equipment_type)
                self.assertIsNone(unit.equipment_type)
                self.assertNotIn("equipment", row)
                expected = "drone" if spec.unit_type is UnitType.AIR else "tank"
                self.assertEqual(profile_key(spec), expected)
                self.assertEqual(profile_key(unit), expected)
                self.assertEqual(profile_key(spec.unit_type), expected)
        self.assertEqual(scene_world(scene), {"width": 900, "height": 600})
        with tempfile.TemporaryDirectory() as directory:
            paths = export_run(simulation, directory)
            payload = json.loads(paths["run"].read_text(encoding="utf-8"))
            self.assertTrue(all("equipment" not in row for row in payload["scene"]["units"]))
            with paths["states"].open(encoding="utf-8-sig", newline="") as stream:
                self.assertNotIn("equipment", csv.DictReader(stream).fieldnames)
            replay = load_replay(paths["run"])
            self.assertEqual(replay.scene, scene)
            self.assertTrue(all(unit.equipment_type is None for unit in replay.units))

    def test_explicit_equipment_does_not_replace_custom_speed_or_sensor_parameters(self):
        scene = scene_from_data(equipment_config())
        self.assertEqual(scene.terrain, "detailed_virtual")
        self.assertEqual([spec.equipment_type for spec in scene.units],
                         [EquipmentType.TANK, EquipmentType.AIRPLANE,
                          EquipmentType.ARMORED_CAR, EquipmentType.DRONE])
        for spec in scene.units:
            self.assertEqual(spec.speed, 73.5)
            self.assertEqual(spec.sensor_range, 321.5)

    def test_model_constructors_reject_incompatible_equipment_and_preserve_positional_arguments(self):
        spec = scene_from_data(valid_config()).units[0]
        self.assertEqual(replace(spec, equipment_type="armored_car").equipment_type,
                         EquipmentType.ARMORED_CAR)
        with self.assertRaises(ValueError):
            replace(spec, equipment_type=EquipmentType.AIRPLANE)
        unit = Unit(spec.id, spec.team, spec.unit_type, spec.position, equipment_type="tank")
        self.assertEqual(unit.equipment_type, EquipmentType.TANK)
        with self.assertRaises(ValueError):
            Unit(spec.id, spec.team, spec.unit_type, spec.position, equipment_type="drone")

    def test_invalid_equipment_and_movement_combinations_report_the_equipment_field(self):
        for equipment in ("airplane", "drone", "unknown", "", None, True, [], {}):
            config = valid_config()
            config["units"][0]["equipment"] = equipment
            with self.subTest(equipment=equipment), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config, "equipment.json")
            self.assertEqual(caught.exception.field, "units[0].equipment")
            self.assertIn("equipment.json", str(caught.exception))
        for equipment in ("tank", "armored_car"):
            config = valid_config()
            config["units"][1]["equipment"] = equipment
            with self.subTest(equipment=equipment), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config)
            self.assertEqual(caught.exception.field, "units[1].equipment")

    def test_unknown_terrain_is_rejected_at_world_field(self):
        for terrain in ("real_world", "", None, True, {}):
            config = valid_config()
            config["world"]["terrain"] = terrain
            with self.subTest(terrain=terrain), self.assertRaises(SceneConfigError) as caught:
                scene_from_data(config)
            self.assertEqual(caught.exception.field, "world.terrain")

    def test_equipment_uses_the_existing_air_and_ground_obstacle_navigation(self):
        for profile in equipment_choices():
            config = valid_config()
            config["units"] = [{"id": "mover", "team": "red", "type": profile.unit_type.value,
                                "equipment": profile.key, "x": 80, "y": 150, "speed": 100,
                                "waypoints": [{"x": 800, "y": 150}]}]
            config["obstacles"] = [{"id": "wall", "x": 400, "y": 0, "width": 40, "height": 600}]
            simulation = Simulation(scene_from_data(config))
            simulation.start()
            with self.subTest(equipment=profile.key):
                self.assertEqual(simulation.units[0].behavior.value,
                                 "NAVIGATING" if profile.unit_type is UnitType.AIR else "BLOCKED")

    def test_document_creates_all_choices_and_keeps_world_metadata_through_edit_save(self):
        with tempfile.TemporaryDirectory() as directory:
            config = equipment_config()
            config["units"] = []
            config["world"]["georeference"] = {"center_latitude": 30,
                                               "center_longitude": 120, "meters_per_unit": 1}
            document = SceneDocument(config, user_root=directory)
            for profile in equipment_choices():
                index = document.add_unit(equipment=profile.key)
                self.assertEqual(profile_key(document.data["units"][index]), profile.key)
                self.assertEqual(document.data["units"][index]["type"], profile.unit_type.value)
            before = document.data
            document.update_item("units", 0, speed=111, sensor_range=222)
            self.assertTrue(document.undo())
            self.assertEqual(document.data, before)
            self.assertTrue(document.redo())
            saved = document.save(Path(directory) / "saved.json")
            restored = SceneDocument.open(saved, user_root=directory)
            self.assertEqual(restored.data, document.data)
            self.assertEqual(scene_world(restored.validate()), config["world"])
            self.assertEqual(restored.validate().units[0].speed, 111)

    def test_document_legacy_add_and_explicit_incompatible_add_are_safe(self):
        document = SceneDocument()
        index = document.add_unit()
        self.assertEqual(document.data["units"][index]["type"], "ground")
        self.assertNotIn("equipment", document.data["units"][index])
        before = document.data
        with self.assertRaises(ValueError):
            document.add_unit(unit_type="ground", equipment="airplane")
        self.assertEqual(document.data, before)

    def test_basic_template_remains_first_and_detailed_template_is_available(self):
        self.assertEqual(next(iter(TEMPLATES)), "基础场景")
        self.assertEqual(TEMPLATES["大型精细场景"].name, "detailed_scene.json")

    def test_recording_csv_snapshots_reset_and_replay_keep_all_choices_in_both_formats(self):
        for game in (False, True):
            with self.subTest(game=game), tempfile.TemporaryDirectory() as directory:
                simulation = Simulation(scene_from_data(equipment_config(game=game)))
                simulation.start()
                simulation.advance(.125)
                simulation.pause()
                paths = export_run(simulation, directory)
                payload = json.loads(paths["run"].read_text(encoding="utf-8"))
                expected = [profile_key(spec) for spec in simulation.scene.units]
                self.assertEqual(payload["format_version"], 3 if game else 2)
                self.assertEqual(payload["scene"]["world"]["terrain"], "detailed_virtual")
                self.assertEqual([row["equipment"] for row in payload["scene"]["units"]], expected)
                self.assertTrue(all([row["equipment"] for row in frame["units"]] == expected
                                    for frame in payload["snapshots"]))
                with paths["states"].open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["equipment"] for row in rows], expected * len(payload["snapshots"]))
                replay = load_replay(paths["run"])
                self.assertEqual(replay.scene, simulation.scene)
                for index, frame in enumerate(payload["snapshots"]):
                    replay.seek(index)
                    self.assertEqual(replay.current_snapshot, frame)
                    self.assertEqual([profile_key(unit) for unit in replay.units], expected)
                    self.assertEqual([unit.equipment_type for unit in replay.units],
                                     [spec.equipment_type for spec in simulation.scene.units])
                simulation.reset()
                self.assertEqual([profile_key(unit) for unit in simulation.units], expected)
                self.assertEqual([row["equipment"] for row in simulation.snapshot()["units"]], expected)

    def test_replay_rejects_corrupt_equipment_identity_before_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            simulation = Simulation(scene_from_data(equipment_config()))
            path = export_run(simulation, directory)["run"]
            original = json.loads(path.read_text(encoding="utf-8"))
            for equipment in ("armored_car", "airplane", "unknown", None, True, []):
                payload = copy.deepcopy(original)
                payload["snapshots"][0]["units"][0]["equipment"] = equipment
                payload["result"] = copy.deepcopy(payload["snapshots"][0])
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(equipment=equipment), self.assertRaises(ReplayError) as caught:
                    load_replay(path)
                self.assertEqual(caught.exception.field, "snapshots[0].units[0].equipment")


if __name__ == "__main__":
    unittest.main()
