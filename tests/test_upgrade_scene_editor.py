"""Scene editing risks: independent drafts, atomic saves, and Tk workflows."""

import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sim_app.scene import SceneConfigError, load_scene
from sim_app.scene_document import TEMPLATES, SceneDocument, blank_scene
from sim_app.scene_editor import SceneEditor
from sim_app.tk_runtime import create_root


def editing_scene():
    data = blank_scene()
    data["units"] = [
        {"id": "red_1", "team": "red", "type": "ground", "x": 100, "y": 100,
         "speed": 80, "waypoints": [{"x": 150, "y": 100}], "return_home": True},
        {"id": "blue_1", "team": "blue", "type": "air", "x": 700, "y": 400},
    ]
    data["obstacles"] = [{"id": "wall", "x": 350, "y": 200, "width": 80, "height": 60}]
    return data


class SceneDocumentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.document = SceneDocument(editing_scene(), user_root=self.root / "用户场景")

    def test_input_and_returned_snapshots_never_mutate_draft(self):
        source = editing_scene()
        document = SceneDocument(source)
        source["units"][0]["waypoints"][0]["x"] = 900
        snapshot = document.data
        snapshot["units"][0]["waypoints"][0]["x"] = 500
        self.assertEqual(document.data["units"][0]["waypoints"][0]["x"], 150)
        self.assertFalse(document.dirty)

    def test_edit_complete_scene_without_json_then_save_reopen(self):
        doc = self.document
        doc.edit(lambda data: data.update(name="全新教学场景 中文 空格"))
        unit_index = doc.add_unit(team="blue", unit_type="air", x=300, y=150)
        doc.update_item("units", unit_index, id="自建实体", speed=120, sensor_range=200, return_home=True)
        obstacle_index = doc.add_obstacle(x=450, y=250)
        doc.update_item("obstacles", obstacle_index, id="自建障碍", width=100, height=120)
        first = doc.add_waypoint(unit_index, 500, 500)
        second = doc.add_waypoint(unit_index, 550, 450)
        doc.update_waypoint(unit_index, first, 510, 490)
        self.assertEqual(doc.move_waypoint(unit_index, second, -1), 0)
        doc.delete_waypoint(unit_index, 1)
        saved = doc.save()
        reopened = SceneDocument.open(saved)
        self.assertEqual(reopened.data, doc.data)
        self.assertEqual(reopened.validate(), doc.validate())
        self.assertEqual(saved.parent, (self.root / "用户场景").resolve())
        self.assertFalse(doc.dirty)
        self.assertIsNone(reopened.path, "Opening a source must start an independent copy")

    def test_undo_redo_restore_saved_state_and_nested_tasks(self):
        saved = self.document.save()
        original = self.document.data
        self.document.add_waypoint(0, 400, 350)
        self.assertTrue(self.document.dirty)
        self.assertTrue(self.document.undo())
        self.assertEqual(self.document.data, original)
        self.assertFalse(self.document.dirty)
        self.assertTrue(self.document.redo())
        self.assertEqual(len(self.document.data["units"][0]["waypoints"]), 2)
        self.assertTrue(self.document.dirty)
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), original)

    def test_unit_obstacle_and_waypoint_deletion_are_undoable(self):
        for change in (lambda: self.document.delete_item("units", 0),
                       lambda: self.document.delete_item("obstacles", 0),
                       lambda: self.document.delete_waypoint(0, 0)):
            with self.subTest(change=change):
                before = self.document.data
                change()
                self.assertNotEqual(self.document.data, before)
                self.assertTrue(self.document.undo())
                self.assertEqual(self.document.data, before)

    def test_invalid_drafts_report_paths_and_preserve_saved_file(self):
        path = self.document.save()
        original_bytes = path.read_bytes()
        changes = (
            (lambda data: data["units"][1].update(id="red_1"), "units[1].id"),
            (lambda data: data["units"][1].update(id="wall"), "units[1].id"),
            (lambda data: data["world"].update(width=float("nan")), "world.width"),
            (lambda data: data["world"].update(width=True), "world.width"),
            (lambda data: data["units"][0].update(x=900), "units[0].x"),
            (lambda data: data["obstacles"][0].update(width=-5), "obstacles[0].width"),
            (lambda data: data["obstacles"][0].update(width=700), "obstacles[0].width"),
            (lambda data: data["world"].update(width=200), "spawn_points.blue.x"),
            (lambda data: data["units"][0]["waypoints"][0].update(y=600), "units[0].waypoints[0].y"),
        )
        for change, field in changes:
            with self.subTest(field=field):
                before = self.document.data
                self.document.edit(change)
                with self.assertRaises(SceneConfigError) as caught:
                    self.document.save(path)
                self.assertEqual(caught.exception.field, field)
                self.assertEqual(path.read_bytes(), original_bytes)
                self.assertTrue(self.document.dirty)
                self.document.undo()
                self.assertEqual(self.document.data, before)

    def test_virtual_rule_validation_is_shared_with_scene_loader(self):
        self.document.edit(lambda data: data.update(rules={"score_limit": 2, "time_limit": 20}))
        self.assertEqual(self.document.validate().rules.score_limit, 2)
        self.document.edit(lambda data: data["rules"].update(score_limit=True))
        with self.assertRaises(SceneConfigError) as caught:
            self.document.save()
        self.assertEqual(caught.exception.field, "rules.score_limit")

    def test_atomic_replace_failure_keeps_draft_file_and_save_identity(self):
        path = self.document.save()
        old_bytes = path.read_bytes()
        self.document.update_item("units", 0, speed=95)
        with patch("sim_app.scene_document.os.replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                self.document.save()
        self.assertEqual(path.read_bytes(), old_bytes)
        self.assertTrue(self.document.dirty)
        self.assertEqual(self.document.path, path)
        self.assertEqual(self.document.data["units"][0]["speed"], 95)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_first_save_failure_does_not_mark_document_saved(self):
        self.document.add_unit()
        with patch("sim_app.scene_document.os.replace", side_effect=OSError("cannot replace")):
            with self.assertRaises(OSError):
                self.document.save()
        self.assertIsNone(self.document.path)
        self.assertTrue(self.document.dirty)
        self.assertFalse(any(self.document.user_root.glob("*.json")))

    def test_templates_are_protected_and_first_save_is_a_copy(self):
        for template in TEMPLATES.values():
            with self.subTest(template=template.name):
                original = template.read_bytes()
                document = SceneDocument.open(template, user_root=self.root)
                document.edit(lambda data: data.update(name="修改后的副本"))
                with self.assertRaises(SceneConfigError):
                    document.save(template)
                self.assertEqual(template.read_bytes(), original)
                saved = document.save()
                self.assertNotEqual(saved, template)
                self.assertEqual(document.source_path, template)
                self.assertEqual(load_scene(saved).name, "修改后的副本")

    def test_invalid_open_and_load_do_not_replace_existing_draft(self):
        source = self.root / "broken.json"
        source.write_text('{"name": "bad",}', encoding="utf-8")
        with self.assertRaises(SceneConfigError) as caught:
            SceneDocument.open(source)
        self.assertEqual(caught.exception.field, "$")
        before = self.document.data
        with self.assertRaises(SceneConfigError):
            self.document.load({"name": "bad"})
        self.assertEqual(self.document.data, before)


class SceneEditorTests(unittest.TestCase):
    """Programmatic Tk widget checks, separate from the final real GUI audit."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.tkroot = create_root()
            cls.tkroot.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tkroot"):
            cls.tkroot.destroy()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.runs = []
        self.editor = SceneEditor(self.tkroot, self.runs.append)
        self.editor.pack(fill="both", expand=True)
        self.editor.load_document(editing_scene())
        self.editor.get_document().user_root = self.root / "用户 场景"
        self.error_patch = patch("sim_app.scene_editor.messagebox.showerror")
        self.error = self.error_patch.start()
        self.addCleanup(self.error_patch.stop)
        self.addCleanup(self.editor.destroy)

    def test_forms_create_entity_obstacle_and_ordered_tasks_without_json(self):
        self.assertTrue(self.editor.add_unit())
        self.editor._item_vars["id"].set("表单新建实体")
        self.editor._item_vars["team"].set("blue")
        self.editor._item_vars["type"].set("air")
        self.editor._item_vars["x"].set("280")
        self.editor._item_vars["y"].set("330")
        self.editor._item_vars["return_home"].set(True)
        self.assertTrue(self.editor.apply_changes())
        self.assertTrue(self.editor.add_waypoint(450, 300))
        self.assertTrue(self.editor.add_waypoint(550, 330))
        self.assertTrue(self.editor.reorder_waypoint(-1))
        data = self.editor.get_document().data
        self.assertEqual(data["units"][2]["waypoints"], [{"x": 550, "y": 330}, {"x": 450, "y": 300}])
        self.assertTrue(self.editor.add_obstacle())
        self.editor._item_vars["id"].set("表单新建障碍")
        self.editor._item_vars["width"].set("40")
        self.editor._item_vars["height"].set("90")
        self.assertTrue(self.editor.apply_changes())
        saved = self.editor.save_document()
        self.assertIsNotNone(saved)
        reopened = SceneDocument.open(saved)
        self.assertEqual(reopened.data, self.editor.get_document().data)
        self.assertFalse(self.editor.dirty)
        self.assertFalse(self.error.called)

    def test_world_rules_and_team_point_forms(self):
        self.editor._scene_vars["name"].set("场景规则表单")
        self.editor._scene_vars["world.width"].set("1000")
        self.editor._scene_vars["spawn_points.red.x"].set("120")
        self.editor._scene_vars["return_points.blue.y"].set("480")
        self.editor._scene_vars["rules_enabled"].set(True)
        self.editor._scene_vars["rules.score_limit"].set("5")
        self.editor._scene_vars["rules.time_limit"].set("30")
        self.assertTrue(self.editor.validate_document())
        data = self.editor.get_document().data
        self.assertEqual(data["world"]["width"], 1000)
        self.assertEqual(data["spawn_points"]["red"]["x"], 120)
        self.assertEqual(data["return_points"]["blue"]["y"], 480)
        self.assertEqual(self.editor.get_document().validate().rules.score_limit, 5)

    def test_unsaved_form_changes_are_protected_before_open_and_close(self):
        before = self.editor.get_document().data
        self.editor._scene_vars["name"].set("尚未提交表单")
        self.assertTrue(self.editor.dirty)
        self.assertFalse(self.editor.get_document().dirty)
        with patch("sim_app.scene_editor.messagebox.askyesnocancel", return_value=None):
            self.assertFalse(self.editor.load_document(blank_scene()))
            self.assertFalse(self.editor.can_close())
        self.assertEqual(self.editor.get_document().data, before)
        self.assertEqual(self.editor._scene_vars["name"].get(), "尚未提交表单")
        with patch("sim_app.scene_editor.messagebox.askyesnocancel", return_value=False):
            self.assertTrue(self.editor.load_document(blank_scene()))
        self.assertFalse(self.editor.dirty)
        self.assertEqual(self.editor.get_document().data, blank_scene())

    def test_unsaved_save_choice_cannot_close_when_validation_fails(self):
        self.editor.select(("units", 0))
        self.editor._item_vars["x"].set("900")
        with patch("sim_app.scene_editor.messagebox.askyesnocancel", return_value=True):
            self.assertFalse(self.editor.can_close())
        self.assertTrue(self.editor.dirty)
        self.assertIsNone(self.editor.get_document().path)
        self.assertIn("units[0].x", str(self.error.call_args))

    def test_unsaved_save_choice_saves_before_replacing_draft(self):
        self.editor._scene_vars["name"].set("保存后切换")
        with patch("sim_app.scene_editor.messagebox.askyesnocancel", return_value=True):
            self.assertTrue(self.editor.load_document(blank_scene()))
        saved_files = list((self.root / "用户 场景").glob("*.json"))
        self.assertEqual(len(saved_files), 1)
        self.assertEqual(load_scene(saved_files[0]).name, "保存后切换")
        self.assertEqual(self.editor.get_document().data, blank_scene())

    def test_run_callback_only_receives_valid_saved_independent_input(self):
        self.editor.select(("units", 0))
        self.editor._item_vars["speed"].set("120")
        self.assertTrue(self.editor.run_document())
        first = Path(self.runs[0])
        self.assertTrue(first.exists())
        self.assertEqual(load_scene(first).units[0].speed, 120)
        first_bytes = first.read_bytes()
        self.editor._item_vars["speed"].set("160")
        self.assertTrue(self.editor.run_document())
        self.assertNotEqual(Path(self.runs[1]), first)
        self.assertEqual(first.read_bytes(), first_bytes)
        self.assertEqual(load_scene(self.runs[1]).units[0].speed, 160)
        second_bytes = Path(self.runs[1]).read_bytes()
        self.editor._item_vars["speed"].set("180")
        self.assertIsNotNone(self.editor.save_document())
        self.assertEqual(first.read_bytes(), first_bytes)
        self.assertEqual(Path(self.runs[1]).read_bytes(), second_bytes)
        self.assertNotEqual(self.editor.get_document().path, Path(self.runs[1]))
        self.editor._item_vars["x"].set("nan")
        self.assertFalse(self.editor.run_document())
        self.assertEqual(len(self.runs), 2)
        self.assertIn("units[0].x", str(self.error.call_args))

    def test_invalid_form_keeps_previous_document_and_all_pending_values(self):
        self.editor.select(("units", 0))
        before = self.editor.get_document().data
        self.editor._scene_vars["name"].set("不应部分应用")
        self.editor._item_vars["x"].set("True")
        self.assertFalse(self.editor.apply_changes())
        self.assertEqual(self.editor.get_document().data, before)
        self.assertTrue(self.editor.dirty)
        self.assertEqual(self.editor._scene_vars["name"].get(), "不应部分应用")
        self.assertTrue(self.editor.undo())
        self.assertFalse(self.editor.dirty)

    def test_map_list_property_link_and_mouse_move_are_undoable(self):
        self.editor.canvas.configure(width=600, height=400)
        self.tkroot.update_idletasks()
        self.editor.select(("units", 0))
        self.assertEqual(self.editor.tree.selection(), ("units:0",))
        self.assertEqual(self.editor._item_vars["id"].get(), "red_1")
        x, y = self.editor._xy(100, 100)
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=0))
        tx, ty = self.editor._xy(240, 210)
        self.editor._canvas_drag(SimpleNamespace(x=tx, y=ty))
        self.editor._canvas_release(SimpleNamespace(x=tx, y=ty))
        moved = self.editor.get_document().data["units"][0]
        self.assertAlmostEqual(moved["x"], 240, places=2)
        self.assertAlmostEqual(moved["y"], 210, places=2)
        self.assertEqual(float(self.editor._item_vars["x"].get()), moved["x"])
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data["units"][0]["x"], 100)
        self.assertEqual(self.editor.get_document().data["units"][0]["y"], 100)

    def test_team_return_marker_and_task_points_can_be_moved_and_deleted(self):
        self.editor.select(("return_points", "red"))
        self.editor._item_vars["x"].set("220")
        self.editor._item_vars["y"].set("180")
        self.assertTrue(self.editor.apply_changes())
        self.assertEqual(self.editor.get_document().data["return_points"]["red"], {"x": 220, "y": 180})
        self.assertFalse(self.editor.delete_selected())
        self.editor.select(("waypoints", 0, 0))
        self.editor._item_vars["x"].set("320")
        self.assertTrue(self.editor.apply_changes())
        self.assertTrue(self.editor.delete_selected())
        self.assertEqual(self.editor.get_document().data["units"][0]["waypoints"], [])
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data["units"][0]["waypoints"][0]["x"], 320)

    def test_invalid_load_never_discards_dirty_document(self):
        self.editor._scene_vars["name"].set("保留草稿")
        with patch("sim_app.scene_editor.messagebox.askyesnocancel") as prompt:
            with self.assertRaises(SceneConfigError):
                self.editor.load_document({"name": "invalid"})
        self.assertFalse(prompt.called)
        self.assertTrue(self.editor.dirty)
        self.assertEqual(self.editor._scene_vars["name"].get(), "保留草稿")


if __name__ == "__main__":
    unittest.main()
