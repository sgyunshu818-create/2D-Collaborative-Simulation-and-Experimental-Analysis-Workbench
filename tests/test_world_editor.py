"""World-map editing safeguards in a real, withdrawn Tk widget tree."""

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sim_app.geography import GeoReference
from sim_app.scene_document import SceneDocument, blank_scene
from sim_app.scene_editor import SceneEditor
from sim_app.tk_runtime import create_root


def example_scene(*, geographic=False):
    data = blank_scene()
    data['units'] = [
        {'id': 'red_1', 'team': 'red', 'type': 'ground', 'x': 180, 'y': 180,
         'waypoints': [{'x': 450, 'y': 250}]},
        {'id': 'blue_1', 'team': 'blue', 'type': 'air', 'x': 650, 'y': 420},
    ]
    if geographic:
        data['world']['georeference'] = {
            'center_latitude': 31.2, 'center_longitude': 121.4, 'meters_per_unit': 150.0,
        }
    return data


class WorldEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = create_root()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f'Tk display unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'root'):
            cls.root.destroy()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.editor = SceneEditor(self.root, lambda path: None)
        self.editor.pack(fill='both', expand=True)
        self.addCleanup(self.editor.destroy)
        # A withdrawn root has no visible geometry. Keep actual Tk widgets,
        # while supplying the map viewport used for geographic hit testing.
        self.viewport = [600, 400]
        for method, index in (('winfo_width', 0), ('winfo_height', 1)):
            mock = patch.object(self.editor.canvas, method, side_effect=lambda i=index: self.viewport[i])
            mock.start()
            self.addCleanup(mock.stop)
        errors = patch('sim_app.scene_editor.messagebox.showerror')
        self.error = errors.start()
        self.addCleanup(errors.stop)
        self.assertTrue(self.editor.load_document(example_scene()))
        self.editor.get_document().user_root = Path(self.directory.name)

    def load_geo(self):
        self.assertTrue(self.editor.load_document(example_scene(geographic=True)))
        self.assertTrue(self.editor._is_world_map())

    def fill_reference(self, **overrides):
        values = {'enabled': True, 'center_latitude': '31.2',
                  'center_longitude': '121.4', 'meters_per_unit': '150'}
        values.update(overrides)
        for key, value in values.items():
            self.editor._geo_vars[key].set(value)

    def test_old_scene_has_no_real_position_and_can_browse_global(self):
        original = self.editor.get_document().data
        self.assertEqual(self.editor.map_view.get(), '虚拟场景')
        self.assertFalse(self.editor._geo_vars['enabled'].get())
        self.assertEqual(self.editor._geo_vars['center_latitude'].get(), '')
        self.editor.focus_global(animate=False)
        self.assertTrue(self.editor._is_world_map())
        self.assertEqual(self.editor._world_view.hit_rects, {})
        self.assertFalse(self.editor.focus_scene(animate=False))
        self.assertIn('地理参考', self.editor.status_var.get())
        self.assertEqual(self.editor.get_document().data, original)
        self.assertFalse(self.editor.dirty)

    def test_geo_template_loads_world_mode_and_focuses_reference(self):
        self.load_geo()
        self.assertTrue(self.editor.focus_scene(animate=False))
        lon, lat = self.editor._world_view.to_lonlat((300, 200))
        self.assertAlmostEqual(lon, 121.4, places=6)
        self.assertAlmostEqual(lat, 31.2, places=6)
        self.assertGreater(self.editor._world_view.camera.zoom, 1)
        self.assertFalse(self.editor.dirty)

    def test_world_unit_drag_selects_list_and_only_pans_camera(self):
        self.load_geo()
        self.editor.select(None)
        before = self.editor.get_document().data
        center = self.editor._world_view.camera.actual_center
        x, y = self.editor._world_view.local_to_screen(180, 180, before['world'])
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=1))
        self.assertEqual(self.editor._selected, ('units', 0))
        self.assertEqual(self.editor.tree.selection(), ('units:0',))
        self.assertIsNone(self.editor._drag)
        self.editor._canvas_drag(SimpleNamespace(x=x+60, y=y+30))
        self.editor._canvas_release(SimpleNamespace(x=x+60, y=y+30))
        self.assertNotEqual(self.editor._world_view.camera.actual_center, center)
        self.assertEqual(self.editor.get_document().data, before)
        self.assertFalse(self.editor.dirty)

    def test_geographic_fields_save_reload_and_undo(self):
        before = self.editor.get_document().data
        self.fill_reference()
        self.assertTrue(self.editor.dirty)
        self.assertFalse(self.editor.get_document().dirty)
        self.assertTrue(self.editor.apply_georeference())
        reference = self.editor.get_document().data['world']['georeference']
        self.assertEqual(reference['meters_per_unit'], 150)
        saved = self.editor.save_document(Path(self.directory.name) / '地理场景.json')
        self.assertIsNotNone(saved)
        reopened = SceneDocument.open(saved)
        self.assertEqual(reopened.data, self.editor.get_document().data)
        self.assertEqual(reopened.validate().georeference, GeoReference(31.2, 121.4, 150))
        self.assertFalse(self.editor.dirty)
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data, before)
        self.assertFalse(self.editor._geo_vars['enabled'].get())
        self.assertTrue(self.editor.redo())
        self.assertEqual(self.editor.get_document().data['world']['georeference'], reference)

    def test_disabling_reference_is_undoable_and_scene_form_preserves_it(self):
        self.load_geo()
        reference = self.editor.get_document().data['world']['georeference']
        self.editor._scene_vars['name'].set('保留参考')
        self.editor._scene_vars['world.width'].set('1000')
        self.assertTrue(self.editor.apply_changes())
        self.assertEqual(self.editor.get_document().data['world']['georeference'], reference)
        self.editor._geo_vars['enabled'].set(False)
        self.assertTrue(self.editor.apply_georeference())
        self.assertNotIn('georeference', self.editor.get_document().data['world'])
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data['world']['georeference'], reference)
        self.assertTrue(self.editor._geo_vars['enabled'].get())

    def test_bad_reference_keeps_draft_and_pending_form(self):
        original = self.editor.get_document().data
        for invalid in ({'center_longitude': '181'}, {'meters_per_unit': '0'},
                        {'center_longitude': '179.99'}, {'center_latitude': 'NaN'}):
            with self.subTest(invalid=invalid):
                self.fill_reference(**invalid)
                self.assertFalse(self.editor.apply_georeference())
                self.assertEqual(self.editor.get_document().data, original)
                self.assertTrue(self.editor._geo_pending)
                self.assertTrue(self.editor.dirty)
                self.assertTrue(self.editor.undo())
        self.assertTrue(self.error.called)

    def test_geo_pending_is_protected_on_close_and_undoes_without_history(self):
        original = self.editor.get_document().data
        self.fill_reference()
        with patch('sim_app.scene_editor.messagebox.askyesnocancel', return_value=None):
            self.assertFalse(self.editor.can_close())
            self.assertFalse(self.editor.load_document(blank_scene()))
        self.assertEqual(self.editor.get_document().data, original)
        self.assertTrue(self.editor.undo())
        self.assertFalse(self.editor.dirty)
        self.assertFalse(self.editor._geo_vars['enabled'].get())

    def test_expand_restore_preserves_panes_camera_selection_and_fields(self):
        self.load_geo()
        self.editor.select(('units', 0))
        self.root.update_idletasks()
        original_panes = self.editor.body.panes()
        original_sashes = tuple(self.editor.body.sashpos(i) for i in range(2))
        world = self.editor.get_document().data
        center = self.editor._world_view.camera.actual_center
        zoom = self.editor._world_view.camera.zoom
        self.editor.toggle_map_expanded()
        self.viewport[:] = [1000, 600]
        self.editor._draw()
        self.assertEqual(len(self.editor.body.panes()), 1)
        self.assertEqual(self.editor.expand_button.cget('text'), '恢复面板')
        self.assertEqual(self.editor._world_view.camera.actual_center, center)
        self.assertEqual(self.editor._world_view.camera.zoom, zoom)
        self.editor.toggle_map_expanded()
        self.viewport[:] = [600, 400]
        self.root.update_idletasks()
        self.editor._draw()
        self.assertEqual(self.editor.body.panes(), original_panes)
        self.assertEqual(tuple(self.editor.body.sashpos(i) for i in range(2)), original_sashes)
        self.assertEqual(self.editor._selected, ('units', 0))
        self.assertEqual(self.editor._item_vars['id'].get(), 'red_1')
        self.assertEqual(self.editor.get_document().data, world)
        self.assertEqual(self.editor._world_view.camera.actual_center, center)

    def test_world_wheel_callbacks_remain_bounded_and_cancel_on_destroy(self):
        self.load_geo()
        callbacks = []
        for _ in range(30):
            self.editor.zoom_map(1.02, (300, 200))
            callbacks.append(self.editor._world_animation_after)
        active = set(self.root.tk.call('after', 'info'))
        self.assertLessEqual(sum(value in active for value in callbacks if value is not None), 1)
        callback = self.editor._world_animation_after
        self.editor.destroy()
        self.assertIsNone(self.editor._world_animation_after)
        if callback is not None:
            self.assertNotIn(callback, self.root.tk.call('after', 'info'))
        self.root.update_idletasks()

    def test_world_motion_reports_lonlat_and_empty_margin(self):
        self.editor.focus_global(animate=False)
        self.editor._canvas_motion(SimpleNamespace(x=300, y=200))
        self.assertIn('地图', self.editor.coordinate_var.get())
        self.assertIn('°', self.editor.coordinate_var.get())
        self.editor._canvas_motion(SimpleNamespace(x=300, y=-1000))
        self.assertIn('地图外', self.editor.coordinate_var.get())
        self.assertNotIn('°', self.editor.coordinate_var.get())

    def test_virtual_transform_and_unit_editing_survive_world_browsing(self):
        self.editor.zoom_map(1.7, (300, 200), animate=False)
        original_transform = self.editor._transform
        original = self.editor.get_document().data
        self.editor.focus_global(animate=False)
        self.editor.map_view.set('虚拟场景')
        self.editor._map_view_changed()
        self.assertEqual(self.editor._transform, original_transform)
        self.assertEqual(self.editor.get_document().data, original)
        x, y = self.editor._xy(180, 180)
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=0))
        nx, ny = self.editor._xy(250, 230)
        self.editor._canvas_drag(SimpleNamespace(x=nx, y=ny))
        self.editor._canvas_release(SimpleNamespace(x=nx, y=ny))
        self.assertAlmostEqual(self.editor.get_document().data['units'][0]['x'], 250, places=2)
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data, original)


if __name__ == '__main__':
    unittest.main()
