"""Large-map editing, hit testing and native text remain coherent."""
import copy
import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from sim_app.equipment import equipment_choices, profile_key
from sim_app.models import Point
from sim_app.renderer import Renderer, WINDOW_SIZE, TEXT
from sim_app.scene import load_scene
from sim_app.scene_document import CONFIG_ROOT, SceneDocument
from sim_app.scene_editor import SceneEditor
from sim_app.simulation import Simulation
from sim_app.tk_runtime import create_root
from sim_app.visual_assets import equipment_asset_available, unit_glyph_surface


def detailed_data():
    return json.loads((CONFIG_ROOT / 'detailed_scene.json').read_text(encoding='utf-8'))


class DetailedPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.font.init()

    def test_new_world_exceeds_twenty_times_each_previous_dimension_and_has_four_types(self):
        old = load_scene(CONFIG_ROOT / 'mountain_scene.json')
        new = load_scene(CONFIG_ROOT / 'detailed_scene.json')
        self.assertGreaterEqual(new.width, old.width * 20)
        self.assertGreaterEqual(new.height, old.height * 20)
        self.assertEqual(new.terrain, 'astra_atlas')
        self.assertEqual({profile_key(u) for u in new.units}, {p.key for p in equipment_choices()})
        self.assertEqual(len(new.units), 8)

    def test_four_equipment_choices_have_distinct_transparent_glyphs(self):
        pixels = []
        for profile in equipment_choices():
            with self.subTest(equipment=profile.key):
                self.assertTrue(equipment_asset_available(profile))
                glyph = unit_glyph_surface(profile, (82, 199, 243), size=56)
                self.assertEqual(glyph.get_size(), (56, 56))
                self.assertEqual(glyph.get_at((0, 0)).a, 0)
                self.assertGreater(glyph.get_bounding_rect().width, 25)
                pixels.append(pygame.image.tostring(glyph, 'RGBA'))
        self.assertEqual(len(set(pixels)), 4)

    def test_detailed_camera_hit_testing_and_pan_preserve_simulation_facts(self):
        sim = Simulation(load_scene(CONFIG_ROOT / 'detailed_scene.json'))
        renderer = Renderer(pygame.Surface(WINDOW_SIZE))
        renderer.draw(sim)
        self.assertEqual(renderer.camera.max_zoom, 256)
        self.assertEqual(set(renderer.unit_hit_rects), {u.id for u in sim.units})
        before = copy.deepcopy((sim.units, sim.events, sim.snapshots, sim.step_count))
        for unit in sim.units:
            point = renderer.world_to_screen(unit.position, sim)
            self.assertTrue(renderer.select_at(point, sim))
            self.assertEqual(renderer.selected_unit_id, unit.id)
        point = Point(120000, 80000)
        renderer.camera.set_pose(16, point)
        renderer.camera.pan((32, -18))
        renderer.draw(sim)
        pixel = renderer.world_to_screen(point, sim)
        restored = renderer.screen_to_world(pixel, sim)
        self.assertAlmostEqual(restored.x, point.x, delta=1)
        self.assertAlmostEqual(restored.y, point.y, delta=1)
        self.assertEqual(before, (sim.units, sim.events, sim.snapshots, sim.step_count))
        renderer.draw(Simulation(load_scene(CONFIG_ROOT / 'basic_scene.json')))
        self.assertEqual(renderer.camera.max_zoom, 6)

    def test_narrow_chinese_labels_are_rasterized_at_native_size(self):
        renderer = Renderer(pygame.Surface(WINDOW_SIZE))
        with patch('pygame.transform.smoothscale', side_effect=AssertionError('text bitmap resized')):
            rendered = renderer._render_text('装甲车与飞机的完整中文测试编号', 16, TEXT, max_width=64)
        self.assertLessEqual(rendered.get_width(), 64)
        self.assertEqual(rendered.get_height(), renderer.fonts[12].get_height())


class DetailedEditorTests(unittest.TestCase):
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
        for method, value in (('winfo_width', 800), ('winfo_height', 520)):
            mock = patch.object(self.editor.canvas, method, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch('sim_app.scene_editor.messagebox.showerror')
        self.error = mock.start()
        self.addCleanup(mock.stop)
        self.editor.load_document(detailed_data())
        self.editor.get_document().user_root = Path(self.directory.name)

    def test_selecting_equipment_updates_category_preview_save_and_undo(self):
        self.editor.select(('units', 0))
        self.assertEqual(self.editor.preview_type.get(), '坦克')
        original = self.editor.get_document().data
        self.editor._item_vars['equipment'].set('飞机')
        self.editor._item_fields['equipment'].event_generate('<<ComboboxSelected>>')
        self.root.update_idletasks()
        self.assertEqual(self.editor._item_vars['type'].get(), 'air')
        self.assertEqual(self.editor.preview_type.get(), '飞机')
        self.assertEqual(self.editor.get_document().data, original)
        self.assertTrue(self.editor.apply_changes())
        saved = self.editor.save_document(Path(self.directory.name) / 'equipment.json')
        reopened = SceneDocument.open(saved)
        self.assertEqual(reopened.data['units'][0]['equipment'], 'airplane')
        self.assertEqual(reopened.data['units'][0]['type'], 'air')
        self.assertEqual(reopened.data['world']['terrain'], 'astra_atlas')
        self.assertTrue(self.editor.undo())
        self.assertEqual(self.editor.get_document().data, original)
        self.assertEqual(self.editor.preview_type.get(), '坦克')
        self.assertTrue(self.editor.redo())
        self.assertEqual(self.editor.preview_type.get(), '飞机')
        self.assertFalse(self.error.called)

    def test_large_map_zoom_anchor_and_left_drag_keep_draft_unchanged(self):
        before = self.editor.get_document().data
        self.assertEqual(self.editor._max_virtual_zoom(), 256)
        anchor = (610, 370)
        world_point = self.editor._world_xy(*anchor)
        self.editor.zoom_map(16, anchor, animate=False)
        restored = self.editor._world_xy(*anchor)
        for actual, expected in zip(restored, world_point):
            self.assertAlmostEqual(actual, expected, places=6)
        origin = self.editor._camera_pan
        self.editor.pan_mode.set(True)
        self.editor._canvas_press(SimpleNamespace(x=400, y=260, state=0))
        self.editor._canvas_drag(SimpleNamespace(x=470, y=300))
        self.editor._canvas_release(SimpleNamespace(x=470, y=300))
        self.assertNotEqual(self.editor._camera_pan, origin)
        self.assertEqual(self.editor.get_document().data, before)
        self.assertFalse(self.editor.dirty)
        self.editor.zoom_map(100, animate=False)
        self.assertEqual(self.editor._camera_zoom, 256)

    def test_restore_recovers_columns_when_startup_sashes_were_zero(self):
        # A mapped widget tree checks actual sash positions, not mocked calls.
        self.root.geometry('1280x800')
        self.root.deiconify()
        try:
            self.root.update()
            self.editor.toggle_map_expanded()
            self.editor._saved_sashes = (0, 0)
            self.editor.toggle_map_expanded()
            self.root.update()
            self.assertEqual(len(self.editor.body.panes()), 3)
            first, second = (self.editor.body.sashpos(i) for i in range(2))
            self.assertGreater(first, 120)
            self.assertGreater(second, first + 250)
            self.assertLess(second, self.editor.body.winfo_width() - 120)
        finally:
            self.root.withdraw()
