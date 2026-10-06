"""Camera/edit invariants and interruptible callback lifetime on a Tk root."""

import tkinter as tk
from tkinter import font as tkfont, ttk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sim_app.scene_editor import SceneEditor
from sim_app.tk_runtime import create_root
from sim_app.ui_theme import UI_FONT, DATA_FONT, BooleanToggle, CollapsibleFrame, NavigationRail, apply_theme
from tests.test_upgrade_scene_editor import editing_scene


class TkMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = create_root()
            cls.root.withdraw()
            apply_theme(cls.root)
        except tk.TclError as error:
            raise unittest.SkipTest(f'Tk display unavailable: {error}')

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.editor = SceneEditor(self.root, lambda _: None)
        self.addCleanup(self.editor.destroy)
        self.editor.load_document(editing_scene())
        for method, value in (('winfo_width', 600), ('winfo_height', 440)):
            context = patch.object(self.editor.canvas, method, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        self.editor._draw()

    def run_loop(self, duration=230):
        self.root.after(duration, self.root.quit)
        self.root.mainloop()

    def assert_point(self, first, second):
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, places=8)

    def test_font_family_covers_named_fonts_and_map_text(self):
        self.assertEqual(UI_FONT[0], DATA_FONT[0])
        actual = tkfont.nametofont('TkDefaultFont', root=self.root).actual('family')
        for name in tkfont.names(self.root):
            self.assertEqual(tkfont.nametofont(name, root=self.root).actual('family'), actual)
        for item in self.editor.canvas.find_all():
            if self.editor.canvas.type(item) == 'text':
                font = tkfont.Font(root=self.root, font=self.editor.canvas.itemcget(item, 'font'))
                self.assertEqual(font.actual('family'), actual)

    def test_zoom_keeps_cursor_world_point_on_every_drawn_frame(self):
        anchor = (210, 180)
        point = self.editor._world_xy(*anchor)
        frames = []
        draw = self.editor._draw
        def observe():
            draw()
            frames.append(self.editor._world_xy(*anchor))
        with patch.object(self.editor, '_draw', side_effect=observe):
            self.editor.zoom_map(2, anchor)
            self.run_loop()
        self.assertGreater(len(frames), 3)
        for frame in frames:
            self.assert_point(frame, point)
        self.assertAlmostEqual(self.editor._camera_zoom, 2)
        self.assertIsNone(self.editor._zoom_motion.after_id)

    def test_continuous_wheel_replaces_transition_and_stops_at_bounds(self):
        anchor = (240, 200)
        point = self.editor._world_xy(*anchor)
        for _ in range(10):
            self.editor._canvas_zoom(SimpleNamespace(delta=120, x=anchor[0], y=anchor[1]))
        self.assertAlmostEqual(self.editor._zoom_target, 1.12 ** 10)
        self.run_loop()
        self.assertAlmostEqual(self.editor._camera_zoom, 1.12 ** 10)
        self.assert_point(self.editor._world_xy(*anchor), point)
        self.editor.zoom_map(100, anchor, animate=False)
        self.assertEqual(self.editor._camera_zoom, 6)
        self.assertTrue(self.editor.zoom_in_button.instate(['disabled']))
        self.editor.fit_map(animate=False)
        self.assertEqual(self.editor._camera_zoom, 1)
        self.assertEqual(self.editor._camera_pan, (0, 0))
        self.assertTrue(self.editor.zoom_out_button.instate(['disabled']))

    def test_pan_space_and_focus_only_change_view(self):
        before = self.editor.get_document().data
        self.editor.zoom_map(2, animate=False)
        original = self.editor._xy(100, 100)
        self.editor._space_press(None)
        self.editor._canvas_press(SimpleNamespace(x=200, y=180, state=0))
        self.editor._canvas_drag(SimpleNamespace(x=230, y=220))
        self.editor._canvas_release(SimpleNamespace(x=230, y=220))
        moved = self.editor._xy(100, 100)
        self.assert_point(moved, (original[0]+30, original[1]+40))
        self.assertEqual(self.editor.get_document().data, before)
        self.assertFalse(self.editor.dirty)
        self.editor._pan_start(SimpleNamespace(x=200, y=180))
        self.editor._view_focus_out(None)
        self.assertIsNone(self.editor._pan_drag)
        self.assertFalse(self.editor._space_down)

    def test_left_blank_drag_pans_and_pan_mode_never_edits_units(self):
        before = self.editor.get_document().data
        self.editor.zoom_map(2, animate=False)
        original = self.editor._xy(100, 100)
        with patch.object(self.editor.canvas, 'find_overlapping', return_value=()):
            self.editor._canvas_press(SimpleNamespace(x=260, y=220, state=0))
        self.editor._canvas_drag(SimpleNamespace(x=290, y=240))
        self.editor._canvas_release(SimpleNamespace(x=290, y=240))
        self.assert_point(self.editor._xy(100, 100), (original[0]+30, original[1]+20))
        self.editor.pan_button.invoke()
        self.assertTrue(self.editor.pan_mode.get())
        x, y = self.editor._xy(100, 100)
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=0))
        self.assertIsNone(self.editor._drag)
        self.editor._canvas_drag(SimpleNamespace(x=x+45, y=y+35))
        self.editor._canvas_release(SimpleNamespace(x=x+45, y=y+35))
        self.assertEqual(self.editor.get_document().data, before)
        self.assertFalse(self.editor.dirty)
        self.editor.pan_button.invoke()
        self.assertFalse(self.editor.pan_mode.get())

    def test_zoomed_unit_and_waypoint_drag_save_real_coordinates(self):
        self.editor.zoom_map(3, self.editor._xy(100, 100), animate=False)
        self.editor.select(('units', 0))
        x, y = self.editor._xy(100, 100)
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=0))
        x, y = self.editor._xy(160, 130)
        self.editor._canvas_drag(SimpleNamespace(x=x, y=y))
        self.editor._canvas_release(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.editor.get_document().data['units'][0]['x'], 160)
        self.assertEqual(self.editor.get_document().data['units'][0]['y'], 130)
        self.editor.undo()
        self.editor.select(('waypoints', 0, 0))
        x, y = self.editor._xy(150, 100)
        self.editor._canvas_press(SimpleNamespace(x=x, y=y, state=0))
        x, y = self.editor._xy(200, 140)
        self.editor._canvas_drag(SimpleNamespace(x=x, y=y))
        self.editor._canvas_release(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.editor.get_document().data['units'][0]['waypoints'][0], {'x': 200, 'y': 140})
        self.editor.undo()
        self.assertEqual(self.editor.get_document().data, editing_scene())

    def test_resize_keeps_world_center_and_terrain_is_viewport_bounded(self):
        self.editor.zoom_map(6, (210, 160), animate=False)
        center = self.editor._world_xy(300, 220)
        with patch.object(self.editor.canvas, 'winfo_width', return_value=450), \
                patch.object(self.editor.canvas, 'winfo_height', return_value=360):
            self.editor._draw()
            self.assert_point(self.editor._world_xy(225, 180), center)
            if self.editor._terrain_image is not None:
                self.assertLessEqual(self.editor._terrain_image.width(), 450)
                self.assertLessEqual(self.editor._terrain_image.height(), 360)
        self.assertLessEqual(len(self.editor._terrain_images), 4)

    def test_navigation_and_disclosure_reverse_without_delaying_state(self):
        calls = []
        nav = NavigationRail(self.root, lambda index: (calls.append(index), nav.select(index)))
        self.addCleanup(nav.destroy)
        nav._click(SimpleNamespace(y=112+3*52+20))
        self.assertEqual(calls, [3])
        self.assertEqual(nav.selected, 3)
        self.assertNotEqual(nav._indicator, 3)
        self.root.after(35, lambda: nav.select(1))
        group = CollapsibleFrame(self.root, '技术详情')
        self.addCleanup(group.destroy)
        value = tk.StringVar(master=self.root, value='retain')
        ttk.Entry(group.body, textvariable=value).grid()
        self.root.update_idletasks()
        group.toggle()
        self.assertTrue(group.expanded)
        self.root.after(35, group.toggle)
        self.run_loop(260)
        self.assertEqual(nav.selected, 1)
        self.assertEqual(nav._indicator, 1)
        self.assertFalse(group.expanded)
        self.assertEqual(value.get(), 'retain')
        self.assertFalse(group.clip.winfo_manager())
        self.assertIsNone(group._transition.after_id)

    def test_destroy_cancels_pending_camera_navigation_disclosure_and_toggle(self):
        parent = ttk.Frame(self.root)
        nav = NavigationRail(parent, lambda _: None)
        group = CollapsibleFrame(parent, '详情')
        variable = tk.BooleanVar(master=self.root, value=False)
        switch = BooleanToggle(parent, variable)
        nav.select(3)
        group.toggle()
        switch._toggle()
        self.assertTrue(variable.get())
        self.editor.zoom_map(1.2)
        self.editor._request_draw()
        transitions = [nav._transition, group._transition, switch._transition, self.editor._zoom_motion]
        pending = [item.after_id for item in transitions] + [self.editor._redraw_after]
        parent.destroy()
        self.editor.destroy()
        for transition in transitions:
            self.assertIsNone(transition.after_id)
        current = self.root.tk.call('after', 'info')
        self.assertFalse(set(pending) & set(current))
        self.run_loop(200)


if __name__ == '__main__':
    unittest.main()
