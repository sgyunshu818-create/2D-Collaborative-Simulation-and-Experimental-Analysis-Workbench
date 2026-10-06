"""World-map gestures, legacy coordinates and replay controls without a window."""
import copy
import os
from pathlib import Path
import tempfile
from time import monotonic
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from sim_app.app import App
from sim_app.geography import scene_world
from sim_app.models import Point, RunState
from sim_app.recording import export_run
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.replay import load_replay
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation
from tests.test_scene import valid_config


def model(*, referenced=True, moving=False):
    data = valid_config()
    if referenced:
        data['world']['georeference'] = dict(center_latitude=32, center_longitude=122,
                                             meters_per_unit=1000)
    if moving:
        data['obstacles'] = []
        for unit in data['units']:
            unit.update(speed=80, waypoints=[dict(x=450, y=300)])
    return Simulation(scene_from_data(data))


class WorldRendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.display.init()
        pygame.font.init()

    def setUp(self):
        self.renderer = Renderer(pygame.Surface(WINDOW_SIZE))

    def app(self, sim):
        app = App(sim, self.renderer)
        self.renderer.draw(sim)
        return app

    def click(self, app, action):
        controls = (*self.renderer.buttons, *self.renderer.map_buttons,
                    *self.renderer.map_navigation_buttons, *self.renderer.extra_buttons)
        button = next(b for b in controls if b.action == action)
        changed = app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=button.rect.center))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=button.rect.center))
        self.renderer.draw(app.simulation)
        return changed

    @staticmethod
    def facts(sim):
        return copy.deepcopy((sim.units, sim.events, getattr(sim, 'snapshots', None),
                              sim.step_count, sim.sim_time, sim.state))

    def test_reference_defaults_to_world_and_roundtrip_preserves_local_coordinates(self):
        sim = model()
        self.app(sim)
        self.assertEqual(self.renderer.map_mode, 'world')
        self.assertGreater(self.renderer.world_map.camera.zoom, 1)
        point = Point(450, 300)
        pixel = self.renderer.world_to_screen(point, sim)
        restored = self.renderer.screen_to_world(pixel, sim)
        # Renderer pixels are rounded; this view shows about two local units
        # per pixel. The inverse should remain within that pixel's footprint.
        self.assertAlmostEqual(restored.x, point.x, delta=2)
        self.assertAlmostEqual(restored.y, point.y, delta=2)
        self.assertEqual(self.renderer.world_to_screen(restored, sim), pixel)
        self.assertEqual(self.renderer.camera.zoom, 1)

    def test_global_scene_and_locate_buttons_only_change_display(self):
        sim = model()
        app = self.app(sim)
        saved = self.facts(sim)
        selected = self.renderer.selected_unit_id
        self.assertFalse(self.click(app, 'world_view'))
        self.renderer.world_map.camera.update(monotonic()+1)
        self.renderer.draw(sim)
        self.assertEqual(self.renderer.world_map.camera.zoom, 1)
        self.assertFalse(self.click(app, 'focus_scene'))
        self.assertGreater(self.renderer.world_map.camera.zoom, 1)
        self.assertFalse(self.click(app, 'scene_view'))
        self.assertEqual(self.renderer.map_mode, 'scene')
        self.assertFalse(self.click(app, 'focus_scene'))
        self.assertEqual(self.renderer.map_mode, 'world')
        self.assertEqual(self.renderer.selected_unit_id, selected)
        self.assertEqual(saved, self.facts(sim))

    def test_unreferenced_scene_browses_globe_without_fabricating_unit_positions(self):
        sim = model(referenced=False)
        app = self.app(sim)
        saved = self.facts(sim)
        self.assertEqual(self.renderer.map_mode, 'scene')
        self.assertIsNone(self.renderer.world_map)
        self.click(app, 'world_view')
        self.assertEqual(self.renderer.map_mode, 'world')
        self.assertEqual(self.renderer.unit_hit_rects, {})
        self.assertIsNone(self.renderer.world_to_screen(sim.units[0].position, sim))
        self.assertIsNone(self.renderer.screen_to_world(self.renderer.map_rect.center, sim))
        self.assertFalse(self.renderer.select_at(self.renderer.map_rect.center, sim))
        self.click(app, 'focus_scene')
        self.assertTrue(self.renderer.map_notice)
        self.assertIn(self.renderer.label('地理参考', 'geographic reference'), app.notice)
        self.click(app, 'scene_view')
        self.assertTrue(self.renderer.unit_hit_rects)
        self.assertEqual(saved, self.facts(sim))

    def test_world_click_drag_and_cursor_wheel_share_the_rendered_camera(self):
        sim = model()
        app = self.app(sim)
        before = self.facts(sim)
        unit = sim.units[-1]
        pixel = self.renderer.world_to_screen(unit.position, sim)
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pixel))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pixel))
        self.assertEqual(self.renderer.selected_unit_id, unit.id)
        origin = self.renderer.map_rect.center
        center = self.renderer.world_map.camera.actual_center
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=origin))
        app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(origin[0]+50, origin[1]+20)))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(origin[0]+50, origin[1]+20)))
        self.assertNotEqual(center, self.renderer.world_map.camera.actual_center)
        anchor = (self.renderer.map_rect.x+300, self.renderer.map_rect.y+200)
        local_anchor = (anchor[0]-self.renderer.map_rect.x, anchor[1]-self.renderer.map_rect.y)
        lonlat = self.renderer.world_map.to_lonlat(local_anchor)
        with patch('pygame.mouse.get_pos', return_value=anchor):
            app.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=2))
        self.renderer.world_map.camera.update(monotonic()+1)
        after = self.renderer.world_map.to_lonlat(local_anchor)
        self.assertAlmostEqual(lonlat[0], after[0], places=6)
        self.assertAlmostEqual(lonlat[1], after[1], places=6)
        self.assertEqual(self.renderer.camera.zoom, 1)
        self.assertEqual(before, self.facts(sim))

    def test_expand_restore_retains_pose_selection_and_controls_and_ignores_hidden_rows(self):
        sim = model()
        app = self.app(sim)
        normal_rect = self.renderer.map_rect.copy()
        center = self.renderer.world_map.camera.actual_center
        zoom = self.renderer.zoom
        selected = self.renderer.selected_unit_id
        hidden_row = self.renderer.unit_rows[-1][0].center
        self.click(app, 'map_maximize')
        self.assertGreaterEqual(self.renderer.map_rect.width, 1180)
        self.assertGreaterEqual(self.renderer.map_rect.height, 570)
        self.assertEqual(self.renderer.unit_rows, [])
        self.assertEqual(self.renderer.event_rows, [])
        self.assertEqual(self.renderer.world_map.camera.actual_center, center)
        self.assertEqual(self.renderer.zoom, zoom)
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=hidden_row))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=hidden_row))
        self.assertEqual(self.renderer.selected_unit_id, selected)
        self.assertTrue(self.click(app, 'start'))
        self.assertEqual(sim.state, RunState.RUNNING)
        self.assertTrue(self.click(app, 'pause'))
        self.assertEqual(sim.state, RunState.PAUSED)
        self.click(app, 'map_maximize')
        self.assertEqual(self.renderer.map_rect, normal_rect)
        self.assertTrue(self.renderer.unit_rows)
        self.assertEqual(self.renderer.selected_unit_id, selected)
        self.assertEqual(self.renderer.world_map.camera.actual_center, center)

    def test_world_rows_use_display_interpolation_without_mutating_records(self):
        sim = model(moving=True)
        sim.start()
        sim.advance(.1875)
        before = self.facts(sim)
        self.renderer.draw(sim)
        rows = self.renderer._geographic_units(sim)
        for unit, row in zip(sim.units, rows):
            shown = self.renderer._display_positions[unit.id]
            self.assertEqual((row['x'], row['y']), (shown.x, shown.y))
        self.assertNotEqual(rows[0]['x'], sim.units[0].position.x)
        self.assertEqual(before, self.facts(sim))

    def test_world_view_is_cached_across_repeated_mode_switches(self):
        sim = model()
        app = self.app(sim)
        world_map = self.renderer.world_map
        for _ in range(3):
            self.click(app, 'scene_view')
            self.click(app, 'world_view')
        self.assertIs(self.renderer.world_map, world_map)

    def test_replay_keeps_saved_reference_and_expanded_timeline_reachable(self):
        sim = model(moving=True)
        sim.start()
        sim.advance(.5)
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(export_run(sim, Path(directory))['run'])
        app = self.app(replay)
        saved = copy.deepcopy(replay._snapshots)
        self.assertEqual(scene_world(replay.scene)['georeference'], scene_world(sim.scene)['georeference'])
        self.click(app, 'map_maximize')
        self.assertGreaterEqual(self.renderer.map_rect.width, 1180)
        self.assertGreater(self.renderer.map_rect.height, 500)
        pos = (self.renderer.timeline.right-1, self.renderer.timeline.centery)
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))
        self.renderer.draw(replay)
        self.assertEqual(replay.index, replay.count-1)
        self.assertEqual(replay.state, RunState.PAUSED)
        unit = replay.units[-1]
        pixel = self.renderer.world_to_screen(unit.position, replay)
        self.assertTrue(self.renderer.select_at(pixel, replay))
        self.assertEqual(self.renderer.selected_unit_id, unit.id)
        self.click(app, 'scene_view')
        self.click(app, 'focus_scene')
        self.assertEqual(replay._snapshots, saved)


if __name__ == '__main__':
    unittest.main()
