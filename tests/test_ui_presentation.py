"""Map gestures and display interpolation must never change recorded facts."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

from sim_app.app import App
from sim_app.camera import MapCamera
from sim_app.models import Point
from sim_app.motion import MotionPresenter
from sim_app.recording import export_run
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.replay import load_replay
from sim_app.scene import scene_from_data
from sim_app.simulation import Simulation

PROJECT = Path(__file__).resolve().parents[1]


def simulation():
    data = json.loads((PROJECT / 'configs/basic_scene.json').read_text(encoding='utf-8'))
    data['fixed_dt'] = .1
    data.pop('rules')
    return Simulation(scene_from_data(data, PROJECT / 'configs/basic_scene.json'))


class PresentationTests(unittest.TestCase):
    def test_zoom_anchor_and_interruptions_preserve_world_point(self):
        camera = MapCamera()
        camera.bind((900, 600), (250, 190, 700, 460))
        anchor = (500, 360)
        world = camera.to_world(anchor)
        camera.zoom_by(2, anchor, now=10)
        for tick in (10.02, 10.05, 10.09):
            camera.update(tick)
            point = camera.to_screen(world)
            self.assertAlmostEqual(point[0], anchor[0])
            self.assertAlmostEqual(point[1], anchor[1])
        camera.zoom_by(2, anchor, now=10.10)
        camera.update(10.3)
        self.assertEqual(camera.zoom, 4)
        camera.reset(now=10.4)
        camera.update(10.6)
        self.assertEqual(camera.zoom, 1)
        self.assertEqual(camera.actual_center, Point(450, 300))

    def test_live_fractional_steps_interpolate_without_changing_state(self):
        sim = simulation()
        sim.start()
        sim.advance(.15)
        saved = copy.deepcopy((sim.events, sim.snapshots, sim.units))
        points = MotionPresenter().positions(sim)
        for unit in sim.units:
            a, b = sim.scene.units[[u.id for u in sim.units].index(unit.id)].position, unit.position
            self.assertAlmostEqual(points[unit.id].x, (a.x+b.x)/2)
            self.assertAlmostEqual(points[unit.id].y, (a.y+b.y)/2)
        self.assertEqual(saved, (sim.events, sim.snapshots, sim.units))

    def test_pause_resume_holds_until_next_fixed_step_without_reversing(self):
        sim = simulation()
        presenter = MotionPresenter()
        sim.start(); sim.advance(.15)
        presenter.positions(sim)
        sim.pause(); presenter.sync(sim)
        stopped = {u.id: u.position for u in sim.units}
        self.assertEqual(presenter.positions(sim), stopped)
        sim.resume(); presenter.sync(sim)
        for _ in range(4):
            sim.advance(.01)
            self.assertEqual(presenter.positions(sim), stopped)
        sim.advance(.02)
        shown = presenter.positions(sim)
        for unit in sim.units:
            self.assertGreater((shown[unit.id].x-stopped[unit.id].x) *
                               (unit.position.x-stopped[unit.id].x), 0)

    def test_teleport_is_not_interpolated_from_old_trail(self):
        sim = simulation(); sim.start(); sim.advance(.15)
        sim.units[0].position = Point(600, 500)
        self.assertEqual(MotionPresenter().positions(sim)[sim.units[0].id], Point(600, 500))

    def test_sparse_replay_interpolation_and_seek_preserve_snapshots(self):
        sim = simulation(); sim.start(); sim.advance(2.4)
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(export_run(sim, Path(directory))['run'])
            presenter = MotionPresenter()
            original = copy.deepcopy(replay._snapshots)
            replay.start(); replay.advance(.5)
            self.assertEqual(replay.sim_time, 0)
            shown = presenter.positions(replay)
            self.assertNotEqual(shown[replay.units[0].id], replay.units[0].position)
            replay.pause(); presenter.sync(replay)
            self.assertEqual(presenter.positions(replay), {u.id:u.position for u in replay.units})
            replay.seek(replay.count-1); presenter.sync(replay)
            self.assertEqual(presenter.positions(replay), {u.id:u.position for u in replay.units})
            self.assertEqual(replay._snapshots, original)

    def test_left_drag_moves_camera_and_plain_click_still_selects_unit(self):
        pygame.font.init()
        sim = simulation()
        renderer = Renderer(pygame.Surface(WINDOW_SIZE))
        renderer.draw(sim)
        app = App(sim, renderer)
        before = copy.deepcopy(sim.units)
        origin = renderer.map_rect.center
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=origin))
        app.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(origin[0]+70,origin[1]+35)))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(origin[0]+70,origin[1]+35)))
        renderer.draw(sim)
        point = renderer.world_to_screen(Point(450, 300), sim)
        self.assertAlmostEqual(point[0], origin[0]+70, delta=1)
        self.assertAlmostEqual(point[1], origin[1]+35, delta=1)
        unit = sim.units[-1]
        rect = renderer.unit_hit_rects[unit.id]
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=rect.center))
        self.assertEqual(renderer.selected_unit_id, unit.id)
        self.assertEqual(sim.units, before)
        self.assertIsNone(renderer._drag_button)

    def test_pan_focus_loss_and_sidebar_scroll_are_isolated(self):
        pygame.font.init()
        sim = simulation(); renderer = Renderer(pygame.Surface(WINDOW_SIZE)); renderer.draw(sim)
        renderer.ui_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=renderer.map_rect.center), sim)
        renderer.ui_event(pygame.event.Event(pygame.WINDOWFOCUSLOST), sim)
        self.assertIsNone(renderer._drag_button)
        with patch('pygame.mouse.get_pos', return_value=renderer.unit_scroll_rect.center):
            renderer.ui_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3), sim)
        self.assertEqual(renderer.camera.target_zoom, 1)


if __name__ == '__main__':
    unittest.main()
