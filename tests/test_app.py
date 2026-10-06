"""Real Pygame input events against the app and renderer, without a window."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from sim_app.app import App
from sim_app.models import Point, RunState
from sim_app.renderer import Renderer
from sim_app.scene import load_scene
from sim_app.simulation import Simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy"})
        cls.environment.start()
        cls.addClassCleanup(cls.environment.stop)
        if not pygame.display.get_init():
            pygame.display.init()
            cls.addClassCleanup(pygame.display.quit)
        if not pygame.font.get_init():
            pygame.font.init()
            cls.addClassCleanup(pygame.font.quit)
        cls.renderer = Renderer(pygame.Surface((1280, 800)))
        cls.scene = load_scene(PROJECT_ROOT / "configs" / "default_scene.json")

    def setUp(self):
        self.simulation = Simulation(self.scene)
        self.app = App(self.simulation, self.renderer)
        # Input tests exercise controls; export is covered with temporary paths.
        saver = patch.object(self.app, "save_if_needed", return_value=True)
        saver.start()
        self.addCleanup(saver.stop)

    def key(self, key):
        return self.app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))

    def click(self, action, button=1):
        control = next(control for control in self.renderer.buttons if control.action == action)
        return self.app.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=control.rect.center)
        )

    def test_keyboard_controls_update_state_clock_and_exit(self):
        simulation = self.simulation
        initial_positions = [unit.position for unit in simulation.units]
        self.assertFalse(self.key(pygame.K_SPACE))
        self.assertEqual(simulation.state, RunState.READY)
        self.assertTrue(self.key(pygame.K_RETURN))
        self.assertEqual(simulation.state, RunState.RUNNING)
        self.assertEqual(simulation.advance(self.scene.fixed_dt * 3), 3)
        self.assertFalse(self.key(pygame.K_RETURN))
        self.assertEqual(simulation.step_count, 3)
        self.assertTrue(self.key(pygame.K_SPACE))
        self.assertEqual(simulation.state, RunState.PAUSED)
        self.assertEqual(simulation.advance(100), 0)
        self.assertEqual(simulation.step_count, 3)
        self.assertTrue(self.key(pygame.K_SPACE))
        self.assertEqual(simulation.state, RunState.RUNNING)
        self.assertEqual(simulation.advance(self.scene.fixed_dt), 1)
        self.assertEqual(simulation.step_count, 4)
        simulation.units[0].position = Point(1, 2)
        self.assertTrue(self.key(pygame.K_r))
        self.assertEqual(simulation.state, RunState.READY)
        self.assertEqual((simulation.step_count, simulation.sim_time), (0, 0))
        self.assertEqual([unit.position for unit in simulation.units], initial_positions)
        self.assertTrue(self.app.active)
        self.assertFalse(self.key(pygame.K_ESCAPE))
        self.assertFalse(self.app.active)

    def test_keypad_enter_starts_and_unrelated_events_do_nothing(self):
        self.assertFalse(self.app.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_RETURN)))
        self.assertFalse(self.key(pygame.K_a))
        self.assertEqual(self.simulation.state, RunState.READY)
        self.assertTrue(self.key(pygame.K_KP_ENTER))
        self.assertEqual(self.simulation.state, RunState.RUNNING)
        self.assertTrue(self.app.active)

    def test_mouse_controls_ignore_disabled_buttons_and_right_clicks(self):
        self.assertFalse(self.click("pause"))
        self.assertFalse(self.click("resume"))
        self.assertFalse(self.click("start", button=3))
        self.assertEqual(self.simulation.state, RunState.READY)
        self.assertTrue(self.click("start"))
        self.assertEqual(self.simulation.state, RunState.RUNNING)
        self.simulation.advance(self.scene.fixed_dt * 2)
        self.assertFalse(self.click("start"))
        self.assertFalse(self.click("resume"))
        self.assertEqual(self.simulation.step_count, 2)
        self.assertTrue(self.click("pause"))
        self.assertEqual(self.simulation.state, RunState.PAUSED)
        self.assertFalse(self.click("start"))
        self.assertFalse(self.click("pause"))
        self.assertTrue(self.click("resume"))
        self.assertEqual(self.simulation.state, RunState.RUNNING)
        self.assertTrue(self.click("reset"))
        self.assertEqual(self.simulation.state, RunState.READY)
        self.assertEqual((self.simulation.step_count, self.simulation.sim_time), (0, 0))
        self.assertFalse(self.click("exit"))
        self.assertFalse(self.app.active)

    def test_window_close_event_stops_the_app(self):
        self.key(pygame.K_RETURN)
        self.assertFalse(self.app.handle_event(pygame.event.Event(pygame.QUIT)))
        self.assertFalse(self.app.active)


if __name__ == "__main__":
    unittest.main()
