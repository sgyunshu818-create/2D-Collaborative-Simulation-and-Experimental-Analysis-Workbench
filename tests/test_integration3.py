"""Stage 3 controls preserve runs and read recorded facts through the UI."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from sim_app.app import App, PROJECT_ROOT
from sim_app.models import RunState
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.scene import load_scene
from sim_app.simulation import Simulation


class Stage3IntegrationTests(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.output = Path(self.workspace.name)
        self.scene_path = PROJECT_ROOT / "configs" / "interaction_scene.json"
        self.sim = Simulation(load_scene(self.scene_path))
        self.renderer = Renderer(pygame.Surface(WINDOW_SIZE))
        self.app = App(self.sim, self.renderer, self.scene_path, self.output)
        caption = patch.object(self.app, "_set_caption")
        caption.start()
        self.addCleanup(caption.stop)

    def click(self, action):
        button = next(b for b in (*self.renderer.buttons, *self.renderer.extra_buttons) if b.action == action)
        return self.app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=button.rect.center))

    def test_scene_switch_keeps_paused_game_records_and_restores_ready_business_scene(self):
        self.assertTrue(self.click("sharing"))
        self.assertFalse(self.sim.sharing_enabled)
        self.assertTrue(self.click("start"))
        self.sim.advance(0.5)
        self.assertFalse(self.click("scene"))
        self.assertTrue(self.click("pause"))
        self.assertTrue(self.click("scene"))
        self.assertEqual(self.app.simulation.state, RunState.READY)
        self.assertEqual(self.app.simulation.scene.name, "stage4_basic")
        self.assertIsNotNone(self.app.simulation.scene.rules)
        self.assertTrue(self.app.simulation.has_missions)
        saved = json.loads(self.app.last_run_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["result"]["state"], "PAUSED")
        self.assertEqual(saved["result"]["step"], 30)
        self.assertFalse(saved["result"]["sharing_enabled"])
        self.assertEqual(len(list(self.output.glob("run_*"))), 1)

    def completed_replay(self):
        self.sim.start()
        self.sim.advance(100)
        self.assertTrue(self.sim.finished)
        self.assertTrue(self.app.save_if_needed())
        self.renderer.replay_available = True
        self.assertTrue(self.click("replay"))
        return self.app.simulation

    def test_replay_controls_do_not_overwrite_or_export_the_recording(self):
        replay = self.completed_replay()
        recorded = self.app.last_run_path.read_bytes()
        self.assertTrue(replay.is_replay)
        self.assertEqual(replay.state, RunState.READY)
        self.assertFalse(self.click("sharing"))
        self.assertTrue(self.click("start"))
        replay.advance(100)
        self.assertTrue(replay.finished)
        self.assertTrue(self.click("reset"))
        self.assertEqual((replay.index, replay.state), (0, RunState.READY))
        self.assertTrue(self.app.save_if_needed())
        self.assertEqual(len(list(self.output.glob("run_*"))), 1)
        self.assertEqual(self.app.last_run_path.read_bytes(), recorded)

    def test_timeline_and_step_buttons_restore_exact_saved_frames(self):
        replay = self.completed_replay()
        self.renderer.draw(replay)
        pos = self.renderer.timeline.topright
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(pos[0] - 1, self.renderer.timeline.centery))
        self.assertTrue(self.app.handle_event(event))
        self.assertEqual(replay.state, RunState.PAUSED)
        self.assertEqual(replay.index, replay.count - 1)
        self.assertEqual(replay.current_snapshot, replay.result_snapshot)
        self.assertTrue(self.click("previous"))
        previous = replay.current_snapshot
        self.assertEqual(replay.state, RunState.PAUSED)
        self.assertTrue(self.click("next"))
        self.assertEqual(replay.current_snapshot, replay.result_snapshot)
        self.assertNotEqual(previous, replay.current_snapshot)
        self.renderer.draw(replay)

    def test_keyboard_sharing_scene_and_replay_shortcuts_reach_the_same_controls(self):
        event = lambda key: pygame.event.Event(pygame.KEYDOWN, key=key)
        self.assertTrue(self.app.handle_event(event(pygame.K_s)))
        self.assertFalse(self.sim.sharing_enabled)
        self.sim.start()
        self.sim.advance(100)
        self.assertTrue(self.app.handle_event(event(pygame.K_l)))
        replay = self.app.simulation
        self.assertTrue(replay.is_replay)
        self.assertTrue(self.app.handle_event(event(pygame.K_RIGHT)))
        self.assertTrue(self.app.handle_event(event(pygame.K_LEFT)))
        self.assertEqual(replay.index, 0)
        self.assertTrue(self.app.handle_event(event(pygame.K_c)))
        self.assertFalse(getattr(self.app.simulation, "is_replay", False))
        self.assertEqual(self.app.simulation.state, RunState.READY)


if __name__ == "__main__":
    unittest.main()
