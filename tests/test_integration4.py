"""Preserve live data on failed writes and keep all three demonstrations usable."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from sim_app.app import App
from sim_app.models import RunState
from sim_app.renderer import Renderer, WINDOW_SIZE
from sim_app.scenarios import DEFAULT_SCENE, SCENE_CHOICES, artifact_prefix, stage_name
from sim_app.scene import SceneConfigError, load_scene
from sim_app.simulation import Simulation


class Stage4IntegrationTests(unittest.TestCase):
    def setUp(self):
        pygame.font.init()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.sim = Simulation(load_scene(DEFAULT_SCENE))
        self.renderer = Renderer(pygame.Surface(WINDOW_SIZE))
        self.app = App(self.sim, self.renderer, DEFAULT_SCENE, self.output)
        caption = patch.object(self.app, "_set_caption")
        caption.start()
        self.addCleanup(caption.stop)

    def paused(self):
        self.sim.start()
        self.sim.advance(2)
        self.sim.pause()

    def test_business_cycle_saves_paused_run_then_recreates_ready_scene(self):
        self.assertEqual([p.name for p in SCENE_CHOICES],
                         ["basic_scene.json", "obstacle_scene.json", "sharing_scene.json"])
        self.paused()
        for expected in ("stage4_obstacle", "stage4_sharing", "stage4_basic"):
            self.assertTrue(self.app.dispatch("scene"))
            current = self.app.simulation
            self.assertEqual((current.scene.name, current.state, current.step_count),
                             (expected, RunState.READY, 0))
            self.renderer.draw(current)
        saved = json.loads(self.app.last_run_path.read_text(encoding="utf-8"))
        self.assertEqual((saved["scene"]["name"], saved["result"]["state"], saved["result"]["step"]),
                         ("stage4_basic", "PAUSED", 120))
        self.assertEqual(len(list(self.output.glob("run_*"))), 1)

    def test_failed_reset_preserves_snapshot_and_can_save_before_retry(self):
        self.paused()
        before = self.sim.snapshot()
        events = list(self.sim.events)
        with patch("sim_app.app.export_run", side_effect=OSError("read-only output")), contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.app.dispatch("reset"))
        self.assertEqual(self.sim.snapshot(), before)
        self.assertEqual(self.sim.events, events)
        self.assertTrue(self.app.active)
        self.assertFalse(self.app.saved_for_run)
        self.assertTrue(self.app.dispatch("reset"))
        saved = json.loads(self.app.last_run_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["result"], before)
        self.assertEqual((self.sim.state, self.sim.step_count), (RunState.READY, 0))

    def test_failed_exit_pauses_live_run_and_retry_saves_before_closing(self):
        self.sim.start()
        self.sim.advance(2)
        positions = [u.position for u in self.sim.units]
        with patch("sim_app.app.export_run", side_effect=OSError("disk write failed")), contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.app.dispatch("exit"))
        self.assertTrue(self.app.active)
        self.assertEqual((self.sim.state, self.sim.step_count), (RunState.PAUSED, 120))
        self.assertEqual([u.position for u in self.sim.units], positions)
        self.assertEqual(self.sim.advance(30), 0)
        self.assertFalse(self.app.dispatch("exit"))
        self.assertFalse(self.app.active)
        self.assertTrue(self.app.last_run_path.is_file())

    def test_window_close_cannot_discard_a_failed_export(self):
        self.paused()
        before = self.sim.snapshot()
        event = pygame.event.Event(pygame.QUIT)
        with patch("sim_app.app.export_run", side_effect=OSError("write failed")), contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(self.app.handle_event(event))
        self.assertTrue(self.app.active)
        self.assertEqual(self.sim.snapshot(), before)
        self.assertFalse(self.app.handle_event(event))
        self.assertFalse(self.app.active)

    def test_failed_scene_export_retains_the_current_world(self):
        self.paused()
        before = self.sim.snapshot()
        with patch("sim_app.app.export_run", side_effect=OSError("write failed")), contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(self.app.dispatch("scene"))
        self.assertIs(self.app.simulation, self.sim)
        self.assertEqual(self.sim.snapshot(), before)
        self.assertEqual(self.app.scene_path, DEFAULT_SCENE)

    def test_unreadable_next_scene_retains_current_state_and_can_retry(self):
        missing = self.output / "missing_scene.json"
        before = self.sim.snapshot()
        with patch("sim_app.app.SCENE_CHOICES", (DEFAULT_SCENE, missing)):
            self.assertFalse(self.app.dispatch("scene"))
        self.assertIs(self.app.simulation, self.sim)
        self.assertEqual(self.sim.snapshot(), before)
        self.assertIn("missing_scene.json", self.app.notice)
        self.assertTrue(self.app.dispatch("scene"))
        self.assertEqual(self.app.simulation.scene.name, "stage4_obstacle")

    def test_invalid_recording_does_not_replace_the_live_world(self):
        self.paused()
        self.assertTrue(self.app.save_if_needed())
        self.app.last_run_path.write_text("{", encoding="utf-8")
        before = self.sim.snapshot()
        self.assertFalse(self.app.dispatch("replay"))
        self.assertIs(self.app.simulation, self.sim)
        self.assertEqual(self.sim.snapshot(), before)
        self.assertIn("run.json", self.app.notice)

    def test_capture_failure_stops_only_optional_capture(self):
        self.app.capture_dir = self.output / "capture"
        self.paused()
        before = self.sim.snapshot()
        with patch("pygame.image.save", side_effect=OSError("capture write failed")), contextlib.redirect_stderr(io.StringIO()):
            self.app.capture_frame()
        self.assertIsNone(self.app.capture_dir)
        self.assertTrue(self.app.active)
        self.assertEqual(self.sim.snapshot(), before)
        self.assertTrue(self.app.save_if_needed())

    def test_deep_json_reports_configuration_error_without_recursion_traceback(self):
        path = self.output / "deep.json"
        path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
        with self.assertRaisesRegex(SceneConfigError, "nesting"):
            load_scene(path)

    def test_stage_names_and_artifacts_preserve_explicit_legacy_scenes(self):
        for filename, stage, prefix in (("basic_scene.json", "Stage 4", "stage4_basic"),
                                        ("interaction_scene.json", "Stage 3", "stage3"),
                                        ("navigation_scene.json", "Stage 2", "stage2"),
                                        ("default_scene.json", "Stage 1", "stage1")):
            scene = load_scene(DEFAULT_SCENE.parent / filename)
            sim = Simulation(scene)
            self.assertEqual(stage_name(scene, sim.has_missions), stage)
            self.assertEqual(artifact_prefix(scene, sim.has_missions), prefix)


if __name__ == "__main__":
    unittest.main()
