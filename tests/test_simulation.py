"""Deterministic clock and control checks, with no sleeping or Pygame window."""

import json
import tempfile
import unittest
from pathlib import Path

from sim_app.models import BehaviorState, Point, RunState
from sim_app.scene import load_scene
from sim_app.simulation import Simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    def setUp(self):
        # A binary-exact step isolates control semantics from rounding tests.
        config = json.loads((PROJECT_ROOT / "configs" / "default_scene.json").read_text(encoding="utf-8"))
        config["fixed_dt"] = 0.125
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.scene = load_scene(path)
        self.simulation = Simulation(self.scene)

    @staticmethod
    def unit_snapshot(units):
        return [(unit.id, unit.team, unit.unit_type, unit.position, unit.behavior) for unit in units]

    def test_initial_state_does_not_advance(self):
        simulation = self.simulation
        self.assertEqual(simulation.state, RunState.READY)
        self.assertEqual(simulation.step_count, 0)
        self.assertEqual(simulation.sim_time, 0)
        self.assertEqual(len(simulation.units), 4)
        self.assertTrue(all(unit.behavior == BehaviorState.IDLE for unit in simulation.units))
        self.assertEqual(simulation.advance(10), 0)
        self.assertEqual((simulation.step_count, simulation.sim_time), (0, 0))

    def test_running_uses_fixed_steps_and_keeps_remainder(self):
        simulation = self.simulation
        self.assertTrue(simulation.start())
        self.assertEqual(simulation.advance(0.3125), 2)
        self.assertEqual(simulation.step_count, 2)
        self.assertEqual(simulation.sim_time, 0.25)
        self.assertEqual(simulation.advance(0.0625), 1)
        self.assertEqual(simulation.step_count, 3)
        self.assertEqual(simulation.sim_time, 3 * self.scene.fixed_dt)

    def test_paused_time_is_discarded_on_resume(self):
        simulation = self.simulation
        simulation.start()
        simulation.advance(0.25)
        self.assertTrue(simulation.pause())
        self.assertEqual(simulation.state, RunState.PAUSED)
        self.assertEqual(simulation.advance(100), 0)
        self.assertEqual((simulation.step_count, simulation.sim_time), (2, 0.25))
        self.assertTrue(simulation.resume())
        self.assertEqual(simulation.state, RunState.RUNNING)
        self.assertEqual(simulation.advance(0.125), 1)
        self.assertEqual((simulation.step_count, simulation.sim_time), (3, 0.375))

    def test_invalid_and_repeated_controls_leave_state_and_clock_unchanged(self):
        simulation = self.simulation
        transitions = (
            ("pause", False, RunState.READY),
            ("resume", False, RunState.READY),
            ("start", True, RunState.RUNNING),
            ("start", False, RunState.RUNNING),
            ("resume", False, RunState.RUNNING),
            ("pause", True, RunState.PAUSED),
            ("pause", False, RunState.PAUSED),
            ("start", False, RunState.PAUSED),
            ("resume", True, RunState.RUNNING),
            ("resume", False, RunState.RUNNING),
        )
        for operation, changed, state in transitions:
            with self.subTest(operation=operation, state=state):
                self.assertEqual(getattr(simulation, operation)(), changed)
                self.assertEqual(simulation.state, state)
                self.assertEqual((simulation.step_count, simulation.sim_time), (0, 0))

    def test_static_units_remain_at_their_initial_positions(self):
        simulation = self.simulation
        before = self.unit_snapshot(simulation.units)
        simulation.start()
        simulation.advance(10)
        simulation.pause()
        simulation.advance(20)
        simulation.resume()
        simulation.advance(5)
        self.assertEqual(self.unit_snapshot(simulation.units), before)
        self.assertEqual(simulation.step_count, 120)

    def test_reset_from_every_state_restores_data_and_clears_time_remainder(self):
        for state in RunState:
            with self.subTest(state=state):
                simulation = Simulation(self.scene)
                initial = self.unit_snapshot(simulation.units)
                original_units = list(simulation.units)
                if state != RunState.READY:
                    simulation.start()
                    simulation.advance(0.3125)
                if state == RunState.PAUSED:
                    simulation.pause()
                simulation.units[0].position = Point(10, 20)
                self.assertTrue(simulation.reset())
                self.assertEqual(simulation.state, RunState.READY)
                self.assertEqual((simulation.step_count, simulation.sim_time), (0, 0))
                self.assertEqual(self.unit_snapshot(simulation.units), initial)
                self.assertIsNot(simulation.units[0], original_units[0])
                simulation.start()
                self.assertEqual(simulation.advance(0.0625), 0)
                self.assertEqual(simulation.advance(0.0625), 1)

    def test_runtime_units_are_independent_of_scene_and_other_simulations(self):
        first = self.simulation
        second = Simulation(self.scene)
        original = self.scene.units[0].position
        first.units[0].position = Point(10, 20)
        self.assertEqual(self.scene.units[0].position, original)
        self.assertEqual(second.units[0].position, original)
        first.reset()
        self.assertEqual(first.units[0].position, original)
        self.assertIsNot(first.units[0], second.units[0])

    def test_equal_elapsed_time_in_different_frame_chunks_has_equal_steps(self):
        scene = load_scene(PROJECT_ROOT / "configs" / "default_scene.json")
        frame_sequences = (
            [1.0],
            [0.1] * 10,
            [1 / 60] * 60,
            [0.04, 0.06] * 10,
            [0.001] * 1000,
        )
        for frame_times in frame_sequences:
            with self.subTest(frames=len(frame_times)):
                simulation = Simulation(scene)
                simulation.start()
                total_returned_steps = sum(simulation.advance(frame_dt) for frame_dt in frame_times)
                self.assertEqual(total_returned_steps, 60)
                self.assertEqual(simulation.step_count, 60)
                self.assertAlmostEqual(simulation.sim_time, 1.0)


if __name__ == "__main__":
    unittest.main()
