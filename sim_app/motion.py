"""Pure display interpolation over existing fixed steps and recorded facts."""
from bisect import bisect_right
from collections import OrderedDict
from math import hypot

from .models import BehaviorState, Point, RunState
from .navigation import segment_clear


class MotionPresenter:
    def __init__(self):
        self._model = None
        self._segments = OrderedDict()
        self._next_key = None
        self._next_rows = {}
        self._last_state = None
        self._hold_step = None

    def sync(self, sim):
        """Discard live interpolation history at a user timing transition."""
        self._hold_step = (sim, sim.step_count)
        self._last_state = sim.state

    def _safe(self, sim, unit, before, after, interval):
        distance = hypot(after.x - before.x, after.y - before.y)
        if interval <= 0 or distance > max(0, unit.speed) * interval * 1.25 + 1e-7:
            return False
        key = (unit.unit_type, before, after)
        if key in self._segments:
            self._segments.move_to_end(key)
            return self._segments[key]
        result = segment_clear(sim.scene, before, after, unit.unit_type)
        if len(self._segments) >= 128:
            self._segments.popitem(last=False)
        self._segments[key] = result
        return result

    @staticmethod
    def _between(before, after, alpha):
        return Point(before.x + (after.x - before.x) * alpha,
                     before.y + (after.y - before.y) * alpha)

    def positions(self, sim):
        positions = {unit.id: unit.position for unit in sim.units}
        if self._model is not sim:
            self._model = sim
            self._segments.clear()
            self._next_key = None
            self._next_rows.clear()
            if self._hold_step is not None and self._hold_step[0] is not sim:
                self._hold_step = None
            self._last_state = None
        if sim.state is not RunState.RUNNING:
            self._hold_step = (sim, sim.step_count)
            self._last_state = sim.state
            return positions
        if self._last_state is not None and self._last_state is not RunState.RUNNING:
            self._hold_step = (sim, sim.step_count)
        self._last_state = sim.state
        if getattr(sim, 'is_replay', False):
            # Read the recording already owned by Replay. Never copy its whole
            # history each frame, seek it, or write interpolated values into it.
            frames, times = getattr(sim, '_snapshots', ()), getattr(sim, '_times', ())
            future = bisect_right(times, sim.sim_time)
            if not frames or future >= len(frames):
                return positions
            span = times[future] - sim.sim_time
            clock = getattr(sim, '_playback_time', sim.sim_time)
            if not sim.sim_time <= clock <= times[future] or span <= 0:
                return positions
            key = (sim.index, future)
            if key != self._next_key:
                self._next_key = key
                self._next_rows = {row['id']: row for row in frames[future]['units']}
            alpha = min(1.0, max(0.0, (clock - sim.sim_time) / span))
            for unit in sim.units:
                saved = self._next_rows.get(unit.id)
                if saved is None or unit.behavior not in (BehaviorState.NAVIGATING, BehaviorState.RETURNING):
                    continue
                end = Point(saved['x'], saved['y'])
                if saved['distance_travelled'] < unit.distance_travelled:
                    continue
                if self._safe(sim, unit, unit.position, end, span):
                    positions[unit.id] = self._between(unit.position, end, alpha)
            return positions
        interval = sim.scene.fixed_dt
        if self._hold_step == (sim, sim.step_count):
            return positions
        self._hold_step = None
        alpha = min(1.0, max(0.0, getattr(sim, '_accumulator', 0.0) / interval))
        frame = getattr(sim, '_render_step', None)
        endpoints = frame[1] if frame is not None and frame[0] == sim.step_count else {}
        for unit in sim.units:
            sample = endpoints.get(unit.id)
            if unit.behavior not in (BehaviorState.NAVIGATING, BehaviorState.RETURNING) or sample is None:
                continue
            identity, before, after = sample
            # Edits, teleports and replaced entities invalidate old endpoints.
            if identity == id(unit) and after == unit.position and self._safe(sim, unit, before, after, interval):
                positions[unit.id] = self._between(before, unit.position, alpha)
        return positions
