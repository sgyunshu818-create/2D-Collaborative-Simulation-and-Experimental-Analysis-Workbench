"""Play recorded snapshots without rerunning movement, sensing or game rules.

Version 2 recordings are sparse (usually one frame per simulated second).
Playback holds each recorded frame until the next recorded time. It never
interpolates missing positions or invents planned paths and trails.
"""

import copy
import json
import math
from bisect import bisect_left, bisect_right
from collections import deque
from pathlib import Path

from .equipment import profile_key
from .models import BehaviorState, Contact, EquipmentType, Point, RunState, Team, Unit
from .scene import SceneConfigError, scene_from_data


class ReplayError(ValueError):
    """A recording error carrying the original filename and offending field."""

    def __init__(self, path: str | Path, field: str, message: str):
        self.path = Path(path)
        self.field = field
        super().__init__(f"{self.path}: {field}: {message}")


class _Validator:
    def __init__(self, path, scene, events):
        self.path, self.scene, self.events = path, scene, events
        self.specs = {unit.id: unit for unit in scene.units}

    def fail(self, field, message):
        raise ReplayError(self.path, field, message)

    def obj(self, value, field):
        if not isinstance(value, dict):
            self.fail(field, "must be a JSON object")
        return value

    def array(self, value, field):
        if not isinstance(value, list):
            self.fail(field, "must be a JSON array")
        return value

    def number(self, value, field, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(field, "must be a finite non-negative number")
        try:
            number = float(value)
        except (ValueError, OverflowError):
            self.fail(field, "must be a finite non-negative number")
        if not math.isfinite(number) or number < 0 or (positive and number == 0):
            self.fail(field, "must be a finite positive number" if positive else "must be a finite non-negative number")
        return number

    def integer(self, value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            self.fail(field, "must be a non-negative integer")
        return value

    def string(self, value, field, empty=True):
        if not isinstance(value, str) or (not empty and not value.strip()):
            self.fail(field, "must be a string" if empty else "must be a non-empty string")
        return value

    def boolean(self, value, field):
        if not isinstance(value, bool):
            self.fail(field, "must be a JSON boolean")
        return value

    def point(self, value, field):
        x, y = self.number(value.get("x"), field + ".x"), self.number(value.get("y"), field + ".y")
        if x >= self.scene.width or y >= self.scene.height:
            self.fail(field, "position must be inside the world")

    def clock(self, value, field):
        step = self.integer(value.get("step"), field + ".step")
        self.number(step, field + ".step")
        time = self.number(value.get("time"), field + ".time")
        if not math.isclose(time, step * self.scene.fixed_dt, rel_tol=1e-9, abs_tol=1e-9):
            self.fail(field + ".time", "must agree with step * scene.fixed_dt")
        return step, time

    def reference(self, value, field, optional=False):
        if optional and value is None:
            return
        if not isinstance(value, str) or value not in self.specs:
            self.fail(field, "must reference a unit in the recorded scene")

    def snapshot(self, value, field, version):
        value = self.obj(value, field)
        step, time = self.clock(value, field)
        if not isinstance(value.get("state"), str) or value["state"] not in {state.value for state in RunState}:
            self.fail(field + ".state", "unknown run state")
        found = set()
        for index, unit in enumerate(self.array(value.get("units"), field + ".units")):
            where = f"{field}.units[{index}]"
            unit = self.obj(unit, where)
            self.reference(unit.get("id"), where + ".id")
            if unit["id"] in found:
                self.fail(where + ".id", "duplicate unit")
            found.add(unit["id"])
            spec = self.specs[unit["id"]]
            if unit.get("team") != spec.team.value or unit.get("type") != spec.unit_type.value:
                self.fail(where, "team and type must agree with the scene unit")
            if "equipment" in unit:
                try:
                    equipment_type = EquipmentType(unit["equipment"])
                except (ValueError, TypeError):
                    self.fail(where + ".equipment", "unknown equipment type")
                if equipment_type.unit_type is not spec.unit_type or equipment_type.value != profile_key(spec):
                    self.fail(where + ".equipment", "equipment must agree with the scene unit")
            self.point(unit, where)
            if not isinstance(unit.get("behavior"), str) or unit["behavior"] not in {state.value for state in BehaviorState}:
                self.fail(where + ".behavior", "unknown behavior state")
            waypoint_index = self.integer(unit.get("waypoint_index"), where + ".waypoint_index")
            if waypoint_index > len(spec.waypoints):
                self.fail(where + ".waypoint_index", "exceeds the recorded route length")
            self.string(unit.get("reason"), where + ".reason")
            self.number(unit.get("distance_travelled"), where + ".distance_travelled")
            if version != 3:
                continue
            self.number(unit.get("sensor_range"), where + ".sensor_range", positive=True)
            targets = set()
            for cindex, contact in enumerate(self.array(unit.get("contacts"), where + ".contacts")):
                cwhere = f"{where}.contacts[{cindex}]"
                contact = self.obj(contact, cwhere)
                self.reference(contact.get("target_id"), cwhere + ".target_id")
                self.reference(contact.get("source_id"), cwhere + ".source_id")
                target_id = contact["target_id"]
                if target_id in targets or self.specs[target_id].team is spec.team:
                    self.fail(cwhere + ".target_id", "must be a unique opposing unit")
                targets.add(target_id)
                if self.specs[contact["source_id"]].team is not spec.team:
                    self.fail(cwhere + ".source_id", "observer must belong to the receiving team")
                self.point(contact, cwhere)
                if self.integer(contact.get("observed_step"), cwhere + ".observed_step") > step:
                    self.fail(cwhere + ".observed_step", "cannot be in the future")
                self.boolean(contact.get("shared"), cwhere + ".shared")
            tagged = self.array(unit.get("tagged_targets"), where + ".tagged_targets")
            checked = set()
            for tindex, target_id in enumerate(tagged):
                twhere = f"{where}.tagged_targets[{tindex}]"
                self.reference(target_id, twhere)
                if target_id in checked or self.specs[target_id].team is spec.team:
                    self.fail(twhere, "must be a unique opposing unit")
                checked.add(target_id)
            self.integer(unit.get("tag_count"), where + ".tag_count")
            self.integer(unit.get("tag_flash_until_step"), where + ".tag_flash_until_step")
        if found != set(self.specs):
            self.fail(field + ".units", "must contain every scene unit exactly once")
        if version == 3:
            scores = self.obj(value.get("scores"), field + ".scores")
            if set(scores) != {team.value for team in Team}:
                self.fail(field + ".scores", "must contain red and blue scores")
            for team in Team:
                self.integer(scores[team.value], field + ".scores." + team.value)
            self.boolean(value.get("sharing_enabled"), field + ".sharing_enabled")
            self.string(value.get("finish_reason"), field + ".finish_reason")
            if value.get("winner") not in (None, "", "red", "blue", "draw"):
                self.fail(field + ".winner", "must be empty, null, red, blue or draw")
            count = self.integer(value.get("event_count"), field + ".event_count")
            if count > len(self.events):
                self.fail(field + ".event_count", "exceeds the event count")
            if count and self.events[count - 1]["step"] > step:
                self.fail(field + ".event_count", "exposes events from a future step")
        return step, time


def _legacy_event_counts(snapshots, events):
    """Recover v2 control boundaries, including pause/resume at the same step."""
    markers = {}
    cursor = 0
    previous = None
    for index, snapshot in enumerate(snapshots):
        state = snapshot["state"]
        if state != previous and state != RunState.READY.value:
            kind = {RunState.RUNNING.value: "resumed" if previous == RunState.PAUSED.value else "run_started",
                    RunState.PAUSED.value: "paused", RunState.FINISHED.value: "run_finished"}[state]
            for event_index in range(cursor, len(events)):
                event = events[event_index]
                if event["step"] > snapshot["step"]:
                    break
                if event["kind"] == kind and event["step"] == snapshot["step"]:
                    markers[index] = event_index
                    cursor = event_index + 1
                    break
        previous = state
    counts = []
    event_index = 0
    for index, snapshot in enumerate(snapshots):
        while event_index < len(events) and events[event_index]["step"] <= snapshot["step"]:
            event_index += 1
        count = event_index
        if snapshot["state"] == RunState.READY.value:
            count = 0
        else:
            for next_index in range(index + 1, len(snapshots)):
                if snapshots[next_index]["step"] != snapshot["step"]:
                    break
                if next_index in markers:
                    count = min(count, markers[next_index])
                    break
        counts.append(count)
    return counts


class Replay:
    """A presentation model exposing the same display fields as Simulation.

    ``state`` controls playback; ``source_state`` is the saved simulation state.
    Seeking pauses playback, including at the last frame. Normal playback ends
    in FINISHED even when the saved run was incomplete or paused.
    """

    is_replay = True
    TRAIL_LIMIT = 300

    def __init__(self, scene, payload, path):
        self.scene = scene
        self.source_path = Path(path)
        self.format_version = payload["format_version"]
        self._snapshots = copy.deepcopy(payload["snapshots"])
        self._events = copy.deepcopy(payload["events"])
        self.metadata = copy.deepcopy(payload.get("metadata", {}))
        self._times = [frame["time"] for frame in self._snapshots]
        self._previous_time_indices = []
        previous = -1
        for index, frame in enumerate(self._snapshots):
            if index and frame["time"] > self._snapshots[index - 1]["time"]:
                previous = index - 1
            self._previous_time_indices.append(previous)
        # Only frame references are indexed; displayed trails hold <=300 points.
        self._trail_change_indices = {unit.id: [] for unit in scene.units}
        last_positions = {}
        for index, frame in enumerate(self._snapshots):
            for saved in frame["units"]:
                position = (saved["x"], saved["y"])
                if last_positions.get(saved["id"]) != position:
                    self._trail_change_indices[saved["id"]].append(index)
                    last_positions[saved["id"]] = position
        self._trails = {unit.id: deque(maxlen=self.TRAIL_LIMIT) for unit in scene.units}
        self._applied_index = -1
        self._event_counts = ([frame["event_count"] for frame in self._snapshots]
                              if self.format_version == 3 else
                              _legacy_event_counts(self._snapshots, self._events))
        self.state = RunState.READY
        self.index = 0
        self._apply(0)
        self._playback_time = self.sim_time

    @property
    def count(self):
        return len(self._snapshots)

    @property
    def snapshots(self):
        return copy.deepcopy(self._snapshots)

    @property
    def current_snapshot(self):
        return copy.deepcopy(self._snapshots[self.index])

    @property
    def result_snapshot(self):
        return copy.deepcopy(self._snapshots[-1])

    @property
    def events(self):
        return copy.deepcopy(self._events[:self._event_counts[self.index]])

    @property
    def has_missions(self):
        return self.scene.rules is not None or any(unit.waypoints or unit.return_home for unit in self.scene.units)

    @property
    def finished(self):
        return self.state is RunState.FINISHED

    def _apply(self, index):
        self._update_trails(index)
        self.index = index
        frame = self._snapshots[index]
        self.step_count, self.sim_time = frame["step"], frame["time"]
        self.source_state = RunState(frame["state"])
        self.scores = copy.deepcopy(frame.get("scores", {"red": 0, "blue": 0}))
        self.sharing_enabled = frame.get("sharing_enabled", False)
        self.finish_reason = frame.get("finish_reason", "")
        self.winner = frame.get("winner")
        specs = {unit.id: unit for unit in self.scene.units}
        previous_index = self._previous_time_indices[index]
        previous_frame = self._snapshots[previous_index] if previous_index >= 0 else None
        previous_units = {row["id"]: row for row in previous_frame["units"]} if previous_frame else {}
        self.recorded_speeds = {}
        self.units = []
        for saved in frame["units"]:
            spec = specs[saved["id"]]
            position = Point(saved["x"], saved["y"])
            unit = Unit(spec.id, spec.team, spec.unit_type, position,
                        behavior=BehaviorState(saved["behavior"]), speed=spec.speed,
                        waypoints=spec.waypoints, return_home=spec.return_home,
                        waypoint_index=saved["waypoint_index"], reason=saved["reason"],
                        distance_travelled=saved["distance_travelled"], trail=list(self._trails[spec.id]),
                        sensor_range=saved.get("sensor_range", spec.sensor_range),
                        equipment_type=spec.equipment_type)
            unit.contacts = {contact["target_id"]: Contact(
                contact["target_id"], Point(contact["x"], contact["y"]),
                contact["observed_step"], contact["source_id"], contact["shared"])
                for contact in saved.get("contacts", [])}
            unit.tagged_targets = set(saved.get("tagged_targets", []))
            unit.tag_count = saved.get("tag_count", 0)
            unit.tag_flash_until_step = saved.get("tag_flash_until_step", 0)
            self.units.append(unit)
            speed = None
            if previous_frame is not None:
                delta = saved["distance_travelled"] - previous_units[spec.id]["distance_travelled"]
                if delta >= 0:
                    speed = delta / (frame["time"] - previous_frame["time"])
                    if not math.isfinite(speed):
                        speed = None
            self.recorded_speeds[spec.id] = speed
        self._applied_index = index

    def _update_trails(self, index):
        """Increment normal playback; rebuild only the bounded tail when seeking."""
        if index == self._applied_index + 1:
            for saved in self._snapshots[index]["units"]:
                point = Point(saved["x"], saved["y"])
                trail = self._trails[saved["id"]]
                if not trail or point != trail[-1]:
                    trail.append(point)
            return
        for unit_id, indices in self._trail_change_indices.items():
            stop = bisect_right(indices, index)
            trail = self._trails[unit_id]
            trail.clear()
            for frame_index in indices[max(0, stop - self.TRAIL_LIMIT):stop]:
                saved = next(row for row in self._snapshots[frame_index]["units"] if row["id"] == unit_id)
                trail.append(Point(saved["x"], saved["y"]))

    def start(self):
        if self.state is not RunState.READY:
            return False
        self.state = RunState.FINISHED if self.index == self.count - 1 else RunState.RUNNING
        return True

    def pause(self):
        if self.state is not RunState.RUNNING:
            return False
        self.state = RunState.PAUSED
        return True

    def resume(self):
        if self.state is not RunState.PAUSED:
            return False
        self.state = RunState.FINISHED if self.index == self.count - 1 else RunState.RUNNING
        return True

    def reset(self):
        changed = self.index != 0 or self.state is not RunState.READY or self._playback_time != self._snapshots[0]["time"]
        self.state = RunState.READY
        self._apply(0)
        self._playback_time = self.sim_time
        return changed

    def seek(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.count:
            return False
        self.state = RunState.PAUSED
        self._apply(index)
        self._playback_time = self.sim_time
        return True

    def previous(self):
        return self.seek(self.index - 1)

    def seek_time(self, seconds):
        """Hold the last recorded frame at/before seconds; never interpolate."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return False
        try:
            seconds = float(seconds)
        except (ValueError, OverflowError):
            return False
        if not math.isfinite(seconds) or seconds < 0:
            return False
        return self.seek(max(0, bisect_right(self._times, seconds) - 1))

    def seek_event(self, event_index):
        """Select the first frame exposing this zero-based original event."""
        if isinstance(event_index, bool) or not isinstance(event_index, int) or not 0 <= event_index < len(self._events):
            return False
        index = bisect_left(self._event_counts, event_index + 1)
        return self.seek(index) if index < self.count else False

    def next(self):
        return self.seek(self.index + 1)

    def advance(self, real_dt):
        if isinstance(real_dt, bool) or not isinstance(real_dt, (float, int)):
            raise ValueError("real_dt must be a finite, non-negative number")
        try:
            elapsed = float(real_dt)
        except (ValueError, OverflowError) as error:
            raise ValueError("real_dt must be a finite, non-negative number") from error
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("real_dt must be a finite, non-negative number")
        if self.state is not RunState.RUNNING:
            return 0
        increment = min(elapsed, max(0.0, self._snapshots[-1]["time"] - self._playback_time))
        self._playback_time = min(self._snapshots[-1]["time"], math.fsum((self._playback_time, increment)))
        previous = self.index
        tolerance = self.scene.fixed_dt * 1e-9
        target = bisect_right(self._times, self._playback_time + tolerance) - 1
        if target > self.index:
            self._apply(target)
        if self.index == self.count - 1:
            self.state = RunState.FINISHED
        return self.index - previous


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number {value}")
    return number


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value {value}")


def _load_recording(path: str | Path):
    """Read format 2/3 JSON and validate recorded facts before opening a UI."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig") as stream:
            payload = json.load(stream, parse_constant=_reject_constant, parse_float=_finite_float)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise ReplayError(path, "$", f"cannot read recording: {error}") from error
    if not isinstance(payload, dict):
        raise ReplayError(path, "$", "must be a JSON object")
    version = payload.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in (2, 3):
        raise ReplayError(path, "format_version", "supported versions are 2 and 3")
    try:
        scene = scene_from_data(payload.get("scene"), path)
    except SceneConfigError as error:
        raise ReplayError(path, "scene." + error.field, error.message) from error
    if (scene.rules is not None) != (version == 3):
        raise ReplayError(path, "scene.rules", "game rules require format_version 3")
    events = payload.get("events")
    validator = _Validator(path, scene, events)
    events = validator.array(events, "events")
    previous_clock = (-1, -1.0)
    for index, event in enumerate(events):
        where = f"events[{index}]"
        event = validator.obj(event, where)
        clock = validator.clock(event, where)
        if clock[0] < previous_clock[0] or clock[1] < previous_clock[1]:
            validator.fail(where, "events must be ordered by non-decreasing step and time")
        previous_clock = clock
        validator.string(event.get("kind"), where + ".kind", empty=False)
        validator.string(event.get("message"), where + ".message")
        validator.reference(event.get("unit_id"), where + ".unit_id", optional=True)
        for field in ("target_id", "source_id"):
            if field in event:
                validator.reference(event[field], where + "." + field, optional=True)
        if "details" in event:
            details = validator.obj(event["details"], where + ".details")
            for field in ("target_id", "source_id"):
                if field in details:
                    validator.reference(details[field], where + ".details." + field, optional=True)
    snapshots = validator.array(payload.get("snapshots"), "snapshots")
    if not snapshots:
        validator.fail("snapshots", "at least one recorded frame is required")
    previous_clock = (-1, -1.0)
    previous_count = 0
    for index, snapshot in enumerate(snapshots):
        where = f"snapshots[{index}]"
        clock = validator.snapshot(snapshot, where, version)
        if clock[0] < previous_clock[0] or clock[1] < previous_clock[1]:
            validator.fail(where, "frames must be ordered by non-decreasing step and time")
        previous_clock = clock
        if version == 3:
            if snapshot["event_count"] < previous_count:
                validator.fail(where + ".event_count", "cannot decrease between frames")
            previous_count = snapshot["event_count"]
    if events and events[-1]["step"] > snapshots[-1]["step"]:
        validator.fail("events", "contains events after the final recorded frame")
    if version == 3 and previous_count != len(events):
        validator.fail("snapshots[-1].event_count", "final frame must expose every recorded event")
    if payload.get("result") != snapshots[-1]:
        validator.fail("result", "must exactly match the final snapshot")
    return scene, payload


def load_recording(path: str | Path) -> dict:
    """Load validated original facts without constructing playback copies."""
    return _load_recording(path)[1]


def load_replay(path: str | Path) -> Replay:
    scene, payload = _load_recording(path)
    return Replay(scene, payload, path)
