"""Fixed-step navigation and an optional fictional observation-and-tag game."""

import math

from .engagement import adjudicate_tags, duration_steps, refresh_contacts, tag_candidates
from .models import BehaviorState, Point, RunState, Scene, Team, Unit
from .navigation import plan_path, point_clear, segment_clear
from .recording import new_run_metadata, utc_timestamp

# Rules scenes record one frame every this many logic steps, plus every frame
# whose outcome changes. Three steps is 20 Hz at the default 1/60 s step, which
# keeps the curves exact and replay smooth while cutting the record by about two
# thirds. Replay never interpolates, so a longer stride reads as visible
# stepping for the fastest units.
RECORD_STRIDE = 3

# Recorded contact coordinates retain full precision for replay validation.


class Simulation:
    def __init__(self, scene: Scene, record_every_step: bool = False) -> None:
        self.scene = scene
        self.record_every_step = bool(record_every_step)
        self.state = RunState.READY
        self.units = self._initial_units()
        self.step_count = 0
        self._accumulator = 0.0
        self._render_step = None
        self._last_snapshot_second = 0
        self.events: list[dict] = []
        self.snapshots: list[dict] = []
        self.sharing_enabled = scene.rules.sharing_enabled if scene.rules is not None else False
        self.scores: dict[Team, int] = {team: 0 for team in Team}
        self.finish_reason = ""
        self.winner = ""
        self._last_tag_steps: dict[str, int] = {}
        self.metadata = new_run_metadata()
        self._last_export_signature = None
        self._record_snapshot()

    @property
    def sim_time(self) -> float:
        """Derive elapsed time from integer steps rather than repeated additions."""
        return self.step_count * self.scene.fixed_dt

    @property
    def has_missions(self) -> bool:
        return any(spec.waypoints or spec.return_home for spec in self.scene.units)

    @property
    def finished(self) -> bool:
        return self.state is RunState.FINISHED

    def _initial_units(self) -> list[Unit]:
        return [
            Unit(
                spec.id, spec.team, spec.unit_type, spec.position,
                speed=spec.speed, waypoints=spec.waypoints, return_home=spec.return_home,
                trail=[spec.position], sensor_range=spec.sensor_range, equipment_type=spec.equipment_type,
            )
            for spec in self.scene.units
        ]

    def _event(self, kind: str, message: str, unit: Unit | None = None, **details) -> None:
        event = {
            "step": self.step_count,
            "time": self.sim_time,
            "kind": kind,
            "unit_id": unit.id if unit is not None else None,
            "message": message,
        }
        if details:
            event["details"] = details
        self.events.append(event)

    def snapshot(self) -> dict:
        """Return a detached current state; legacy scenes retain their shape."""
        snapshot = {
            "step": self.step_count,
            "time": self.sim_time,
            "state": self.state.value,
            "units": [{
                "id": unit.id,
                "team": unit.team.value,
                "type": unit.unit_type.value,
                **({"equipment": unit.equipment_type.value} if unit.equipment_type is not None else {}),
                "x": unit.position.x,
                "y": unit.position.y,
                "behavior": unit.behavior.value,
                "waypoint_index": unit.waypoint_index,
                "reason": unit.reason,
                "distance_travelled": unit.distance_travelled,
            } for unit in self.units],
        }
        if self.scene.rules is not None:
            snapshot.update(
                scores={team.value: self.scores[team] for team in Team},
                sharing_enabled=self.sharing_enabled,
                finish_reason=self.finish_reason,
                winner=self.winner,
                event_count=len(self.events),
            )
            for row, unit in zip(snapshot["units"], self.units):
                row.update(
                    sensor_range=unit.sensor_range,
                    contacts=[{
                        "target_id": contact.target_id,
                        "x": contact.position.x,
                        "y": contact.position.y,
                        "observed_step": contact.observed_step,
                        "source_id": contact.source_id,
                        "shared": contact.shared,
                    } for _, contact in sorted(unit.contacts.items())],
                    tagged_targets=sorted(unit.tagged_targets),
                    tag_count=unit.tag_count,
                    tag_flash_until_step=unit.tag_flash_until_step,
                )
        return snapshot

    def _record_snapshot(self) -> None:
        snapshot = self.snapshot()
        if not self.snapshots or snapshot != self.snapshots[-1]:
            self.snapshots.append(snapshot)

    @staticmethod
    def _outcome_key(snapshot: dict) -> tuple:
        """Frame-level scalars that must be dated exactly, whatever the stride."""
        scores = snapshot.get("scores")
        return (snapshot.get("state"),
                tuple(sorted(scores.items())) if isinstance(scores, dict) else None,
                snapshot.get("finish_reason"), snapshot.get("winner"),
                snapshot.get("event_count"))

    def _record_step_snapshot(self) -> None:
        """Record one logic step of a rules scene under the reduced-rate policy.

        ``RECORD_STRIDE`` samples the run, and any frame whose outcome scalars
        moved since the last recorded frame is kept regardless, so every scoring
        step and every event step survives. The recorded tail is therefore
        allowed to lag the current step mid-run; ``export_run`` appends the live
        frame, and control transitions still record unconditionally.
        """
        snapshot = self.snapshot()
        if (not self.record_every_step and self.step_count % RECORD_STRIDE
                and self.snapshots
                and self._outcome_key(snapshot) == self._outcome_key(self.snapshots[-1])):
            return
        if not self.snapshots or snapshot != self.snapshots[-1]:
            self.snapshots.append(snapshot)

    def _append_trail(self, unit: Unit) -> None:
        if not unit.trail or unit.position != unit.trail[-1]:
            unit.trail.append(unit.position)
            if len(unit.trail) > 300:
                del unit.trail[:-300]

    def _block_unit(self, unit: Unit, reason: str) -> None:
        unit.behavior = BehaviorState.BLOCKED
        unit.path = ()
        unit.path_index = 0
        unit.reason = reason
        self._append_trail(unit)
        self._event("path_blocked", reason, unit)

    def _stop_unit(self, unit: Unit, reason: str) -> None:
        unit.behavior = BehaviorState.STOPPED
        unit.target = None
        unit.path = ()
        unit.path_index = 0
        unit.reason = reason
        self._append_trail(unit)
        self._event("unit_stopped", reason, unit)

    def _complete_leg(self, unit: Unit) -> None:
        if unit.behavior is BehaviorState.RETURNING:
            self._stop_unit(unit, "Returned to the team return point")
        else:
            self._event(
                "waypoint_reached", f"Reached waypoint {unit.waypoint_index + 1}", unit
            )
            unit.waypoint_index += 1

    def _plan_next_leg(self, unit: Unit) -> None:
        """Plan a leg, consuming already-reached waypoints without recursion."""
        while unit.behavior not in (BehaviorState.STOPPED, BehaviorState.BLOCKED):
            if unit.waypoint_index < len(unit.waypoints):
                unit.behavior = BehaviorState.NAVIGATING
                goal = unit.waypoints[unit.waypoint_index]
            elif unit.return_home:
                if unit.behavior is not BehaviorState.RETURNING:
                    self._event("return_started", "All waypoints reached; returning home", unit)
                unit.behavior = BehaviorState.RETURNING
                goal = self.scene.return_points[unit.team]
            else:
                self._stop_unit(unit, "Preset route completed")
                return

            unit.target = goal
            if (
                isinstance(unit.speed, bool)
                or not isinstance(unit.speed, (int, float))
                or not math.isfinite(unit.speed)
                or unit.speed <= 0
            ):
                self._block_unit(unit, "Target is unreachable: navigation speed must be positive and finite")
                return
            if not point_clear(self.scene, unit.position, unit.unit_type):
                self._block_unit(unit, "Target is unreachable: starting point is blocked or outside the traversable world")
                return
            if not point_clear(self.scene, goal, unit.unit_type):
                self._block_unit(unit, "Target is unreachable: destination is blocked or outside the traversable world")
                return
            path = plan_path(self.scene, unit.position, goal, unit.unit_type)
            if path is None:
                self._block_unit(unit, "Target is unreachable: no collision-free route connects the two points")
                return
            unit.path = path
            unit.path_index = 0
            unit.reason = ""
            self._event("route_planned", f"Planned {len(path)} route segment(s)", unit)
            if path:
                return
            self._complete_leg(unit)

    def _finish_if_done(self) -> None:
        if self.scene.rules is not None:
            self._finish_round_if_done()
            return
        if self.has_missions and all(
            unit.behavior in (BehaviorState.IDLE, BehaviorState.STOPPED, BehaviorState.BLOCKED)
            for unit in self.units
        ):
            self.state = RunState.FINISHED
            self.metadata["ended_at"] = utc_timestamp()
            self._accumulator = 0.0
            self._event("run_finished", "All assigned missions have stopped or become blocked")
            self._record_snapshot()

    def _finish_round_if_done(self) -> None:
        rules = self.scene.rules
        if rules is None or self.finished:
            return
        if any(score >= rules.score_limit for score in self.scores.values()):
            reason = "score_limit"
        elif self.step_count >= duration_steps(rules.time_limit, self.scene.fixed_dt):
            reason = "time_limit"
        elif self.has_missions and all(
            unit.behavior in (BehaviorState.IDLE, BehaviorState.STOPPED, BehaviorState.BLOCKED)
            for unit in self.units
        ):
            reason = "missions_complete"
        else:
            return
        self.finish_reason = reason
        red, blue = self.scores[Team.RED], self.scores[Team.BLUE]
        self.winner = "red" if red > blue else "blue" if blue > red else "draw"
        for unit in self.units:
            if unit.behavior not in (BehaviorState.STOPPED, BehaviorState.BLOCKED):
                self._stop_unit(unit, f"Virtual round ended: {reason}")
        self.state = RunState.FINISHED
        self.metadata["ended_at"] = utc_timestamp()
        self._accumulator = 0.0
        self._event(
            "round_finished", f"Virtual round ended by {reason}; result: {self.winner}",
            finish_reason=reason, winner=self.winner,
        )
        self._record_snapshot()

    def _update_game(self) -> None:
        rules = self.scene.rules
        if rules is None:
            return
        by_id = {unit.id: unit for unit in self.units}
        for event in refresh_contacts(self.scene, self.units, self.step_count, self.sharing_enabled):
            self._event(event.kind, event.message, by_id[event.unit_id], **event.details)
        candidates = tag_candidates(
            self.units, self.step_count, self.scene.fixed_dt, rules, self._last_tag_steps
        )
        # Every referee decision precedes all mutations, including score-limit
        # checks. A same-step reply therefore counts even when it ties a round.
        accepted = adjudicate_tags(self.scene, self.units, candidates)
        for candidate in accepted:
            actor = by_id[candidate.actor_id]
            actor.tagged_targets.add(candidate.target_id)
            actor.tag_count += 1
            actor.tag_flash_until_step = self.step_count + duration_steps(0.5, self.scene.fixed_dt)
            self._last_tag_steps[actor.id] = self.step_count
            self.scores[actor.team] += 1
            self._event(
                "virtual_tag", f"Virtually tagged {candidate.target_id}; +1 point", actor,
                target_id=candidate.target_id,
            )
        self._finish_round_if_done()

    def _update_units(self) -> None:
        for unit in self.units:
            if unit.behavior not in (BehaviorState.NAVIGATING, BehaviorState.RETURNING):
                continue
            budget = unit.speed * self.scene.fixed_dt
            while budget > 0 and unit.behavior in (BehaviorState.NAVIGATING, BehaviorState.RETURNING):
                vertex = unit.path[unit.path_index]
                dx, dy = vertex.x - unit.position.x, vertex.y - unit.position.y
                distance = math.hypot(dx, dy)
                if distance <= budget:
                    position = vertex
                    travelled = distance
                else:
                    fraction = budget / distance
                    position = Point(unit.position.x + dx * fraction, unit.position.y + dy * fraction)
                    travelled = budget
                if not segment_clear(self.scene, unit.position, position, unit.unit_type):
                    self._block_unit(unit, "Target is unreachable: the next movement segment is obstructed")
                    break
                unit.position = position
                unit.distance_travelled += travelled
                budget = max(0.0, budget - travelled)
                if position == vertex:
                    # Preserve every corner even if a large step crosses several
                    # vertices. The trail remains a faithful collision-free route.
                    self._append_trail(unit)
                    unit.path_index += 1
                    if unit.path_index == len(unit.path):
                        self._complete_leg(unit)
                        if unit.behavior is not BehaviorState.STOPPED:
                            self._plan_next_leg(unit)
                else:
                    break
            if self.step_count % 10 == 0:
                self._append_trail(unit)
        if self.scene.rules is None:
            self._finish_if_done()

    def start(self) -> bool:
        if self.state is not RunState.READY:
            return False
        self.state = RunState.RUNNING
        self.metadata["started_at"] = utc_timestamp()
        self._event("run_started", "Simulation started")
        for unit in self.units:
            if unit.waypoints or unit.return_home:
                self._plan_next_leg(unit)
        self._finish_if_done()
        self._record_snapshot()
        return True

    def pause(self) -> bool:
        if self.state is not RunState.RUNNING:
            return False
        self.state = RunState.PAUSED
        self._event("paused", "Simulation paused")
        self._record_snapshot()
        return True

    def resume(self) -> bool:
        if self.state is not RunState.PAUSED:
            return False
        self.state = RunState.RUNNING
        self._event("resumed", "Simulation resumed")
        self._record_snapshot()
        return True

    def set_sharing(self, enabled: bool) -> bool:
        """Change team broadcasts; disabling immediately removes shared memory."""
        if not isinstance(enabled, bool):
            raise ValueError("sharing must be a boolean")
        if self.scene.rules is None or self.finished or enabled == self.sharing_enabled:
            return False
        self.sharing_enabled = enabled
        if not enabled:
            for unit in sorted(self.units, key=lambda unit: unit.id):
                for target_id, contact in sorted(tuple(unit.contacts.items())):
                    if contact.shared:
                        del unit.contacts[target_id]
                        self._event(
                            "contact_lost", f"Removed shared observation of {target_id}", unit,
                            target_id=target_id, source_id=contact.source_id,
                        )
        self._event("sharing_changed", f"Team sharing {'enabled' if enabled else 'disabled'}", enabled=enabled)
        self._record_snapshot()
        return True

    def reset(self) -> bool:
        initial_units = self._initial_units()
        changed = (
            self.state is not RunState.READY
            or self.step_count != 0
            or self._accumulator != 0.0
            or self.units != initial_units
            or self.sharing_enabled != (self.scene.rules.sharing_enabled if self.scene.rules is not None else False)
            or any(self.scores.values()) or self.finish_reason != "" or self.winner != ""
        )
        self.state = RunState.READY
        self.step_count = 0
        self._accumulator = 0.0
        self.units = initial_units
        self._render_step = None
        self.events.clear()
        self.snapshots.clear()
        self._last_snapshot_second = 0
        self.sharing_enabled = self.scene.rules.sharing_enabled if self.scene.rules is not None else False
        self.scores = {team: 0 for team in Team}
        self.finish_reason = ""
        self.winner = ""
        self._last_tag_steps.clear()
        if changed:
            self.metadata = new_run_metadata()
            self._last_export_signature = None
        self._record_snapshot()
        return changed

    def advance(self, real_dt: float) -> int:
        """Consume active real time and return the actual completed logic steps.

        Calls in READY, PAUSED and FINISHED discard elapsed time. A fractional
        step earned before pausing remains available after resuming. Every unit
        uses the fixed step duration regardless of how rendering divides frames.
        """
        if isinstance(real_dt, bool) or not isinstance(real_dt, (int, float)):
            raise ValueError("real_dt must be a finite, non-negative number")
        try:
            elapsed = float(real_dt)
        except (OverflowError, ValueError) as error:
            raise ValueError("real_dt must be a finite, non-negative number") from error
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("real_dt must be a finite, non-negative number")
        if self.state is not RunState.RUNNING:
            return 0

        fixed_dt = self.scene.fixed_dt
        accumulated = math.fsum((self._accumulator, elapsed))
        # Correct round-off near a single-step boundary without changing time.
        tolerance = min(fixed_dt * 1e-9, max(fixed_dt * 1e-12, 4 * math.ulp(accumulated)))
        steps = math.floor((accumulated + tolerance) / fixed_dt)
        self._accumulator = max(0.0, accumulated - steps * fixed_dt)
        completed = 0
        for _ in range(steps):
            # Retain only the last fixed-step endpoints for presentation.
            # Trails are sampled every ten steps and cannot serve as animation
            # endpoints. This transient cache is excluded from all recordings.
            render_before = {unit.id: (id(unit), unit.position) for unit in self.units}
            self.step_count += 1
            completed += 1
            self._update_units()
            if self.scene.rules is not None:
                self._update_game()
                self._record_step_snapshot()
            second = math.floor(self.sim_time + fixed_dt * 1e-9)
            if self.scene.rules is None and second > self._last_snapshot_second:
                self._last_snapshot_second = second
                self._record_snapshot()
            self._render_step = (self.step_count, {
                unit.id: (*render_before[unit.id], unit.position)
                for unit in self.units if unit.id in render_before
            })
            if self.finished:
                break
        return completed
