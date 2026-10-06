"""Observation, team knowledge and refereeing for a fictional tag game.

Only ``observe_opponents`` reads opponents' real positions for sensing. Tag
selection consumes immutable contacts; ``adjudicate_tags`` independently checks
the real geometry as a game referee. No weapons or damage are simulated.
"""

from dataclasses import dataclass, field
from math import ceil, hypot, isfinite
from typing import Mapping, Sequence

from .models import Contact, GameRules, Point, Scene, Unit, UnitType


@dataclass(frozen=True)
class ContactEvent:
    kind: str
    unit_id: str
    message: str
    details: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TagCandidate:
    actor_id: str
    target_id: str


def duration_steps(seconds: float, fixed_dt: float) -> int:
    """Round a positive duration up to the next complete logic step."""
    ratio = seconds / fixed_dt
    if isfinite(ratio):
        return max(1, ceil(ratio - 1e-12))
    # Both inputs can be valid finite numbers even when their quotient exceeds
    # float range. Integer ratios avoid an otherwise surprising overflow.
    numerator, denominator = float(seconds).as_integer_ratio()
    dt_numerator, dt_denominator = float(fixed_dt).as_integer_ratio()
    numerator *= dt_denominator
    denominator *= dt_numerator
    return max(1, (numerator + denominator - 1) // denominator)


def _intersects_obstacle(scene: Scene, start: Point, end: Point) -> bool:
    # Unlike movement clearance, observation uses the original rectangles.
    # Closed edges count as occlusion, including a tangent along a wall.
    for obstacle in scene.obstacles:
        entry, exit_ = 0.0, 1.0
        for origin, delta, lower, upper in (
            (start.x, end.x - start.x, obstacle.x, obstacle.x + obstacle.width),
            (start.y, end.y - start.y, obstacle.y, obstacle.y + obstacle.height),
        ):
            if delta == 0:
                if origin < lower or origin > upper:
                    break
                continue
            near, far = (lower - origin) / delta, (upper - origin) / delta
            if near > far:
                near, far = far, near
            entry, exit_ = max(entry, near), min(exit_, far)
            if entry > exit_:
                break
        else:
            return True
    return False


def line_of_sight(scene: Scene, start: Point, end: Point, observer_type: UnitType) -> bool:
    """Aircraft use the fixed-height simplification; ground lines may occlude."""
    return observer_type is UnitType.AIR or not _intersects_obstacle(scene, start, end)


def observe_opponents(scene: Scene, observer: Unit, units: Sequence[Unit], step: int) -> dict[str, Contact]:
    """Return only directly visible opponents, as separate immutable records."""
    observed = {}
    for target in sorted(units, key=lambda unit: unit.id):
        if target.team is observer.team:
            continue
        distance = hypot(target.position.x - observer.position.x, target.position.y - observer.position.y)
        if distance <= observer.sensor_range and line_of_sight(
            scene, observer.position, target.position, observer.unit_type
        ):
            observed[target.id] = Contact(
                target.id, Point(target.position.x, target.position.y), step, observer.id
            )
    return observed


def refresh_contacts(scene: Scene, units: Sequence[Unit], step: int, sharing_enabled: bool) -> list[ContactEvent]:
    """Refresh direct observations, broadcast those observations once, then age.

    The broadcast source is the direct-observation table constructed for this
    step. Remembered or previously shared contacts never enter that table, so a
    relay cannot keep a lost target alive. Direct observations have priority.
    """
    if scene.rules is None:
        return []
    ordered = sorted(units, key=lambda unit: unit.id)
    direct = {unit.id: observe_opponents(scene, unit, ordered, step) for unit in ordered}
    ttl_steps = duration_steps(scene.rules.contact_ttl, scene.fixed_dt)
    events = []
    for receiver in ordered:
        previous = receiver.contacts
        refreshed = dict(direct[receiver.id])
        for target_id, contact in refreshed.items():
            old = previous.get(target_id)
            if old is None or old.shared:
                events.append(ContactEvent(
                    "object_discovered", receiver.id, f"Directly observed {target_id}",
                    {"target_id": target_id, "source_id": receiver.id},
                ))
        if sharing_enabled:
            for sender in ordered:
                if sender.id == receiver.id or sender.team is not receiver.team:
                    continue
                for target_id, contact in direct[sender.id].items():
                    # Sorted senders provide a stable source when several see
                    # the same target. A receiver's own direct sight wins.
                    if target_id in refreshed:
                        continue
                    shared = Contact(
                        target_id, Point(contact.position.x, contact.position.y),
                        contact.observed_step, contact.source_id, shared=True,
                    )
                    refreshed[target_id] = shared
                    old = previous.get(target_id)
                    if old is None or not old.shared or old.source_id != shared.source_id:
                        events.append(ContactEvent(
                            "info_shared", receiver.id,
                            f"Received observation of {target_id} from {sender.id}",
                            {"target_id": target_id, "source_id": sender.id},
                        ))
        for target_id, old in sorted(previous.items()):
            if target_id in refreshed:
                continue
            if step - old.observed_step >= ttl_steps or (old.shared and not sharing_enabled):
                events.append(ContactEvent(
                    "contact_lost", receiver.id, f"Observation of {target_id} expired",
                    {"target_id": target_id, "source_id": old.source_id},
                ))
            else:
                refreshed[target_id] = old
        receiver.contacts = refreshed
    return events


def tag_candidates(
    units: Sequence[Unit], step: int, fixed_dt: float, rules: GameRules,
    last_tag_steps: Mapping[str, int],
) -> tuple[TagCandidate, ...]:
    """Choose at most one eligible remembered target per actor for this step.

    This function deliberately cannot access a scene or resolve a target ID
    against actual units. The nearest remembered point is a simple fixed rule.
    """
    cooldown = duration_steps(rules.tag_cooldown, fixed_dt)
    candidates = []
    for actor in sorted(units, key=lambda unit: unit.id):
        last_step = last_tag_steps.get(actor.id)
        if last_step is not None and step - last_step < cooldown:
            continue
        eligible = []
        for contact in actor.contacts.values():
            if contact.target_id in actor.tagged_targets:
                continue
            distance = hypot(contact.position.x - actor.position.x, contact.position.y - actor.position.y)
            if distance <= rules.tag_range:
                eligible.append((distance, contact.target_id))
        if eligible:
            candidates.append(TagCandidate(actor.id, min(eligible)[1]))
    return tuple(candidates)


def adjudicate_tags(scene: Scene, units: Sequence[Unit], candidates: Sequence[TagCandidate]) -> tuple[TagCandidate, ...]:
    """Check all candidates against one unchanged world before score updates."""
    if scene.rules is None:
        return ()
    by_id = {unit.id: unit for unit in units}
    valid = []
    seen_actors = set()
    for candidate in candidates:
        actor, target = by_id.get(candidate.actor_id), by_id.get(candidate.target_id)
        if actor is None or target is None or actor.team is target.team:
            continue
        if actor.id in seen_actors or target.id in actor.tagged_targets:
            continue
        distance = hypot(actor.position.x - target.position.x, actor.position.y - target.position.y)
        if distance <= scene.rules.tag_range and line_of_sight(
            scene, actor.position, target.position, actor.unit_type
        ):
            valid.append(candidate)
            seen_actors.add(actor.id)
    return tuple(valid)
