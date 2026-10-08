"""Export scene, events and state snapshots with the Python standard library."""

from __future__ import annotations

import csv
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from .geography import scene_world

if TYPE_CHECKING:
    from .simulation import Simulation

APP_VERSION = "2026.10-workbench"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_run_metadata() -> dict:
    """Capture the installed code/environment once, before this run starts."""
    source = hashlib.sha256()
    source_files = []
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        relative = path.relative_to(Path(__file__).parent).as_posix()
        source.update(relative.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        source_files.append(relative)
    try:
        pygame_version = importlib.metadata.version("pygame")
    except importlib.metadata.PackageNotFoundError:
        pygame_version = None
    return {
        "schema_version": 1, "app_version": APP_VERSION,
        "run_id": uuid.uuid4().hex, "created_at": utc_timestamp(),
        "started_at": None, "ended_at": None, "saved_at": None,
        "source_sha256": source.hexdigest(), "source_files": source_files,
        "dependencies": {"python": platform.python_version(), "pygame": pygame_version},
        "environment": {"platform": platform.platform(), "implementation": platform.python_implementation(),
                        "executable": sys.executable},
    }


def _export_metadata(sim, scene_data: dict, current: dict) -> dict:
    metadata = copy.deepcopy(getattr(sim, "metadata", {}))
    canonical = json.dumps(scene_data, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    metadata["config_sha256"] = hashlib.sha256(canonical).hexdigest()
    signature = (current["step"], current["state"], len(sim.events))
    if metadata.get("saved_at") is None or getattr(sim, "_last_export_signature", None) != signature:
        metadata["saved_at"] = utc_timestamp()
    metadata["save_status"] = ("completed" if current["state"] == "FINISHED" else
                               "not_started" if current["state"] == "READY" else "interrupted")
    return metadata


def _scene_data(sim: Simulation) -> dict:
    scene = sim.scene
    data = {
        "name": scene.name,
        "world": scene_world(scene),
        "fixed_dt": scene.fixed_dt,
        "spawn_points": {team.value: {"x": p.x, "y": p.y}
                         for team, p in scene.spawn_points.items()},
        "return_points": {team.value: {"x": p.x, "y": p.y}
                          for team, p in scene.return_points.items()},
        "obstacles": [
            {"id": o.id, "x": o.x, "y": o.y, "width": o.width, "height": o.height}
            for o in scene.obstacles
        ],
        "units": [
            {"id": u.id, "team": u.team.value, "type": u.unit_type.value,
             **({"equipment": u.equipment_type.value} if u.equipment_type is not None else {}),
             "x": u.position.x, "y": u.position.y,
             **({"speed": u.speed, "waypoints": [{"x": p.x, "y": p.y} for p in u.waypoints],
                 "return_home": u.return_home} if u.speed > 0 else {}),
             **({"sensor_range": u.sensor_range}
                if getattr(scene, "rules", None) or u.sensor_range != (160.0 if u.unit_type.value == "ground" else 240.0)
                else {})}
            for u in scene.units
        ],
    }
    if getattr(scene, "rules", None) is not None:
        data["rules"] = asdict(scene.rules)
    return data


def _unit_state(unit) -> dict:
    return {
        "id": unit.id, "team": unit.team.value, "type": unit.unit_type.value,
        **({"equipment": unit.equipment_type.value} if unit.equipment_type is not None else {}),
        "x": unit.position.x, "y": unit.position.y, "behavior": unit.behavior.value,
        "waypoint_index": unit.waypoint_index, "reason": unit.reason,
        "distance_travelled": unit.distance_travelled,
    }


def export_run(sim: Simulation, directory: str | Path,
               source_scene: str | Path | None = None) -> dict[str, Path]:
    """Save the current run, including incomplete or blocked runs, without mutating it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current = sim.snapshot() if hasattr(sim, "snapshot") else {
        "step": sim.step_count, "time": sim.sim_time, "state": sim.state.value,
        "units": [_unit_state(u) for u in sim.units]}
    snapshots = copy.deepcopy(sim.snapshots)
    if not snapshots or snapshots[-1] != current:
        snapshots.append(current)
    payload = {
        "format_version": 3 if getattr(sim.scene, "rules", None) is not None else 2,
        "source_scene": str(Path(source_scene).resolve()) if source_scene else None,
        "scene": _scene_data(sim),
        "result": current,
        "events": copy.deepcopy(sim.events),
        "snapshots": snapshots,
    }
    payload["metadata"] = _export_metadata(sim, payload["scene"], current)
    payload["metadata"]["recording_format_version"] = payload["format_version"]
    paths = {"run": directory / "run.json", "events": directory / "events.csv",
             "states": directory / "states.csv"}
    paths["run"].write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                            encoding="utf-8")
    with paths["events"].open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["step", "time", "kind", "unit_id", "message"]
        if payload["format_version"] == 3:
            fields += ["details"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for event in payload["events"]:
            row = dict(event)
            if payload["format_version"] == 3:
                row["details"] = json.dumps({k: v for k, v in event.items()
                                             if k not in ("step", "time", "kind", "unit_id", "message")},
                                            ensure_ascii=False, allow_nan=False, sort_keys=True)
            writer.writerow(row)
    with paths["states"].open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["step", "time", "state", "id", "team", "type", "x", "y", "behavior",
                  "waypoint_index", "reason", "distance_travelled"]
        if any("equipment" in unit for snapshot in snapshots for unit in snapshot["units"]):
            fields.insert(fields.index("type") + 1, "equipment")
        if payload["format_version"] == 3:
            fields += ["score_red", "score_blue", "sharing_enabled", "finish_reason", "winner",
                       "event_count", "sensor_range", "contact_count", "shared_count",
                       "tagged_targets_count", "tag_count", "tag_flash_until_step"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for snapshot in snapshots:
            for unit in snapshot["units"]:
                row = {"step": snapshot["step"], "time": snapshot["time"],
                       "state": snapshot["state"], **unit}
                if payload["format_version"] == 3:
                    row.update(score_red=snapshot["scores"]["red"], score_blue=snapshot["scores"]["blue"],
                               sharing_enabled=snapshot["sharing_enabled"],
                               finish_reason=snapshot["finish_reason"], winner=snapshot["winner"],
                               event_count=snapshot.get("event_count", ""))
                    # Counts, not the contact rows themselves: a spreadsheet cell
                    # holding a nested JSON blob is unreadable, and the full
                    # detail already lives in run.json next to this file.
                    row["contact_count"] = len(unit["contacts"])
                    row["shared_count"] = sum(contact["shared"] is True for contact in unit["contacts"])
                    row["tagged_targets_count"] = len(unit["tagged_targets"])
                writer.writerow(row)
    if hasattr(sim, "metadata"):
        sim.metadata = copy.deepcopy(payload["metadata"])
        sim._last_export_signature = (current["step"], current["state"], len(sim.events))
    return paths
