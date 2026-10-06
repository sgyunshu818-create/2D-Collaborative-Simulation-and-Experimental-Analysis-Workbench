"""Independent, undoable JSON drafts for the scene editor.

The immutable runtime Scene is created only by the existing scene validator.
Draft edits may temporarily be invalid; validation and saving are explicit.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .equipment import equipment_profile
from .models import EquipmentType, Scene, UnitType
from .scene import SceneConfigError, scene_from_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
USER_SCENE_ROOT = CONFIG_ROOT / "user_scenes"
TEMPLATES = {
    "基础场景": CONFIG_ROOT / "basic_scene.json",
    "障碍场景": CONFIG_ROOT / "obstacle_scene.json",
    "共享场景": CONFIG_ROOT / "sharing_scene.json",
    "地理场景": CONFIG_ROOT / "geographic_scene.json",
    "大型精细场景": CONFIG_ROOT / "detailed_scene.json",
}


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def blank_scene() -> dict[str, Any]:
    """A complete editable scene; no runtime objects are shared with it."""
    return {
        "name": "自定义场景",
        "world": {"width": 900, "height": 600},
        "fixed_dt": 1 / 60,
        "spawn_points": {"red": {"x": 80, "y": 120}, "blue": {"x": 800, "y": 460}},
        "return_points": {"red": {"x": 80, "y": 120}, "blue": {"x": 800, "y": 460}},
        "units": [],
        "obstacles": [],
    }


class SceneDocument:
    """A mutable draft behind defensive snapshots and atomic, validated saves.

    ``data`` returns a deep copy. Use ``replace`` or the edit methods to change
    the draft. ``source_path`` records provenance; ``path`` is the saved user
    copy and is never initialized to a supplied template. Failed writes leave
    both the old file and the current draft available for recovery.
    """

    def __init__(self, payload: dict[str, Any] | None = None, *, source_path: str | Path | None = None,
                 user_root: str | Path | None = None, history_limit: int = 100) -> None:
        self.user_root = Path(user_root) if user_root is not None else USER_SCENE_ROOT
        self.history_limit = max(1, history_limit)
        self._data = copy.deepcopy(payload if payload is not None else blank_scene())
        self._clean = copy.deepcopy(self._data)
        self._undo: list[dict[str, Any]] = []
        self._redo: list[dict[str, Any]] = []
        self._source_path = Path(source_path) if source_path is not None else None
        self._path = None

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def dirty(self) -> bool:
        return self._data != self._clean

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def load(self, payload: dict[str, Any], *, source_path: str | Path | None = None) -> None:
        """Replace the complete document after validation, resetting history."""
        candidate = copy.deepcopy(payload)
        scene_from_data(candidate, source_path or "<draft>")
        self._data = candidate
        self._clean = copy.deepcopy(candidate)
        self._undo.clear()
        self._redo.clear()
        self._source_path = Path(source_path) if source_path is not None else None
        self._path = None

    @classmethod
    def open(cls, path: str | Path, **kwargs: Any) -> SceneDocument:
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise SceneConfigError(path, "$", f"invalid JSON at line {error.lineno}, column {error.colno}") from error
        except (OSError, UnicodeError, RecursionError) as error:
            raise SceneConfigError(path, "$", f"cannot read configuration: {error}") from error
        scene_from_data(payload, path)
        return cls(payload, source_path=path, **kwargs)

    def replace(self, payload: dict[str, Any]) -> bool:
        """Commit one undoable draft edit without forcing premature validity."""
        candidate = copy.deepcopy(payload)
        if candidate == self._data:
            return False
        self._undo.append(copy.deepcopy(self._data))
        del self._undo[:-self.history_limit]
        self._redo.clear()
        self._data = candidate
        return True

    def edit(self, change: Callable[[dict[str, Any]], None]) -> bool:
        candidate = self.data
        change(candidate)
        return self.replace(candidate)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(copy.deepcopy(self._data))
        self._data = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(copy.deepcopy(self._data))
        self._data = self._redo.pop()
        return True

    def validate(self) -> Scene:
        return scene_from_data(self._data, self._path or self._source_path or "<draft>")

    def unique_id(self, prefix: str) -> str:
        used = {item.get("id") for key in ("units", "obstacles") for item in self._data[key]}
        number = 1
        while f"{prefix}_{number:02d}" in used:
            number += 1
        return f"{prefix}_{number:02d}"

    def add_unit(self, *, team: str = "red", unit_type: str | None = None, x: float | None = None,
                 y: float | None = None, equipment: EquipmentType | str | None = None) -> int:
        profile = equipment_profile(equipment if equipment is not None else unit_type or "ground")
        movement = UnitType(unit_type) if unit_type is not None else profile.unit_type
        if equipment is not None and profile.unit_type is not movement:
            raise ValueError(f"equipment {profile.key!r} requires type {profile.unit_type.value!r}")
        unit_type = movement.value
        position = self._data["spawn_points"][team]
        prefix = profile.key if equipment is not None else unit_type
        unit = {"id": self.unique_id(f"{team}_{prefix}"), "team": team, "type": unit_type,
                "x": position["x"] if x is None else x, "y": position["y"] if y is None else y,
                "speed": profile.default_speed, "sensor_range": profile.default_sensor_range,
                "return_home": False, "waypoints": []}
        if equipment is not None:
            unit["equipment"] = profile.key
        index = len(self._data["units"])
        self.edit(lambda data: data["units"].append(unit))
        return index

    def add_obstacle(self, *, x: float | None = None, y: float | None = None) -> int:
        world = self._data["world"]
        width, height = world["width"], world["height"]
        obstacle = {"id": self.unique_id("obstacle"), "x": width * .4 if x is None else x,
                    "y": height * .4 if y is None else y, "width": width * .1, "height": height * .1}
        index = len(self._data["obstacles"])
        self.edit(lambda data: data["obstacles"].append(obstacle))
        return index

    def update_item(self, collection: str, index: int, **values: Any) -> None:
        if collection not in ("units", "obstacles"):
            raise ValueError("collection must be units or obstacles")
        self.edit(lambda data: data[collection][index].update(copy.deepcopy(values)))

    def delete_item(self, collection: str, index: int) -> None:
        if collection not in ("units", "obstacles"):
            raise ValueError("collection must be units or obstacles")
        self.edit(lambda data: data[collection].pop(index))

    def add_waypoint(self, unit_index: int, x: float, y: float) -> int:
        index = len(self._data["units"][unit_index].get("waypoints", []))
        self.edit(lambda data: data["units"][unit_index].setdefault("waypoints", []).append({"x": x, "y": y}))
        return index

    def update_waypoint(self, unit_index: int, waypoint_index: int, x: float, y: float) -> None:
        self.edit(lambda data: data["units"][unit_index]["waypoints"][waypoint_index].update(x=x, y=y))

    def delete_waypoint(self, unit_index: int, waypoint_index: int) -> None:
        self.edit(lambda data: data["units"][unit_index]["waypoints"].pop(waypoint_index))

    def move_waypoint(self, unit_index: int, waypoint_index: int, offset: int) -> int:
        points = self._data["units"][unit_index].get("waypoints", [])
        new_index = waypoint_index + offset
        if not 0 <= new_index < len(points):
            return waypoint_index
        def move(data: dict[str, Any]) -> None:
            values = data["units"][unit_index]["waypoints"]
            values.insert(new_index, values.pop(waypoint_index))
        self.edit(move)
        return new_index

    def suggested_path(self) -> Path:
        name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", str(self._data.get("name", "scene"))).strip(" .")
        name = (name or "scene")[:80]
        # Prefixing also avoids Windows reserved device filenames.
        stem = f"scene_{name}_{datetime.now():%Y%m%d_%H%M%S}"
        candidate = self.user_root / f"{stem}.json"
        suffix = 1
        while candidate.exists():
            candidate = self.user_root / f"{stem}_{suffix}.json"
            suffix += 1
        return candidate

    def save(self, path: str | Path | None = None) -> Path:
        """Validate, serialize, and atomically replace only a user-owned file."""
        self.validate()
        target = Path(path) if path is not None else self._path or self.suggested_path()
        target = target.resolve()
        if _within(target, CONFIG_ROOT) and not _within(target, USER_SCENE_ROOT):
            raise SceneConfigError(target, "$", "templates are read-only; save a copy in configs/user_scenes")
        try:
            encoded = json.dumps(self._data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError, RecursionError) as error:
            raise SceneConfigError(target, "$", f"draft cannot be saved as JSON: {error}") from error
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=target.parent, prefix=f".{target.name}.",
                                             suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._path = target
        self._clean = copy.deepcopy(self._data)
        return target
