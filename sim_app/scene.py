"""Load a JSON scene with errors tied to its filename and field path."""

import json
import math
from pathlib import Path
from typing import Any

from .models import EquipmentType, GameRules, Obstacle, Point, Scene, Team, UnitSpec, UnitType
from .geography import GeoReference


class SceneConfigError(ValueError):
    """A readable scene error suitable for the command-line entry point."""

    def __init__(self, path: str | Path, field: str, message: str) -> None:
        self.path = Path(path)
        self.field = field
        self.message = message
        super().__init__(f"{self.path}: {field}: {message}")


class _Validator:
    def __init__(self, path: Path) -> None:
        self.path = path

    def fail(self, field: str, message: str) -> None:
        raise SceneConfigError(self.path, field, message)

    def object(self, value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.fail(field, "must be a JSON object")
        return value

    def array(self, value: Any, field: str) -> list[Any]:
        if not isinstance(value, list):
            self.fail(field, "must be a JSON array")
        return value

    def required(self, obj: dict[str, Any], key: str, field: str) -> Any:
        if key not in obj:
            self.fail(field, "required field is missing")
        return obj[key]

    def string(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            self.fail(field, "must be a non-empty string")
        return value

    def boolean(self, value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            self.fail(field, "must be a JSON boolean")
        return value

    def positive_integer(self, value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            self.fail(field, "must be a positive JSON integer (booleans are not accepted)")
        return value

    def number(self, value: Any, field: str, *, positive: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(field, "must be a finite number (booleans are not accepted)")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            self.fail(field, "must be a finite number")
        if not math.isfinite(number):
            self.fail(field, "must be a finite number")
        if positive and number <= 0:
            self.fail(field, "must be greater than zero")
        return number

    def point(self, obj: Any, field: str, width: float, height: float) -> Point:
        obj = self.object(obj, field)
        x = self.number(self.required(obj, "x", f"{field}.x"), f"{field}.x")
        y = self.number(self.required(obj, "y", f"{field}.y"), f"{field}.y")
        if not 0 <= x < width:
            self.fail(f"{field}.x", f"must satisfy 0 <= x < {width:g}")
        if not 0 <= y < height:
            self.fail(f"{field}.y", f"must satisfy 0 <= y < {height:g}")
        return Point(x, y)

    def team_points(
        self, obj: Any, field: str, width: float, height: float
    ) -> dict[Team, Point]:
        obj = self.object(obj, field)
        for key in obj:
            if key not in {team.value for team in Team}:
                self.fail(f"{field}.{key}", "unknown team; expected 'red' or 'blue'")
        return {
            team: self.point(
                self.required(obj, team.value, f"{field}.{team.value}"),
                f"{field}.{team.value}",
                width,
                height,
            )
            for team in Team
        }


def load_scene(path: str | Path) -> Scene:
    """Read a scene without creating a window or changing the working directory."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig") as source:
            data = json.load(source)
    except json.JSONDecodeError as error:
        raise SceneConfigError(
            path, "$", f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    except RecursionError as error:
        raise SceneConfigError(path, "$", "JSON nesting exceeds the supported depth") from error
    except (OSError, UnicodeError) as error:
        raise SceneConfigError(path, "$", f"cannot read configuration: {error}") from error

    return scene_from_data(data, path)


def scene_from_data(data: Any, path: str | Path = "<memory>") -> Scene:
    """Validate a decoded scene, also used by records without temporary files."""
    path = Path(path)
    validator = _Validator(path)

    data = validator.object(data, "$")
    name = validator.string(validator.required(data, "name", "name"), "name")
    world = validator.object(validator.required(data, "world", "world"), "world")
    width = validator.number(validator.required(world, "width", "world.width"), "world.width", positive=True)
    height = validator.number(validator.required(world, "height", "world.height"), "world.height", positive=True)
    terrain = None
    if "terrain" in world:
        terrain = validator.string(world["terrain"], "world.terrain")
        if terrain not in ("detailed_virtual", "astra_mountain", "astra_atlas"):
            validator.fail("world.terrain", "unknown terrain; expected 'detailed_virtual', 'astra_mountain' or 'astra_atlas'")
    reference = None
    if 'georeference' in world:
        raw = validator.object(world['georeference'], 'world.georeference')
        for key in raw:
            if key not in ('center_latitude', 'center_longitude', 'meters_per_unit'):
                validator.fail(f'world.georeference.{key}', 'unknown geographic reference field')
        values = {key: validator.number(validator.required(raw, key, f'world.georeference.{key}'),
                                       f'world.georeference.{key}')
                  for key in ('center_latitude', 'center_longitude', 'meters_per_unit')}
        try:
            reference = GeoReference(**values)
            reference.validate_extent(width, height)
        except ValueError as error:
            validator.fail('world.georeference', str(error))
    fixed_dt = validator.number(validator.required(data, "fixed_dt", "fixed_dt"), "fixed_dt", positive=True)
    spawn_points = validator.team_points(
        validator.required(data, "spawn_points", "spawn_points"), "spawn_points", width, height
    )
    return_points = validator.team_points(
        validator.required(data, "return_points", "return_points"), "return_points", width, height
    )

    rules = None
    if "rules" in data:
        rule_data = validator.object(data["rules"], "rules")
        defaults = GameRules()
        known_fields = set(GameRules.__dataclass_fields__)
        for key in rule_data:
            if key not in known_fields:
                validator.fail(f"rules.{key}", "unknown virtual game rule")
        rule_values = {
            "sharing_enabled": validator.boolean(
                rule_data.get("sharing_enabled", defaults.sharing_enabled), "rules.sharing_enabled"
            ),
            "score_limit": validator.positive_integer(
                rule_data.get("score_limit", defaults.score_limit), "rules.score_limit"
            ),
        }
        for key in ("tag_range", "tag_cooldown", "time_limit", "contact_ttl"):
            rule_values[key] = validator.number(
                rule_data.get(key, getattr(defaults, key)), f"rules.{key}", positive=True
            )
        # Zero means "no limit", so this one accepts zero but never a negative reach.
        range_value = validator.number(
            rule_data.get("sharing_range", defaults.sharing_range), "rules.sharing_range"
        )
        if range_value < 0:
            validator.fail("rules.sharing_range", "must not be negative")
        rule_values["sharing_range"] = range_value
        rules = GameRules(**rule_values)

    identifiers: set[str] = set()

    def identifier(obj: dict[str, Any], field: str) -> str:
        value = validator.string(validator.required(obj, "id", f"{field}.id"), f"{field}.id")
        if value in identifiers:
            validator.fail(f"{field}.id", f"duplicate identifier {value!r}")
        identifiers.add(value)
        return value

    obstacles = []
    for index, item in enumerate(validator.array(validator.required(data, "obstacles", "obstacles"), "obstacles")):
        field = f"obstacles[{index}]"
        item = validator.object(item, field)
        obstacle_id = identifier(item, field)
        position = validator.point(item, field, width, height)
        obstacle_width = validator.number(
            validator.required(item, "width", f"{field}.width"), f"{field}.width", positive=True
        )
        obstacle_height = validator.number(
            validator.required(item, "height", f"{field}.height"), f"{field}.height", positive=True
        )
        if position.x + obstacle_width > width:
            validator.fail(f"{field}.width", "rectangle extends beyond the world right boundary")
        if position.y + obstacle_height > height:
            validator.fail(f"{field}.height", "rectangle extends beyond the world bottom boundary")
        obstacles.append(Obstacle(obstacle_id, position.x, position.y, obstacle_width, obstacle_height))

    units = []
    for index, item in enumerate(validator.array(validator.required(data, "units", "units"), "units")):
        field = f"units[{index}]"
        item = validator.object(item, field)
        unit_id = identifier(item, field)
        team_value = validator.required(item, "team", f"{field}.team")
        type_value = validator.required(item, "type", f"{field}.type")
        try:
            team = Team(team_value)
        except (ValueError, TypeError):
            validator.fail(f"{field}.team", "must be 'red' or 'blue'")
        try:
            unit_type = UnitType(type_value)
        except (ValueError, TypeError):
            validator.fail(f"{field}.type", "must be 'ground' or 'air'")
        equipment_type = None
        if "equipment" in item:
            try:
                equipment_type = EquipmentType(item["equipment"])
            except (ValueError, TypeError):
                validator.fail(f"{field}.equipment", "must be 'drone', 'airplane', 'armored_car' or 'tank'")
            if equipment_type.unit_type is not unit_type:
                validator.fail(f"{field}.equipment",
                               f"{equipment_type.value!r} requires type {equipment_type.unit_type.value!r}")
        position = validator.point(item, field, width, height)
        waypoints = tuple(
            validator.point(point, f"{field}.waypoints[{point_index}]", width, height)
            for point_index, point in enumerate(
                validator.array(item.get("waypoints", []), f"{field}.waypoints")
            )
        )
        return_home = validator.boolean(item.get("return_home", False), f"{field}.return_home")
        if "speed" in item:
            speed = validator.number(item["speed"], f"{field}.speed", positive=True)
        elif waypoints or return_home:
            speed = 90.0 if unit_type is UnitType.GROUND else 140.0
        else:
            speed = 0.0
        sensor_range = validator.number(
            item.get("sensor_range", 160.0 if unit_type is UnitType.GROUND else 240.0),
            f"{field}.sensor_range", positive=True,
        )
        units.append(UnitSpec(unit_id, team, unit_type, position, speed, waypoints, return_home,
                              sensor_range, equipment_type))

    if rules is not None and {unit.team for unit in units} != set(Team):
        validator.fail("units", "a virtual game requires at least one unit from each team")
    return Scene(name, width, height, fixed_dt, spawn_points, return_points, tuple(obstacles),
                 tuple(units), rules, reference, terrain)
