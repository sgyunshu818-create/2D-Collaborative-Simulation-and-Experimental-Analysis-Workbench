"""Data shared by the scene loader, simulation clock and renderer."""

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping
from .geography import GeoReference


class Team(str, Enum):
    RED = "red"
    BLUE = "blue"


class UnitType(str, Enum):
    GROUND = "ground"
    AIR = "air"


class EquipmentType(str, Enum):
    """Displayable equipment families using the existing movement categories."""

    DRONE = "drone"
    AIRPLANE = "airplane"
    ARMORED_CAR = "armored_car"
    TANK = "tank"

    @property
    def unit_type(self) -> UnitType:
        return UnitType.AIR if self in (self.DRONE, self.AIRPLANE) else UnitType.GROUND


def _check_equipment(equipment_type, unit_type):
    if equipment_type is None:
        return None
    equipment_type = EquipmentType(equipment_type)
    if equipment_type.unit_type != unit_type:
        raise ValueError(f"equipment_type {equipment_type.value!r} requires type {equipment_type.unit_type.value!r}")
    return equipment_type


class BehaviorState(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    RETURNING = "RETURNING"
    STOPPED = "STOPPED"
    BLOCKED = "BLOCKED"


class RunState(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Obstacle:
    id: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class GameRules:
    """Rules for the entirely virtual observation-and-tag classroom game."""

    sharing_enabled: bool = True
    tag_range: float = 90.0
    tag_cooldown: float = 1.0
    score_limit: int = 3
    time_limit: float = 45.0
    contact_ttl: float = 2.0


@dataclass(frozen=True)
class Contact:
    """A remembered observation, never a live reference to a target unit."""

    target_id: str
    position: Point
    observed_step: int
    source_id: str
    shared: bool = False


@dataclass(frozen=True)
class UnitSpec:
    id: str
    team: Team
    unit_type: UnitType
    position: Point
    speed: float = 0.0
    waypoints: tuple[Point, ...] = ()
    return_home: bool = False
    sensor_range: float = 160.0
    equipment_type: EquipmentType | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", tuple(self.waypoints))
        object.__setattr__(self, "equipment_type", _check_equipment(self.equipment_type, self.unit_type))


@dataclass(frozen=True)
class Scene:
    name: str
    width: float
    height: float
    fixed_dt: float
    spawn_points: Mapping[Team, Point]
    return_points: Mapping[Team, Point]
    obstacles: tuple[Obstacle, ...]
    units: tuple[UnitSpec, ...]
    rules: GameRules | None = None
    georeference: GeoReference | None = None
    terrain: str | None = None

    def __post_init__(self) -> None:
        # A frozen dataclass alone does not protect mutable dictionaries/lists.
        object.__setattr__(self, "spawn_points", MappingProxyType(dict(self.spawn_points)))
        object.__setattr__(self, "return_points", MappingProxyType(dict(self.return_points)))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "units", tuple(self.units))


@dataclass
class Unit:
    id: str
    team: Team
    unit_type: UnitType
    position: Point
    behavior: BehaviorState = BehaviorState.IDLE
    speed: float = 0.0
    waypoints: tuple[Point, ...] = ()
    return_home: bool = False
    target: Point | None = None
    path: tuple[Point, ...] = ()
    path_index: int = 0
    waypoint_index: int = 0
    trail: list[Point] = field(default_factory=list)
    reason: str = ""
    distance_travelled: float = 0.0
    sensor_range: float = 160.0
    contacts: dict[str, Contact] = field(default_factory=dict)
    tagged_targets: set[str] = field(default_factory=set)
    tag_count: int = 0
    tag_flash_until_step: int = 0
    equipment_type: EquipmentType | None = None

    def __post_init__(self) -> None:
        self.equipment_type = _check_equipment(self.equipment_type, self.unit_type)
