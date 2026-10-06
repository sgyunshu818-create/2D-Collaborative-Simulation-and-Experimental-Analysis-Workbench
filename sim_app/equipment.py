"""Equipment choices for presentation and editable classroom scene defaults.

These defaults are game parameters, not real-world equipment specifications.
Navigation continues to use only the existing air and ground categories.
"""

from dataclasses import dataclass
from typing import Mapping

from .models import EquipmentType, UnitType


@dataclass(frozen=True)
class EquipmentProfile:
    equipment_type: EquipmentType
    label: str
    default_speed: float
    default_sensor_range: float

    @property
    def key(self) -> str:
        return self.equipment_type.value

    @property
    def unit_type(self) -> UnitType:
        return self.equipment_type.unit_type


EQUIPMENT_PROFILES = (
    EquipmentProfile(EquipmentType.DRONE, "无人机", 140.0, 240.0),
    EquipmentProfile(EquipmentType.AIRPLANE, "飞机", 140.0, 240.0),
    EquipmentProfile(EquipmentType.ARMORED_CAR, "装甲车", 90.0, 160.0),
    EquipmentProfile(EquipmentType.TANK, "坦克", 90.0, 160.0),
)
_PROFILES = {profile.key: profile for profile in EQUIPMENT_PROFILES}


def profile_key(value) -> str:
    """Resolve a unit, spec, draft row or key; legacy air/ground still display."""
    if isinstance(value, EquipmentProfile):
        return value.key
    if isinstance(value, EquipmentType):
        return value.value
    if isinstance(value, Mapping):
        equipment = value.get("equipment")
        movement = value.get("type")
    elif hasattr(value, "unit_type"):
        equipment = getattr(value, "equipment_type", None)
        movement = value.unit_type
    else:
        equipment, movement = value, value
    if equipment is not None:
        if equipment in (UnitType.AIR, UnitType.GROUND):
            movement = equipment
        else:
            return EquipmentType(equipment).value
    movement = UnitType(movement)
    return EquipmentType.DRONE.value if movement is UnitType.AIR else EquipmentType.TANK.value


def equipment_profile(value) -> EquipmentProfile:
    return _PROFILES[profile_key(value)]


def equipment_label(value) -> str:
    return equipment_profile(value).label


def equipment_choices(unit_type: UnitType | str | None = None) -> tuple[EquipmentProfile, ...]:
    """Return all choices, or choices compatible with a movement category."""
    if unit_type is None:
        return EQUIPMENT_PROFILES
    movement = UnitType(unit_type)
    return tuple(profile for profile in EQUIPMENT_PROFILES if profile.unit_type is movement)
