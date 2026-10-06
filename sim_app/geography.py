"""Explicit display reference between local teaching coordinates and WGS84.

The model still navigates a flat local rectangle. This reference is an
equirectangular mapping at its centre latitude, not a geodesic mover or DEM.
"""
from dataclasses import dataclass
from math import asin, cos, isfinite, radians, sin, sqrt

EARTH_RADIUS_M = 6371008.8
METERS_PER_DEGREE = EARTH_RADIUS_M * radians(1)


@dataclass(frozen=True)
class GeoReference:
    center_latitude: float
    center_longitude: float
    meters_per_unit: float = 1.0

    def __post_init__(self):
        values = (self.center_latitude, self.center_longitude, self.meters_per_unit)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in values):
            raise ValueError('地理参考需要有限数值')
        if not -80 <= self.center_latitude <= 80:
            raise ValueError('中心纬度须在 −80° 到 80° 之间')
        if not -180 <= self.center_longitude <= 180:
            raise ValueError('中心经度须在 −180° 到 180° 之间')
        if self.meters_per_unit <= 0:
            raise ValueError('每单位米数须大于零')

    @classmethod
    def from_world(cls, world):
        data = world.get('georeference')
        return cls(**data) if data is not None else None

    def to_lonlat(self, x, y, width, height):
        lon = self.center_longitude + (x-width/2) * self.meters_per_unit / (
            METERS_PER_DEGREE * cos(radians(self.center_latitude)))
        lat = self.center_latitude - (y-height/2) * self.meters_per_unit / METERS_PER_DEGREE
        return lon, lat

    def to_local(self, lon, lat, width, height):
        return (width/2 + (lon-self.center_longitude) * METERS_PER_DEGREE * cos(
            radians(self.center_latitude)) / self.meters_per_unit,
                height/2 - (lat-self.center_latitude) * METERS_PER_DEGREE / self.meters_per_unit)

    def bounds(self, width, height):
        west, north = self.to_lonlat(0, 0, width, height)
        east, south = self.to_lonlat(width, height, width, height)
        return west, south, east, north

    def validate_extent(self, width, height):
        west, south, east, north = self.bounds(width, height)
        if west < -180 or east > 180:
            raise ValueError('场景范围跨越 ±180°；请调整中心经度、世界尺寸或每单位米数')
        if south < -85 or north > 85:
            raise ValueError('场景范围超出 ±85°；请调整中心纬度、世界尺寸或每单位米数')

    def as_dict(self):
        return {'center_latitude': self.center_latitude, 'center_longitude': self.center_longitude,
                'meters_per_unit': self.meters_per_unit}


def scene_world(scene):
    world = {'width': scene.width, 'height': scene.height}
    reference = getattr(scene, 'georeference', None)
    if reference is not None:
        world['georeference'] = reference.as_dict()
    terrain = getattr(scene, 'terrain', None)
    if terrain is not None:
        world['terrain'] = terrain
    return world


def format_lonlat(lon, lat):
    return f'{abs(lat):.3f}°{"N" if lat >= 0 else "S"}  {abs(lon):.3f}°{"E" if lon >= 0 else "W"}'


def distance_m(lon1, lat1, lon2, lat2):
    a = sin(radians(lat2-lat1)/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(radians(lon2-lon1)/2)**2
    return 2 * EARTH_RADIUS_M * asin(min(1, sqrt(max(0, a))))
