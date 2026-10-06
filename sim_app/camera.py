"""A bounded, interruptible map camera; no simulation or GUI state is changed."""
from math import isfinite
from time import monotonic

from .models import Point

MIN_ZOOM = 1.0
MAX_ZOOM = 6.0
ZOOM_DURATION = .160


class MapCamera:
    def __init__(self, *, min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM):
        self.min_zoom, self.max_zoom = float(min_zoom), float(max_zoom)
        self.zoom = self.target_zoom = 1.0
        self.center = None
        self.world_size = None
        self.viewport = (0, 0, 1, 1)
        self._transition = None
        self._base_scale = 1.0
        self._default_center = Point(.5, .5)
        self._screen_center = (.5, .5)

    def bind(self, world_size, viewport):
        world_size = tuple(float(v) for v in world_size)
        viewport = tuple(float(v) for v in viewport)
        if self.world_size is not None and world_size != self.world_size:
            self.set_pose(1.0, None)
        if viewport != self.viewport:
            # Resizing retains the current pose; an old screen anchor belongs
            # to the old viewport and must not continue moving the new one.
            self._transition = None
            self.target_zoom = self.zoom
        self.world_size, self.viewport = world_size, viewport
        self._base_scale = min(viewport[2] / world_size[0], viewport[3] / world_size[1])
        self._default_center = Point(world_size[0] / 2, world_size[1] / 2)
        self._screen_center = (viewport[0] + viewport[2] / 2, viewport[1] + viewport[3] / 2)

    @property
    def base_scale(self):
        return self._base_scale

    @property
    def actual_center(self):
        return self.center or self._default_center

    @property
    def screen_center(self):
        return self._screen_center

    @property
    def animating(self):
        return self._transition is not None

    def set_pose(self, zoom, center):
        self.zoom = self.target_zoom = min(self.max_zoom, max(self.min_zoom, float(zoom)))
        self.center = center
        self._transition = None

    def to_screen(self, point):
        x, y = self.screen_center
        center, scale = self.actual_center, self.base_scale * self.zoom
        return (x + (point.x - center.x) * scale, y + (point.y - center.y) * scale)

    def to_world(self, screen):
        x, y = self.screen_center
        center, scale = self.actual_center, self.base_scale * self.zoom
        return Point(center.x + (screen[0] - x) / scale, center.y + (screen[1] - y) / scale)

    def visible_world(self):
        x, y, width, height = self.viewport
        corner = self.to_world((x, y))
        scale = self.base_scale * self.zoom
        return (corner.x, corner.y, width / scale, height / scale)

    def update(self, now=None):
        if self._transition is None:
            return False
        now = monotonic() if now is None else now
        start, from_zoom, to_zoom, from_center, to_center, anchor = self._transition
        ratio = min(1.0, max(0.0, (now - start) / ZOOM_DURATION))
        eased = ratio * ratio * (3 - 2 * ratio)
        self.zoom = from_zoom + (to_zoom - from_zoom) * eased
        if anchor is not None:
            screen, world = anchor
            x, y = self.screen_center
            scale = self.base_scale * self.zoom
            self.center = Point(world.x - (screen[0] - x) / scale,
                                world.y - (screen[1] - y) / scale)
        else:
            self.center = Point(from_center.x + (to_center.x - from_center.x) * eased,
                                from_center.y + (to_center.y - from_center.y) * eased)
        if ratio >= 1:
            self.zoom, self.center = to_zoom, to_center
            self._transition = None
        return True

    def zoom_by(self, factor, anchor=None, now=None):
        if not isfinite(factor) or factor <= 0:
            return False
        now = monotonic() if now is None else now
        self.update(now)
        target = min(self.max_zoom, max(self.min_zoom, self.target_zoom * factor))
        if abs(target - self.target_zoom) < 1e-10:
            return False
        screen = tuple(anchor or self.screen_center)
        world = self.to_world(screen)
        x, y = self.screen_center
        scale = self.base_scale * target
        end = Point(world.x - (screen[0] - x) / scale, world.y - (screen[1] - y) / scale)
        self.target_zoom = target
        self._transition = (now, self.zoom, target, self.actual_center, end, (screen, world))
        return True

    def pan(self, delta, now=None):
        self.update(now)
        center, scale = self.actual_center, self.base_scale * self.zoom
        width, height = self.world_size
        # Keeping the center within the world always leaves some scene visible,
        # including at maximum zoom after an unusually large drag.
        center = Point(min(width, max(0, center.x - delta[0] / scale)),
                       min(height, max(0, center.y - delta[1] / scale)))
        self.set_pose(self.zoom, center)

    def reset(self, now=None, animate=True):
        now = monotonic() if now is None else now
        self.update(now)
        width, height = self.world_size or (1, 1)
        end = Point(width / 2, height / 2)
        if not animate:
            self.set_pose(1.0, None)
            return
        self.target_zoom = 1.0
        self._transition = (now, self.zoom, 1.0, self.actual_center, end, None)
