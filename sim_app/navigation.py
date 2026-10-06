"""Deterministic navigation for the second-stage teaching simulation.

Ground units move through a visibility graph around expanded rectangular
obstacles. Air units fly directly at a fixed height and ignore these ground
obstacles. Neither planner approximates the world with a grid.
"""

from bisect import bisect_left, bisect_right
from heapq import heappop, heappush
from math import hypot, inf, isfinite, nextafter

from .models import Point, Scene, UnitType


GROUND_CLEARANCE = 14.0

# Rectangle bounds are (left, top, right, bottom), including their edges.
_Rectangle = tuple[float, float, float, float]


def _rectangles(scene: Scene) -> tuple[_Rectangle, ...]:
    return tuple(
        (
            obstacle.x - GROUND_CLEARANCE,
            obstacle.y - GROUND_CLEARANCE,
            obstacle.x + obstacle.width + GROUND_CLEARANCE,
            obstacle.y + obstacle.height + GROUND_CLEARANCE,
        )
        for obstacle in scene.obstacles
    )


def _within_world(scene: Scene, point: Point, unit_type: UnitType) -> bool:
    if not all(isfinite(value) for value in (point.x, point.y, scene.width, scene.height)):
        return False
    if unit_type == UnitType.AIR:
        return 0.0 <= point.x < scene.width and 0.0 <= point.y < scene.height
    if unit_type == UnitType.GROUND:
        # A center exactly 14 units from a world edge is permitted.
        return (
            GROUND_CLEARANCE <= point.x <= scene.width - GROUND_CLEARANCE
            and GROUND_CLEARANCE <= point.y <= scene.height - GROUND_CLEARANCE
        )
    return False


def _inside_rectangle(point: Point, rectangle: _Rectangle) -> bool:
    left, top, right, bottom = rectangle
    return left <= point.x <= right and top <= point.y <= bottom


def point_clear(
    scene: Scene, point: Point, unit_type: UnitType = UnitType.GROUND
) -> bool:
    """Return whether a unit center may occupy ``point``.

    Ground obstacle edges, after expansion by ``GROUND_CLEARANCE``, are
    occupied. Air centers use the world's half-open coordinate range.
    """
    if not _within_world(scene, point, unit_type):
        return False
    return unit_type == UnitType.AIR or not any(
        _inside_rectangle(point, rectangle) for rectangle in _rectangles(scene)
    )


def _intersects_rectangle(start: Point, end: Point, rectangle: _Rectangle) -> bool:
    """Intersect a closed line segment and rectangle using the slab method.

    Testing the entire segment prevents a fast unit from passing through an
    obstacle even when both endpoints themselves are free.
    """
    left, top, right, bottom = rectangle
    entry, exit_ = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start.x, end.x - start.x, left, right),
        (start.y, end.y - start.y, top, bottom),
    ):
        if delta == 0.0:
            if origin < lower or origin > upper:
                return False
            continue
        near, far = (lower - origin) / delta, (upper - origin) / delta
        if near > far:
            near, far = far, near
        entry = max(entry, near)
        exit_ = min(exit_, far)
        if entry > exit_:
            return False
    return True


def _segment_clear(
    scene: Scene,
    start: Point,
    end: Point,
    unit_type: UnitType,
    rectangles: tuple[_Rectangle, ...],
) -> bool:
    if not _within_world(scene, start, unit_type) or not _within_world(scene, end, unit_type):
        return False
    # The valid world region is convex, so checking its two endpoints suffices.
    return unit_type == UnitType.AIR or not any(
        _intersects_rectangle(start, end, rectangle) for rectangle in rectangles
    )


def segment_clear(
    scene: Scene,
    start: Point,
    end: Point,
    unit_type: UnitType = UnitType.GROUND,
) -> bool:
    """Return whether the complete movement segment stays in free space."""
    return _segment_clear(scene, start, end, unit_type, _rectangles(scene))


def _outside_edge(
    value: float,
    direction: int,
    coordinates: list[float],
    minimum: float,
    maximum: float,
    epsilon: float,
) -> float:
    # Keep the tiny corner offset inside narrow gaps rather than crossing a
    # neighboring rectangle edge. Clamping also preserves routes beside walls.
    if direction < 0:
        index = bisect_left(coordinates, value) - 1
        gap = value - coordinates[index] if index >= 0 else inf
    else:
        index = bisect_right(coordinates, value)
        gap = coordinates[index] - value if index < len(coordinates) else inf
    offset = min(epsilon, gap / 4.0)
    result = value + direction * offset
    if result == value:
        result = nextafter(value, -inf if direction < 0 else inf)
    return min(maximum, max(minimum, result))


def _corner_nodes(scene: Scene, rectangles: tuple[_Rectangle, ...]) -> list[Point]:
    minimum = GROUND_CLEARANCE
    maximum_x, maximum_y = scene.width - minimum, scene.height - minimum
    x_edges = sorted({minimum, maximum_x, *(x for rect in rectangles for x in (rect[0], rect[2]))})
    y_edges = sorted({minimum, maximum_y, *(y for rect in rectangles for y in (rect[1], rect[3]))})
    epsilon = max(1e-7, max(scene.width, scene.height) * 1e-9)
    corners: list[Point] = []
    seen: set[Point] = set()
    for left, top, right, bottom in rectangles:
        for x, x_direction, y, y_direction in (
            (left, -1, top, -1),
            (right, 1, top, -1),
            (right, 1, bottom, 1),
            (left, -1, bottom, 1),
        ):
            point = Point(
                _outside_edge(x, x_direction, x_edges, minimum, maximum_x, epsilon),
                _outside_edge(y, y_direction, y_edges, minimum, maximum_y, epsilon),
            )
            if point not in seen and _within_world(scene, point, UnitType.GROUND) and not any(
                _inside_rectangle(point, rectangle) for rectangle in rectangles
            ):
                seen.add(point)
                corners.append(point)
    return corners


def plan_path(
    scene: Scene, start: Point, goal: Point, unit_type: UnitType
) -> tuple[Point, ...] | None:
    """Plan deterministic waypoints, excluding ``start`` and including ``goal``.

    Valid equal endpoints return an empty tuple. Invalid endpoints or an
    unreachable destination return ``None``. Each returned movement segment is
    checked against the same collision geometry as ``segment_clear``.
    """
    if not point_clear(scene, start, unit_type) or not point_clear(scene, goal, unit_type):
        return None
    if start == goal:
        return ()
    rectangles = _rectangles(scene)
    if _segment_clear(scene, start, goal, unit_type, rectangles):
        return (goal,)

    nodes = [start, goal]
    seen = {start, goal}
    for corner in _corner_nodes(scene, rectangles):
        if corner not in seen:
            nodes.append(corner)
            seen.add(corner)

    neighbors: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for first, start_node in enumerate(nodes):
        for second in range(first + 1, len(nodes)):
            end_node = nodes[second]
            if _segment_clear(scene, start_node, end_node, unit_type, rectangles):
                distance = hypot(end_node.x - start_node.x, end_node.y - start_node.y)
                neighbors[first].append((second, distance))
                neighbors[second].append((first, distance))

    distances = [inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distances[0] = 0.0
    pending = [(0.0, 0)]
    while pending:
        distance, current = heappop(pending)
        if distance != distances[current]:
            continue
        if current == 1:
            route = []
            while current != 0:
                route.append(nodes[current])
                parent = previous[current]
                assert parent is not None
                current = parent
            return tuple(reversed(route))
        for next_node, length in neighbors[current]:
            candidate = distance + length
            if candidate < distances[next_node]:
                distances[next_node] = candidate
                previous[next_node] = current
                heappush(pending, (candidate, next_node))
    return None
