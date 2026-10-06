"""Geometry checks for platform clearance and deterministic obstacle routing."""

import unittest

from sim_app.models import Obstacle, Point, Scene, Team, UnitType
from sim_app.navigation import GROUND_CLEARANCE, plan_path, point_clear, segment_clear


def navigation_scene(obstacles=(), width=300, height=300):
    anchors = {Team.RED: Point(40, 150), Team.BLUE: Point(260, 150)}
    return Scene("navigation_test", width, height, 0.125, anchors, anchors, obstacles, ())


class NavigationTests(unittest.TestCase):
    def assert_safe_route(self, scene, start, goal, route):
        self.assertIsNotNone(route)
        self.assertIsInstance(route, tuple)
        self.assertEqual(route[-1], goal)
        self.assertNotEqual(route[0], start)
        previous = start
        for point in route:
            self.assertTrue(point_clear(scene, point))
            self.assertTrue(segment_clear(scene, previous, point), (previous, point))
            previous = point

    def test_unobstructed_path_is_direct(self):
        scene = navigation_scene()
        start, goal = Point(40, 150), Point(260, 150)
        self.assertEqual(plan_path(scene, start, goal, UnitType.GROUND), (goal,))
        self.assertEqual(plan_path(scene, start, goal, UnitType.AIR), (goal,))

    def test_same_valid_start_and_goal_needs_no_motion(self):
        scene = navigation_scene()
        point = Point(40, 150)
        self.assertEqual(plan_path(scene, point, point, UnitType.GROUND), ())
        self.assertEqual(plan_path(scene, point, point, UnitType.AIR), ())

    def test_ground_routes_around_obstacle_and_air_crosses_directly(self):
        scene = navigation_scene((Obstacle("block", 125, 90, 50, 120),))
        start, goal = Point(40, 150), Point(260, 150)
        self.assertFalse(segment_clear(scene, start, goal))
        route = plan_path(scene, start, goal, UnitType.GROUND)
        self.assert_safe_route(scene, start, goal, route)
        self.assertGreater(len(route), 1)
        self.assertTrue(segment_clear(scene, start, goal, UnitType.AIR))
        self.assertEqual(plan_path(scene, start, goal, UnitType.AIR), (goal,))

    def test_clearance_protects_world_edge_and_obstacle_margin(self):
        scene = navigation_scene((Obstacle("block", 125, 90, 50, 120),))
        self.assertGreater(GROUND_CLEARANCE, 0)
        for point in (Point(GROUND_CLEARANCE / 2, 150), Point(125 - GROUND_CLEARANCE / 2, 150)):
            with self.subTest(point=point):
                self.assertFalse(point_clear(scene, point, UnitType.GROUND))
                self.assertTrue(point_clear(scene, point, UnitType.AIR))
        self.assertFalse(segment_clear(scene, Point(40, 85), Point(260, 85)))

    def test_full_height_wall_is_unreachable_for_ground(self):
        scene = navigation_scene((Obstacle("wall", 140, 0, 20, 300),))
        start, goal = Point(40, 150), Point(260, 150)
        self.assertIsNone(plan_path(scene, start, goal, UnitType.GROUND))
        self.assertEqual(plan_path(scene, start, goal, UnitType.AIR), (goal,))

    def test_gap_narrower_than_ground_clearance_is_not_a_route(self):
        scene = navigation_scene((
            Obstacle("upper", 140, 0, 20, 140),
            Obstacle("lower", 140, 160, 20, 140),
        ))
        self.assertIsNone(plan_path(scene, Point(40, 150), Point(260, 150), UnitType.GROUND))

    def test_overlapping_obstacles_can_be_routed_around_as_a_union(self):
        scene = navigation_scene((
            Obstacle("one", 110, 90, 60, 90),
            Obstacle("two", 140, 130, 60, 90),
        ))
        start, goal = Point(40, 150), Point(260, 150)
        self.assert_safe_route(scene, start, goal, plan_path(scene, start, goal, UnitType.GROUND))

    def test_out_of_world_endpoints_and_obstacle_endpoints_are_rejected(self):
        scene = navigation_scene((Obstacle("block", 125, 90, 50, 120),))
        valid = Point(40, 150)
        for point in (Point(-1, 150), Point(300, 150), Point(150, -1), Point(150, 300)):
            for unit_type in UnitType:
                with self.subTest(point=point, unit_type=unit_type):
                    self.assertFalse(point_clear(scene, point, unit_type))
                    self.assertFalse(segment_clear(scene, valid, point, unit_type))
                    self.assertIsNone(plan_path(scene, valid, point, unit_type))
                    self.assertIsNone(plan_path(scene, point, valid, unit_type))
        inside = Point(150, 150)
        self.assertIsNone(plan_path(scene, valid, inside, UnitType.GROUND))
        self.assertIsNone(plan_path(scene, inside, valid, UnitType.GROUND))
        self.assertEqual(plan_path(scene, valid, inside, UnitType.AIR), (inside,))

    def test_symmetric_route_has_repeatable_tie_breaking(self):
        scene = navigation_scene((Obstacle("block", 125, 90, 50, 120),))
        start, goal = Point(40, 150), Point(260, 150)
        expected = plan_path(scene, start, goal, UnitType.GROUND)
        self.assert_safe_route(scene, start, goal, expected)
        for _ in range(5):
            self.assertEqual(plan_path(scene, start, goal, UnitType.GROUND), expected)


if __name__ == "__main__":
    unittest.main()
