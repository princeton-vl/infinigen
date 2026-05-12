from typing import Callable

import bpy
import numpy as np
import procfunc as pf

import infinigen_v2.generators.scenes.collision_collection as ccol
from infinigen_v2.generators.cameras.util import total_bbox
from infinigen_v2.util.errors import RejectedScene


class RRTPolicyError(ValueError):
    pass


class _RRTPlanner:
    def __init__(
        self,
        rng: pf.RNG,
        colliders: ccol.CollisionSet,
        depsgraph: bpy.types.Depsgraph,
        bbox: tuple[np.ndarray, np.ndarray],
        validate_node: Callable[[np.ndarray], bool],
        step_range: tuple[float, float] = (1.0, 1.0),
        stride_range: tuple[int, int] = (16, 32),
        min_node_dist_to_obstacle: float = 0.2,
        max_iter: int = 2000,
    ):
        self.rng = rng
        self.colliders = colliders
        self._depsgraph = depsgraph
        self.validate_node = validate_node
        self.bbox_min = np.asarray(bbox[0], dtype=np.float64)
        self.bbox_max = np.asarray(bbox[1], dtype=np.float64)
        self.step_range = step_range
        self.stride_range = stride_range
        self.max_iter = max_iter
        self.step = float(self.rng.uniform(*self.step_range))
        self.vertices: dict[
            tuple[float, float, float], tuple[tuple[float, float, float] | None, float]
        ] = {}

        self.collision_check_dirs: list[np.ndarray] = []
        if min_node_dist_to_obstacle > 0:
            thetas = [2 * np.pi * i / 8 for i in range(8)]
            phis = [np.pi * i / 4 for i in range(5)]
            for theta in thetas:
                for phi in phis:
                    self.collision_check_dirs.append(
                        min_node_dist_to_obstacle
                        * np.array(
                            [
                                np.cos(theta) * np.sin(phi),
                                np.sin(theta) * np.sin(phi),
                                np.cos(phi),
                            ],
                            dtype=np.float64,
                        )
                    )

    def _is_in_bbox(self, coord: np.ndarray) -> bool:
        return not (np.any(coord < self.bbox_min) or np.any(coord > self.bbox_max))

    def _line_not_valid(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        bbox: tuple[np.ndarray, np.ndarray] | None = None,
        dist: float | None = None,
    ) -> bool:
        p1 = np.asarray(p1, dtype=np.float64)
        p2 = np.asarray(p2, dtype=np.float64)
        if bbox is not None:
            if not self._is_in_bbox(p1) or not self._is_in_bbox(p2):
                return True

        delta = p2 - p1
        length = float(np.linalg.norm(delta)) if dist is None else float(dist)
        if length < 1e-8:
            return False
        direction = delta / np.linalg.norm(delta)
        hit, *_ = bpy.context.scene.ray_cast(
            self._depsgraph,
            p1,
            direction,
            distance=length,
        )
        return bool(hit)

    def _prox_check(self, x: np.ndarray) -> bool:
        for direction in self.collision_check_dirs:
            if self._line_not_valid(x, x + direction):
                return False
        return True

    def _is_valid(self, node: np.ndarray) -> bool:
        if not self._is_in_bbox(node):
            return False
        if not self.validate_node(node):
            return False
        return True

    def _rand_node(self) -> np.ndarray:
        return self.rng.uniform(self.bbox_min, self.bbox_max)

    def _rand_valid_node(self, max_iter: int = 500) -> np.ndarray:
        for _ in range(max_iter):
            node = self._rand_node()
            if self._is_valid(node):
                return node
        raise RRTPolicyError("RRT could not find a valid random node")

    def _parent(
        self, x: tuple[float, float, float]
    ) -> tuple[float, float, float] | None:
        return self.vertices[x][0] if x in self.vertices else None

    def _cost(self, x: tuple[float, float, float]) -> float | None:
        return self.vertices[x][1] if x in self.vertices else None

    def _get_vertices(self) -> np.ndarray:
        return np.asarray(list(self.vertices.keys()), dtype=np.float64)

    def _wireup(
        self,
        x: tuple[float, float, float],
        y: tuple[float, float, float],
    ) -> None:
        y_cost = self._cost(y)
        assert y_cost is not None
        self.vertices[x] = (y, y_cost + self._dist(np.asarray(x), np.asarray(y)))

    @staticmethod
    def _dist(pos1: np.ndarray, pos2: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(pos1) - np.asarray(pos2)))

    def _sample_free(self, target: np.ndarray, bias: float = 0.1) -> np.ndarray:
        if self.rng.random() < bias:
            return np.asarray(target)
        return self._rand_node()

    def _nearest(self, x: np.ndarray) -> tuple[float, float, float]:
        vertices = self._get_vertices()
        dists = np.linalg.norm(vertices - x[None, :], axis=1)
        return tuple(vertices[np.argmin(dists)])

    def _neighborhood(
        self,
        x: np.ndarray,
        radius: float | None = None,
        max_iter: int = 10,
    ) -> np.ndarray:
        vertices = self._get_vertices()
        num_verts = len(vertices)
        if num_verts == 0:
            return np.empty((0, 3), dtype=np.float64)

        gamma = 5.0
        eta = self.step
        nearpoints = np.empty((0, 3), dtype=np.float64)
        if radius is None:
            safe_n = max(num_verts, 2)
            r = min(gamma * ((np.log(safe_n) / safe_n) ** (1 / 3)), eta)
        else:
            r = radius

        i = 0
        while len(nearpoints) == 0:
            if i > max_iter:
                return np.empty((0, 3), dtype=np.float64)
            inside = np.linalg.norm(vertices - x[None, :], axis=1) < r
            nearpoints = vertices[inside]
            i += 1
            r += eta
        return nearpoints

    def _steer(
        self, x: np.ndarray, direction: np.ndarray
    ) -> tuple[float, float, float]:
        if np.allclose(x, direction):
            return tuple(x)
        dist = self._dist(x, direction)
        step = min(dist, self.step)
        increment = (direction - x) / dist * step
        return tuple(x + increment)

    def _choose_parent(
        self,
        xnew: tuple[float, float, float],
        xnear_arr: np.ndarray,
    ) -> tuple[tuple[float, float, float] | None, list[bool]]:
        xmin = None
        cmin = None
        collisions: list[bool] = []
        for xnear_np in xnear_arr:
            xnear = tuple(xnear_np)
            xnear_cost = self._cost(xnear)
            if xnear_cost is None:
                collisions.append(True)
                continue
            c1 = xnear_cost + self._dist(np.asarray(xnew), np.asarray(xnear))
            collide = self._line_not_valid(np.asarray(xnew), np.asarray(xnear))
            collisions.append(collide)
            if not collide and (xmin is None or c1 < cmin):
                xmin, cmin = xnear, c1
        return xmin, collisions

    def _rewire_neighborhood(
        self,
        xnew: tuple[float, float, float],
        xnear_arr: np.ndarray,
        collisions: list[bool],
    ) -> None:
        xnew_cost = self._cost(xnew)
        if xnew_cost is None:
            return
        for i, xnear_np in enumerate(xnear_arr):
            xnear = tuple(xnear_np)
            xnear_cost = self._cost(xnear)
            if xnear_cost is None:
                continue
            c2 = xnew_cost + self._dist(np.asarray(xnew), np.asarray(xnear))
            if not collisions[i] and c2 < xnear_cost:
                self._wireup(xnear, xnew)

    def generate_path(
        self,
        start: np.ndarray | tuple[float, float, float] | None = None,
        goal: np.ndarray | tuple[float, float, float] | None = None,
    ) -> list[tuple[float, float, float]]:
        x0 = (
            tuple(self._rand_valid_node())
            if start is None
            else tuple(np.asarray(start))
        )
        xt = tuple(self._rand_valid_node()) if goal is None else tuple(np.asarray(goal))
        if not self._is_valid(np.asarray(x0)):
            raise RRTPolicyError(f"RRT started with invalid node {x0}")
        if not self._is_valid(np.asarray(xt)):
            raise RRTPolicyError(f"RRT goal is invalid node {xt}")

        self.vertices = {x0: (None, 0.0)}
        n_iter = 0
        while n_iter < self.max_iter:
            xrand = self._sample_free(np.asarray(xt))
            xnearest = self._nearest(xrand)
            xnew = self._steer(np.asarray(xnearest), np.asarray(xrand))
            xnew_np = np.asarray(xnew)

            if self._prox_check(xnew_np):
                xnear_arr = self._neighborhood(xnew_np)
                xmin, collisions = self._choose_parent(xnew, xnear_arr)
                if xmin is not None:
                    self._wireup(xnew, xmin)
                    self._rewire_neighborhood(xnew, xnear_arr, collisions)

                    if self._dist(
                        np.asarray(xnew), np.asarray(xt)
                    ) < self.step and not self._line_not_valid(
                        np.asarray(xnew), np.asarray(xt)
                    ):
                        break

            self.step = float(self.rng.uniform(*self.step_range))
            n_iter += 1

        near_goal = self._neighborhood(np.asarray(xt), self.step, max_iter=1000)
        if len(near_goal) == 0:
            raise RRTPolicyError(f"RRT could not find path from {x0} to {xt}")
        costs = [self._cost(tuple(x)) for x in near_goal]
        finite_costs = [c if c is not None else np.inf for c in costs]
        x = tuple(near_goal[int(np.argmin(finite_costs))])

        path: list[tuple[float, float, float]] = []
        while x != x0:
            path.append(x)
            parent = self._parent(x)
            if parent is None:
                raise RRTPolicyError(
                    "RRT path construction failed due to missing parent"
                )
            x = parent
        path.reverse()
        return path

    def next_goal(
        self,
        start: np.ndarray | tuple[float, float, float],
        max_iter: int = 100,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=np.float64)
        r_range = (
            self.step_range[0] * self.stride_range[0],
            self.step_range[1] * self.stride_range[1],
        )
        theta_range = (0.0, 2 * np.pi)

        for _ in range(max_iter):
            r = self.rng.uniform(*r_range)
            z = self.rng.uniform(self.bbox_min[2], self.bbox_max[2])
            theta = self.rng.uniform(*theta_range)

            translation = (
                np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64) * r
            )
            translation[2] = z - start[2]
            nxt = start + translation
            if self._is_valid(nxt):
                return nxt
        raise RRTPolicyError(
            f"RRT could not find next goal node from start {tuple(start)}"
        )


def _validate_rrt_node(
    colliders: ccol.CollisionSet,
    depsgraph: bpy.types.Depsgraph,
    node: np.ndarray,
    probe_size: float = 0.1,
    max_vertical_ray: float = 100.0,
    max_lateral_ray: float = 100.0,
    require_enclosed: bool = False,
) -> bool:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = node
    if ccol.box_intersection_test(colliders, transform=transform, size=probe_size):
        return False
    if not require_enclosed:
        return True

    def _has_hit(direction: np.ndarray, max_distance: float) -> bool:
        hit, *_ = bpy.context.scene.ray_cast(
            depsgraph,
            node,
            direction,
            distance=max_distance,
        )
        return bool(hit)

    if not _has_hit(np.array([0.0, 0.0, 1.0], dtype=np.float64), max_vertical_ray):
        return False
    if not _has_hit(np.array([0.0, 0.0, -1.0], dtype=np.float64), max_vertical_ray):
        return False
    for direction in (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
    ):
        if not _has_hit(direction, max_lateral_ray):
            return False
    return True


def _segment_valid(
    start: np.ndarray,
    end: np.ndarray,
    predicate: Callable[[np.ndarray], bool],
    n_checks: int,
) -> bool:
    for t in np.linspace(0, 1, n_checks + 2)[1:-1]:
        if not predicate(start + t * (end - start)):
            return False
    return True


def _sample_enclosed_start(
    planner: _RRTPlanner,
    colliders: ccol.CollisionSet,
    depsgraph: bpy.types.Depsgraph,
    max_iter: int = 1000,
) -> np.ndarray:
    for _ in range(max_iter):
        candidate = planner._rand_valid_node()
        if _validate_rrt_node(
            colliders=colliders,
            depsgraph=depsgraph,
            node=candidate,
            require_enclosed=True,
        ):
            return candidate
    raise RejectedScene("Could not find an indoor valid camera start")


def _path_or_fallback(
    rng: pf.RNG,
    planner: _RRTPlanner,
    start_loc: np.ndarray,
    max_goal_attempts: int,
    max_path_retries: int,
    step_range: tuple[float, float],
) -> list[tuple[float, float, float]]:
    for _ in range(max_path_retries):
        try:
            goal = planner.next_goal(start=start_loc, max_iter=max_goal_attempts)
            path = planner.generate_path(start=start_loc, goal=goal)
        except RRTPolicyError:
            continue
        if len(path) > 0:
            return path

    for _ in range(max_goal_attempts):
        r = float(rng.uniform(step_range[0] * 0.25, step_range[1] * 1.0))
        theta = float(rng.uniform(0.0, 2 * np.pi))
        zoff = float(rng.uniform(-0.1, 0.1))
        candidate = start_loc + np.array(
            [r * np.cos(theta), r * np.sin(theta), zoff], dtype=np.float64
        )
        if not planner._is_valid(candidate):
            continue
        if planner._line_not_valid(start_loc, candidate):
            continue
        return [tuple(candidate)]

    return [tuple(start_loc)]


def rrt_camera(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    margin: float = 0.05,
    step_range: tuple[float, float] = (1, 2),
    stride_range: tuple[int, int] = (64, 128),
    min_node_dist_to_obstacle: float = 0.4,
    max_rrt_iter: int = 2000,
    max_goal_attempts: int = 200,
    max_path_retries: int = 80,
    speed_mps_range: tuple[float, float] = (1.0, 1.5),
    rot_std_deg: tuple[float, float, float] = (20.0, 20.0, 20.0),
    max_abs_roll_deg: float = 25.0,
    max_abs_pitch_offset_deg: float = 25.0,
    step_predicate: Callable[[np.ndarray], bool] | None = None,
    n_intermediate_checks: int = 4,
) -> pf.CameraObject:
    camera = pf.ops.primitives.perspective_camera(focal_length_mm=focal_length_mm)
    bbox = total_bbox(objects)

    bbox_min, bbox_max = np.asarray(bbox[0]).copy(), np.asarray(bbox[1]).copy()
    bbox_min += margin
    bbox_max -= margin
    if np.any(bbox_min >= bbox_max):
        raise RejectedScene("RRT camera bbox is invalid after applying margins")

    depsgraph = bpy.context.evaluated_depsgraph_get()

    planner = _RRTPlanner(
        rng=rng,
        colliders=colliders,
        depsgraph=depsgraph,
        bbox=(bbox_min, bbox_max),
        validate_node=lambda node: _validate_rrt_node(
            colliders=colliders,
            depsgraph=depsgraph,
            node=node,
            require_enclosed=False,
        )
        and (step_predicate is None or step_predicate(node)),
        step_range=step_range,
        stride_range=stride_range,
        min_node_dist_to_obstacle=min_node_dist_to_obstacle,
        max_iter=max_rrt_iter,
    )

    frame_start_f = float(frame_start)
    frame_end_f = float(frame_end)
    frame_curr = frame_start_f
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base

    # Always initialize from an RRT-valid node to avoid default camera origin leakage.
    start_loc = _sample_enclosed_start(
        planner=planner,
        colliders=colliders,
        depsgraph=depsgraph,
    )

    init_rot = (np.pi / 2, 0.0, float(rng.uniform(-np.pi, np.pi)))
    pf.ops.object.set_transform(camera, location=start_loc, rotation_euler=init_rot)

    camera.item().keyframe_insert("location", frame=frame_start)
    camera.item().keyframe_insert("rotation_euler", frame=frame_start)

    path: list[tuple[float, float, float]] = []
    path_ind = 0
    max_abs_roll_rad = np.deg2rad(max_abs_roll_deg)
    max_abs_pitch_offset_rad = np.deg2rad(max_abs_pitch_offset_deg)
    while frame_curr < frame_end_f - 1e-6:
        if path_ind >= len(path):
            path = _path_or_fallback(
                rng=rng,
                planner=planner,
                start_loc=start_loc,
                max_goal_attempts=max_goal_attempts,
                max_path_retries=max_path_retries,
                step_range=step_range,
            )
            path_ind = 0

        waypoint = np.asarray(path[path_ind], dtype=np.float64)
        segment = waypoint - start_loc
        segment_dist = float(np.linalg.norm(segment))
        if segment_dist < 1e-6:
            path_ind += 1
            continue

        speed = float(rng.uniform(*speed_mps_range))
        duration_frames = max(1.0, segment_dist / max(speed, 1e-4) * fps)
        frame_next = min(frame_end_f, frame_curr + duration_frames)
        frac = (frame_next - frame_curr) / duration_frames

        next_loc = start_loc + segment * frac
        if step_predicate is not None and not _segment_valid(
            start_loc, next_loc, step_predicate, n_intermediate_checks
        ):
            path_ind += 1
            continue

        rot_jitter = np.deg2rad(rng.normal(0.0, np.asarray(rot_std_deg)))
        next_rot = (
            np.asarray(camera.item().rotation_euler, dtype=np.float64)
            + rot_jitter * frac
        )
        # Keep camera mostly level by default: clamp roll and pitch.
        next_rot[0] = np.clip(
            next_rot[0],
            np.pi / 2 - max_abs_pitch_offset_rad,
            np.pi / 2 + max_abs_pitch_offset_rad,
        )
        next_rot[1] = np.clip(next_rot[1], -max_abs_roll_rad, max_abs_roll_rad)
        pf.ops.object.set_transform(
            camera,
            location=next_loc,
            rotation_euler=next_rot,
        )

        keyframe = int(round(frame_next))
        keyframe = max(keyframe, int(np.floor(frame_curr)) + 1)
        camera.item().keyframe_insert("location", frame=keyframe)
        camera.item().keyframe_insert("rotation_euler", frame=keyframe)

        start_loc = next_loc
        frame_curr = frame_next
        if frac >= 1.0 - 1e-6:
            path_ind += 1

    return camera


def rrt_camera_fast(
    rng: pf.RNG,
    colliders: ccol.CollisionSet,
    objects: list[pf.MeshObject],
    frame_start: int = 1,
    frame_end: int = 1,
    focal_length_mm: float = 15,
    speed_mps_range: tuple[float, float] = (5.0, 7.5),
    stride_range: tuple[int, int] = (320, 640),
    step_predicate: Callable[[np.ndarray], bool] | None = None,
    n_intermediate_checks: int = 4,
    **kwargs,
) -> pf.CameraObject:
    return rrt_camera(
        rng=rng,
        colliders=colliders,
        objects=objects,
        frame_start=frame_start,
        frame_end=frame_end,
        focal_length_mm=focal_length_mm,
        speed_mps_range=speed_mps_range,
        stride_range=stride_range,
        step_predicate=step_predicate,
        n_intermediate_checks=n_intermediate_checks,
        **kwargs,
    )
