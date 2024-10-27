# Copyright (c) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Dylan Li: primary author

import logging
from functools import partial
from math import ceil

import bpy
import gin
import numpy as np
from mathutils import Euler, Vector, bvhtree
from numpy.matlib import repmat
from numpy.random import uniform as U

from infinigen.core.placement.animation_policy import PolicyError
from infinigen.core.util.random import random_general

logger = logging.getLogger(__name__)


@gin.configurable
def validate_node_nature(node, obj_groups):
    MAX_ITERS = 100
    EPS = 0.00000000001
    direction = Vector((0, 0, 1))

    # check if node is in an object (or under a plane)
    for obj in obj_groups[0]:
        mat = obj.matrix_local.inverted()
        origin = mat @ Vector(node)
        num_hits = 0
        deps = bpy.context.evaluated_depsgraph_get()
        bvh = bvhtree.BVHTree.FromObject(obj, deps)
        loc, _, face, _ = bvh.ray_cast(origin, direction)

        while loc is not None and num_hits < MAX_ITERS:
            num_hits += 1
            new_face = face
            curr_eps = EPS
            while new_face == face:
                fudge_loc = loc + direction * curr_eps
                new_loc, _, new_face, _ = bvh.ray_cast(fudge_loc, direction)
                if new_face == face:
                    curr_eps *= 10
                else:
                    loc = new_loc
            face = new_face

        if (num_hits % 2) != 0:
            return False

    return True


@gin.configurable
def validate_node_indoors(node, obj_groups):
    # check if inside building using room objs
    for obj in obj_groups[0]:
        deps = bpy.context.evaluated_depsgraph_get()
        bvh = bvhtree.BVHTree.FromObject(obj, deps)
        cast_up = bvh.ray_cast(node, Vector((0, 0, 1)))
        cast_down = bvh.ray_cast(node, Vector((0, 0, -1)))
        if cast_up[0] is not None and cast_down[0] is not None:
            break
    else:
        return False

    MAX_ITERS = 100
    EPS = 0.00000000001
    direction = Vector((0, 0, 1))

    # check if node is in a nonroom object
    for obj in obj_groups[1]:
        num_hits = 0
        deps = bpy.context.evaluated_depsgraph_get()
        bvh = bvhtree.BVHTree.FromObject(obj, deps)
        loc, _, face, _ = bvh.ray_cast(node, direction)
        if loc is not None:
            f = bvh.ray_cast(node, Vector((0, 0, -1)))
            if f[0] is None:
                return False

        while loc is not None and num_hits < MAX_ITERS:
            num_hits += 1
            new_face = face
            curr_eps = EPS
            while new_face == face:
                fudge_loc = loc + direction * curr_eps
                new_loc, _, new_face, _ = bvh.ray_cast(fudge_loc, direction)
                if new_face == face:
                    curr_eps *= 10
                else:
                    loc = new_loc
            face = new_face

        if (num_hits % 2) != 0:
            return False

    return True


@gin.configurable
class RRT:
    def __init__(
        self,
        obj_groups,
        validate_node,
        bbox=None,
        step_range=(1, 1),
        stride_range=(16, 32),
        min_node_dist_to_obstacle=0.2,
        max_iter=2000,
    ):
        self.validate_node = partial(
            validate_node,
            obj_groups=obj_groups,
        )
        self.bbox = self.get_bbox() if bbox is None else bbox
        self.step_range = step_range
        self.max_iter = max_iter
        self.step = U(*step_range)
        self.stride_range = stride_range
        self.vertices = {}
        self.collision_check_dirs = []

        # create directions to check proximity to obstacles
        thetas = [2 * np.pi * i / 8 for i in range(8)]
        phis = [np.pi * i / 4 for i in range(5)]
        for theta in thetas:
            for phi in phis:
                self.collision_check_dirs.append(
                    min_node_dist_to_obstacle
                    * Vector(
                        (
                            np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi),
                        )
                    )
                )

    def generate_path(self, start=None, goal=None):
        bound_added = 0
        if start is None:
            x0 = tuple(self.rand_valid_node())
        else:
            x0 = tuple(start)
            if not self.is_valid(x0):
                raise PolicyError(f"RRT started with invalid node {x0}")
            self.bbox = np.vstack((self.bbox, start))
            bound_added += 1
        if goal is None:
            xt = tuple(self.rand_valid_node())
        else:
            xt = tuple(goal)
            self.bbox = np.vstack((self.bbox, goal))
            bound_added += 1

        path = []
        self.vertices = {}
        self.vertices[x0] = (None, 0.0)  # (parent, cost)
        iter = 0
        while iter < self.max_iter:
            xrand = self.sample_free(xt)
            xnearest = self.nearest(xrand)
            xnew = self.steer(xnearest, xrand)

            if self.prox_check(xnew):
                Xnear = self.neighborhood(xnew)
                # connecting along minimal cost path
                xmin, cmin = None, None
                collisions = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    c1 = self.cost(xnear) + self.getDist(xnew, xnear)
                    collide = self.line_not_valid(xnew, xnear, self.bbox)
                    collisions.append(collide)
                    if not collide:
                        if xmin is None:
                            xmin, cmin = xnear, c1
                        elif c1 < cmin:
                            xmin, cmin = xnear, c1

                if xmin is not None:
                    self.wireup(xnew, xmin)
                    # rewire nodes near xnew to xnew if beneficial
                    for i in range(len(Xnear)):
                        collide = collisions[i]
                        xnear = tuple(Xnear[i])
                        c2 = self.cost(xnew) + self.getDist(xnew, xnear)
                        if not collide and c2 < self.cost(xnear):
                            self.wireup(xnear, xnew)

                    # stop if xnew is near the goal node
                    if self.getDist(xnew, xt) < self.step and not self.line_not_valid(
                        xnew, xt
                    ):
                        break

            self.step = U(*self.step_range)
            iter += 1

        if bound_added > 0:
            self.bbox = self.bbox[:-bound_added]
        xn = self.neighborhood(xt, self.step, max_iter=1000)
        if len(xn) == 0:
            raise PolicyError(f"RRT could not find path from {x0} to {xt}")
        c = [self.cost(tuple(x)) for x in xn]
        xncmin = tuple(xn[np.argmin(c)])
        x = xncmin

        path = []
        while x != x0:
            path.append(x)
            x = self.parent(x)
        # path.append(x0)
        path.reverse()

        return path

    def next_goal(self, start, max_iter=100):
        r_range = (
            self.step_range[0] * self.stride_range[0],
            self.step_range[1] * self.stride_range[1],
        )
        z_range = (self.bbox.min(axis=0)[2], self.bbox.max(axis=0)[2])
        theta_range = (0, 2 * np.pi)
        iter = 0
        while iter < max_iter:
            r = np.random.uniform(*r_range)
            z = np.random.uniform(*z_range)
            theta = np.random.uniform(*theta_range)

            translation = Vector((1, 0, 0))
            translation.rotate(Euler((0, 0, theta)))
            translation *= r
            translation[2] = z
            next = Vector(start) + translation
            if self.is_valid(next):
                return next
            iter += 1
        raise PolicyError(
            f"RRT could not find the next goal node with start node {start}"
        )

    def prox_check(self, x):
        for dir in self.collision_check_dirs:
            # ray cast from x to check proximity to obstacles
            if self.line_not_valid(x, Vector(x) + dir):
                return False

        return True

    def node_in_object(self, coord):
        for obj in bpy.context.scene.objects:
            if (
                obj.type == "MESH"
                and not obj.hide_render
                and "atmosphere" not in obj.name
            ):
                if self.is_inside_obj(obj, coord):
                    return True
        return False

    def is_valid(self, node):
        if not self.is_in_bbox(node, self.bbox):
            return False

        if not self.validate_node(node=node):
            return False

        return True

    def rand_valid_node(self):
        x = self.rand_node()
        i = 0

        while not self.is_valid(x):
            x = self.rand_node()
            i += 1
            if i > 100:
                return (0.0, 0.0, 0.0)
        return x

    def rand_node(self):
        return np.random.uniform(self.bbox.min(axis=0), self.bbox.max(axis=0))

    def parent(self, x):
        if x in self.vertices:
            return self.vertices[x][0]
        else:
            return None

    def cost(self, x):
        if x in self.vertices:
            return self.vertices[x][1]
        else:
            return None

    def get_vertices(self):
        return np.array(list(self.vertices.keys()))

    def wireup(self, x, y):
        self.vertices[x] = (y, self.cost(y) + self.getDist(x, y))

    def getDist(self, pos1, pos2):
        return np.sqrt(
            sum(
                [
                    (pos1[0] - pos2[0]) ** 2,
                    (pos1[1] - pos2[1]) ** 2,
                    (pos1[2] - pos2[2]) ** 2,
                ]
            )
        )

    def sample_free(self, xt, bias=0.1):
        if np.random.random() < bias:
            return np.array(xt)
        else:
            return self.rand_node()

    def nearest(self, x):
        V = self.get_vertices()
        xr = repmat(x, len(V), 1)
        dists = np.linalg.norm(xr - V, axis=1)
        nearest = tuple(V[np.argmin(dists)])
        return nearest

    def neighborhood(self, x, radius=None, max_iter=10):
        V = self.get_vertices()
        num_verts = len(V)

        gamma = 5
        eta = self.step
        nearpoints = []
        r = (
            min(gamma * ((np.log(num_verts) / num_verts) ** (1 / 3)), eta)
            if radius is None
            else radius
        )
        i = 0
        while len(nearpoints) == 0:
            if i > max_iter:
                return np.array([])
            xr = repmat(x, num_verts, 1)
            inside = np.linalg.norm(xr - V, axis=1) < r
            nearpoints = V[inside]
            i += 1
            r += eta

        return np.array(nearpoints)

    def steer(self, x, direction):
        if np.equal(x, direction).all():
            return x
        dist = self.getDist(x, direction)
        # don't overshoot direction node
        step = min(dist, self.step)
        increment = (direction - x) / dist * step
        xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])

        return xnew

    def get_bbox(self, objects=None):
        def np_matmul_coords(coordinates, matrix, space=None):
            M = (space @ matrix @ space.inverted() if space else matrix).transposed()
            ones = np.ones((coordinates.shape[0], 1))
            coords4d = np.hstack((coordinates, ones))
            return np.dot(coords4d, M)[:, :-1]

        # get the global coordinates of all object bounding box corners
        if objects is None:
            objects = [
                obj
                for obj in bpy.data.objects
                if obj.type == "MESH"
                and "atmosphere" not in obj.name.lower()
                and not obj.hide_render
            ]
        coords = np.vstack(
            tuple(
                np_matmul_coords(np.array(o.bound_box), o.matrix_world.copy())
                for o in objects
            )
        )

        return coords

    def is_in_bbox(self, coord, bbox):
        return not (
            np.any(coord < bbox.min(axis=0)) or np.any(coord > bbox.max(axis=0))
        )

    def line_not_valid(self, p1, p2, bbox=None, dist=None):
        """see if line intersects obstacle"""
        p1_np, p2_np = np.array(p1), np.array(p2)

        if bbox is not None:
            if (not self.is_in_bbox(p2_np, bbox)) or (not self.is_in_bbox(p1_np, bbox)):
                return True

        if dist is None:
            dist = np.linalg.norm(p2_np - p1_np)
        deps = bpy.context.evaluated_depsgraph_get()
        f = bpy.context.scene.ray_cast(deps, p1_np, p2_np - p1_np, distance=dist)
        return f[0]


@gin.configurable
class AnimPolicyRRT:
    def __init__(
        self,
        rrt=None,
        obj_groups=None,
        speed=("uniform", 1.0, 1.0),
        rot=("normal", 0, [20, 20, 20], 3),
        start=None,
        goal=None,
    ):
        self.speed = speed
        self.rot = rot
        self.start = start
        self.goal = goal
        self.rrt = RRT(obj_groups=obj_groups) if rrt is None else rrt
        self.ind = 0

    def reset(self):
        self.path = (
            self.rrt.generate_path(start=self.start, goal=self.goal)
            if self.rrt is not None
            else []
        )
        self.ind = 0

    def retry(self):
        self.ind = max(0, self.ind - 1)

    def __call__(self, obj, frame_curr, bvh, retry_pct=0):
        start = obj.location if self.start is None else self.start

        if retry_pct != 0:
            self.ind = max(0, self.ind - 1)

        if self.ind == 0:
            self.path = self.rrt.generate_path(start=start, goal=self.goal)

        if self.ind >= len(self.path):
            self.path = self.rrt.generate_path(start=start, goal=self.goal)
            self.ind = 0

        if len(self.path) < 1:
            raise PolicyError(f"RRTStar created path of len {len(self.path)} < 1")

        speed = random_general(self.speed)

        if self.ind >= len(self.path) or self.ind < 0:
            raise PolicyError(
                f"index {self.ind} out of bounds for path of len {len(self.path)}"
            )

        pos = Vector(self.path[self.ind])
        time = np.linalg.norm(pos - obj.location) / speed
        rot_diff = np.deg2rad(random_general(self.rot)) if self.ind > 0 else np.zeros(3)
        rot = np.array(obj.rotation_euler) + rot_diff
        self.ind += 1

        return Vector(pos), Vector(rot), time, "BEZIER"


@gin.configurable
def validate_cam_pose_rrt(
    cam,
    max_sky_percent=0.7,  # max percent of sky in view frame
    max_proxim_percent=0.5,  # max percent of view frame too close to cam
    min_obj_dist=None,  # distance that is too close to cam
    min_pixels_check=100,  # min number of pixels to check in view frame for sky/proxim
):
    origin = cam.location
    if not cam.type == "CAMERA":
        cam = cam.children[0]
    if not cam.type == "CAMERA":
        raise ValueError(f"{cam.name=} had {cam.type=}")

    bpy.context.view_layer.update()

    frame = cam.data.view_frame(scene=bpy.context.scene)
    topRight = frame[0]
    bottomLeft = frame[2]
    topLeft = frame[3]
    z = frame[0][2]
    eps = abs(z) if min_obj_dist is None else min_obj_dist

    # number of pixels in viewframe to ray cast through
    resolutionX = bpy.context.scene.render.resolution_x * (
        bpy.context.scene.render.resolution_percentage / 100
    )
    resolutionY = bpy.context.scene.render.resolution_y * (
        bpy.context.scene.render.resolution_percentage / 100
    )
    num_pix_x = ceil((min_pixels_check * resolutionX / resolutionY) ** 0.5)
    num_pix_y = ceil((min_pixels_check * resolutionY / resolutionX) ** 0.5)

    # setup vectors to match pixels
    xRange = np.linspace(topLeft[0], topRight[0], num_pix_x)
    yRange = np.linspace(topLeft[1], bottomLeft[1], num_pix_y)

    deps = bpy.context.evaluated_depsgraph_get()
    sky_hits = 0
    prox_hits = 0
    rotation = cam.matrix_world.to_quaternion()
    num_pixels = len(xRange) * len(yRange)
    sky_threshold = max_sky_percent * num_pixels
    prox_threshold = max_proxim_percent * num_pixels
    for x in xRange:
        for y in yRange:
            # ray cast through view frame
            dir = Vector((x, y, z))
            dir.rotate(rotation)
            hit_obj, loc, *_ = bpy.context.scene.ray_cast(deps, origin, dir)

            if hit_obj:
                if (loc - origin).length < eps:
                    prox_hits += 1
            else:
                sky_hits += 1

            if sky_hits > sky_threshold:
                logger.warning(f"Sky hits {sky_hits / num_pixels}")
                return False
            if prox_hits > prox_threshold:
                logger.warning(f"Prox hits {prox_hits / num_pixels}")
                return False

    return True
