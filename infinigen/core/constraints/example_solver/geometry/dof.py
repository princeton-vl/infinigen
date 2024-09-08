# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import logging

import bpy
import gin
import numpy as np
import trimesh
from mathutils import Vector
from shapely.geometry import Point

import infinigen.core.constraints.example_solver.geometry.validity as validity
import infinigen.core.util.blender as butil
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints.constraint_language import util as iu
from infinigen.core.constraints.example_solver import state_def
from infinigen.core.constraints.example_solver.geometry import stability

logger = logging.getLogger(__name__)


def stable_against_matrix(point, normal):
    """
    Given a point and normal defining a plane, return a 3x3 matrix that
    restricts movement perpendicular to the plane.
    """
    # Normalize the normal vector
    normal = np.array(normal)
    normalized_normal = normal / np.linalg.norm(normal)

    # Create a matrix that restricts movement along the normal direction
    restriction_matrix = np.identity(3) - np.outer(normalized_normal, normalized_normal)
    return restriction_matrix


def combined_stability_matrix(parent_planes):
    """
    Given a list of relations (each a tuple of point and normal),
    compute the combined 3x3 matrix M.
    """

    M = np.identity(3)
    for name, poly in parent_planes:
        obj = bpy.data.objects[name]
        poly = obj.data.polygons[poly]
        point = obj.data.vertices[poly.vertices[0]]
        M = np.dot(M, stable_against_matrix(point, poly.normal))
    return M


def rotation_constraint(normal):
    """
    Given a normal defining a plane, return the axis of rotation allowed by this constraint.
    """
    # Normalize the normal vector
    normal = np.array(normal)
    normalized_normal = normal / np.linalg.norm(normal)

    return normalized_normal


def combine_rotation_constraints(parent_planes, eps=0.01):
    """
    Given a list of normals, compute the combined axis of rotation.
    If there are conflicting constraints, return None.
    """

    normals = [
        bpy.data.objects[name].data.polygons[poly].normal
        for name, poly in parent_planes
    ]

    # Start with the first constraint
    combined_axis = rotation_constraint(normals[0])

    for normal in normals[1:]:
        axis = rotation_constraint(normal)

        # If the axes are not parallel, there's a conflict
        if not np.isclose(combined_axis.dot(axis), 1, atol=eps):
            return None

        # Otherwise, keep the current axis (since they're parallel)

    return combined_axis


def rotate_object_around_axis(obj, axis, std, angle=None):
    """
    Rotate an object around a given axis.
    """
    # Normalize the axis
    axis = np.array(axis)
    normalized_axis = axis / np.linalg.norm(axis)

    # If no angle is provided, generate a random angle between 0 and 2*pi
    if angle is None:
        angle = np.random.normal(0, std)

    obj.rotation_mode = "AXIS_ANGLE"
    obj.rotation_axis_angle = Vector([angle] + list(normalized_axis))


def check_init_valid(
    state: state_def.State,
    name: str,
    obj_planes: list,
    assigned_planes: list,
    margins: list,
    rev_normals: list[bool],
):
    """
    Check that the plane assignments to the object is valid. First checks that the rotations can be satisfied, then
    checks that the translations can be satisfied. Returns a boolean indicating if the assignments are valid, the number
    of degrees of freedom remaining, and the translation vector if the assignments are valid.
    """
    if len(obj_planes) == 0:
        raise ValueError(f"{check_init_valid.__name__} for {name=} got {obj_planes=}")
    if len(obj_planes) > 3:
        raise ValueError(
            f"{check_init_valid.__name__} for {name=} got {len(obj_planes)=}"
        )

    def get_rot(ind):
        """
        Get the rotation axis and angle needed to align the object's plane with the assigned plane.
        """
        try:
            a = obj_planes[ind][0]
            b = assigned_planes[ind][0]
        except IndexError:
            raise ValueError(f"Invalid {ind=} {obj_planes=} {assigned_planes=}")

        a_plane = obj_planes[ind]
        b_plane = assigned_planes[ind]
        rev_normal = rev_normals[ind]
        a_obj = bpy.data.objects[a]
        b_obj = bpy.data.objects[b]

        a_poly_index = a_plane[1]
        a_poly = a_obj.data.polygons[a_poly_index]
        b_poly_index = b_plane[1]
        b_poly = b_obj.data.polygons[b_poly_index]
        plane_normal_a = butil.global_polygon_normal(a_obj, a_poly)
        plane_normal_b = butil.global_polygon_normal(b_obj, b_poly, rev_normal)
        plane_normal_b = -plane_normal_b
        rotation_axis = np.cross(plane_normal_a, plane_normal_b)

        if not np.isclose(np.linalg.norm(rotation_axis), 0, atol=1e-03):
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        else:
            rotation_axis = np.array([0, 0, 1])
        dot = plane_normal_a.dot(plane_normal_b)
        rotation_angle = np.arccos(np.clip(dot, -1, 1))
        if np.isnan(rotation_angle):
            raise ValueError(f"Invalid {rotation_angle=}")
        return a, b, rotation_axis, rotation_angle, plane_normal_b

    def is_rotation_allowed(rotation_axis, reference_normal):
        # Check if rotation axis is the same as the reference normal (with some tolerance)
        res = np.allclose(rotation_axis, reference_normal, atol=1e-02) or np.allclose(
            rotation_axis, -reference_normal, atol=1e-02
        )
        if not res:
            dot = rotation_axis.dot(reference_normal)
            logger.debug(
                f"{is_rotation_allowed.__name__} got {res=} with {rotation_axis=} {reference_normal=} {dot=}"
            )
        return res

    a, b, rotation_axis, rotation_angle, plane_normal_b = get_rot(0)
    iu.rotate(state.trimesh_scene, a, rotation_axis, rotation_angle)
    first_plane_normal = plane_normal_b  # Save the normal of the first plane

    dof_remaining = True  # Degree of freedom remaining after the first alignment

    # Check and apply rotations for subsequent planes
    for i in range(1, len(obj_planes)):
        a, b, rotation_axis, rotation_angle, plane_normal_b = get_rot(i)

        if np.isclose(np.linalg.norm(rotation_angle), 0, atol=1e-01):
            logger.debug(f"no rotation needed for {i=} of {len(obj_planes)}")
            continue

        rot_allowed = is_rotation_allowed(rotation_axis, first_plane_normal)
        if dof_remaining and rot_allowed:
            # Rotate around the normal of the first plane
            iu.rotate(state.trimesh_scene, a, rotation_axis, rotation_angle)
            dof_remaining = False  # No more degrees of freedom remaining
            logger.debug(f"rotated {a=} to satisfy assignment {i=}")
        else:
            logger.debug(
                f"dofs failed for {i=} of {len(obj_planes)=}, {rot_allowed=} {dof_remaining=}"
            )
            return False, None, None

    # Construct the system of linear equations for translation
    A = []
    c = []
    for i in range(len(obj_planes)):
        a_obj_name, a_poly_index = obj_planes[i]
        b_obj_name, b_poly_index = assigned_planes[i]
        margin = margins[i]
        rev_normal = rev_normals[i]

        a_obj = bpy.data.objects[a_obj_name]
        b_obj = bpy.data.objects[b_obj_name]

        a_poly = a_obj.data.polygons[a_poly_index]
        b_poly = b_obj.data.polygons[b_poly_index]

        # Get global coordinates and normals
        plane_point_a = butil.global_vertex_coordinates(
            a_obj, a_obj.data.vertices[a_poly.vertices[0]]
        )
        plane_point_b = butil.global_vertex_coordinates(
            b_obj, b_obj.data.vertices[b_poly.vertices[0]]
        )
        plane_normal_b = butil.global_polygon_normal(b_obj, b_poly, rev_normal)
        plane_point_b += plane_normal_b * margin

        # Append to the matrix A and vector b for Ax = c
        A.append(plane_normal_b)
        c.append(plane_normal_b.dot(plane_point_b - plane_point_a))

    # Solve the linear system
    A = np.array(A)
    c = np.array(c)

    t, residuals, rank, s = np.linalg.lstsq(A, c, rcond=None)
    a_obj_name, a_poly_index = obj_planes[0]

    a_obj = bpy.data.objects[a_obj_name]

    # Check if the solution is valid
    # You can define a threshold to determine if the residuals are acceptable
    # Manually compute residuals if m <= n
    if residuals.size == 0:
        computed_residuals = np.dot(A, t) - c
        residuals_sum = np.sum(computed_residuals**2)
        if residuals_sum < 1e-03:
            return True, A.shape[1] - rank, t  # Solution is valid
        else:
            logger.debug(f"{check_init_valid.__name__} failed with {residuals_sum=}")
            return False, None, None  # Solution is not valid
    else:
        if np.all(residuals < 1e-03):
            return True, A.shape[1] - rank, t  # Solution is valid
        else:
            logger.debug(f"{check_init_valid.__name__} failed with {residuals=}")
            return False, None, None  # No valid solution


def project(points, plane_normal):
    to_2D = trimesh.geometry.plane_transform(origin=(0, 0, 0), normal=plane_normal)
    vertices_2D = trimesh.transformations.transform_points(points, to_2D)[:, :2]
    return vertices_2D


def apply_relations_surfacesample(
    state: state_def.State,
    name: str,
):
    """
    Apply the relation constraints to the object. Place it in the scene according to the constraints.
    """

    obj_state = state.objs[name]
    obj_name = obj_state.obj.name

    def relation_sort_key(relation_state):
        return isinstance(relation_state.relation, cl.CoPlanar)

    obj_state.relations = sorted(obj_state.relations, key=relation_sort_key)

    parent_objs = []
    parent_planes = []
    obj_planes = []
    margins = []
    parent_tag_list = []
    relations = []
    rev_normals = []

    if len(obj_state.relations) == 0:
        raise ValueError(f"Object {name} has no relations")
    elif len(obj_state.relations) > 3:
        raise ValueError(
            f"Object {name} has more than 3 relations, not supported. {obj_state.relations=}"
        )

    for i, relation_state in enumerate(obj_state.relations):
        if isinstance(relation_state.relation, cl.AnyRelation):
            raise ValueError(
                f"Got {relation_state.relation} for {name=} {relation_state.target_name=}"
            )

        parent_obj = state.objs[relation_state.target_name].obj
        obj_plane, parent_plane = state.planes.get_rel_state_planes(
            state, name, relation_state
        )

        if obj_plane is None:
            continue
        if parent_plane is None:
            continue

        obj_planes.append(obj_plane)
        parent_planes.append(parent_plane)
        parent_objs.append(parent_obj)
        match relation_state.relation:
            case cl.StableAgainst(
                _child_tags, parent_tags, margin, _check_z, rev_normal
            ):
                margins.append(margin)
                parent_tag_list.append(parent_tags)
                relations.append(relation_state.relation)
                rev_normals.append(rev_normal)
            case cl.SupportedBy(_parent_tags, parent_tags):
                margins.append(0)
                parent_tag_list.append(parent_tags)
                relations.append(relation_state.relation)
                rev_normals.append(False)
            case cl.CoPlanar(_child_tags, parent_tags, margin, rev_normal):
                margins.append(margin)
                parent_tag_list.append(parent_tags)
                relations.append(relation_state.relation)
                rev_normals.append(rev_normal)
            case cl.Touching(_child_tags, parent_tags, margin):
                margins.append(margin)
                parent_tag_list.append(parent_tags)
                relations.append(relation_state.relation)
                rev_normals.append(False)
            case _:
                raise NotImplementedError

    valid, dof, T = check_init_valid(
        state, name, obj_planes, parent_planes, margins, rev_normals
    )
    if not valid:
        rels = [(rels.relation, rels.target_name) for rels in obj_state.relations]
        logger.warning(f"Init was invalid for {name=} {rels=}")
        return None

    if dof == 0:
        iu.translate(state.trimesh_scene, obj_name, T)
    elif dof == 1:
        assert len(parent_planes) == 2, (name, len(parent_planes))

        parent_obj1 = parent_objs[0]
        parent_obj2 = parent_objs[1]
        parent_plane1 = parent_planes[0]
        parent_plane2 = parent_planes[1]
        parent_tags1 = parent_tag_list[0]
        parent_tags2 = parent_tag_list[1]
        margin1 = margins[0]
        margin2 = margins[1]
        obj_plane1 = obj_planes[0]
        obj_plane2 = obj_planes[1]
        relation2 = relations[1]
        rev_normal1 = rev_normals[0]
        rev_normal2 = rev_normals[1]

        parent1_trimesh = state.planes.get_tagged_submesh(
            state.trimesh_scene, parent_obj1.name, parent_tags1, parent_plane1
        )
        parent2_trimesh = state.planes.get_tagged_submesh(
            state.trimesh_scene, parent_obj2.name, parent_tags2, parent_plane2
        )

        parent1_poly_index = parent_plane1[1]
        parent1_poly = parent_obj1.data.polygons[parent1_poly_index]
        plane_normal_1 = butil.global_polygon_normal(parent_obj1, parent1_poly)
        pts = parent2_trimesh.vertices
        projected = project(pts, plane_normal_1)
        p1_to_p1 = trimesh.path.polygons.projected(
            parent1_trimesh, plane_normal_1, (0, 0, 0)
        )

        if p1_to_p1 is None:
            raise ValueError(
                f"Failed to project {parent1_trimesh=} {plane_normal_1=} for {name=}"
            )

        if all(
            [p1_to_p1.buffer(1e-1).contains(Point(pt[0], pt[1])) for pt in projected]
        ) and (not isinstance(relation2, cl.CoPlanar)):
            face_mask = tagging.tagged_face_mask(parent_obj2, parent_tags2)
            stability.move_obj_random_pt(
                state, obj_name, parent_obj2.name, face_mask, parent_plane2
            )
            stability.snap_against(
                state.trimesh_scene,
                obj_name,
                parent_obj2.name,
                obj_plane2,
                parent_plane2,
                margin=margin2,
                rev_normal=rev_normal2,
            )
            stability.snap_against(
                state.trimesh_scene,
                obj_name,
                parent_obj1.name,
                obj_plane1,
                parent_plane1,
                margin=margin1,
                rev_normal=rev_normal1,
            )
        else:
            face_mask = tagging.tagged_face_mask(parent_obj1, parent_tags1)
            stability.move_obj_random_pt(
                state, obj_name, parent_obj1.name, face_mask, parent_plane1
            )
            stability.snap_against(
                state.trimesh_scene,
                obj_name,
                parent_obj1.name,
                obj_plane1,
                parent_plane1,
                margin=margin1,
                rev_normal=rev_normal1,
            )
            stability.snap_against(
                state.trimesh_scene,
                obj_name,
                parent_obj2.name,
                obj_plane2,
                parent_plane2,
                margin=margin2,
                rev_normal=rev_normal2,
            )

    elif dof == 2:
        # assert len(parent_planes) == 1, (name, len(parent_planes))
        for i, relation_state in enumerate(obj_state.relations):
            parent_obj = state.objs[relation_state.target_name].obj
            obj_plane, parent_plane = state.planes.get_rel_state_planes(
                state, name, relation_state
            )
            if obj_plane is None:
                continue
            if parent_plane is None:
                continue
            iu.set_rotation(
                state.trimesh_scene,
                obj_name,
                (0, 0, 2 * np.pi * np.random.randint(0, 4) / 4),
            )
            face_mask = tagging.tagged_face_mask(
                parent_obj, relation_state.relation.parent_tags
            )

            match relation_state.relation:
                case cl.CoPlanar:
                    stability.snap_against(
                        state.trimesh_scene,
                        obj_name,
                        parent_obj.name,
                        obj_plane,
                        parent_plane,
                        margin=margin,
                        rev_normal=relation_state.relation.rev_normal,
                    )

                case cl.StableAgainst(_, parent_tags, margin, _check_z, rev_normal):
                    stability.move_obj_random_pt(
                        state, obj_name, parent_obj.name, face_mask, parent_plane
                    )
                    stability.snap_against(
                        state.trimesh_scene,
                        obj_name,
                        parent_obj.name,
                        obj_plane,
                        parent_plane,
                        margin=margin,
                        rev_normal=rev_normal,
                    )

                case cl.SupportedBy(_, parent_tags):
                    stability.move_obj_random_pt(
                        state, obj_name, parent_obj.name, face_mask, parent_plane
                    )
                    stability.snap_against(
                        state.trimesh_scene,
                        obj_name,
                        parent_obj.name,
                        obj_plane,
                        parent_plane,
                        margin=0,
                        rev_normal=False,
                    )
                case _:
                    raise NotImplementedError
    return parent_planes


def validate_relations_feasible(state: state_def.State, name: str) -> bool:
    assignments = state.objs[name].relations
    targets = [rel.target_name for rel in assignments]

    rooms = {targ for targ in targets if t.Semantics.Room in state.objs[targ].tags}
    if len(rooms) > 1:
        raise ValueError(f"Object {name} has multiple room targets {rooms}")


@gin.configurable
def try_apply_relation_constraints(
    state: state_def.State, name: str, n_try_resolve=10, visualize=False
):
    """
    name is in objs.name
    name has been recently reassigned or added or swapped
    it needs snapping, and dof updates

    Result:
    dof_mat and dof axis for name are updated
    objstate for name has update location rotaton etc

    """

    validate_relations_feasible(state, name)

    for retry in range(n_try_resolve):
        obj_state = state.objs[name]
        if iu.blender_objs_from_names(obj_state.obj.name)[0].dimensions[2] > 2.5:
            logger.warning(
                f"Object {obj_state.obj.name} is too tall for the room: {obj_state.obj.dimensions[2]}, 2.5"
            )
        parent_planes = apply_relations_surfacesample(state, name)

        # assignments not valid
        if parent_planes is None:
            logger.debug(f"Found {parent_planes=} for {name=} {retry=}")
            if visualize:
                vis = butil.copy(obj_state.obj)
                vis.name = obj_state.obj.name[:30] + "_noneplanes_" + str(retry)
                butil.save_blend("test.blend")
            return False

        if validity.check_post_move_validity(state, name):
            obj_state.dof_matrix_translation = combined_stability_matrix(parent_planes)
            obj_state.dof_rotation_axis = combine_rotation_constraints(parent_planes)
            return True

        if visualize:
            vis = butil.copy(obj_state.obj)
            vis.name = obj_state.obj.name[:30] + "_failure_" + str(retry)
            butil.save_blend("test.blend")

    logger.debug(f"Exhausted {n_try_resolve=} tries for {name=}")
    return False
