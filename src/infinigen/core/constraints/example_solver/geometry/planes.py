# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

from __future__ import annotations

import logging

import bpy
import gin
import numpy as np
import trimesh

import infinigen.core.util.blender as butil
from infinigen.core import tagging
from infinigen.core import tags as t
from infinigen.core.constraints.constraint_language.util import (
    blender_objs_from_names,
    meshes_from_names,
)

logger = logging.getLogger(__name__)


class Planes:
    def __init__(self):
        self._mesh_hashes = {}  # Dictionary to store mesh hashes for each object
        self._cached_planes = {}  # Dictionary to store computed planes, keyed by object and face_mask hash
        self._cached_plane_masks = {}  # Dictionary to store computed plane masks, keyed by object, plane, and face_mask hash

    def calculate_mesh_hash(self, obj):
        # Simple hash based on counts of vertices, edges, and polygons
        mesh = obj.data
        hash_str = (
            f"{obj.name}_{len(mesh.vertices)}_{len(mesh.edges)}_{len(mesh.polygons)}"
        )
        return hash(hash_str)

    def hash_face_mask(self, face_mask):
        # Hash the face_mask to use as part of the key for caching
        return hash(face_mask.tostring())

    def get_all_planes_cached(self, obj, face_mask, tolerance=1e-4):
        current_mesh_hash = self.calculate_mesh_hash(obj)
        current_face_mask_hash = self.hash_face_mask(face_mask)
        cache_key = (obj.name, current_face_mask_hash)

        # Check if mesh has been modified or planes have not been computed before for this object and face_mask
        if (
            cache_key not in self._cached_planes
            or self._mesh_hashes.get(obj.name) != current_mesh_hash
        ):
            self._mesh_hashes[obj.name] = (
                current_mesh_hash  # Update the hash for this object
            )
            # Recompute planes for this object and face_mask and update cache
            # logger.info(f'Cache MISS planes for {obj.name=}')
            self._cached_planes[cache_key] = self.compute_all_planes_fast(
                obj, face_mask, tolerance
            )

        # logger.info(f'Cache HIT planes for {obj.name=}')
        return self._cached_planes[cache_key]

    @staticmethod
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @staticmethod
    def hash_plane(normal, point, tolerance=1e-4):
        normal_normalized = normal / np.linalg.norm(normal)
        distance = np.dot(normal_normalized, point)
        return (
            tuple(np.round(normal_normalized / tolerance).astype(int)),
            round(distance / tolerance),
        )

    def compute_all_planes_fast(self, obj, face_mask, tolerance=1e-4):
        # Cache computations

        vertex_cache = {
            v.index: butil.global_vertex_coordinates(obj, v) for v in obj.data.vertices
        }
        normal_cache = {
            p.index: butil.global_polygon_normal(obj, p)
            for p in obj.data.polygons
            if face_mask[p.index]
        }

        unique_planes = {}

        for polygon in obj.data.polygons:
            if not face_mask[polygon.index]:
                continue

            # Get the normal and a vertex to represent the plane
            normal = normal_cache[polygon.index]

            if np.linalg.norm(normal) < 1e-6:
                continue

            vertex = vertex_cache[polygon.vertices[0]]

            # Hash the plane using both normal and the point
            plane_hash = self.hash_plane(normal, vertex, tolerance)

            if plane_hash not in unique_planes:
                unique_planes[plane_hash] = (obj.name, polygon.index)

        return list(unique_planes.values())

    def get_all_planes_deprecated(
        self, obj, face_mask, tolerance=1e-4
    ) -> tuple[str, int]:
        "get all unique planes formed by faces in face_mask"
        # ASSUMES: object is triangulated, no quads/polygons
        unique_planes = []
        for polygon in obj.data.polygons:
            if not face_mask[polygon.index]:
                continue
            vertex = butil.global_vertex_coordinates(
                obj, obj.data.vertices[polygon.vertices[0]]
            )
            normal = butil.global_polygon_normal(obj, polygon)
            belongs_to_existing_plane = False
            for name, polygon2_index in unique_planes:
                polygon2 = obj.data.polygons[polygon2_index]
                plane_vertex = butil.global_vertex_coordinates(
                    obj, obj.data.vertices[polygon2.vertices[0]]
                )
                plane_normal = butil.global_polygon_normal(obj, polygon2)
                if np.allclose(
                    np.cross(normal, plane_normal), 0, rtol=tolerance
                ) and np.allclose(
                    np.dot(vertex - plane_vertex, plane_normal), 0, rtol=tolerance
                ):
                    belongs_to_existing_plane = True
                    break
            if (
                not belongs_to_existing_plane
                and polygon.normal
                and polygon.normal.length > 0
            ):
                unique_planes.append((obj.name, polygon.index))
        return unique_planes

    @gin.configurable
    def get_tagged_planes(self, obj: bpy.types.Object, tags: set, fast=True):
        """
        get all unique planes formed by faces tagged with tags
        """

        tags = t.to_tag_set(tags)

        mask = tagging.tagged_face_mask(obj, tags)
        if not mask.any():
            obj_tags = tagging.union_object_tags(obj)
            logger.warning(
                f"Attempted to get_tagged_planes {obj.name=} {tags=} but mask was empty, {obj_tags=}"
            )
            return []

        if fast:
            planes = self.get_all_planes_cached(obj, mask)
        else:
            planes = self.compute_all_planes_fast(obj, mask)
        return planes

    def get_rel_state_planes(self, state, name: str, relation_state: tuple):
        obj = state.objs[name].obj
        relation = relation_state.relation

        parent_obj = state.objs[relation_state.target_name].obj
        obj_tags = relation.child_tags
        parent_tags = relation.parent_tags

        parent_all_planes = self.get_tagged_planes(parent_obj, parent_tags)
        obj_all_planes = self.get_tagged_planes(obj, obj_tags)

        # for i, p in enumerate(parent_all_planes):
        #    splitted_parent = planes.extract_tagged_plane(parent_obj, parent_tags, p)
        #    splitted_parent.name = f'parent_plane_{i}'
        # for i, p in enumerate(obj_all_planes):
        #    splitted_parent = planes.extract_tagged_plane(parent_obj, obj_tags, p)
        #    splitted_parent.name = f'obj_plane_{i}'
        # return

        if relation_state.parent_plane_idx >= len(parent_all_planes):
            logging.warning(
                f"{parent_obj.name=} had too few planes ({len(parent_all_planes)}) for {relation_state}"
            )
            parent_plane = None
        else:
            parent_plane = parent_all_planes[relation_state.parent_plane_idx]

        if relation_state.child_plane_idx >= len(obj_all_planes):
            logging.warning(
                f"{obj.name=} had too few planes ({len(obj_all_planes)}) for {relation_state}"
            )
            obj_plane = None
        else:
            obj_plane = obj_all_planes[relation_state.child_plane_idx]

        return obj_plane, parent_plane

    @staticmethod
    def planerep_to_poly(planerep):
        name, idx = planerep
        return bpy.data.objects[name].data.polygons[idx]

    def extract_tagged_plane(self, obj: bpy.types.Object, tags: set, plane: int):
        """
        get a single plane formed by faces tagged with tags
        """

        if obj.type != "MESH":
            raise TypeError("Object is not a mesh!")

        face_mask = tagging.tagged_face_mask(obj, tags)
        mask = self.tagged_plane_mask(obj, face_mask, plane)

        if not mask.any():
            obj_tags = tagging.union_object_tags(obj)
            logger.warning(
                f"Attempted to extract_tagged_plane {obj.name=} {tags=} but mask was empty, {obj_tags=}"
            )

        butil.select(obj)
        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="FACE")
        bpy.ops.mesh.select_all(action="DESELECT")
        # Set initial selection for polygons to False
        bpy.ops.object.mode_set(mode="OBJECT")

        for poly in obj.data.polygons:
            poly.select = mask[poly.index]

        # Switch to Edit mode, duplicate the selection, and separate it
        old_set = set(bpy.data.objects[:])
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.duplicate()
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        new_set = set(bpy.data.objects[:]) - old_set
        return new_set.pop()

    def get_tagged_submesh(
        self, scene: trimesh.Scene, name: str, tags: set, plane: int
    ):
        obj = blender_objs_from_names(name)[0]
        face_mask = tagging.tagged_face_mask(obj, tags)
        mask = self.tagged_plane_mask(obj, face_mask, plane)
        tmesh = meshes_from_names(scene, name)[0]
        geom = tmesh.submesh(np.where(mask), append=True)
        return geom

    def tagged_plane_mask(
        self,
        obj: bpy.types.Object,
        face_mask: np.ndarray,
        plane: tuple[str, int],
        hash_tolerance=1e-4,
        plane_tolerance=1e-2,
        fast=True,
    ) -> np.ndarray:
        if not fast:
            return self._compute_tagged_plane_mask(
                obj, face_mask, plane, plane_tolerance
            )
        obj_id = obj.name
        current_hash = self.calculate_mesh_hash(obj)  # Calculate current mesh hash
        face_mask_hash = self.hash_face_mask(face_mask)  # Calculate hash for face_mask
        ref_poly = self.planerep_to_poly(plane)
        ref_vertex = butil.global_vertex_coordinates(
            obj, obj.data.vertices[ref_poly.vertices[0]]
        )
        ref_normal = butil.global_polygon_normal(obj, ref_poly)
        plane_hash = self.hash_plane(
            ref_normal, ref_vertex, hash_tolerance
        )  # Calculate hash for plane

        # Composite key now includes face_mask_hash
        cache_key = (obj_id, plane_hash, face_mask_hash)

        # Check if the mesh has been modified since last calculation or if the face mask has changed
        mesh_or_face_mask_changed = (
            cache_key not in self._cached_plane_masks
            or self._mesh_hashes.get(obj_id) != current_hash
        )

        if not mesh_or_face_mask_changed:
            # logger.info(f'Cache HIT plane mask for {obj.name=}')
            return self._cached_plane_masks[cache_key]["mask"]

        # If mesh or face mask changed, update the hash and recompute
        self._mesh_hashes[obj_id] = current_hash

        # Compute and cache the plane mask
        # logger.info(f'Cache MISS plane mask for {obj.name=}')
        plane_mask = self._compute_tagged_plane_mask(
            obj, face_mask, plane, plane_tolerance
        )

        # Update the cache with the new result
        self._cached_plane_masks[cache_key] = {
            "mask": plane_mask,
        }

        return plane_mask

    def _compute_tagged_plane_mask(self, obj, face_mask, plane, tolerance):
        """
        Given a plane, return a mask of all polygons in obj that are coplanar with the plane.
        """
        plane_mask = np.zeros(len(obj.data.polygons), dtype=bool)
        ref_poly = self.planerep_to_poly(plane)
        ref_vertex = butil.global_vertex_coordinates(
            obj, obj.data.vertices[ref_poly.vertices[0]]
        )
        ref_normal = butil.global_polygon_normal(obj, ref_poly)

        for candidate_polygon in obj.data.polygons:
            if not face_mask[candidate_polygon.index]:
                continue

            candidate_vertex = butil.global_vertex_coordinates(
                obj, obj.data.vertices[candidate_polygon.vertices[0]]
            )
            candidate_normal = butil.global_polygon_normal(obj, candidate_polygon)
            diff_vec = ref_vertex - candidate_vertex
            if not np.isclose(np.linalg.norm(diff_vec), 0):
                diff_vec /= np.linalg.norm(diff_vec)

            ndot = np.dot(ref_normal, candidate_normal)
            pdot = np.dot(diff_vec, candidate_normal)

            in_plane = np.allclose(ndot, 1, atol=tolerance) and np.allclose(
                pdot, 0, atol=tolerance
            )

            plane_mask[candidate_polygon.index] = in_plane

        return plane_mask
