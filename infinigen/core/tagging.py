# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Yihan Wang, Karhan Kayan: face based tagging, canonical surface tagging, mask extraction, support tag


import json
import logging
from typing import Union

import bpy
import numpy as np
from mathutils import Vector

import infinigen.core.util.blender as butil
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util.logging import lazydebug

from . import tags as t

logger = logging.getLogger(__name__)

PREFIX = "TAG_"
COMBINED_ATTR_NAME = "MaskTag"


class AutoTag:
    tag_dict = {}

    def __init__(self):
        self.tag_dict = {}

    def clear(self):
        self.tag_dict = {}

    # This function now only supports APPLIED OBJECTS
    # PLEASE KEEP ALL THE GEOMETRY APPLIED BEFORE SCATTERING THEM ON THE TERRAIN
    # PLEASE DO NOT USE BOOLEAN TAGS FOR OTHER USE
    def save_tag(self, path="./MaskTag.json"):
        with open(path, "w") as f:
            json.dump(self.tag_dict, f)

    def load_tag(self, path="./MaskTag.json"):
        with open(path, "r") as f:
            self.tag_dict = json.load(f)

    def _extract_incoming_tagmasks(self, obj):
        new_attr_names = [
            name for name in obj.data.attributes.keys() if name.startswith(PREFIX)
        ]

        n_poly = len(obj.data.polygons)
        for name in new_attr_names:
            attr = obj.data.attributes[name]
            if attr.domain != "FACE":
                raise ValueError(
                    f"Incoming attribute {obj.name=} {attr.name=} had invalid {attr.domain=}, expected FACE"
                )
            if len(attr.data) != n_poly:
                raise ValueError(
                    f"Incoming attribute {obj.name=} {attr.name=} had invalid {len(attr.data)=}, expected {n_poly=}"
                )

        new_attrs = {
            name[len(PREFIX) :]: surface.read_attr_data(obj, name, "FACE")
            for name in new_attr_names
        }

        for name, vals in new_attrs.items():
            if vals.dtype == bool:
                continue
            elif vals.dtype.kind == "f":
                new_attrs[name] = vals > 0.5
            elif vals.dtype.kind == "i":
                new_attrs[name] = vals > 0
            else:
                raise ValueError(
                    f"Incoming attribute {obj.name=} had invalid np dtype {vals.dtype} {vals.dtype.kind=}, expected float or ideally boolean "
                )

        for name, arr in new_attrs.items():
            if arr.dtype != bool:
                raise ValueError(
                    f"Retrieved incoming tag mask {name=} had {arr.dtype=}, expected bool"
                )

        for name in new_attr_names:
            obj.data.attributes.remove(obj.data.attributes[name])

        return new_attrs

    def _specialize_tag_name(self, vi, name, tag_name_lookup):
        if "." in name:
            raise ValueError(f'{name=} should not contain separator character "."')

        if vi == 0:
            return name

        existing = tag_name_lookup[vi - 1]
        parts = set(existing.split("."))

        if name in parts:
            return existing

        parts.add(name)
        return ".".join(sorted(list(parts)))

    def _relabel_obj_single(self, obj, tag_name_lookup):
        n_poly = len(obj.data.polygons)
        new_attrs = self._extract_incoming_tagmasks(obj)

        if COMBINED_ATTR_NAME in obj.data.attributes.keys():
            domain = obj.data.attributes[COMBINED_ATTR_NAME].domain
            if domain != "FACE":
                raise ValueError(
                    f"{obj.name=} had {COMBINED_ATTR_NAME} on {domain=}, expected FACE"
                )
            tagint = surface.read_attr_data(obj, COMBINED_ATTR_NAME, domain="FACE")
        else:
            tagint = np.full(n_poly, 0, np.int64)

        assert tagint.dtype == np.int64, tagint.dtype

        for name, new_mask in new_attrs.items():
            affected_tagints = np.unique(tagint[new_mask])

            for vi in affected_tagints:
                affected_mask = new_mask * (tagint == vi)
                if not affected_mask.any():
                    continue

                new_tag_name = self._specialize_tag_name(vi, name, tag_name_lookup)
                tag_value = self.tag_dict.get(new_tag_name)

                if tag_value is None:
                    tag_value = len(self.tag_dict) + 1
                    self.tag_dict[new_tag_name] = tag_value
                    tag_name_lookup.append(new_tag_name)

                assert (
                    len(self.tag_dict) == len(tag_name_lookup)
                ), f"{len(self.tag_dict)=} yet {len(tag_name_lookup)=}, out of sync at {vi=} {new_tag_name=}"
                assert new_tag_name in tag_name_lookup

                lazydebug(
                    logger,
                    lambda: f"{self._relabel_obj_single.__name__} updating {vi=} to {new_tag_name=} with {affected_mask.mean()=:.2f} for {obj.name=}",
                )

                tagint[affected_mask] = tag_value

        if COMBINED_ATTR_NAME not in obj.data.attributes.keys():
            mask_tag_attr = obj.data.attributes.new(COMBINED_ATTR_NAME, "INT", "FACE")
        else:
            mask_tag_attr = obj.data.attributes[COMBINED_ATTR_NAME]

        mask_tag_attr.data.foreach_set("value", tagint)

    def relabel_obj(self, root_obj):
        tag_name_lookup = [None] * len(self.tag_dict)

        for name, tag_id in self.tag_dict.items():
            key = tag_id - 1
            if key >= len(tag_name_lookup):
                raise IndexError(
                    f"{name} had {tag_id=} {key=} yet {len(self.tag_dict)=}"
                )
            if tag_name_lookup[key] is not None:
                raise ValueError(
                    f"{name=} {tag_id=} {key=} attempted to overwrite {tag_name_lookup[key]=}"
                )
            tag_name_lookup[key] = name

        for obj in butil.iter_object_tree(root_obj):
            if obj.type != "MESH":
                continue
            self._relabel_obj_single(obj, tag_name_lookup)

        return root_obj


tag_system = AutoTag()


def print_segments_summary(obj: bpy.types.Object):
    tagint = surface.read_attr_data(obj, COMBINED_ATTR_NAME, domain="FACE")

    results = []
    for vi in np.unique(tagint):
        mask = tagint == vi
        results.append((vi, mask.mean()))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"Tag Segments Summary for {obj.name=}")
    for vi, mean in results:
        name = _name_for_tagval(vi)
        print(f"  {mean*100:.1f}% {vi=} {name}")


def tag_object(obj, name=None, mask=None):
    if name is not None:
        name = t.to_string(name)

    for o in butil.iter_object_tree(obj):
        if o.type != "MESH":
            continue

        if name is not None:
            n_poly = len(o.data.polygons)

            if n_poly == 0:
                lazydebug(
                    logger,
                    lambda: f"{tag_object.__name__} had {n_poly=} for {o.name=} {name=} child of {obj.name=}",
                )
                continue

            mask_o = np.full(n_poly, 1, dtype=bool) if mask is None else mask

            assert isinstance(mask_o, np.ndarray)
            assert len(mask_o) == n_poly

            lazydebug(
                logger,
                lambda: f"{tag_object.__name__} applying {name=} {mask_o.mean()=:.2f} to {o.name=}",
            )
            surface.write_attr_data(
                obj=o, attr=(PREFIX + name), data=mask_o, type="BOOLEAN", domain="FACE"
            )

        tag_system.relabel_obj(obj)


def vert_mask_to_tri_mask(obj, vert_mask, require_all=True):
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("vertices", arr)
    face_vert_idxs = arr.reshape(-1, 3).astype(int)

    if require_all:
        return (
            vert_mask[face_vert_idxs[:, 0]]
            * vert_mask[face_vert_idxs[:, 1]]
            * vert_mask[face_vert_idxs[:, 2]]
        )
    else:
        return (
            vert_mask[face_vert_idxs[:, 0]]
            | vert_mask[face_vert_idxs[:, 1]]
            | vert_mask[face_vert_idxs[:, 2]]
        )


CANONICAL_TAGS = [t.Subpart.Back, t.Subpart.Front, t.Subpart.Top, t.Subpart.Bottom]
CANONICAL_TAG_MEANINGS = {
    t.Subpart.Back: (np.min, 0),
    t.Subpart.Front: (np.max, 0),
    t.Subpart.Bottom: (np.min, 2),
    t.Subpart.Top: (np.max, 2),
}


def tag_canonical_surfaces(obj, rtol=0.01):
    obj.update_from_editmode()

    n_vert = len(obj.data.vertices)
    len(obj.data.polygons)

    verts = np.empty(n_vert * 3, dtype=float)
    obj.data.vertices.foreach_get("co", verts)
    verts = verts.reshape(n_vert, 3)

    for tag in CANONICAL_TAGS:
        gather_func, axis_idx = CANONICAL_TAG_MEANINGS[tag]
        target_axis_val = gather_func(verts[:, axis_idx])

        atol = rtol * obj.dimensions[axis_idx]
        vert_mask = np.isclose(verts[:, axis_idx], target_axis_val, atol=atol)

        face_mask = vert_mask_to_tri_mask(obj, vert_mask, require_all=True)

        if not face_mask.any():
            logger.warning(
                f"{tag_canonical_surfaces.__name__} found got {face_mask.mean()=:.2f} for {tag=} on {obj.name=}"
            )

        lazydebug(
            logger,
            lambda: f"{tag_canonical_surfaces.__name__} applying {tag=} {face_mask.mean()=:.2f} to {obj.name=}",
        )
        surface.write_attr_data(
            obj, PREFIX + tag.value, face_mask, type="BOOLEAN", domain="FACE"
        )

    tag_system.relabel_obj(obj)


def tag_nodegroup(nw: NodeWrangler, input_node, name: t.Tag, selection=None):
    name = PREFIX + t.to_string(name)
    sel = surface.eval_argument(nw, selection)
    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": input_node,
            "Name": name,
            "Selection": sel,
            "Value": True,
        },
        attrs={"domain": "FACE", "data_type": "BOOLEAN"},
    )
    return store_named_attribute


def _name_for_tagval(i: int) -> str | None:
    if i == 0:
        # index 0 represents an untagged face
        return None

    name = next((k for k, v in tag_system.tag_dict.items() if v == i), None)

    if name is None:
        raise ValueError(f"Found {name=} for {i=} in {tag_system.tag_dict=}")

    return name


def union_object_tags(obj):
    if COMBINED_ATTR_NAME not in obj.data.attributes:
        return set()

    masktag = surface.read_attr_data(obj, COMBINED_ATTR_NAME)
    res = set()
    for v in np.unique(masktag):
        if v == 0:
            continue
        res = res.union(_name_for_tagval(v).split("."))

    def try_convert(x):
        try:
            return t.to_tag(x)
        except ValueError:
            return x

    return {try_convert(x) for x in res}


def tagged_face_mask(obj: bpy.types.Object, tags: Union[t.Subpart]) -> np.ndarray:
    # ASSUMES: object is triangulated, no quads/polygons

    tags = t.to_tag_set(tags)
    pos_tags = [
        t.to_string(tagval) for tagval in tags if not isinstance(tagval, t.Negated)
    ]
    neg_tags = [
        t.to_string(tagval.tag) for tagval in tags if isinstance(tagval, t.Negated)
    ]
    del tags

    n_poly = len(obj.data.polygons)
    if COMBINED_ATTR_NAME not in obj.data.attributes:
        return np.ones(n_poly, dtype=bool)
    masktag = surface.read_attr_data(obj, COMBINED_ATTR_NAME, domain="FACE")
    face_mask = np.zeros(n_poly, dtype=bool)

    for v in np.unique(masktag):
        if v == 0:
            name_parts = []
        else:
            name_parts = _name_for_tagval(v).split(".")

        v_mask = masktag == v

        if len(pos_tags) > 0 and not all(tag in name_parts for tag in pos_tags):
            continue
        if len(neg_tags) > 0 and any(tag in name_parts for tag in neg_tags):
            continue

        face_mask |= v_mask

    lazydebug(
        logger,
        lambda: f"{obj.name=} had {face_mask.mean()=:.2f} for {pos_tags=} {neg_tags=}",
    )

    return face_mask


def extract_tagged_faces(
    obj: bpy.types.Object, tags: set, nonempty=False
) -> bpy.types.Object:
    "extract the surface that satisfies all tags"

    # Ensure we're dealing with a mesh object
    if obj.type != "MESH":
        raise TypeError("Object is not a mesh!")

    face_mask = tagged_face_mask(obj, tags)

    if nonempty and not face_mask.any():
        raise ValueError(
            f"extract_tagged_faces({obj.name=}, {tags=}, {nonempty=}) got empty mask for {len(obj.data.polygons)}"
        )

    return extract_mask(obj, face_mask, nonempty=nonempty)


def extract_mask(
    obj: bpy.types.Object, face_mask: np.array, nonempty=False
) -> bpy.types.Object:
    if not face_mask.any():
        if nonempty:
            raise ValueError(f"extract_mask({obj.name=}) got empty mask")
        return butil.spawn_vert()

    orig_hide_viewport = obj.hide_viewport
    obj.hide_viewport = False

    # Switch to Edit mode, duplicate the selection, and separate it
    with butil.SelectObjects(obj, active=0):
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="FACE")
            bpy.ops.mesh.select_all(action="DESELECT")

        for poly in obj.data.polygons:
            poly.select = face_mask[poly.index]
        if nonempty and len([p for p in obj.data.polygons if p.select]) == 0:
            raise ValueError(
                f"extract_mask({obj.name=}, {nonempty=}) failed to select polygons"
            )

        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")

        res = next((o for o in bpy.context.selected_objects if o != obj), None)

    obj.hide_viewport = orig_hide_viewport

    if nonempty:
        if res is None:
            raise ValueError(
                f"extract_mask({obj.name=}) got {res=} for {face_mask.mean()=}"
            )
        if len(res.data.polygons) == 0:
            raise ValueError(
                f"extract_mask({obj.name=}) got {res=} with {len(res.data.polygons)=}"
            )
    elif res is None:
        logger.warning(f"extract_mask({obj.name=}) failed to extract any faces")
        return butil.spawn_vert()

    return res


def tag_support_surfaces(obj, angle_threshold=0.1):
    """
    Tags faces of the object (or its mesh children) whose normal is close to the +z direction with the "support" tag.

    Args:
    obj (bpy.types.Object): The object to tag (can be any type of object).
    angle_threshold (float): The cosine of the maximum angle deviation from +z to be considered a support surface.
    """

    def process_mesh(mesh_obj):
        up_vector = Vector((0, 0, 1))

        n_poly = len(mesh_obj.data.polygons)
        support_mask = np.zeros(n_poly, dtype=bool)

        for poly in mesh_obj.data.polygons:
            global_normal = butil.global_polygon_normal(mesh_obj, poly)
            if global_normal.dot(up_vector) > 1 - angle_threshold:
                support_mask[poly.index] = True

        if t.Subpart.SupportSurface.value not in tag_system.tag_dict:
            tag_system.tag_dict[t.Subpart.SupportSurface.value] = (
                len(tag_system.tag_dict) + 1
            )

        tag_object(mesh_obj, name=t.Subpart.SupportSurface.value, mask=support_mask)

        print(
            f"Tagged {support_mask.sum()} faces as 'support' in object {mesh_obj.name}"
        )

    def process_object(obj):
        if obj.type == "MESH":
            process_mesh(obj)
        for child in obj.children:
            process_object(child)

    process_object(obj)
