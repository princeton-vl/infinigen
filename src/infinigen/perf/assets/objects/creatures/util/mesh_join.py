# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Data-level stand-ins for two Blender operators used while assembling creatures.

`origin_to_geometry` and `join_objects_exact` match `origin_set` and `join` bit for bit,
without their dependency-graph updates and datablock scans. Inputs they do not cover,
such as shape keys or vertex groups, go through the operator.
"""

import logging

import bpy
import numpy as np

from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)

# attribute types whose CustomData default is all-zero, which is what a mesh lacking the
# layer contributes to a join: RNA field name, width, numpy dtype
_ATTR_TYPES = {
    "FLOAT": ("value", 1, np.float32),
    "INT": ("value", 1, np.int32),
    "INT8": ("value", 1, np.int32),
    "BOOLEAN": ("value", 1, bool),
    "FLOAT_VECTOR": ("vector", 3, np.float32),
    "FLOAT2": ("vector", 2, np.float32),
    "INT32_2D": ("value", 2, np.int32),
}
_STRUCTURAL = {"position", ".edge_verts", ".corner_vert", ".corner_edge"}
_SELECT = {
    ".select_vert": "vertices",
    ".select_edge": "edges",
    ".select_poly": "polygons",
}
_DOMAIN = {"POINT": "vertices", "EDGE": "edges", "FACE": "polygons", "CORNER": "loops"}


class _UseOperator(Exception):
    """Raised when an input needs the operator."""


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _mul_m4_v3(mat, co):
    """Blender's `mul_m4_v3` on float32 positions, summed in the C code's order."""
    m = _f32(mat)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    out = np.empty_like(co)
    for r in range(3):
        t = x * m[r, 0]
        t = t + y * m[r, 1]
        t = t + m[r, 2] * z
        t = t + m[r, 3]
        out[:, r] = t
    return out


# ---------------------------------------------------------------------------
# origin_set


def _identity_local_basis(obj):
    """True when the 3x3 part of `BKE_object_to_mat4` is the identity, so the offset the
    operator adds to the location is the geometry center itself."""
    if tuple(obj.scale) != (1.0, 1.0, 1.0) or tuple(obj.delta_scale) != (1.0, 1.0, 1.0):
        return False
    mode = obj.rotation_mode
    if mode == "QUATERNION":
        return tuple(obj.rotation_quaternion) == (1.0, 0.0, 0.0, 0.0) and tuple(
            obj.delta_rotation_quaternion
        ) == (1.0, 0.0, 0.0, 0.0)
    if tuple(obj.delta_rotation_euler) != (0.0, 0.0, 0.0):
        return False
    if mode == "AXIS_ANGLE":
        return obj.rotation_axis_angle[0] == 0.0
    return tuple(obj.rotation_euler) == (0.0, 0.0, 0.0)


def _origin_operator(obj):
    with butil.SelectObjects(obj):
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")


def _read_co(mesh):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def origin_to_geometry(obj, no_children=False):
    """`origin_set(type="ORIGIN_GEOMETRY")` (median center) for one mesh object, with
    the operator's float32 arithmetic and no dependency-graph updates. `no_children`
    skips the children scan when the caller knows there are none."""
    fast = (
        obj.type == "MESH"
        and obj.mode == "OBJECT"
        and obj.data.shape_keys is None
        and obj.data.users == 1
        and len(obj.data.vertices) > 0
        and _identity_local_basis(obj)
        and (no_children or not obj.children)
    )
    if not fast:
        _origin_operator(obj)
        return
    mesh = obj.data
    co = _read_co(mesh)
    cent = np.cumsum(co, axis=0, dtype=np.float32)[-1]
    cent = cent * (np.float32(1.0) / np.float32(len(co)))
    if np.any(cent == 0):
        # only the sign of a zero could differ from the operator; let it decide
        _origin_operator(obj)
        return
    mesh.vertices.foreach_set("co", (co + (-cent)).ravel())
    loc = _f32(obj.location) + cent
    obj.location = (float(loc[0]), float(loc[1]), float(loc[2]))


# ---------------------------------------------------------------------------
# join


def _read_mesh(obj):
    mesh = obj.data
    n_v, n_e, n_l, n_f = (
        len(mesh.vertices),
        len(mesh.edges),
        len(mesh.loops),
        len(mesh.polygons),
    )
    d = {"n": (n_v, n_e, n_l, n_f), "pos": _read_co(mesh)}
    edges = np.empty(n_e * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    d["edges"] = edges.reshape(-1, 2)
    d["loop_vert"] = np.empty(n_l, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", d["loop_vert"])
    d["loop_edge"] = np.empty(n_l, dtype=np.int32)
    mesh.loops.foreach_get("edge_index", d["loop_edge"])
    d["loop_start"] = np.empty(n_f, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", d["loop_start"])
    attrs = []
    d["material_index"] = None
    for attr in mesh.attributes:
        name = attr.name
        if name in _STRUCTURAL or name in _SELECT:
            continue
        if attr.data_type not in _ATTR_TYPES or (
            name.startswith(".") and name != "material_index"
        ):
            raise _UseOperator(f"attribute {name!r} of type {attr.data_type}")
        field, width, dtype = _ATTR_TYPES[attr.data_type]
        buf = np.empty(len(attr.data) * width, dtype=dtype)
        attr.data.foreach_get(field, buf)
        if name == "material_index":
            if attr.domain != "FACE" or attr.data_type != "INT":
                raise _UseOperator("material_index layout")
            d["material_index"] = buf
        else:
            attrs.append(
                (
                    name,
                    attr.domain,
                    attr.data_type,
                    buf.reshape(-1, width) if width > 1 else buf,
                )
            )
    d["attrs"] = attrs
    d["select"] = {}
    for name, coll in _SELECT.items():
        if name in mesh.attributes:
            buf = np.empty(len(getattr(mesh, coll)), dtype=bool)
            getattr(mesh, coll).foreach_get("select", buf)
            d["select"][name] = buf
    d["materials"] = [s.material for s in obj.material_slots]
    if any(s.link != "DATA" for s in obj.material_slots):
        raise _UseOperator("object-linked material slot")
    return d


def _check_join_inputs(objs):
    active = objs[0]
    for o in objs:
        if o.type != "MESH" or o.mode != "OBJECT":
            raise _UseOperator(f"{o.name} is not a mesh in object mode")
        if o.data.shape_keys is not None or len(o.vertex_groups) > 0:
            raise _UseOperator(f"{o.name} has shape keys or vertex groups")
        if o.hide_get():
            raise _UseOperator(f"{o.name} is hidden")
    if active.parent is not None and any(o is active.parent for o in objs[1:]):
        raise _UseOperator("active object's parent is joined")


def _mat_key(ma):
    return ma.as_pointer() if ma is not None else None


def _join_data(active, others):
    """Everything the operator would write, computed from the inputs."""
    datas = [_read_mesh(active)] + [_read_mesh(o) for o in others]
    # transforms: the operator maps every other mesh through inverse(active world) @ world,
    # computed by Blender's own invert_m4_m4 / mul_m4_m4m4; Object.matrix_local evaluates
    # exactly that product for a child of the active object from the cached world matrices
    for o, d in zip(others, datas[1:]):
        o.parent = active
        if o.parent is not active:
            raise _UseOperator(f"cannot parent {o.name} to {active.name}")
        d["pos"] = _mul_m4_v3(np.array(o.matrix_local), d["pos"])
    # material slots: the active object's, then every other object's not seen yet
    matar = list(datas[0]["materials"])
    totmat = sum(len(d["materials"]) for d in datas)
    for d in datas[1:]:
        if d["n"][0] == 0:
            continue
        for ma in d["materials"]:
            if not any(_mat_key(ma) == _mat_key(m) for m in matar):
                matar.append(ma)
    if any(ma is None for ma in matar):
        raise _UseOperator("empty material slot")
    totcol = len(matar)
    # face material indices: remapped per mesh through a table that the operator only
    # refreshes for the slots of a mesh that has faces (stale entries carry over)
    matmap = np.zeros(max(totmat, 1), dtype=np.int32)
    have_mi_layer = any(d["material_index"] is not None for d in datas)
    make_mi_layer = have_mi_layer or (totcol > 1 and any(d["n"][3] > 0 for d in datas))
    mi_parts = []
    for d in datas:
        n_f = d["n"][3]
        if n_f == 0:
            continue
        if totmat > 0:
            for a, ma in enumerate(d["materials"]):
                matmap[a] = next(
                    b for b, m in enumerate(matar) if _mat_key(m) == _mat_key(ma)
                )
        src = (
            d["material_index"]
            if d["material_index"] is not None
            else np.zeros(n_f, dtype=np.int32)
        )
        if totmat == 0:
            mi_parts.append(np.zeros(n_f, dtype=np.int32))
        else:
            if src.min() < 0 or src.max() >= totmat:
                raise _UseOperator("material_index out of range")
            mi_parts.append(matmap[src])
    # attribute layout: the active mesh's layers, then new ones in order of appearance
    layout = []
    seen = {}
    for d in datas:
        for name, domain, atype, _ in d["attrs"]:
            if name in seen:
                if seen[name] != (domain, atype):
                    raise _UseOperator(f"attribute {name!r} with two layouts")
                continue
            seen[name] = (domain, atype)
            layout.append((name, domain, atype))
    return datas, matar, layout, make_mi_layer, mi_parts


def _write_join(active, datas, matar, layout, make_mi_layer, mi_parts):
    mesh = active.data
    n_v = sum(d["n"][0] for d in datas)
    n_e = sum(d["n"][1] for d in datas)
    n_l = sum(d["n"][2] for d in datas)
    n_f = sum(d["n"][3] for d in datas)
    vert_ofs = np.cumsum([0] + [d["n"][0] for d in datas])
    edge_ofs = np.cumsum([0] + [d["n"][1] for d in datas])
    loop_ofs = np.cumsum([0] + [d["n"][2] for d in datas])
    domain_sizes = {"POINT": n_v, "EDGE": n_e, "CORNER": n_l, "FACE": n_f}
    domain_index = {"POINT": 0, "EDGE": 1, "CORNER": 2, "FACE": 3}

    mesh.clear_geometry()
    mesh.vertices.add(n_v)
    mesh.edges.add(n_e)
    mesh.loops.add(n_l)
    mesh.polygons.add(n_f)
    mesh.vertices.foreach_set("co", np.concatenate([d["pos"] for d in datas]).ravel())
    mesh.edges.foreach_set(
        "vertices",
        np.concatenate([d["edges"] + vert_ofs[i] for i, d in enumerate(datas)]).ravel(),
    )
    mesh.polygons.foreach_set(
        "loop_start",
        np.concatenate([d["loop_start"] + loop_ofs[i] for i, d in enumerate(datas)]),
    )
    mesh.loops.foreach_set(
        "vertex_index",
        np.concatenate([d["loop_vert"] + vert_ofs[i] for i, d in enumerate(datas)]),
    )
    mesh.loops.foreach_set(
        "edge_index",
        np.concatenate([d["loop_edge"] + edge_ofs[i] for i, d in enumerate(datas)]),
    )
    # material slots first: clearing them drops any material_index layer
    mesh.materials.clear()
    for ma in matar:
        mesh.materials.append(ma)
    for name, domain, atype in layout:
        field, width, dtype = _ATTR_TYPES[atype]
        parts = []
        for d in datas:
            n = d["n"][domain_index[domain]]
            own = next((a for a in d["attrs"] if a[0] == name), None)
            parts.append(
                own[3]
                if own is not None
                else np.zeros((n, width) if width > 1 else n, dtype=dtype)
            )
        attr = mesh.attributes.new(name, atype, domain)
        attr.data.foreach_set(field, np.concatenate(parts).ravel())
    if make_mi_layer:
        attr = mesh.attributes.new("material_index", "INT", "FACE")
        attr.data.foreach_set(
            "value",
            np.concatenate(mi_parts) if mi_parts else np.zeros(0, dtype=np.int32),
        )
    for name, coll in _SELECT.items():
        if any(name in d["select"] for d in datas):
            parts = [
                d["select"].get(
                    name,
                    np.zeros(
                        d["n"][
                            domain_index[
                                {
                                    "vertices": "POINT",
                                    "edges": "EDGE",
                                    "polygons": "FACE",
                                }[coll]
                            ]
                        ],
                        dtype=bool,
                    ),
                )
                for d in datas
            ]
            getattr(mesh, coll).foreach_set("select", np.concatenate(parts))


def _same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            a is not None
            and b is not None
            and a.shape == b.shape
            and a.dtype == b.dtype
            and np.array_equal(a, b)
        )
    return a == b


def join_objects_exact(objs, check_attributes=False):
    """`butil.join_objects` with the join done at the data level and the same result:
    `objs[0]` gets the others' geometry in the view layer's base order, the others are
    removed, and it is returned selected and active."""
    if check_attributes:
        return butil.join_objects(objs, check_attributes=True)
    empty_objs = [o for o in objs if len(o.data.vertices) == 0]
    objs = [o for o in objs if len(o.data.vertices) > 0]
    butil.delete(empty_objs)
    if len(objs) < 2:
        return butil.join_objects(objs)
    active = objs[0]
    try:
        _check_join_inputs(objs)
        # bring the cached world matrices up to date, as the operator wrapper does
        bpy.context.view_layer.update()
        order = {o.name: i for i, o in enumerate(bpy.context.view_layer.objects)}
        if any(o.name not in order for o in objs):
            raise _UseOperator("object outside the view layer")
        others = sorted(objs[1:], key=lambda o: order[o.name])
        prepared = _join_data(active, others)
    except _UseOperator as e:
        logger.info(f"join_objects_exact: using the operator ({e})")
        return butil.join_objects(objs)
    _write_join(active, *prepared)
    _remove_objects_only(others)
    butil.select_none()
    active.select_set(True)
    bpy.context.view_layer.objects.active = active
    return active


def _remove_objects_only(objs):
    """Free the joined objects as the operator does, leaving their meshes with no users
    for the pipeline's garbage collection."""
    bpy.data.batch_remove(objs)
