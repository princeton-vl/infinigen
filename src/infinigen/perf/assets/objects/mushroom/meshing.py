# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

import numpy as np

from infinigen.perf.assets.utils.mesh_io import write_positions


def triangle_budget(profile, face_size):
    # A voxel surface contributes faces in proportion to its area projected
    # onto the three coordinate planes. Integrate |nx|+|ny|+|nz| around each
    # revolved profile segment, then count two triangles per voxel quad.
    rz = profile[:, [0, 2]].astype(np.float64)
    delta = np.abs(np.diff(rz, axis=0))
    radius = (rz[:-1, 0] + rz[1:, 0]) / 2
    projected_area = np.sum(radius * (8 * delta[:, 1] + 2 * np.pi * delta[:, 0]))
    return max(4, int(round(2 * projected_area / face_size**2)))


def _native_spin(
    anchors,
    vector_locations=(),
    resolution=None,
    rotation_resolution=None,
    axis=(0, 0, 1),
    loop=False,
    dupli=False,
):
    """`draw.spin` extruded by `infinigen_gpl.perf.spin.spin_mesh` instead of Blender's
    spin operator, whose face order changes from run to run."""
    import bpy

    from infinigen.assets.utils import draw
    from infinigen.core.util import blender as butil
    from infinigen.infinigen_gpl.perf.spin import spin_mesh

    obj = draw.bezier_curve(anchors, vector_locations, resolution)
    co = draw.read_co(obj)
    mean_radius = np.mean(
        np.linalg.norm(
            co - (co @ np.array(axis))[:, np.newaxis] * np.array(axis), axis=-1
        )
    )
    if rotation_resolution is None:
        rotation_resolution = min(int(2 * np.pi * mean_radius / 5e-3), 128)
    butil.modify_mesh(obj, "WELD", merge_threshold=1e-3)
    if loop:
        with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill()
        draw.remesh_fill(obj)
    spin_mesh(obj, rotation_resolution, axis, dupli)
    with butil.ViewportMode(obj, "EDIT"), butil.Suppress():
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=1e-3)
    return obj


def spin(anchors, vector_locations=(), face_size=None, **kwargs):
    original = _native_spin

    if face_size is None:
        return original(anchors, vector_locations, **kwargs)
    from infinigen.assets.utils.draw import bezier_curve
    from infinigen.core.util import blender as butil
    from infinigen.perf.assets.objects.mushroom._meshing import (
        profile_mesh_with_edges,
        profile_triangle_count,
    )

    obj = bezier_curve(anchors, vector_locations, kwargs.get("resolution"))
    mesh = obj.data
    profile = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get("co", profile.ravel())
    # the profile mesher needs x >= 0 along the whole profile
    if profile[:, 0].min() < -1e-7:
        butil.delete(obj)
        return original(anchors, vector_locations, **kwargs)
    # match the triangle count of the voxel remesh the default path applies
    step = float(face_size)
    target = triangle_budget(profile, step)
    for _ in range(2):
        triangles = profile_triangle_count(profile, step)
        if abs(triangles / target - 1) < 0.005:
            break
        step *= np.sqrt(triangles / target)
    co, loops, counts, edges, corner_edges = profile_mesh_with_edges(profile, step)
    mesh.clear_geometry()
    mesh.vertices.add(len(co))
    write_positions(mesh, co)
    mesh.loops.add(len(loops))
    mesh.attributes[".corner_vert"].data.foreach_set("value", loops)
    mesh.polygons.add(len(counts))
    from infinigen.perf.assets.utils.mesh_io import write_faces

    write_faces(mesh, counts)
    mesh.edges.add(len(edges))
    attr = mesh.attributes.get(".edge_verts")
    if attr is not None:
        attr.data.foreach_set("value", edges.ravel())
    else:
        mesh.edges.foreach_set("vertices", edges.ravel())
    mesh.attributes[".corner_edge"].data.foreach_set("value", corner_edges)
    mesh.update()
    obj["_mushroom_direct_mesh"] = True
    return obj


def remesh_with_attrs(obj, face_size, **kwargs):
    if obj.get("_mushroom_direct_mesh", False):
        del obj["_mushroom_direct_mesh"]
        return obj
    from infinigen.core.placement.detail import remesh_with_attrs as original

    return original(obj, face_size, **kwargs)


def cap_attributes(obj, factory):
    from infinigen.core import surface

    mesh = obj.data
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co.ravel())
    co = np.abs(co)
    for k, name in enumerate("xyz"):
        attr = mesh.attributes.get(name) or mesh.attributes.new(name, "FLOAT", "POINT")
        maximum = co[:, k].max(initial=0)
        values = co[:, k] / maximum if maximum else np.zeros(len(co), np.float32)
        attr.data.foreach_set("value", values)
    if factory.is_morel:
        if native_fields_enabled():
            scale, randomness = np.random.uniform(15, 20), np.random.uniform(0.5, 1)
            values = voronoi_mask(obj, scale, randomness, 0.05)
            attr = mesh.attributes.get("morel") or mesh.attributes.new(
                "morel", "FLOAT", "POINT"
            )
            attr.data.foreach_set("value", values)
        else:
            surface.add_geomod(obj, factory.geo_morel, apply=True)
    else:
        np.random.uniform(15, 20)
        np.random.uniform(0.5, 1)


def native_fields_enabled():
    """True on Blender 4.2, whose texture evaluation the native field kernels match."""
    import bpy

    return bpy.app.version[:2] == (4, 2)


def voronoi_mask(obj, scale, randomness, threshold):
    from infinigen.infinigen_gpl.perf.voronoi._voronoi import edge_mask

    co = np.empty((len(obj.data.vertices), 3), np.float32)
    obj.data.attributes["position"].data.foreach_get("vector", co.ravel())
    return edge_mask(co, scale, randomness, threshold)


def web_mask(obj, factory):
    from infinigen.core import surface
    from infinigen.core.nodes.node_info import Nodes

    if not native_fields_enabled():
        surface.add_geomod(obj, factory.geo_voronoi, apply=True)
        return
    name = "_mushroom_web_mask"
    values = voronoi_mask(obj, np.random.uniform(15, 20), 1.0, 0.06)
    attr = obj.data.attributes.new(name, "FLOAT", "POINT")
    attr.data.foreach_set("value", values)

    def separate(nw):
        geometry = nw.new_node(
            Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
        )
        keep = nw.new_node(
            Nodes.NamedAttribute,
            input_kwargs={"Name": name},
            attrs={"data_type": "FLOAT"},
        )
        selected = nw.compare("GREATER_THAN", keep, 0.5)
        geometry = nw.new_node(Nodes.SeparateGeometry, [geometry, selected])
        nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})

    surface.add_geomod(obj, separate, apply=True)
    attr = obj.data.attributes.get(name)
    if attr is not None:
        obj.data.attributes.remove(attr)


def texture_displace(obj, texture, strength=0.008):
    from infinigen.core.util import blender as butil

    supported = (
        native_fields_enabled()
        and texture.type in {"STUCCI", "MARBLE"}
        and texture.noise_basis == "BLENDER_ORIGINAL"
        and texture.intensity == 1
        and texture.contrast == 1
        and not texture.use_color_ramp
        and (texture.type != "MARBLE" or texture.noise_basis_2 == "SIN")
    )
    if not supported:
        butil.modify_mesh(
            obj, "DISPLACE", strength=strength, texture=texture, mid_level=0
        )
        return
    from infinigen.infinigen_gpl.perf.voronoi._voronoi import texture_displace as native

    mesh = obj.data
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co.ravel())
    from infinigen.perf.assets.utils.mesh_io import read_normals

    normals = read_normals(mesh)
    kind = int(texture.type == "MARBLE")
    subtype = (
        ["SOFT", "SHARP", "SHARPER"].index(texture.marble_type)
        if kind
        else ["PLASTIC", "WALL_IN", "WALL_OUT"].index(texture.stucci_type)
    )
    result = native(
        co,
        normals,
        texture.noise_scale,
        kind,
        subtype,
        getattr(texture, "noise_depth", 2),
        texture.turbulence,
        texture.noise_type == "HARD_NOISE",
        strength,
    )
    write_positions(mesh, result)
    mesh.update()


def extension(obj, noise_strength=0.2):
    if not native_fields_enabled():
        from infinigen.assets.utils.decorate import geo_extension
        from infinigen.core import surface

        surface.add_geomod(obj, geo_extension, apply=True, input_args=[noise_strength])
        return
    from infinigen.infinigen_gpl.perf.voronoi._voronoi import extension_positions

    strength = np.random.uniform(noise_strength / 2, noise_strength)
    scale = np.random.uniform(1.4, 2.8)
    offset = np.asarray(np.random.uniform(-1, 1, 3), np.float32)
    mesh = obj.data
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co.ravel())
    result = extension_positions(co, strength, scale, offset)
    write_positions(mesh, result)
    mesh.update()


def displace_vertices(obj, fn):
    from collections.abc import Iterable

    mesh = obj.data
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co.ravel())
    co = co.astype(np.float64)
    if not isinstance(fn, Iterable):
        delta = fn(*co.T)
        for k in range(3):
            co[:, k] += delta[k]
    else:
        co += fn
    write_positions(mesh, co.astype(np.float32))
    mesh.update()


def origin2lowest(obj):
    from infinigen.assets.utils.object import origin2lowest as original

    if (
        obj.type != "MESH"
        or obj.parent is not None
        or obj.data.users != 1
        or obj.modifiers
        or obj.data.shape_keys
        or obj.animation_data is not None
        or obj.constraints
        or tuple(obj.delta_location) != (0.0, 0.0, 0.0)
        or tuple(obj.delta_rotation_euler) != (0.0, 0.0, 0.0)
        or tuple(obj.delta_scale) != (1.0, 1.0, 1.0)
        or tuple(obj.scale) != (1.0, 1.0, 1.0)
        or obj.rotation_mode != "XYZ"
        or tuple(obj.rotation_euler) != (0.0, 0.0, 0.0)
    ):
        return original(obj)
    mesh = obj.data
    if not mesh.vertices:
        return
    co = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.attributes["position"].data.foreach_get("vector", co.ravel())
    from mathutils import Matrix

    mesh.transform(Matrix.Translation(-co[np.argmin(co[:, 2])]))
    obj.location = (0.0, 0.0, 0.0)
    import bpy

    bpy.context.view_layer.update()
