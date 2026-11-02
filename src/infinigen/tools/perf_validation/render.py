# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Shaded and wireframe renders of assets and generated scenes, with triangle counts."""

from __future__ import annotations

import pathlib

import bpy
import numpy as np
from mathutils import Vector

SHADED_RESOLUTION = 900
# The wireframe is a close-up of a fixed fraction of the asset: framed on the
# whole asset, a dense mesh saturates into a solid mass.
WIREFRAME_RESOLUTION = 1600
# fraction of the asset's largest extent
WIREFRAME_ZOOM = 0.18
# The trunk view: the asset's own mesh around the lower trunk, its children (a
# tree's twigs and leaves) hidden. The window spans TRUNK_EDGES median edge
# lengths there, and at most TRUNK_ZOOM of the asset.
TRUNK_ZOOM = 0.06
TRUNK_EDGES = 50
MODES = ("shaded", "wireframe", "trunk")
SHADED_SAMPLES = 16
WIREFRAME_SAMPLES = 16

# (offset from centre, energy, disk size)
LIGHTS = [((2.5, -3.0, 4.0), 500.0, 4.0), ((-2.0, 1.0, 2.5), 220.0, 3.0)]


def triangles(objs: list[bpy.types.Object]) -> int:
    """Triangles in `objs` and everything parented under them."""
    seen, total = set(), 0
    for root in objs:
        for o in [root, *root.children_recursive]:
            if o.type != "MESH" or o.name in seen:
                continue
            seen.add(o.name)
            total += sum(len(p.vertices) - 2 for p in o.data.polygons)
    return total


def asset_triangles() -> dict[str, int]:
    """Triangles per generated asset collection (`assets:<factory>`), which a scene
    draws as instances that its own objects do not include."""
    counts = {}
    for col in bpy.data.collections:
        if not col.name.startswith("assets:"):
            continue
        total = 0
        for obj in col.all_objects:
            if obj.type != "MESH" or obj.data is None:
                continue
            loop_total = np.empty(len(obj.data.polygons), np.int64)
            obj.data.polygons.foreach_get("loop_total", loop_total)
            total += int((loop_total - 2).sum())
        counts[col.name] = total
    return counts


def focus(objs: list[bpy.types.Object]) -> Vector:
    """Close-up aim point: the per-axis median of the world-space vertices, since a
    tree's bounding-box centre is air between branches."""
    pts = []
    for root in objs:
        for o in [root, *root.children_recursive]:
            mesh = getattr(o, "data", None)
            if not isinstance(mesh, bpy.types.Mesh) or not len(mesh.vertices):
                continue
            co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            # a subsample is enough for a median
            if len(co) > 20000:
                co = co[:: len(co) // 20000]
            m = o.matrix_world
            pts.append(np.array([(m @ Vector(p)).to_tuple() for p in co]))
    if not pts:
        return None
    return Vector(np.median(np.concatenate(pts), axis=0).tolist())


def trunk_focus(objs: list[bpy.types.Object]):
    """`(aim, window)` for the trunk view: the median of the lowest fifth of the
    asset's own vertices (children left out), and `TRUNK_EDGES` times their median
    edge length."""
    pts, edges = [], []
    for o in objs:
        mesh = getattr(o, "data", None)
        if not isinstance(mesh, bpy.types.Mesh) or not len(mesh.vertices):
            continue
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        m = np.array(o.matrix_world)
        world = co @ m[:3, :3].T + m[:3, 3]
        pts.append(world)
        ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        edges.append((world, ev))
    if not pts:
        return None, None
    allpts = np.concatenate(pts)
    cut = np.quantile(allpts[:, 2], 0.2)
    aim = np.median(allpts[allpts[:, 2] <= cut], axis=0)
    lengths = [
        np.linalg.norm(w[ev[:, 0]] - w[ev[:, 1]], axis=1)[w[ev[:, 0], 2] <= cut]
        for w, ev in edges
        if len(ev)
    ]
    lengths = np.concatenate(lengths) if lengths else np.zeros(0)
    window = TRUNK_EDGES * float(np.median(lengths)) if len(lengths) else None
    return Vector(aim.tolist()), window


def bounds(objs: list[bpy.types.Object]):
    """World-space centre, extent and radius of everything given."""
    corners = []
    for root in objs:
        for o in [root, *root.children_recursive]:
            if not hasattr(o, "bound_box"):
                continue
            corners += [o.matrix_world @ Vector(c) for c in o.bound_box]
    if not corners:
        return Vector((0, 0, 0)), np.ones(3), 1.0
    pts = np.array([[c.x, c.y, c.z] for c in corners])
    lo, hi = pts.min(0), pts.max(0)
    centre = Vector(((lo + hi) / 2).tolist())
    extent = hi - lo
    # half the largest extent, not the half-diagonal, which is up to sqrt(3)
    # larger and would change the framing and lighting
    radius = max(float(np.max(extent)) * 0.5, 0.01)
    return centre, extent, radius


def look_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


# flat surface tone and wire colour for the overlay
WIRE_SURFACE = (0.82, 0.82, 0.82, 1.0)
WIRE_LINE = (0.0, 0.0, 0.0, 1.0)


def _wire_material(name: str) -> bpy.types.Material:
    """Emission material of one-pixel black wires on a flat surface, so the wires stay
    visible however the scene is lit and at any resolution."""
    mat = bpy.data.materials.new(f"{name}_wire")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeEmission")
    shader.inputs["Strength"].default_value = 1.0

    wire = nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 1.0

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[1].default_value = WIRE_SURFACE
    mix.inputs[2].default_value = WIRE_LINE

    links = mat.node_tree.links
    links.new(wire.outputs["Fac"], mix.inputs[0])
    links.new(mix.outputs[0], shader.inputs["Color"])
    links.new(shader.outputs[0], out.inputs[0])
    return mat


def make_scene(
    name: str, mode: str, centre, extent, radius, aim=None, window=None
) -> bpy.types.Scene:
    """Scene rendering `mode`, one of `MODES`. Shaded frames the whole asset;
    wireframe frames `WIREFRAME_ZOOM` of it around `aim` so the mesh is legible, and
    trunk `TRUNK_ZOOM` of it around the lower trunk."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    scene = bpy.data.scenes.new(f"{name}_{mode}")
    # Cycles on CPU: renders the production materials and needs no GL
    # context, which Workbench does.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SHADED_SAMPLES if mode == "shaded" else WIREFRAME_SAMPLES
    scene.cycles.use_denoising = False
    size = SHADED_RESOLUTION if mode == "shaded" else WIREFRAME_RESOLUTION
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_border = False
    # untiled: tiles spill to EXR files in Blender's temp directory, which may
    # not be writable, and these renders fit in memory
    scene.cycles.use_auto_tile = False

    # Colour management, bounces and the pixel filter stay at Blender's
    # defaults; forcing view_transform to Standard would remove the tone
    # mapping and clip highlights.

    scene.world = bpy.data.worlds.new(f"{name}_{mode}_world")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.025, 0.035, 0.055, 1.0)
    background.inputs["Strength"].default_value = 0.35

    target = centre if mode == "shaded" or aim is None else aim
    scale = float(max(extent))
    camera_data = bpy.data.cameras.new(f"{name}_camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = (
        scale
        * {
            "shaded": 1.30,
            "wireframe": WIREFRAME_ZOOM,
            "trunk": TRUNK_ZOOM,
        }[mode]
    )
    if mode == "trunk" and window:
        camera_data.ortho_scale = min(camera_data.ortho_scale, window)
    camera = bpy.data.objects.new(f"{name}_camera", camera_data)
    # the same direction for both views, so the close-up is recognisably a
    # part of the shaded one; an orthographic camera crops rather than moves
    camera.location = target + Vector((1.5, -2.3, 1.45)) * radius
    scene.collection.objects.link(camera)
    scene.camera = camera
    look_at(camera, target)

    # One scale for position, size and energy: irradiance falls off with the
    # square of distance, so energy grows with the scale squared. Lights move
    # out with subjects larger than unit size, so they stay outside a tree's
    # canopy.
    scale = max(radius, 1.0)
    for index, (offset, energy, size) in enumerate(LIGHTS):
        light_data = bpy.data.lights.new(f"{name}_light_{index}", "AREA")
        light_data.energy = energy * scale**2
        light_data.shape = "DISK"
        light_data.size = size * scale
        light = bpy.data.objects.new(f"{name}_light_{index}", light_data)
        light.location = centre + Vector(offset) * scale
        scene.collection.objects.link(light)
        look_at(light, centre)

    if mode in {"wireframe", "trunk"}:
        scene.view_layers[0].material_override = _wire_material(name)
    return scene


def render(scene, objs: list[bpy.types.Object], path, children=True) -> None:
    """Link `objs` (and, unless `children` is False, everything under them) into
    `scene` and write one image."""
    for root in objs:
        for o in [root, *root.children_recursive] if children else [root]:
            if o.name not in scene.collection.objects:
                scene.collection.objects.link(o)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=scene.name)


def scene_camera(scene: bpy.types.Scene):
    """The camera the pipeline placed, or the first one in the file."""
    if scene.camera is not None:
        return scene.camera
    cameras = sorted(
        (o for o in scene.objects if o.type == "CAMERA"), key=lambda o: o.name
    )
    if not cameras:
        raise RuntimeError(f"{scene.name!r} has no camera to render from")
    scene.camera = cameras[0]
    return scene.camera


def prepare_scene(scene: bpy.types.Scene, mode: str, name: str) -> bpy.types.Scene:
    """Prepare `scene` to render `mode` as it was generated, with displacement set as
    render_image sets it. The wireframe pass adds the overlay, renders untiled, and
    raises the resolution so the triangles do not saturate into a solid mass."""
    if mode not in {"shaded", "wireframe"}:
        raise ValueError(f'mode must be "shaded" or "wireframe", got {mode!r}')

    scene_camera(scene)

    # as render_image does: without true displacement, displaced surfaces such
    # as terrain render flat
    from infinigen.core.rendering.render import set_displacement_mode

    set_displacement_mode()

    for layer in scene.view_layers:
        layer.material_override = _wire_material(name) if mode == "wireframe" else None

    if mode == "wireframe":
        width, height = scene.render.resolution_x, scene.render.resolution_y
        scale = WIREFRAME_RESOLUTION / max(width, height, 1)
        if scale > 1:
            scene.render.resolution_x = max(int(round(width * scale)), 1)
            scene.render.resolution_y = max(int(round(height * scale)), 1)
        # untiled, as in make_scene
        scene.cycles.use_auto_tile = False

    return scene


def use_gpu(scene) -> None:
    """Render `scene` on the visible GPUs, raising rather than falling back to CPU."""
    from infinigen.core.init import configure_cycles_devices

    configure_cycles_devices(use_gpu=True)
    if scene.cycles.device != "GPU":
        raise RuntimeError("--gpu was given but Cycles found no GPU")


def render_scene(scene, path) -> None:
    """Write one image of `scene` as it stands, linking nothing."""
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=scene.name)


def use_writable_temp(directory) -> None:
    """Make `directory` Blender's temporary directory, creating it. Cycles writes tile
    buffers there; if it cannot, a render fails at the end with "Error opening tile
    file" and no image."""
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.temporary_directory = str(path)
