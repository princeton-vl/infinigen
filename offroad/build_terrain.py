"""Build the road-anchored ground as a Blender mesh and save a .blend to inspect.

Run with a Python that has `bpy` available, e.g. the infinigen env:

    ~/anaconda3/envs/infinigen/bin/python -m offroad.build_terrain \
        --out outputs/offroad/proto.blend

What it produces in the .blend:
    - `offroad_ground`   : the conformed terrain mesh, with baked POINT attributes
                           `dist_to_road`, `z_road`, `road_falloff`, and a
                           `road_viz` color attribute (red carriageway, green sides).
    - `road_centerline`  : a POLY curve along the path (for camera/ego FOLLOW_PATH).
    - `road_ribbon`      : a thin flat strip at road elevation, to eyeball the fit.

No Infinigen terrain SDF is used; this is the path-first ground the rest of the
pipeline will scatter vegetation onto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import numpy as np

from offroad.conform_terrain import Heightfield, build_heightfield, road_falloff, validate
from offroad.road_spec import RoadSpec, synthetic_s_curve


def _clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _grid_faces(ny: int, nx: int) -> np.ndarray:
    r = np.arange(ny - 1)
    c = np.arange(nx - 1)
    R, C = np.meshgrid(r, c, indexing="ij")
    v00 = (R * nx + C).ravel()
    v01 = (R * nx + C + 1).ravel()
    v11 = ((R + 1) * nx + C + 1).ravel()
    v10 = ((R + 1) * nx + C).ravel()
    # CCW from above -> +Z normals
    return np.stack([v00, v01, v11, v10], axis=1)


def ground_to_mesh(hf: Heightfield, road: RoadSpec, name: str = "offroad_ground"):
    ny, nx = hf.shape
    verts = np.stack([hf.X.ravel(), hf.Y.ravel(), hf.Z.ravel()], axis=1)
    faces = _grid_faces(ny, nx)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # --- baked attributes (POINT domain, vertex order == hf.ravel()) ---
    dist = hf.dist.ravel()
    zr = hf.z_road.ravel()
    fall = road_falloff(hf.dist, road.half_width, transition=14.0, shoulder=1.5).ravel()

    for attr_name, data in [
        ("dist_to_road", dist),
        ("z_road", zr),
        ("road_falloff", fall),
    ]:
        a = mesh.attributes.new(attr_name, "FLOAT", "POINT")
        a.data.foreach_set("value", data.astype(np.float32))

    # --- visualization color: red carriageway, yellow->green by falloff ---
    on_road = dist <= road.half_width
    rgba = np.zeros((verts.shape[0], 4), dtype=np.float32)
    rgba[:, 3] = 1.0
    # sides: lerp yellow(near, fall=0) -> green(far, fall=1)
    rgba[:, 0] = (1.0 - fall) * 0.9
    rgba[:, 1] = 0.55 + 0.35 * fall
    rgba[:, 2] = 0.10
    # carriageway: red
    rgba[on_road] = (0.85, 0.10, 0.10, 1.0)
    col = mesh.color_attributes.new("road_viz", "FLOAT_COLOR", "POINT")
    col.data.foreach_set("color", rgba.ravel())

    for p in mesh.polygons:
        p.use_smooth = True
    mesh.update()
    return obj


def road_to_curve(road: RoadSpec, name: str = "road_centerline"):
    cd = bpy.data.curves.new(name, "CURVE")
    cd.dimensions = "3D"
    sp = cd.splines.new("POLY")
    pts = road.centerline
    sp.points.add(len(pts) - 1)
    flat = np.concatenate([pts, np.ones((len(pts), 1))], axis=1).astype(np.float32)
    sp.points.foreach_set("co", flat.ravel())
    obj = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def road_ribbon(road: RoadSpec, name: str = "road_ribbon"):
    """A thin flat strip at road elevation, offset +0.05 m, to eyeball the fit."""
    c = road.centerline
    tang = np.gradient(c[:, :2], axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # left perpendicular
    hw = road.half_width
    left = c.copy()
    right = c.copy()
    left[:, :2] += normal * hw
    right[:, :2] -= normal * hw
    left[:, 2] += 0.05
    right[:, 2] += 0.05
    verts = np.concatenate([left, right], axis=0)
    K = len(c)
    faces = []
    for i in range(K - 1):
        faces.append([i, i + 1, K + i + 1, K + i])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/offroad/proto.blend"))
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--amplitude", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=55.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    _clear_scene()

    road = synthetic_s_curve()
    hf = build_heightfield(
        road,
        margin=args.margin,
        spacing=args.spacing,
        amplitude=args.amplitude,
        transition=14.0,
        shoulder=1.5,
        noise_seed=args.seed,
    )
    v = validate(hf, road)
    print("=== VALIDATION ===")
    for k, val in v.items():
        print(f"  {k:28s}: {val}")

    ground_to_mesh(hf, road)
    road_to_curve(road)
    road_ribbon(road)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    print(f"\nSaved {args.out}")
    print(
        "Open with:  ~/anaconda3/envs/infinigen/bin/python -m infinigen.launch_blender "
        f"{args.out}\n"
        "Tip: set viewport shading to 'Vertex Color' / the 'road_viz' attribute to see "
        "the red road vs green sides."
    )


if __name__ == "__main__":
    main()
