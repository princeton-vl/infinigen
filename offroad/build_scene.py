"""End-to-end: xodr -> detailed grassland off-road scene.

    cd <infinigen repo root>
    ~/anaconda3/envs/infinigen/bin/python -m offroad.build_scene \
        --xodr /tmp/mcp_grassland4/scenes/offroad_driving/grassland_llm4/scenario.xodr \
        --out outputs/offroad/grassland.blend --render --trees

Pipeline:
  1. parse xodr -> RoadSpec (centerline + elevation + width + boulders)   [offroad.road_spec]
  2. road-anchored terrain heightfield -> mesh + baked dist_to_road       [offroad.conform_terrain / build_terrain]
  3. init Infinigen gin (base_nature + plain.gin) so asset factories work
  4. materials (grassland soil base + dirt road ribbon) + sky lighting
  5. vegetation: grass + pebbles + xodr boulders (+ optional trees/bushes), all off-corridor
  6. camera along the road; optional EEVEE preview render
  7. save .blend
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import bpy
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
log = logging.getLogger("offroad.build_scene")


def _init_infinigen(seed=77):
    """We deliberately do NOT call Infinigen's gin pipeline (it pulls in the
    terrain/OcMesher stack and a scene_type config we don't want). Asset
    factories are plain gin.configurable functions that fall back to their
    signature defaults when no bindings are present, which is all we need."""
    import numpy as _np

    _np.random.seed(seed)


def _simple_material(name, rgb, roughness=0.95):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _arclen(centerline):
    seg = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_at_s(centerline, s_query):
    s = _arclen(centerline)
    i = int(np.clip(np.searchsorted(s, s_query) - 1, 0, len(s) - 2))
    seg = s[i + 1] - s[i]
    u = 0.0 if seg < 1e-9 else (s_query - s[i]) / seg
    p = centerline[i] * (1 - u) + centerline[i + 1] * u
    d = centerline[i + 1] - centerline[i]
    d = d / (np.linalg.norm(d) + 1e-9)
    return p, d


def _look_at(cam, target):
    import mathutils

    direction = mathutils.Vector(target) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _driving_camera(road, station_m, name="cam_driving"):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = 22.0
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    p, d = _point_at_s(road.centerline, station_m)
    cam.location = mathutils_vec(p + np.array([0, 0, 2.2]) - d * 7.0)
    look = _point_at_s(road.centerline, min(station_m + 25, _arclen(road.centerline)[-1]))[0]
    _look_at(cam, (look[0], look[1], look[2] + 1.0))
    return cam


def _overview_camera(road, name="cam_overview"):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = 28.0
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    c = road.centerline
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    span = max(c[:, 0].ptp(), c[:, 1].ptp())
    cam.location = mathutils_vec((cx - span * 0.2, cy - span * 0.95, c[:, 2].max() + span * 0.5))
    _look_at(cam, (cx, cy, c[:, 2].mean()))
    return cam


def _detail_camera(road, station_m, name="cam_detail"):
    """Low, close, oblique shot of the road surface to show ruts/washboard/potholes."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = 35.0
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    p, d = _point_at_s(road.centerline, station_m)
    cam.location = mathutils_vec(p + np.array([0, 0, 0.9]) - d * 4.5)
    look = _point_at_s(road.centerline, station_m + 4.0)[0]
    _look_at(cam, (look[0], look[1], look[2] + 0.15))
    return cam


def mathutils_vec(arr):
    import mathutils

    return mathutils.Vector((float(arr[0]), float(arr[1]), float(arr[2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xodr", type=str, required=True)
    ap.add_argument("--spec", type=str, default=None,
                    help="spec.json with road.features (default: sibling of xodr)")
    ap.add_argument("--out", type=Path, default=Path("outputs/offroad/grassland.blend"))
    ap.add_argument("--biome", type=str, default=None,
                    help="grassland|desert|mountain|snow|forest (default: spec.json biome)")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--margin", type=float, default=30.0)
    ap.add_argument("--spacing", type=float, default=0.7)
    ap.add_argument("--amplitude", type=float, default=None, help="override biome relief")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--raytrace", action="store_true", help="slow raytraced GI/AO")
    ap.add_argument("--res", type=str, default="960x540")
    args = ap.parse_args()

    np.random.seed(args.seed)

    import json

    from offroad import biomes
    from offroad import build_terrain as bt
    from offroad import road_surface as rs
    from offroad import vegetation as veg
    from offroad.conform_terrain import build_heightfield, validate
    from offroad.road_spec import from_xodr

    # 1) parse xodr + spec (features + biome) -------------------------------
    road = from_xodr(args.xodr)
    spec_path = args.spec or str(Path(args.xodr).with_name("spec.json"))
    features = rs.load_features(spec_path)
    spec_biome = None
    try:
        spec_biome = json.loads(open(spec_path).read()).get("biome")
    except Exception:  # noqa: BLE001
        pass
    biome_name, preset = biomes.resolve(args.biome or spec_biome)
    if args.amplitude is not None:
        preset = {**preset, "amplitude": args.amplitude}
    log.info(f"biome={biome_name}  relief_amp={preset['amplitude']}  "
             f"scatters={preset['scatters']}  features={[f.get('type') for f in features]}")
    c = road.centerline
    log.info(
        f"road: {len(c)} pts, width {road.width} m, length {_arclen(c)[-1]:.1f} m, "
        f"z[{c[:,2].min():.2f},{c[:,2].max():.2f}], {len(road.obstacles)} boulders"
    )

    bt._clear_scene()

    # 2) terrain (relief from biome preset) --------------------------------
    hf = build_heightfield(
        road, margin=args.margin, spacing=args.spacing, amplitude=preset["amplitude"],
        transition=preset["transition"], shoulder=road.width * 0.5 + 1.0, noise_seed=args.seed,
        noise_kwargs={"octaves": preset["octaves"], "base_freq": preset["base_freq"],
                      "persistence": 0.5},
    )
    v = validate(hf, road)
    log.info(f"terrain validate: {v}")
    terrain = bt.ground_to_mesh(hf, road, name="offroad_terrain")
    road_curve = bt.road_to_curve(road)
    # detailed dirt road surface with ruts / washboard / potholes / mud
    road_obj = rs.road_surface_to_mesh(road, features, seed=args.seed)

    # 3) infinigen asset factories (no gin pipeline) ------------------------
    _init_infinigen(seed=args.seed)

    # 4) materials + lighting ----------------------------------------------
    terrain.data.materials.clear()
    terrain.data.materials.append(_simple_material(f"{biome_name}_ground", preset["terrain_color"]))
    # road surface already carries its own dirt/mud materials (road_surface.py)
    scn = bpy.context.scene
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.use_nodes = True
    try:
        from infinigen.assets.lighting import sky_lighting

        sky_lighting.add_lighting()
        log.info("sky lighting (nishita) added")
    except Exception as e:  # noqa: BLE001
        log.warning(f"sky lighting fallback (sun+sky): {e}")
        scn.world.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.7, 0.95, 1)
        scn.world.node_tree.nodes["Background"].inputs[1].default_value = 1.2
        scn.world.node_tree.nodes["Background"].inputs[1].default_value = 1.2

    # explicit LOW sun with shadows so road relief (ruts/washboard/potholes)
    # casts visible shadow bands. Azimuth ~ along the road at the detail feature.
    feat_s0 = next((float(f["s"]) for f in features
                    if str(f.get("type", "")).lower() in ("washboard", "pothole")), 30.0)
    _, tdir = _point_at_s(c, min(feat_s0, _arclen(c)[-1] - 1))
    sun_az = np.arctan2(tdir[1], tdir[0])
    sun = bpy.data.objects.new("RakingSun", bpy.data.lights.new("RakingSun", "SUN"))
    sun.data.energy = preset["sun_energy"]
    sun.data.angle = np.radians(1.0)  # sharpish shadows
    sun.data.use_shadow = True
    sun.rotation_euler = (np.radians(70), 0, sun_az + np.radians(90))  # ~20° above horizon
    scn.collection.objects.link(sun)

    # 5) vegetation per biome (all off-corridor) ---------------------------
    buf = road.width * 0.5 + 0.8
    veg.populate_biome(preset, terrain, road, buffer_m=buf)

    # safety net: never render leftover placeholder cubes
    for coll in bpy.data.collections:
        if coll.name.startswith("placeholders:"):
            coll.hide_render = True
            coll.hide_viewport = True

    # 6) cameras + optional renders ----------------------------------------
    total = _arclen(c)[-1]
    cam_drive = _driving_camera(road, station_m=total * 0.25)
    cam_over = _overview_camera(road)
    # focus the detail cam on the first washboard/pothole feature if present
    feat_s = next((float(f["s"]) for f in features
                   if str(f.get("type", "")).lower() in ("washboard", "pothole")), total * 0.35)
    cam_detail = _detail_camera(road, station_m=max(2.0, feat_s - 4.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.render:
        scn = bpy.context.scene
        scn.render.engine = "BLENDER_EEVEE_NEXT"
        w, h = (int(x) for x in args.res.lower().split("x"))
        scn.render.resolution_x, scn.render.resolution_y = w, h
        try:  # shadow maps (fast) give relief; raytracing too slow on big terrain
            scn.eevee.use_raytracing = bool(args.raytrace)
            scn.eevee.use_shadows = True
            scn.eevee.taa_render_samples = 24
        except Exception:  # noqa: BLE001
            pass
        for cam, tag in [(cam_over, "overview"), (cam_drive, "driving"), (cam_detail, "detail")]:
            try:
                scn.camera = cam
                png = args.out.with_name(f"{args.out.stem}_{tag}.png")
                scn.render.filepath = str(png.resolve())
                bpy.ops.render.render(write_still=True)
                log.info(f"rendered {tag} -> {png}")
            except Exception as e:  # noqa: BLE001
                log.warning(f"render {tag} skipped: {e}")
        scn.camera = cam_drive

    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    log.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
