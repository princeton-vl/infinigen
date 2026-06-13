"""Detailed dirt-road surface: a high-resolution (s, t)-parameterized ribbon with
wheel ruts, washboard ripples, potholes and mud patches displaced into it, plus
per-face dirt/mud material assignment.

`features` is the list from spec.json["road"]["features"], e.g.
  {"type":"washboard","s":25,"length_m":8,"amplitude_m":0.08,"wavelength_m":0.5}
  {"type":"pothole","s":50,"lateral_m":0.5,"radius_m":0.6,"depth_m":0.15}
  {"type":"mud_patch","s":80,"lateral_m":-0.5,"radius_m":1.5,"depth_m":0.05}

Pure-numpy geometry (testable headless); `road_surface_to_mesh` needs bpy.
"""

from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger("offroad.road_surface")


def load_features(spec_path: str) -> list[dict]:
    try:
        spec = json.loads(open(spec_path).read())
        return list(spec.get("road", {}).get("features", []))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not read features from {spec_path}: {e}")
        return []


def _resample_by_arclen(centerline: np.ndarray, ds: float):
    seg = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(2, int(total / ds) + 1)
    s_new = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    for d in range(3):
        out[:, d] = np.interp(s_new, s, centerline[:, d])
    return s_new, out


def build_surface_grid(
    road,
    features: list[dict],
    width: float | None = None,
    ds: float = 0.08,
    dt: float = 0.18,
    lift: float = 0.04,
    rut_offset_frac: float = 0.42,  # wheel track as fraction of half-width
    rut_depth: float = 0.05,
    rut_width: float = 0.22,
    micro_amp: float = 0.015,
    seed: int = 0,
):
    """Return (s_arr, ts, P (NS,NT,3), mud_mask (NS,NT)). z already displaced."""
    width = width or road.width
    hw = width / 2.0
    s_arr, c = _resample_by_arclen(road.centerline, ds)

    d = np.gradient(c[:, :2], axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    left = np.stack([-d[:, 1], d[:, 0]], axis=1)  # (NS,2) left normal in XY

    ts = np.arange(-hw, hw + 1e-6, dt)
    NS, NT = len(s_arr), len(ts)

    P = np.empty((NS, NT, 3))
    P[:, :, 0] = c[:, 0:1] + np.outer(left[:, 0], ts)
    P[:, :, 1] = c[:, 1:2] + np.outer(left[:, 1], ts)
    P[:, :, 2] = c[:, 2:3]

    disp = np.zeros((NS, NT))
    mud = np.zeros((NS, NT), dtype=bool)

    # --- continuous wheel ruts (two longitudinal grooves) ---
    rut_t = rut_offset_frac * hw
    for sign in (-1.0, 1.0):
        disp -= rut_depth * np.exp(-((ts[None, :] - sign * rut_t) ** 2) / (2 * rut_width**2))

    # --- micro surface roughness (fine dirt bumps) ---
    rng = np.random.default_rng(seed)
    micro = (
        np.sin(s_arr[:, None] * 2.3 + ts[None, :] * 5.1 + rng.uniform(0, 6))
        + 0.6 * np.sin(s_arr[:, None] * 7.7 - ts[None, :] * 3.3 + rng.uniform(0, 6))
    )
    disp += micro_amp * micro

    # --- spec features ---
    edge = max(1e-3, 0.0)
    for f in features:
        typ = str(f.get("type", "")).lower()
        s0 = float(f.get("s", 0.0))
        if typ == "washboard":
            L = float(f.get("length_m", 6.0))
            amp = float(f.get("amplitude_m", 0.06))
            wl = max(0.1, float(f.get("wavelength_m", 0.5)))
            band = (s_arr >= s0) & (s_arr <= s0 + L)
            taper = np.clip(np.minimum(s_arr - s0, s0 + L - s_arr) / (L * 0.2 + 1e-6), 0, 1)
            ripple = amp * np.sin(2 * np.pi * (s_arr - s0) / wl) * taper * band
            disp += ripple[:, None]
        elif typ in ("pothole", "mud_patch"):
            tp = float(f.get("lateral_m", 0.0))
            r = max(0.15, float(f.get("radius_m", 0.5)))
            depth = float(f.get("depth_m", 0.12))
            sd = s_arr[:, None] - s0
            td = ts[None, :] - tp
            d2 = sd**2 + td**2
            disp -= depth * np.exp(-d2 / (2 * (r * 0.6) ** 2))
            if typ == "mud_patch":
                mud |= d2 < (r * r)

    P[:, :, 2] += disp + lift
    logger.info(
        f"road surface grid {NS}x{NT}, ruts@±{rut_t:.2f}m, "
        f"{sum(1 for f in features if str(f.get('type','')).lower()=='washboard')} washboard, "
        f"{sum(1 for f in features if str(f.get('type','')).lower()=='pothole')} potholes, "
        f"{int(mud.any())} mud regions"
    )
    return s_arr, ts, P, mud


# --------------------------------------------------------------------------- #
# bpy mesh + materials
# --------------------------------------------------------------------------- #
def _dirt_material():
    import bpy

    mat = bpy.data.materials.new("road_dirt")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    # subtle noise-driven color variation
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 12.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.22, 0.16, 0.11, 1)
    ramp.color_ramp.elements[1].color = (0.40, 0.31, 0.22, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 1.0
    return mat


def _mud_material():
    import bpy

    mat = bpy.data.materials.new("road_mud")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.10, 0.07, 0.05, 1)
    bsdf.inputs["Roughness"].default_value = 0.25  # wet sheen
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.7
    return mat


def road_surface_to_mesh(road, features, seed=0, name="offroad_road", **kw):
    import bpy

    s_arr, ts, P, mud = build_surface_grid(road, features, seed=seed, **kw)
    NS, NT = P.shape[:2]
    verts = P.reshape(-1, 3)

    faces, face_is_mud = [], []
    for i in range(NS - 1):
        for j in range(NT - 1):
            v00 = i * NT + j
            v01 = i * NT + j + 1
            v11 = (i + 1) * NT + j + 1
            v10 = (i + 1) * NT + j
            faces.append((v00, v01, v11, v10))
            face_is_mud.append(mud[i, j] or mud[i, j + 1] or mud[i + 1, j] or mud[i + 1, j + 1])

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    dirt = _dirt_material()
    mud_mat = _mud_material()
    mesh.materials.append(dirt)
    mesh.materials.append(mud_mat)
    for p, is_mud in zip(mesh.polygons, face_is_mud):
        p.material_index = 1 if is_mud else 0
        p.use_smooth = True
    mesh.update()
    return obj
