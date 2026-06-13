"""Road-anchored ground heightfield (the heart of "environment follows road").

Core formula, evaluated per ground point (x, y):

    H(x, y) = z_road(x, y)  +  noise(x, y) * amplitude * falloff(dist_to_road)

where
    z_road(x, y)       elevation of the NEAREST point on the road centerline
                       (the road's own z-profile, extended sideways) -> the anchor
    dist_to_road       horizontal distance to the centerline
    falloff(d)         0 on the carriageway, smoothly ramps to 1 past a
                       transition band, so natural undulation only "grows" away
                       from the road.

Consequences:
    - On the road (d <= half_width): falloff = 0  => H == z_road exactly.
      The road sits perfectly on the ground, no floating, no vertical wall.
    - Far from the road: falloff = 1 => full procedural terrain on top of the
      road-elevation base.
    - In between: a smooth, bounded-slope embankment / cutting.

Everything here is pure NumPy and importable without Blender, so the "no wall"
property can be validated headless. The Blender mesh construction lives in
`build_terrain.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from offroad.road_spec import RoadSpec


# --------------------------------------------------------------------------- #
# Deterministic fractal value noise (no external deps)
# --------------------------------------------------------------------------- #
def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized integer hash -> float in [0, 1) for lattice corners."""
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    h = (ix * np.int64(374761393) + iy * np.int64(668265263)) ^ np.int64(seed)
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int, freq: float) -> np.ndarray:
    xf = x * freq
    yf = y * freq
    x0 = np.floor(xf)
    y0 = np.floor(yf)
    tx = _smoothstep(xf - x0)
    ty = _smoothstep(yf - y0)
    x0i, y0i = x0.astype(np.int64), y0.astype(np.int64)
    v00 = _hash01(x0i, y0i, seed)
    v10 = _hash01(x0i + 1, y0i, seed)
    v01 = _hash01(x0i, y0i + 1, seed)
    v11 = _hash01(x0i + 1, y0i + 1, seed)
    a = v00 * (1 - tx) + v10 * tx
    b = v01 * (1 - tx) + v11 * tx
    return (a * (1 - ty) + b * ty) * 2.0 - 1.0  # -> [-1, 1]


def fractal_noise(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    octaves: int = 5,
    base_freq: float = 0.012,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> np.ndarray:
    """Sum of value-noise octaves, normalized to roughly [-1, 1]."""
    total = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = base_freq
    norm = 0.0
    for o in range(octaves):
        total += amp * _value_noise_2d(x, y, seed + o * 1013, freq)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / max(norm, 1e-9)


# --------------------------------------------------------------------------- #
# Distance + elevation to the road centerline
# --------------------------------------------------------------------------- #
def nearest_road(
    qx: np.ndarray, qy: np.ndarray, centerline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each query point, horizontal distance to the centerline polyline and
    the (interpolated) road elevation at the nearest projection.

    qx, qy : (P,) arrays. centerline : (K, 3). Returns (dist (P,), z_road (P,)).
    Vectorized over points, loops over the K-1 segments (K is small).
    """
    P = qx.shape[0]
    best_d2 = np.full(P, np.inf)
    best_z = np.zeros(P)
    A = centerline[:-1]
    B = centerline[1:]
    for a, b in zip(A, B):
        abx, aby = b[0] - a[0], b[1] - a[1]
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            continue
        apx = qx - a[0]
        apy = qy - a[1]
        t = (apx * abx + apy * aby) / ab2
        t = np.clip(t, 0.0, 1.0)
        projx = a[0] + t * abx
        projy = a[1] + t * aby
        dx = qx - projx
        dy = qy - projy
        d2 = dx * dx + dy * dy
        upd = d2 < best_d2
        best_d2 = np.where(upd, d2, best_d2)
        z_proj = a[2] + t * (b[2] - a[2])
        best_z = np.where(upd, z_proj, best_z)
    return np.sqrt(best_d2), best_z


def nearest_road_smooth(
    qx: np.ndarray, qy: np.ndarray, centerline: np.ndarray, smoothing: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Like `nearest_road`, but the returned elevation is an inverse-distance
    weighted blend over ALL segments (weight 1/(d^2 + smoothing^2)) instead of a
    hard nearest pick. This removes the elevation "seam" that a winding road
    creates where the nearest segment switches between two passes at different
    heights. `dist` is still the true minimum distance.
    """
    P = qx.shape[0]
    best_d2 = np.full(P, np.inf)
    sum_w = np.zeros(P)
    sum_wz = np.zeros(P)
    s2 = smoothing * smoothing
    A, B = centerline[:-1], centerline[1:]
    for a, b in zip(A, B):
        abx, aby = b[0] - a[0], b[1] - a[1]
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            continue
        t = np.clip(((qx - a[0]) * abx + (qy - a[1]) * aby) / ab2, 0.0, 1.0)
        projx = a[0] + t * abx
        projy = a[1] + t * aby
        d2 = (qx - projx) ** 2 + (qy - projy) ** 2
        z_proj = a[2] + t * (b[2] - a[2])
        w = 1.0 / (d2 + s2)
        sum_w += w
        sum_wz += w * z_proj
        best_d2 = np.minimum(best_d2, d2)
    z = sum_wz / np.maximum(sum_w, 1e-12)
    return np.sqrt(best_d2), z


def road_falloff(
    dist: np.ndarray, half_width: float, transition: float, shoulder: float = 0.0
) -> np.ndarray:
    """0 on the carriageway (+optional flat shoulder), smoothstep to 1 over
    `transition` meters past the shoulder edge."""
    flat = half_width + shoulder
    t = (dist - flat) / max(transition, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    return _smoothstep(t)


# --------------------------------------------------------------------------- #
# Heightfield assembly
# --------------------------------------------------------------------------- #
@dataclass
class Heightfield:
    X: np.ndarray  # (ny, nx) meshgrid X
    Y: np.ndarray  # (ny, nx) meshgrid Y
    Z: np.ndarray  # (ny, nx) ground height
    dist: np.ndarray  # (ny, nx) distance to road
    z_road: np.ndarray  # (ny, nx) nearest road elevation (the anchor)
    spacing: float

    @property
    def shape(self):
        return self.X.shape


def build_heightfield(
    road: RoadSpec,
    margin: float = 60.0,
    spacing: float = 1.0,
    amplitude: float = 9.0,
    transition: float = 14.0,
    shoulder: float = 1.5,
    noise_seed: int = 0,
    noise_kwargs: dict | None = None,
    elev_smoothing: float = 4.0,
) -> Heightfield:
    """Build the road-anchored ground heightfield over the road's bounding box.

    `elev_smoothing` > 0 uses a distance-weighted elevation anchor (removes the
    seam where a winding road's nearest segment switches between passes at
    different heights). 0 falls back to hard nearest-segment elevation.
    """
    road = road.resampled(spacing=min(spacing, 1.0))
    x0, y0, x1, y1 = road.xy_bounds(margin)
    nx = int(np.ceil((x1 - x0) / spacing)) + 1
    ny = int(np.ceil((y1 - y0) / spacing)) + 1
    xs = x0 + np.arange(nx) * spacing
    ys = y0 + np.arange(ny) * spacing
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)

    if elev_smoothing > 0:
        dist, z_road = nearest_road_smooth(
            X.ravel(), Y.ravel(), road.centerline, smoothing=elev_smoothing
        )
    else:
        dist, z_road = nearest_road(X.ravel(), Y.ravel(), road.centerline)
    dist = dist.reshape(X.shape)
    z_road = z_road.reshape(X.shape)

    noise = fractal_noise(X, Y, seed=noise_seed, **(noise_kwargs or {}))
    fall = road_falloff(dist, road.half_width, transition, shoulder)

    Z = z_road + noise * amplitude * fall
    return Heightfield(X=X, Y=Y, Z=Z, dist=dist, z_road=z_road, spacing=spacing)


# --------------------------------------------------------------------------- #
# Headless validation: prove there is no "wall"
# --------------------------------------------------------------------------- #
def validate(hf: Heightfield, road: RoadSpec) -> dict:
    """Numeric checks that the road conforms with no vertical wall."""
    on_road = hf.dist <= road.half_width
    # 1) On the carriageway, ground must equal road elevation (the anchor).
    road_err = float(np.abs(hf.Z[on_road] - hf.z_road[on_road]).max()) if on_road.any() else 0.0

    # 2) Max local slope (a wall would be near-vertical, i.e. huge slope).
    gx = np.gradient(hf.Z, hf.spacing, axis=1)
    gy = np.gradient(hf.Z, hf.spacing, axis=0)
    slope = np.sqrt(gx**2 + gy**2)  # rise per meter
    max_slope = float(slope.max())
    max_slope_deg = float(np.degrees(np.arctan(max_slope)))

    # 3) Slope specifically in the transition band (the at-risk zone).
    band = (hf.dist > road.half_width) & (hf.dist <= road.half_width + 16.0)
    band_slope_deg = (
        float(np.degrees(np.arctan(slope[band].max()))) if band.any() else 0.0
    )
    return {
        "grid": hf.shape,
        "road_conform_err_m": road_err,  # want ~0
        "max_slope_deg": max_slope_deg,  # want < ~80 (no vertical wall)
        "transition_max_slope_deg": band_slope_deg,
        "z_range_m": (float(hf.Z.min()), float(hf.Z.max())),
    }


def ascii_cross_section(hf: Heightfield, road: RoadSpec, width: int = 70) -> str:
    """A perpendicular cross-section through the middle of the path, as ASCII,
    so you can 'see' that the road is flat and the sides rise smoothly."""
    ny, nx = hf.shape
    row = ny // 2
    z = hf.Z[row]
    d = hf.dist[row]
    on = d <= road.half_width
    zmin, zmax = z.min(), z.max()
    span = max(zmax - zmin, 1e-6)
    height = 16
    # downsample columns to `width`
    idx = np.linspace(0, nx - 1, width).astype(int)
    cols = []
    for i in idx:
        h = int(round((z[i] - zmin) / span * (height - 1)))
        cols.append((h, on[i]))
    lines = []
    for lv in range(height - 1, -1, -1):
        line = []
        for h, ison in cols:
            if h == lv:
                line.append("#" if ison else "*")
            elif h > lv:
                line.append("#" if ison else ".")
            else:
                line.append(" ")
        lines.append("".join(line))
    legend = f"  (# = road/below-road, * = ground surface; z {zmin:.1f}..{zmax:.1f} m)"
    return "\n".join(lines) + "\n" + legend
