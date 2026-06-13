"""Road path specification (the authoritative input).

A `RoadSpec` is just a 3D centerline polyline (x, y, z in meters) plus a road
width. For now we provide a hand-made synthetic example so we can validate the
terrain-conforming math before wiring up a real OpenDRIVE (.xodr) parser.

`from_xodr` is a stub for the next step.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RoadSpec:
    # centerline: (K, 3) array of [x, y, z] points in meters, ordered along the road
    centerline: np.ndarray
    width: float = 6.0  # full carriageway width in meters
    # optional point obstacles parsed from the source (e.g. xodr <objects>):
    # list of dicts with keys: id, x, y, z, radius, height, kind
    obstacles: list = field(default_factory=list)

    def __post_init__(self):
        self.centerline = np.asarray(self.centerline, dtype=np.float64)
        assert self.centerline.ndim == 2 and self.centerline.shape[1] == 3, (
            f"centerline must be (K,3), got {self.centerline.shape}"
        )

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def xy_bounds(self, margin: float) -> tuple[float, float, float, float]:
        xy = self.centerline[:, :2]
        lo = xy.min(axis=0) - margin
        hi = xy.max(axis=0) + margin
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def resampled(self, spacing: float = 1.0) -> "RoadSpec":
        """Return a copy with the centerline resampled to ~`spacing` meters.

        Dense, evenly spaced points make the point-to-polyline distance accurate.
        """
        c = self.centerline
        seg = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])  # arclength (xy)
        total = s[-1]
        n = max(2, int(np.ceil(total / spacing)) + 1)
        s_new = np.linspace(0.0, total, n)
        out = np.empty((n, 3), dtype=np.float64)
        for d in range(3):
            out[:, d] = np.interp(s_new, s, c[:, d])
        return RoadSpec(out, self.width)


def synthetic_s_curve(
    length: float = 160.0,
    amp: float = 28.0,
    grade: float = 0.04,
    hill_height: float = 6.0,
    width: float = 6.0,
) -> RoadSpec:
    """A deliberately stressful test path:

    - S-shaped in the XY plane (left/right curves),
    - a constant uphill `grade` (rise/run), PLUS
    - a smooth hill bump in the middle of the run.

    The vertical variation is exactly what would create plateaus/canyons under a
    naive flatten, so it is a good test of the road-anchored approach.
    """
    n = 400
    s = np.linspace(0.0, length, n)
    x = s
    y = amp * np.sin(2 * np.pi * s / length)  # one full S
    z = (
        grade * s  # steady climb
        + hill_height * np.exp(-(((s - length * 0.5) / (length * 0.12)) ** 2))  # hill
    )
    centerline = np.stack([x, y, z], axis=1)
    return RoadSpec(centerline, width=width)


def _eval_poly(records: list[tuple[float, float, float, float, float]], s: float) -> float:
    """Evaluate an OpenDRIVE cubic record list [(s0,a,b,c,d), ...] at arclength s.
    z(s) = a + b*ds + c*ds^2 + d*ds^3, ds = s - s0, using the record with the
    largest s0 <= s."""
    if not records:
        return 0.0
    rec = records[0]
    for r in records:
        if r[0] <= s + 1e-9:
            rec = r
        else:
            break
    s0, a, b, c, d = rec
    ds = s - s0
    return a + b * ds + c * ds * ds + d * ds * ds * ds


def from_xodr(path: str, lateral_sign: float = +1.0) -> RoadSpec:
    """Parse a (line-segment planView) OpenDRIVE file into a RoadSpec.

    Handles the dialect emitted by the offroad toolchain: planView is a dense
    list of <geometry><line/> segments; elevationProfile is a list of cubic
    <elevation> records; lane widths give the carriageway width; <objects> give
    point obstacles (boulders) by (s, t).

    `lateral_sign` flips the sign convention for the lateral `t` offset if the
    obstacles end up on the wrong side (OpenDRIVE t>0 is left of travel).
    """
    root = ET.parse(path).getroot()
    road = root.find("road")
    if road is None:
        raise ValueError(f"{path}: no <road> element")

    # --- planView -> centerline XY (collect each geometry start, then final end)
    geoms = road.findall("./planView/geometry")
    if not geoms:
        raise ValueError(f"{path}: empty planView")
    s_list, xy = [], []
    for g in geoms:
        s_list.append(float(g.get("s")))
        xy.append((float(g.get("x")), float(g.get("y"))))
    # append the terminal point of the last segment
    last = geoms[-1]
    hdg = float(last.get("hdg"))
    length = float(last.get("length"))
    s_list.append(float(last.get("s")) + length)
    xy.append((xy[-1][0] + length * np.cos(hdg), xy[-1][1] + length * np.sin(hdg)))
    xy = np.asarray(xy, dtype=np.float64)
    s_arr = np.asarray(s_list, dtype=np.float64)

    # --- elevationProfile -> z(s)
    elev_records = []
    for e in road.findall("./elevationProfile/elevation"):
        elev_records.append(
            (float(e.get("s")), float(e.get("a")), float(e.get("b")),
             float(e.get("c")), float(e.get("d")))
        )
    elev_records.sort(key=lambda r: r[0])
    z = np.array([_eval_poly(elev_records, s) for s in s_arr])

    centerline = np.stack([xy[:, 0], xy[:, 1], z], axis=1)

    # --- lane widths -> carriageway width (sum of |a| over driving lanes)
    width = 0.0
    for w in road.findall(".//laneSection/*/lane/width"):
        width += abs(float(w.get("a", 0.0)))
    if width <= 0.0:
        width = 6.0

    spec = RoadSpec(centerline, width=width)

    # --- objects -> obstacles in world coords (s,t) -> (x,y,z)
    def _frame_at(s_query: float):
        i = int(np.clip(np.searchsorted(s_arr, s_query) - 1, 0, len(s_arr) - 2))
        seg = s_arr[i + 1] - s_arr[i]
        u = 0.0 if seg < 1e-9 else (s_query - s_arr[i]) / seg
        p = centerline[i] * (1 - u) + centerline[i + 1] * u
        d = centerline[i + 1][:2] - centerline[i][:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        left = np.array([-d[1], d[0]])  # left-normal in XY
        return p, left

    for obj in road.findall("./objects/object"):
        s_q = float(obj.get("s", 0.0))
        t = float(obj.get("t", 0.0)) * lateral_sign
        p, left = _frame_at(s_q)
        wx = p[0] + left[0] * t
        wy = p[1] + left[1] * t
        spec.obstacles.append({
            "id": obj.get("id", obj.get("name", "obj")),
            "kind": obj.get("type", "obstacle"),
            "x": float(wx), "y": float(wy), "z": float(p[2]),
            "radius": float(obj.get("radius", 0.4)),
            "height": float(obj.get("height", 0.6)),
        })

    return spec
