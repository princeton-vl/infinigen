"""Headless preview (no Blender): top-down height map + perpendicular cross
sections, to visually confirm the road conforms with no wall.

    python -m offroad.preview --out outputs/offroad/preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offroad.conform_terrain import build_heightfield, nearest_road
from offroad.road_spec import synthetic_s_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/offroad/preview.png"))
    ap.add_argument("--amplitude", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    road = synthetic_s_curve()
    hf = build_heightfield(
        road, margin=55, spacing=1.0, amplitude=args.amplitude,
        transition=14.0, shoulder=1.5, noise_seed=args.seed,
    )
    c = road.centerline

    fig = plt.figure(figsize=(13, 6))

    # (a) top-down height map + road centerline
    ax = fig.add_subplot(1, 2, 1)
    extent = [hf.X.min(), hf.X.max(), hf.Y.min(), hf.Y.max()]
    im = ax.imshow(hf.Z, origin="lower", extent=extent, cmap="terrain", aspect="equal")
    ax.plot(c[:, 0], c[:, 1], "r-", lw=2, label="road centerline")
    # mark the cross-section locations
    xs_marks = np.linspace(c[:, 0].min() + 10, c[:, 0].max() - 10, 3)
    for xm in xs_marks:
        ax.axvline(xm, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("Top-down ground height (terrain cmap) + road")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="height z [m]")

    # (b) perpendicular cross sections at three x positions
    ax2 = fig.add_subplot(1, 2, 2)
    nx = hf.shape[1]
    xs = hf.X[0]
    for xm in xs_marks:
        col = int(np.argmin(np.abs(xs - xm)))
        y = hf.Y[:, col]
        z = hf.Z[:, col]
        d, zr = nearest_road(hf.X[:, col], y, c)
        ax2.plot(y, z, lw=1.5, label=f"x={xm:.0f} m")
        on = d <= road.half_width
        if on.any():
            ax2.plot(y[on], z[on], "r.", ms=4)
    ax2.set_title("Perpendicular cross-sections\n(red dots = carriageway, flat & on-grade)")
    ax2.set_xlabel("y [m]"); ax2.set_ylabel("height z [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal", adjustable="datalim")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
