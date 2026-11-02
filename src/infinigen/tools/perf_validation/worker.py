# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Render one side of a comparison, selecting `--optimized` before anything is built.
Objects keep the implementation they were built with, so the driver runs one process
per side."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import bpy

from infinigen.tools.perf_validation import geometry, render

_SELECTED: str | None = None

# Factories that scenes build through make_leaf_collection, at the
# decimate_rate its leaf callers pass, rather than spawn directly.
COLLECTION_LEAF_TYPES = {
    "LeafFactoryPine": "leaf_pine",
    "LeafFactoryGinko": "leaf_ginko",
    "LeafFactoryMaple": "leaf_maple",
    "LeafFactoryBroadleaf": "leaf_broadleaf",
    "LeafFactoryV2": "leaf_v2",
    "BerryFactory": "berry",
}
PIPELINE_DECIMATE_RATE = 0.97


def _build_asset(factory_cls, attr, seed):
    """Return the meshes a scene would actually contain for this factory."""
    leaf_type = COLLECTION_LEAF_TYPES.get(attr)
    if leaf_type is None:
        return [factory_cls(seed).spawn_asset(seed)]

    from infinigen.assets.objects.trees.generate import (
        make_leaf_collection,
        random_species,
    )

    (_, _, leaf_params), _ = random_species(season=None)
    col = make_leaf_collection(
        seed, leaf_params, 1, leaf_type, decimate_rate=PIPELINE_DECIMATE_RATE
    )
    objs = [o for o in col.objects if o.type == "MESH"]
    for o in objs:  # the builder hides the collection on its way out
        o.hide_viewport = False
        o.hide_render = False
    return objs


def _selected() -> str | None:
    """What this worker was told to select, for the manifest."""
    return _SELECTED


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def do_assets(args) -> dict:
    from infinigen.core.init import apply_gin_configs  # noqa: F401
    from infinigen.core.util.math import FixedSeed

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from infinigen import perf

    print(f"  {perf.status()}", flush=True)

    rows = []

    for factory_name in args.factories:
        module, _, attr = factory_name.rpartition(".")
        factory_cls = getattr(__import__(module, fromlist=[attr]), attr)
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            _reset()
            started = time.perf_counter()
            with FixedSeed(seed):
                objs = _build_asset(factory_cls, attr, seed)
            build_seconds = time.perf_counter() - started
            centre, extent, radius = render.bounds(objs)
            tris = render.triangles(objs)
            samples = f"{attr}_{seed}_samples.npz"
            digest = geometry.record(objs, out / samples)
            # a single-mode rerun must not drop the other view
            prior = pathlib.Path(args.manifest)
            existing = {}
            if prior.exists():
                for row in json.loads(prior.read_text()).get("rows", []):
                    existing[(row["target"], row["seed"])] = row.get("images", {})
            images = dict(existing.get((attr, seed), {}))
            aim = render.focus(objs)
            trunk_aim, trunk_window = (
                render.trunk_focus(objs) if "trunk" in args.modes else (None, None)
            )
            for mode in args.modes:
                path = out / f"{attr}_{seed}_{mode}.png"
                scene = render.make_scene(
                    f"{attr}_{seed}",
                    mode,
                    centre,
                    extent,
                    radius,
                    trunk_aim if mode == "trunk" else aim,
                    window=trunk_window if mode == "trunk" else None,
                )
                render.render(scene, objs, path, children=mode != "trunk")
                images[mode] = path.name
            rows.append(
                {
                    "target": attr,
                    "seed": seed,
                    "triangles": tris,
                    "mesh_digest": digest,
                    "samples": samples,
                    "build_seconds": round(build_seconds, 3),
                    "images": images,
                }
            )
            print(
                f"  {attr} seed {seed}: {tris:,} triangles, "
                f"built in {build_seconds:.2f}s",
                flush=True,
            )

    return {"kind": "assets", "optimized": _selected(), "rows": rows}


def do_scene(args) -> dict:
    """Render a scene blend produced by generate_nature."""
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # a single-mode rerun must not drop the other view, nor its count
    prior = pathlib.Path(args.manifest)
    images, tris, assets = {}, 0, None
    if prior.exists():
        for row in json.loads(prior.read_text()).get("rows", []):
            if row["target"] == args.name:
                images = dict(row.get("images", {}))
                tris = row.get("triangles", 0)
                assets = row.get("asset_triangles")
    for mode in args.modes:
        # reopen per mode: prepare_scene edits the scene itself
        bpy.ops.wm.open_mainfile(filepath=args.blend)
        scene = bpy.context.scene
        if not tris:
            tris = render.triangles(
                [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
            )
        if assets is None:
            assets = render.asset_triangles()
        render.prepare_scene(scene, mode, args.name)
        if args.gpu:
            render.use_gpu(scene)
        path = out / f"{args.name}_{mode}.png"
        started = time.perf_counter()
        render.render_scene(scene, path)
        render_seconds = time.perf_counter() - started
        images[mode] = path.name
    print(
        f"  {args.name}: {tris:,} triangles, rendered in {render_seconds:.1f}s",
        flush=True,
    )
    return {
        "kind": "scene",
        "optimized": _selected(),
        "rows": [
            {
                "target": args.name,
                "seed": 0,
                "triangles": tris,
                "asset_triangles": assets,
                "render_seconds": round(render_seconds, 2),
                "images": images,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimized",
        nargs="?",
        const="all",
        default=None,
        metavar="NAMES",
        help="select optimized implementations: all, or trees,leaves",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    a = sub.add_parser("assets")
    a.add_argument("--factories", nargs="+", required=True)
    a.add_argument("--seeds", type=int, default=10)
    a.add_argument("--seed-start", type=int, default=0)
    a.add_argument(
        "--modes", nargs="+", default=["shaded", "wireframe"], choices=render.MODES
    )
    a.add_argument("--output", required=True)
    a.add_argument("--manifest", required=True)

    s = sub.add_parser("scene")
    s.add_argument("--blend", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--modes", nargs="+", default=["shaded", "wireframe"])
    s.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    global _SELECTED
    _SELECTED = args.optimized
    if args.optimized is not None:
        import infinigen

        infinigen.use_optimized(
            True
            if args.optimized == "all"
            else [n.strip() for n in args.optimized.split(",") if n.strip()]
        )
    # before any render: Cycles picks its scratch directory up front
    render.use_writable_temp(pathlib.Path(args.output) / "_scratch")
    result = do_assets(args) if args.mode == "assets" else do_scene(args)
    pathlib.Path(args.manifest).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
