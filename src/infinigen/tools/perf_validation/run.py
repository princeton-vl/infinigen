# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Compare two selections side by side, in renders and wireframes.

The baseline selects nothing; the other side selects `--optimized` (given before the
subcommand, all by default) as docs/source/OptimizedImplementations.md describes,
e.g. `python -m infinigen.tools.perf_validation.run --optimized=trees assets`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

# Representative factories of the optimized packages; keep the list short
# enough to run. Factories no optimized implementation reaches (BoulderFactory,
# leaf_pine.LeafFactoryPine) are left out: they would compare the default with
# itself.
FACTORIES = [
    "infinigen.assets.objects.trees.generate.TreeFactory",
    "infinigen.assets.objects.trees.generate.BushFactory",
    "infinigen.assets.objects.leaves.leaf_v2.LeafFactoryV2",
    "infinigen.assets.objects.monocot.grasses.GrassesMonocotFactory",
    "infinigen.assets.objects.monocot.agave.AgaveMonocotFactory",
    "infinigen.assets.objects.monocot.pinecone.PineconeFactory",
    "infinigen.assets.objects.small_plants.fern.FernFactory",
    "infinigen.assets.objects.mushroom.generate.MushroomFactory",
    "infinigen.assets.objects.creatures.bird.BirdFactory",
    "infinigen.assets.objects.corals.generate.BrainCoralFactory",
    "infinigen.assets.objects.corals.generate.CauliflowerCoralFactory",
    "infinigen.assets.objects.corals.generate.StarCoralFactory",
]

# The optimized leaves a scene can contain: the leaf_* cases of
# random_tree_child_factory other than leaf_pine, which has no optimized
# implementation.
LEAF_FACTORIES = [
    "infinigen.assets.objects.leaves.leaf_v2.LeafFactoryV2",
    "infinigen.assets.objects.leaves.leaf_broadleaf.LeafFactoryBroadleaf",
    "infinigen.assets.objects.leaves.leaf_maple.LeafFactoryMaple",
    "infinigen.assets.objects.leaves.leaf_ginko.LeafFactoryGinko",
]

SCENES = [
    "arctic",
    "canyon",
    "cave",
    "coast",
    "desert",
    "forest",
    "mountain",
    "plain",
    "river",
    "snowy_mountain",
]

REPO = pathlib.Path(__file__).resolve().parents[4]
EXPECTED = pathlib.Path(__file__).resolve().parent / "expected"


def _select(names: str | None) -> list[str]:
    """Worker arguments that select `names`, or none."""
    return [] if names is None else ["--optimized", names]


def _generate_command(scene, work, tasks, names: str | None) -> list[str]:
    """The generate_nature call for one scene, selecting as a user would, with every
    config in one `--configs` list since generate_nature keeps only the last."""
    configs, overrides = [scene, "simple", "dev"], []
    if names == "all":
        configs.append("performance")
    elif names is not None:
        packages = [n.strip() for n in names.split(",") if n.strip()]
        overrides = ["-p", f"optimized.packages={packages!r}"]
    return [
        sys.executable,
        "-m",
        "infinigen_examples.generate_nature",
        "--seed",
        "0",
        "--task",
        *tasks,
        "--output_folder",
        str(work),
        "--configs",
        *configs,
        *overrides,
    ]


def _one_cpu() -> None:
    """Pin the worker to one CPU before Blender starts and sizes its thread pool.
    Blender's multithreaded Subdivision Surface is not bit-reproducible, which can
    change the triangle count of an asset that raycasts against a subdivided mesh
    (the bird's tail)."""
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def _worker(
    out: pathlib.Path, manifest: pathlib.Path, argv: list[str], one_cpu=False
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "infinigen.tools.perf_validation.worker",
        *argv,
        "--output",
        str(out),
        "--manifest",
        str(manifest),
    ]
    pin = one_cpu and hasattr(os, "sched_setaffinity")
    subprocess.run(cmd, cwd=REPO, check=True, preexec_fn=_one_cpu if pin else None)


def run_assets(args) -> list[pathlib.Path]:
    manifests = []
    if not hasattr(os, "sched_setaffinity"):
        print(
            "   cannot confine the build to one CPU on this platform: a count "
            "may differ from expected/ by a few triangles (see _one_cpu)",
            flush=True,
        )
    sides = (("default", None), ("optimized", args.optimized))
    if getattr(args, "side", "both") != "both":
        sides = tuple(s for s in sides if s[0] == args.side)
    for label, names in sides:
        suffix = f"_{args.seed_start}" if args.seed_start else ""
        out = pathlib.Path(args.output) / f"{label}{suffix}"
        manifest = pathlib.Path(args.output) / f"{label}{suffix}.json"
        print(f"== {label} ({names or 'nothing selected'})", flush=True)
        _worker(
            out,
            manifest,
            [
                *_select(names),
                "assets",
                "--factories",
                *args.factories,
                "--seeds",
                str(args.seeds),
                "--seed-start",
                str(args.seed_start),
                "--modes",
                *args.modes,
            ],
            one_cpu=True,
        )
        manifests.append(manifest)
    return manifests


def run_scenes(args) -> list[pathlib.Path]:
    """Generate each scene for each side, then render them. `--full` also populates the
    scene and meshes the fine terrain, since a coarse scene holds only placeholders;
    `--stage generate` only generates and times it, `--stage render` only renders the
    blends left by a generate stage (on a GPU with `--gpu`)."""
    tasks = ["coarse", "populate", "fine_terrain"] if args.full else ["coarse"]
    stage = getattr(args, "stage", "all")
    manifests = []
    sides = (("default", None), ("optimized", args.optimized))
    if getattr(args, "side", "both") != "both":
        sides = tuple(x for x in sides if x[0] == args.side)
    for label, names in sides:
        out = pathlib.Path(args.output) / label
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for scene in args.scenes:
            work = out / scene
            print(f"== {label}/{scene}", flush=True)
            reuse = stage == "render" or getattr(args, "reuse_blends", False)
            if reuse and (work / "scene.blend").exists():
                print("   reusing the blend already generated", flush=True)
            elif stage == "render":
                print(f"   {scene}: no blend to render, skipped", flush=True)
                continue
            else:
                started = time.perf_counter()
                subprocess.run(
                    _generate_command(scene, work, tasks, names),
                    cwd=REPO,
                    check=False,
                )
                (work / "generate.json").write_text(
                    json.dumps(
                        {
                            "tasks": tasks,
                            "generate_seconds": round(time.perf_counter() - started, 1),
                        }
                    )
                )
            blend = work / "scene.blend"
            if not blend.exists():
                print(f"   {scene}: no blend produced, skipped", flush=True)
                continue
            if stage == "generate":
                continue
            part = out / f"{scene}.json"
            _worker(
                out,
                part,
                [
                    *_select(names),
                    "scene",
                    "--blend",
                    str(blend),
                    "--name",
                    scene,
                    "--modes",
                    *args.modes,
                    *(["--gpu"] if getattr(args, "gpu", False) else []),
                ],
            )
            generated = work / "generate.json"
            for row in json.loads(part.read_text())["rows"]:
                if generated.exists():
                    row["build_seconds"] = json.loads(generated.read_text())[
                        "generate_seconds"
                    ]
                rows.append(row)
        if stage == "generate":
            continue
        manifest = pathlib.Path(args.output) / f"{label}.json"
        manifest.write_text(
            json.dumps(
                {
                    "kind": "scenes",
                    "optimized": names,
                    "tasks": tasks,
                    "rows": rows,
                },
                indent=2,
            )
        )
        manifests.append(manifest)
    return manifests


def _rows(path: pathlib.Path) -> dict:
    """`{(factory, seed): triangles}` from one manifest."""
    part = json.loads(path.read_text())
    return {(r["target"], r["seed"]): r["triangles"] for r in part.get("rows", [])}


def _tolerance(path: pathlib.Path) -> float:
    """Relative count difference a manifest allows, set (with its reason) only for
    implementations that are not reproducible even on one CPU."""
    return float(json.loads(path.read_text()).get("tolerance", 0.0))


def run_check(args) -> int:
    """Compare a run with the triangle counts in `expected/`. Its reference images are
    only for viewing, since Cycles renders vary across machines and versions."""
    problems, checked = [], 0
    for d in sorted(pathlib.Path(p) for p in args.dirs):
        for side in ("default", "optimized"):
            # a run split by --seed-start leaves one manifest per slice
            got_paths = sorted(d.glob(f"{side}*.json"))
            want_path = EXPECTED / d.name / f"{side}.json"
            if not got_paths:
                continue
            if not want_path.exists():
                print(f"   {d.name}/{side}: nothing expected, skipped", flush=True)
                continue
            got, want = {}, _rows(want_path)
            for got_path in got_paths:
                got.update(_rows(got_path))
            tolerance = max(args.tolerance, _tolerance(want_path))
            for key in sorted(want.keys() & got.keys()):
                checked += 1
                a, b = want[key], got[key]
                if a == b:
                    continue
                delta = abs(b - a) / max(a, 1)
                if delta > tolerance:
                    problems.append(
                        f"{d.name}/{side} {key[0]} seed {key[1]}: "
                        f"expected {a:,}, got {b:,} ({delta:+.3%})"
                    )
            for key in sorted(want.keys() - got.keys()):
                problems.append(f"{d.name}/{side} {key[0]} seed {key[1]}: not built")
    for line in problems:
        print(f"   {line}", flush=True)
    print(
        f"{checked} counts checked, {len(problems)} differing"
        + (f" (tolerance {args.tolerance:%})" if args.tolerance else ""),
        flush=True,
    )
    return 1 if problems else 0


def run_merge(args) -> None:
    """Build one page from output directories produced by separate runs."""
    manifests = []
    for d in sorted(pathlib.Path(p) for p in args.dirs):
        # a factory split by --seed-start leaves one manifest per slice
        for base in sorted(d.glob("default*.json")):
            opt = base.with_name("optimized" + base.stem[len("default") :] + ".json")
            if not opt.exists():
                print(
                    f"   {d.name}/{base.stem}: no optimized side, skipped", flush=True
                )
                continue
            manifests += [base, opt]
    if not manifests:
        raise SystemExit("no complete runs to merge")

    from infinigen.tools.perf_validation.page import build

    page = pathlib.Path(args.output) / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    for line in build(manifests, page):
        print(f"   {line}", flush=True)
    print(f"wrote {page} from {len(manifests) // 2} run(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--optimized",
        nargs="?",
        const="all",
        default="all",
        metavar="NAMES",
        help="what the optimized side selects: all, or trees,leaves",
    )
    parser.add_argument("--output", default="visual_check_out")
    sub = parser.add_subparsers(dest="what", required=True)

    a = sub.add_parser("assets")
    a.add_argument("--factories", nargs="+", default=FACTORIES)
    a.add_argument("--seeds", type=int, default=10)
    a.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed, so a long factory can be split across jobs",
    )
    a.add_argument(
        "--modes",
        nargs="+",
        default=["shaded", "wireframe"],
        choices=("shaded", "wireframe", "trunk"),
        help="the views to render: trunk is a close wireframe of the asset's "
        "own mesh with its children hidden (for trees and bushes)",
    )
    a.add_argument(
        "--side",
        choices=("default", "optimized", "both"),
        default="both",
        help="render one side only, to be merged later",
    )

    m = sub.add_parser("merge")
    m.add_argument("dirs", nargs="+")

    c = sub.add_parser(
        "check",
        help="compare a run against the expected counts shipped with the tool",
    )
    c.add_argument("dirs", nargs="+")
    c.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="allowed fractional difference in triangles (default: exact)",
    )

    s = sub.add_parser("scenes")
    s.add_argument("--scenes", nargs="+", default=SCENES)
    s.add_argument(
        "--modes",
        nargs="+",
        default=["shaded"],
        choices=("shaded", "wireframe"),
        help="scenes are compared shaded; a whole-scene wireframe shows little",
    )
    s.add_argument(
        "--side",
        choices=("default", "optimized", "both"),
        default="both",
        help="render one side only, to be merged later",
    )
    s.add_argument(
        "--full",
        action="store_true",
        help="populate the scene and mesh the fine terrain too (much slower)",
    )
    s.add_argument(
        "--stage",
        choices=("all", "generate", "render"),
        default="all",
        help="generate only, or render only the blends already generated",
    )
    s.add_argument(
        "--gpu",
        action="store_true",
        help="render on a GPU (the render stage fails if Cycles finds none)",
    )
    s.add_argument(
        "--reuse-blends",
        action="store_true",
        help="re-render scenes already generated instead of generating again",
    )

    args = parser.parse_args()
    if args.what == "merge":
        run_merge(args)
        return
    if args.what == "check":
        raise SystemExit(run_check(args))
    manifests = run_assets(args) if args.what == "assets" else run_scenes(args)

    if len(manifests) < 2:
        print("one side only; assemble with the merge subcommand")
        return

    from infinigen.tools.perf_validation.page import build

    page = pathlib.Path(args.output) / "index.html"
    for line in build(manifests, page):
        print(f"   {line}", flush=True)
    print(f"\nwrote {page}")


if __name__ == "__main__":
    main()
