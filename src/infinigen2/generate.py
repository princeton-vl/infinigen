#!/usr/bin/env python
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

# PYTHON_ARGCOMPLETE_OK

# ruff: noqa: I001, E402

import argparse
import ast
import inspect
import itertools
import json
import logging
import os
import pprint
import shutil
import signal
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable
import bpy
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# ruff: noqa: E402
# NOTE: logging config has to be before imports that use logging
logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

import procfunc as pf
import procfunc.compute_graph as cg
from procfunc.compute_graph.operators_info import OPERATORS_TO_FUNCTIONS
from procfunc.nodes import NODE_OPERATOR_TABLE
from procfunc.tracer import TraceLevel
from procfunc import codegen

from infinigen2.exporters.util.format import (
    ExportType,
    GT_PASS_DEFAULTS,
    MAINRENDER_PASS_DEFAULTS,
    RenderPass,
    SCENE_PASS_DEFAULTS,
)
from infinigen2.exporters.util.blender_render import DisplacementMode
from infinigen2 import GENERATORS_MANIFEST
from infinigen2.scenes.placement_utils import delete_object
from infinigen2.exporters.realize_mesh import realize_scene
from procfunc.util.manifest import import_item
from procfunc.util.teardown import skip_teardown_on_exit
from infinigen2.util.hardware_info import get_hardware_info
from infinigen2.util.codestats import compute_stats
from infinigen2 import graph_json

logger = logging.getLogger(__name__)


def _parse_seed(value: str) -> int:
    return int(value, 0)


def get_parser():
    """Get the argument parser for the infinigen2 generate command."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "generators",
        nargs="+",
        type=str,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/myresult"),
        help="Output directory for generated scenes and renders",
    )
    parser.add_argument(
        "--seed", type=_parse_seed, help="Random seed for reproducible generation"
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs=2,
        default=(0, 0),
        help="Frame range to render (start, end)",
    )
    parser.add_argument(
        "--exporter_frames",
        type=int,
        nargs=2,
        default=None,
        help="Frame range for Exporter generators only; defaults to --frames. "
        "Lets the scene/camera span the full range while exporters render a subset.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="loglevel",
        default="INFO",
        const="WARNING",
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=logging.getLevelNamesMapping().keys(),
    )
    parser.add_argument(
        "--passes",
        type=ExportType,
        nargs="+",
        default=[ExportType.IMAGE],
    )
    parser.add_argument(
        "--save_blend",
        nargs="?",
        default=None,
        const=True,
        help="Save a blender file to a given path, or to the output folder if not provided",
    )

    parser.add_argument(
        "--displacement_mode",
        type=str,
        choices=[m.name for m in DisplacementMode],
        default=DisplacementMode.DISPLACEMENT_AND_BUMP.name,
    )
    parser.add_argument(
        "--trace",
        nargs="+",
        default=None,
        choices=["codegen", "codestats", "graph"],
        help="Trace execution pipeline. Specify one or more of: codegen, codestats, graph.",
    )
    parser.add_argument(
        "--trace_level",
        type=str,
        choices=[l.name for l in TraceLevel],
        default=TraceLevel.GENERATORS.name,
        help="Trace granularity. choice() peeks through all options at >= RANDOM_CONTROL, resolves to chosen branch when finer.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Resolution of the output images",
    )
    parser.add_argument(
        "-s",
        "--samples",
        type=int,
        default=256,
        help="Number of samples for the rendering",
    )
    parser.add_argument(
        "--focal_length_mm",
        type=float,
        default=15,
        help="Camera focal length in mm",
    )

    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=[0],
        help="Camera indices to render (e.g. --cameras 0 1 for stereo)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second (overrides Blender default of 24)",
    )

    parser.add_argument(
        "--on_interrupt",
        type=str,
        choices=["delete", "cleanstate", "exit", "none"],
        default="none",
    )

    return parser


def _make_interrupt_handler(mode: str, output: Path):
    def handler(signum, frame):
        logger.warning(f"Caught interrupt signal {signum}, on_interrupt={mode!r}")
        if mode == "delete":
            shutil.rmtree(output, ignore_errors=True)
            sys.exit(1)
        elif mode == "cleanstate":
            raise NotImplementedError("cleanstate not yet implemented")
        elif mode == "exit":
            sys.exit(1)

    return handler


_FALLBACK_CATEGORY_SEARCH_STRINGS = [
    ("Object", "objects"),
    ("Scene", "scenes"),
    ("Exporter", "exporters"),
    ("Material", "materials"),
    ("Mask", "masks"),
    ("Material", "shaders.composites"),
]


def _resolve_generator(name: str) -> tuple[Callable, str]:
    manifest_varnames = GENERATORS_MANIFEST["name"].str.split(".").str[-1].values
    matches = GENERATORS_MANIFEST[manifest_varnames == name]

    if len(matches) == 1:
        res_name = matches.iloc[0]["name"]
        res_category = matches.iloc[0]["category"]
        return import_item(res_name), res_category

    if len(matches) > 1:
        raise ValueError(
            f"Multiple generators found for {name}: {matches['name'].values}"
        )

    category = next(
        (
            cat
            for cat, search_str in _FALLBACK_CATEGORY_SEARCH_STRINGS
            if search_str in name
        ),
        None,
    )
    if category is None:
        raise ValueError(
            f"No generator found for {name} - no matches in manifest "
            f"and found no fallback category from options {_FALLBACK_CATEGORY_SEARCH_STRINGS}"
        )

    err = None
    try:
        return import_item(name), category
    except Exception as e:
        err = e

    raise ValueError(
        f"No generator found for {name} - no matches in manifest and import had {err=}"
    )


def _resolve_all_generators(
    generator_strs: list[str],
) -> list[tuple[str, str, Callable]]:
    """Pre-resolve all generators, importing them before tracing starts."""
    return [
        (gen_str, category, func)
        for gen_str in generator_strs
        for func, category in [_resolve_generator(gen_str.strip())]
    ]


def _cleanup_except_returnvals(return_data: dict) -> list[str]:
    valid_objects = list(return_data.get("objects", []))
    valid_objects.extend(return_data.get("cameras", []))
    valid_objects.extend(return_data.get("lights", []))
    if "obj" in return_data:
        valid_objects.append(return_data["obj"])
    valid_objects = [o.item() for o in valid_objects]

    cleaned = []
    for asset in bpy.data.objects:
        if asset not in valid_objects:
            cleaned.append(asset.name)
            asset.name = asset.name + "_CLEANED"
            delete_object(asset)

    return cleaned


def _tight_world_bbox(obj: pf.MeshObject) -> tuple[np.ndarray, np.ndarray]:
    """Tight world-space bbox from evaluated vertices. bbox_min_max(global_coords=True)
    is the local AABB transformed by matrix_world, which inflates rotated objects."""
    item = obj.item()
    eval_obj = item.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = eval_obj.to_mesh()
    n = len(mesh.vertices)
    mat = np.array(item.matrix_world)
    if n == 0:
        eval_obj.to_mesh_clear()
        local = np.array(item.bound_box)
    else:
        local = np.empty(n * 3)
        mesh.vertices.foreach_get("co", local)
        eval_obj.to_mesh_clear()
        local = local.reshape(-1, 3)
    world = (mat[:3, :3] @ local.T).T + mat[:3, 3]
    return world.min(0), world.max(0)


def _bounds(objects: list[pf.MeshObject]) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = zip(*[_tight_world_bbox(o) for o in objects], strict=True)
    return np.minimum.reduce(mins), np.maximum.reduce(maxs)


def _centroid_camera(
    objects: list[pf.MeshObject],
    frac: pf.Vector,
    footprint: pf.MeshObject | None = None,
) -> pf.CameraObject:
    z_min, z_max = _bounds(objects)
    xy_min, xy_max = _bounds([footprint]) if footprint is not None else (z_min, z_max)
    lo = np.array([xy_min[0], xy_min[1], z_min[2]])
    hi = np.array([xy_max[0], xy_max[1], z_max[2]])
    extent = hi - lo
    loc = pf.Vector(lo + extent * np.array(frac))
    # Aim at the scene centroid, biased low so the floor and furniture stay in frame
    target = pf.Vector(lo + extent * np.array((0.5, 0.5, 0.35)))
    rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
    camera = pf.ops.primitives.perspective_camera()
    pf.ops.object.set_transform(camera, loc, rotation_euler)
    camera.item().name = "Camera"
    camera.item().data.lens = 20
    return camera


def _ensure_cameras_and_lights(data: dict):
    """Create dummy camera/lights if none exist. Mutates data in-place."""
    cameras = data.get("cameras", [])
    if not isinstance(cameras, cg.Proxy) and len(cameras) == 0:
        dummy_camera = _centroid_camera(
            data["objects"], (0.2, 0.2, 0.5), footprint=data.get("floor")
        )
        data["cameras"] = [dummy_camera]
        cameras = data["cameras"]

    lights = data.get("lights", [])
    if not isinstance(lights, cg.Proxy) and len(lights) == 0:
        light = pf.ops.primitives.point_lamp(energy=150)
        loc = cameras[0].item().matrix_world @ pf.Vector((0, 0.5, -1))
        pf.ops.object.set_transform(light, location=loc)
        data["lights"] = [light]


def _build_func_resolution_map(toplevel_graph) -> tuple[dict, list[str]]:
    func_resolution = {}

    for op_type, op_func in OPERATORS_TO_FUNCTIONS.items():
        func_resolution[op_func] = op_type

    for oprow in NODE_OPERATOR_TABLE:
        func_resolution[oprow.pf_func] = oprow.operator_type

    for name in dir(pf):
        if name.startswith("_"):
            continue
        obj = getattr(pf, name)
        if isinstance(obj, type) and not isinstance(obj, types.ModuleType):
            func_resolution[obj] = f"pf.{name}"

    default_resolution, import_lines = codegen.default_func_resolution_map(
        toplevel_graph, skip_funcs=set(func_resolution.keys())
    )
    func_resolution.update(default_resolution)

    return func_resolution, import_lines


def _execute_trace(
    modes: list[str],
    output_folder: Path,
    generators: list[str],
    seed: int,
    pipeline_parameters: dict,
    trace_level: TraceLevel = TraceLevel.GENERATORS,
):
    resolved_generators = _resolve_all_generators(generators)
    rng = np.random.default_rng(seed)
    rng_node = pf.compute_graph.ConstantNode(value=rng, metadata={"seed": seed})

    graph = pf.trace(
        execute_generators,
        trace_level=trace_level,
        output_folder=output_folder,
        generators=resolved_generators,
        rng=rng_node,
        pipeline_parameters=pipeline_parameters,
    )

    if "codegen" in modes:
        func_resolution, import_lines = _build_func_resolution_map(graph)
        code = codegen.to_python(
            graph,
            func_resolution=func_resolution,
            import_lines=import_lines,
            toplevel_as_maincall=False,
        )
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")

        dest = output_folder / "codegen.py"
        dest.write_text(code)
        logger.info(f"Generated code written to {dest}")

    if "graph" in modes:
        json_str = graph_json.to_json(graph)
        dest = output_folder / "graph.json"
        dest.write_text(json_str)
        logger.info(f"Graph JSON written to {dest}")

    if "codestats" in modes:
        stats = compute_stats(graph, trace_level=trace_level)
        pprint.pprint(stats)


def _execute_step(
    generator_str: str,
    category: str,
    generator_func: Callable,
    data: dict,
    rng: np.random.Generator,
) -> Any:
    sig = inspect.signature(generator_func)
    step_kwargs = {k: data[k] for k in sig.parameters if k in data}
    if "rng" in sig.parameters:
        step_kwargs["rng"] = rng

    result = generator_func(**step_kwargs)

    if pf.context.globals.current_trace_level is not None:
        varname = generator_func.__name__.removesuffix("_rand")
        for name, val in pf.util.pytree.PyTree(result).items():
            if isinstance(val, pf.compute_graph.Proxy):
                val.node.metadata["varname"] = f"{varname}_{name}" if name else varname

    return result


def _unpack_by_category(category: str, result, data: dict):
    match category:
        case "Material" | "MaterialOverlay":
            data["material"] = result
        case "Mask":
            data["mask"] = result.mask
            data["material"] = pf.Material(
                surface=pf.nodes.shader.diffuse_bsdf(color=result.mask)
            )
        case "Object":
            data["obj"] = result.mesh
            data["objects"].append(result.mesh)
            if hasattr(result, "light") and result.light is not None:
                data.setdefault("lights", []).append(result.light)
        case "Scene":
            data["objects"] += result.all_objects
            data["cameras"] += getattr(result, "cameras", [])
            data["lights"] += getattr(result, "lights", [])
            if hasattr(result, "colliders"):
                data["colliders"] = result.colliders
            if getattr(result, "floor", None) is not None:
                data["floor"] = result.floor
            if getattr(result, "dimensions", None) is not None:
                data["dimensions"] = result.dimensions
        case "Exporter":
            data["exports"] = data["exports"] + [result]
        case "Cameras":
            data["cameras"] = result
        case _:
            raise ValueError(f"Unknown category: {category}")


def execute_generators(
    output_folder: Path,
    generators: list[tuple[str, str, Callable]],
    rng: np.random.Generator,
    pipeline_parameters: dict,
) -> dict[ExportType, list[Path]]:  # noqa: C901
    data = pipeline_parameters.copy()

    data["exports"] = []
    data["lights"] = []
    data["objects"] = []
    data["cameras"] = []
    data["output_folder"] = output_folder

    uv_generators = {
        "material_torus_uv",
        "material_plane_uv",
        "material_plane_horizontal_uv",
    }
    if any(gen_str in uv_generators for gen_str, _, _ in generators):
        data["vector"] = pf.nodes.shader.coord().uv
    else:
        data["vector"] = pf.nodes.shader.geometry().position

    realized = False
    generator_times = {}

    for generator_str, category, generator_func in generators:
        gen_rng, rng = rng.spawn(2)
        start_time = time.perf_counter()

        logger.info(f"Executing {generator_str} as {generator_func.__name__}")

        if (
            category == "Exporter"
            and not realized
            and (
                pipeline_parameters.get("displacement_mode")
                == DisplacementMode.REALIZE_MESH
            )
        ):
            realize_scene()
            realized = True

        if category == "Exporter":
            _ensure_cameras_and_lights(data)
            exp_data = {
                **data,
                "frame_start": data["exporter_frame_start"],
                "frame_end": data["exporter_frame_end"],
            }
            for cam_idx in data.get("camera_indices", [0]):
                exp_data["camera"] = data["cameras"][cam_idx]
                result = _execute_step(
                    generator_str, category, generator_func, exp_data, gen_rng
                )
                _unpack_by_category(category, result, data)
        else:
            result = _execute_step(
                generator_str, category, generator_func, data, gen_rng
            )
            _unpack_by_category(category, result, data)

        if pf.context.globals.current_trace_level is None:
            cleaned = _cleanup_except_returnvals(data)
            if cleaned:
                logger.info(
                    f"Deleting {cleaned!r}, {generator_str} created them but didnt return them"
                )

        elapsed = time.perf_counter() - start_time
        generator_times[generator_str] = elapsed

        logger.info(f"Finished {generator_str} in {elapsed:.3f}s")

    for name, elapsed in generator_times.items():
        logger.info(f"{name}: {elapsed:.3f}s")

    data["generator_times"] = generator_times
    return {k: v for k, v in data.items() if k not in pipeline_parameters}


def resolve_pass_argument(export_type: ExportType) -> RenderPass | None:
    for p in itertools.chain(
        MAINRENDER_PASS_DEFAULTS.values(),
        GT_PASS_DEFAULTS.values(),
        SCENE_PASS_DEFAULTS.values(),
    ):
        if p.type == export_type:
            return p
    raise ValueError(f"No default config found for {export_type}")


class _DebugFilter(logging.Filter):
    """Pass records at/above base_level always; for finer (DEBUG) records, pass only
    those from infinigen/procfunc loggers matching the -d substrings (or all if -d was
    given with no substrings). Applied at the handler so it catches loggers created
    lazily after configuration, which a one-shot per-logger level pass would miss."""

    def __init__(self, base_level: int, substrings: list[str]):
        super().__init__()
        self.base_level = base_level
        self.substrings = substrings

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= self.base_level:
            return True
        if not record.name.startswith(("infinigen", "procfunc")):
            return False
        return not self.substrings or any(s in record.name for s in self.substrings)


def _configure_log_level(args):
    base_level = logging.getLevelNamesMapping()[args.loglevel]
    if args.debug is None:
        logging.root.setLevel(base_level)
        return
    logging.root.setLevel(logging.DEBUG)
    debug_filter = _DebugFilter(base_level, args.debug)
    for handler in logging.root.handlers:
        handler.addFilter(debug_filter)


def _main():  # noqa: C901
    parser = get_parser()
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    _configure_log_level(args)

    # pf.context.globals.set_strict()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
    rng = np.random.default_rng(seed)
    pf.ops.object.clear_scene()

    bpy.context.scene.render.resolution_x = args.resolution[0]
    bpy.context.scene.render.resolution_y = args.resolution[1]
    if args.fps is not None:
        bpy.context.scene.render.fps = args.fps
    bpy.context.scene.frame_start = args.frames[0]
    bpy.context.scene.frame_end = args.frames[1]

    args.output.mkdir(parents=True, exist_ok=True)

    if args.on_interrupt != "none":
        handler = _make_interrupt_handler(args.on_interrupt, args.output)
        signal.signal(signal.SIGUSR1, handler)
        signal.signal(signal.SIGTERM, handler)

    slurm_restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
    exporter_frames = (
        args.exporter_frames if args.exporter_frames is not None else args.frames
    )
    pipeline_parameters = dict(
        output_folder=args.output,
        frame_start=args.frames[0],
        frame_end=args.frames[1],
        exporter_frame_start=exporter_frames[0],
        exporter_frame_end=exporter_frames[1],
        resolution=args.resolution,
        min_samples=32,
        max_samples=args.samples,
        samples_adaptive_threshold=0.005,
        export_passes=args.passes,
        film_exposure=2.0,
        displacement_mode=getattr(DisplacementMode, args.displacement_mode),
        render_passes=[
            resolve_pass_argument(export_type) for export_type in args.passes
        ],
        render_skip_existing=slurm_restart_count > 0,
        focal_length_mm=args.focal_length_mm,
        camera_indices=args.cameras,
    )

    if args.trace is not None:
        _execute_trace(
            modes=args.trace,
            output_folder=args.output,
            generators=args.generators,
            seed=seed,
            pipeline_parameters=pipeline_parameters,
            trace_level=TraceLevel[args.trace_level],
        )
        return

    resolved_generators = _resolve_all_generators(args.generators)
    results = execute_generators(
        output_folder=args.output,
        generators=resolved_generators,
        rng=rng,
        pipeline_parameters=pipeline_parameters,
    )
    exports = {}
    for d in results.get("exports", []):
        for k, v in d.items():
            exports.setdefault(k, []).extend(v)
    generator_times = results.get("generator_times", {})

    if args.save_blend:
        for l in [
            bpy.data.objects,
            bpy.data.materials,
            bpy.data.node_groups,
            bpy.data.meshes,
        ]:
            for obj in l:
                obj.use_fake_user = True
        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if mod.type == "SUBSURF":
                    mod.show_viewport = False
        output_path = (
            Path(args.save_blend)
            if isinstance(args.save_blend, str)
            else args.output / "scene.blend"
        )
        pf.ops.file.save_blend(output_path=output_path)

    missing = [p for p in args.passes if p not in exports]
    if missing:
        logger.warning(f"Requested passes were not produced: {missing}")

    def _serialize_arg(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_serialize_arg(x) for x in v]
        if hasattr(v, "name"):
            return v.name
        return v

    metadata = {
        "args": {k: _serialize_arg(v) for k, v in vars(args).items()},
        "seed": hex(seed),
        "hardware": get_hardware_info(),
        "generator_times": generator_times,
        "exports": {k.value: [str(p) for p in v] for k, v in exports.items()},
    }
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for _export_type, frames in exports.items():
        for f in frames:
            print(f)


def main():
    with skip_teardown_on_exit():
        _main()


if __name__ == "__main__":
    main()
