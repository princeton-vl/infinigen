import argparse
import importlib
import logging
import subprocess
import sys
from functools import partial
from pathlib import Path

import bpy
import procfunc as pf
from procfunc import transforms as tr
from procfunc.transpiler import main as transpiler
from procfunc.util.manifest import import_item

logger = logging.getLogger(__name__)


def _v1_to_blender(v1_str: str, idx: int) -> bpy.types.Material | bpy.types.NodeTree:
    if "/" in v1_str:
        v1_path, v1_name = v1_str.rsplit(":", 1)
        v1_path = Path(v1_path).absolute()
        sys.path.append(str(v1_path.parent))
        v1_filemod = importlib.import_module(v1_path.stem)
        v1_func = getattr(v1_filemod, v1_name)
    else:
        v1_func = import_item(v1_str)

    if v1_func.__name__ == "apply":
        obj = pf.ops.primitives.mesh_cube(size=1)
        v1_func(obj.item())
        res = obj.item().material_slots[0].material
    if "Factory" in v1_func.__name__:
        res = v1_func(idx).spawn_asset(idx)
    else:
        res = v1_func()

    if hasattr(res, "apply"):
        obj = pf.ops.primitives.mesh_cube(size=1)
        res.apply(obj.item())
        # assert len(obj.item().modifiers) == 0, "Object has modifiers, not supported"
        res = obj.item().material_slots[0].material
    elif hasattr(res, "generate"):
        res = res.generate()

    return res

    if isinstance(res, bpy.types.Material):
        return res
    elif isinstance(res, bpy.types.Object):
        return res
    elif isinstance(res, bpy.types.NodeTree):
        return res
    else:
        raise ValueError(f"Invalid result type: {type(res)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infinigen_v1_name", type=str, nargs="+")
    parser.add_argument("infinigen_v2_name", type=str)
    parser.add_argument("--num_instances", type=int, default=1)
    parser.add_argument("--loglevel", type=str, default="INFO")
    parser.add_argument(
        "-d",
        "--debug",
        type=str,
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--append",
        action="store_true",
    )
    parser.add_argument("--format", action="store_true")
    parser.add_argument("--save_v1_blend", type=Path, default=None)
    parser.add_argument("--add_line_comments", action="store_true")
    parser.add_argument("--add_version_comment", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if (
                args.debug == "all"
                or len(args.debug) == 0
                or any(name.endswith(x) for x in args.debug)
            ):
                logging.getLogger(name).setLevel(logging.DEBUG)
    targets = [
        _v1_to_blender(v1_str, idx)
        for v1_str in args.infinigen_v1_name
        for idx in range(args.num_instances)
    ]

    if len(targets) == 0:
        raise ValueError(f"No targets found for {args.infinigen_v1_name}")

    if args.save_v1_blend:
        for target in targets:
            if hasattr(target, "use_fake_user"):  # prevent cleanup
                target.use_fake_user = True
        pf.ops.file.save_blend(output_path=args.save_v1_blend)

    use_transforms = [
        tr.map_subgraphs(tr.remove_v1_name_from_graph),
        tr.map_subgraphs(tr.coerce_shaders_to_materialresult),
        tr.eliminate_duplicate_subgraphs,
        partial(tr.eliminate_duplicate_result_types, uses_threshold=3),
        tr.map_subgraphs(tr.fill_graph_defaults_with_call_node),
        lambda graphs: [
            (
                tr.colors_to_hsv_definition(g)
                if not isinstance(targets[i], bpy.types.NodeTree)
                else g
            )
            for i, g in enumerate(graphs)
        ],
        tr.map_subgraphs(lambda _n, graph: tr.extract_shader_vectors_as_inputs(graph)),
    ]

    if args.num_instances > 1:
        # use_transforms.append(tr.extract_differences_as_inputs)
        use_transforms.append(
            lambda graphs: [graphs[0]] + [tr.infer_distribution_hypercube(graphs)]
        )

    python = transpiler.transpile_targets(
        targets=targets,
        transforms=use_transforms,
        add_line_comments=args.add_line_comments,
        add_version_comment=args.add_version_comment,
    )

    if "/" in args.infinigen_v2_name:
        outpath, outname = args.infinigen_v2_name.rsplit(":", 1)
    else:
        outpath, outname = args.infinigen_v2_name.rsplit(".", 1)
        outpath = Path("src") / outpath.replace(".", "/")
        outpath = outpath.with_suffix(".py")
        assert outpath.parent.exists(), (
            f"Output directory {outpath.parent} does not exist"
        )

    if args.append:
        with outpath.open("a") as f:
            f.write("\n" + python)
    else:
        outpath.write_text(python)

    if args.format:
        subprocess.run(["uv", "run", "ruff", "format", str(outpath)])
        subprocess.run(["uv", "run", "ruff", "check", "--fix", str(outpath)])

    logger.info(f"Wrote output to {outpath}")


if __name__ == "__main__":
    main()
