# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Koray Kunal

"""Export an Infinigen 2 ``.blend`` with baked materials.

Infinigen 2 can write a ``.blend`` with ``--save_blend``.  The V1 exporter
already contains the material-baking and FBX/OBJ/USD conversion code, so this
script is a small cross-platform wrapper around it.  It is intentionally
scoped to individual assets and small scenes; exporting full Infinigen scenes
to FBX is still limited by instancing and memory use.

Example
-------
Generate a single asset and save its blend file::

    uv run infinigen sofa_rand --seed 0 \
        --output /absolute/path/to/outputs/sofa --save_blend

Export it to a baked FBX::

    uv run --with gin-config python scripts/export_v2_asset.py \
        --input /absolute/path/to/outputs/sofa/scene.blend \
        --output /absolute/path/to/outputs/sofa/export \
        --format fbx \
        --resolution 1024
"""

import argparse
import sys
from pathlib import Path

import bpy

FORMATS = ("fbx", "obj", "usdc", "usda", "stl", "ply")


def _mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]


def _keep_objects(names: list[str]) -> None:
    targets = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            available = ", ".join(sorted(item.name for item in bpy.data.objects))
            raise SystemExit(
                f"Object {name!r} was not found in the blend file. "
                f"Available objects: {available}"
            )
        targets.append(obj)

    keep = set()
    for obj in targets:
        keep.add(obj)
        keep.update(obj.children_recursive)
        parent = obj.parent
        while parent is not None:
            keep.add(parent)
            parent = parent.parent

    for obj in list(bpy.data.objects):
        if obj not in keep:
            bpy.data.objects.remove(obj, do_unlink=True)


def _flatten_parent_transforms() -> None:
    # Clearing a parent keeps the object's local transform, which can move it
    # if the parent had a rotation or scale. Bake the full world matrix first.
    for obj in _mesh_objects():
        if obj.parent is None:
            continue
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world


def _remove_non_mesh() -> None:
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)


def _resolve_blend(input_path: Path) -> Path:
    if input_path.is_dir():
        candidates = sorted(input_path.glob("*.blend"))
        if not candidates:
            raise SystemExit(f"No .blend files found in {input_path}")
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise SystemExit(
                f"Found multiple .blend files in {input_path}: {names}. "
                "Pass the specific file with --input."
            )
        return candidates[0]

    if not input_path.is_file():
        raise SystemExit(f"Input does not exist: {input_path}")
    return input_path


def _default_name(blend_path: Path) -> str:
    # Infinigen writes `scene.blend` into a per-run folder, so the folder name
    # is usually more descriptive than the file stem.
    if blend_path.stem == "scene":
        return blend_path.parent.name
    return blend_path.stem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an Infinigen 2 .blend to FBX/OBJ/USD with baked textures. "
            "Generate the input with `infinigen <generator> --save_blend`."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A .blend file, or a directory containing exactly one .blend file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where the exported asset and its textures are written.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Name for the exported subdirectory and file. Defaults to the input "
            "blend file's stem, or its parent folder name when the file is "
            "called scene.blend."
        ),
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="fbx",
        help="Output format. Defaults to fbx.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Resolution of the baked texture maps. Defaults to 1024.",
    )
    parser.add_argument(
        "--object",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Keep only this object (and its children and parents). Can be passed "
            "more than once. By default all mesh objects are exported."
        ),
    )
    parser.add_argument(
        "--keep-non-mesh",
        action="store_true",
        help="Keep cameras, lights and empties. By default they are removed.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Export each mesh object to its own file.",
    )
    parser.add_argument(
        "--vertex-colors",
        action="store_true",
        help="Bake to vertex colors instead of textures (fbx, obj and ply only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    blend_path = _resolve_blend(args.input).resolve()

    if args.resolution < 1:
        raise SystemExit("--resolution must be a positive integer.")

    if args.vertex_colors and args.format not in {"fbx", "obj", "ply"}:
        raise SystemExit(
            "--vertex-colors is only supported for fbx, obj and ply outputs."
        )

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    if args.object:
        _keep_objects(args.object)

    _flatten_parent_transforms()

    if not args.keep_non_mesh:
        _remove_non_mesh()

    if not _mesh_objects():
        raise SystemExit("No mesh objects are left to export.")

    try:
        from infinigen.tools.export import export_curr_scene
    except ModuleNotFoundError as exc:
        if exc.name == "gin":
            raise SystemExit(
                "The V1 baking exporter requires gin-config. Re-run the command "
                "with `uv run --with gin-config ...`."
            ) from exc
        raise

    asset_name = args.name or _default_name(blend_path)
    # Blender resolves relative image paths against the blend file, not the
    # current directory, so keep the export path absolute.
    output_folder = (args.output / asset_name).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "textures").mkdir(exist_ok=True)

    result = export_curr_scene(
        output_folder=output_folder,
        format=args.format,
        image_res=args.resolution,
        vertex_colors=args.vertex_colors,
        individual_export=args.individual,
        omniverse_export=False,
    )

    if result is None:
        # The V1 exporter returns None when individual_export is set.
        print(f"Exported {output_folder}")
    else:
        print(f"Exported {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
