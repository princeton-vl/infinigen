# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Abhishek Joshi
import argparse
from pathlib import Path

import bpy

from infinigen.assets import sim_objects as objects
from infinigen.core.nodes.node_transpiler import transpiler


def load_blender_file(filepath):
    """
    Open a Blender file.
    """
    bpy.ops.wm.open_mainfile(filepath=filepath)


def transpile_simready(args):
    """
    Compiles and transpiles an asset for sim ready form.
    """
    load_blender_file(args.blend_path)
    obj = bpy.data.objects.get(args.blend_obj_name)

    output_name = (
        obj.name.lower() if args.output_name is None else args.output_name.lower()
    )
    output_name.replace(".", "_")

    dependencies = [
        # if your transpile target is using nodegroups taken from some python file,
        # add those filepaths here so the transpiler imports from them rather than creating a duplicate definition.
    ]

    res = transpiler.transpile_object_to_sim_class(
        obj=obj,
        module_dependencies=dependencies,
        output_name=output_name,
    )

    # write the transpiled information to python script
    transpiled_path = Path(objects.__path__[0]) / f"{output_name}.py"
    transpiled_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transpiled_path, "w") as f:
        f.write(res)
    print(f"Generated transpiled code to {transpiled_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bp",
        "--blend_path",
        type=str,
        required=True,
        help="blend file to convert to mjcf scene",
    )
    parser.add_argument(
        "-bon",
        "--blend_obj_name",
        type=str,
        required=True,
        help="name of the objects to transpile and make sim ready",
    )
    parser.add_argument(
        "--output_name", type=str, required=False, help="name of the object"
    )
    args = parser.parse_args()
    transpile_simready(args)
