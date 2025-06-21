# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import argparse
import pprint
from pathlib import Path

from infinigen.core.sim import sim_factory as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name", type=str, required=True, help="name of the asset"
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-exp",
        "--exporter",
        type=str,
        default="mjcf",
        help="exporter to use to spawn asset",
    )
    parser.add_argument(
        "-dir",
        "--export_dir",
        type=str,
        default="./sim_exports",
        help="directory to export asset to",
    )
    parser.add_argument(
        "-c",
        "--include_collisions",
        action="store_true",
        help="directory to export asset to",
    )

    args = parser.parse_args()
    export_path, semantic_mapping = sf.spawn_simready(
        name=args.name,
        seed=args.seed,
        exporter=args.exporter,
        export_dir=Path(args.export_dir),
        visual_only=not args.include_collisions,
    )
    pprint.pprint(semantic_mapping)

    print(f"Exported to {export_path.resolve()}")
