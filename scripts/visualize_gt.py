#!/usr/bin/env -S uv run --no-sync python
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson: original Infinigen v1 ground-truth visualization (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/core/rendering/post_render.py)
# - Alexander Raistrick: port to v2

import argparse
import json
from pathlib import Path

from infinigen2.exporters.util.format import ExportType
from infinigen2.exporters.visualize_gt import visualize_gt

parser = argparse.ArgumentParser(
    description="Visualize ground truth data from a render output folder"
)
parser.add_argument("input", type=Path, help="Output folder containing metadata.json")
args = parser.parse_args()

metadata_path = args.input / "metadata.json"
with open(metadata_path) as f:
    metadata = json.load(f)

exports = {}
for key, paths in metadata["exports"].items():
    export_type = ExportType(key)
    exports[export_type] = [Path(p) for p in paths]

results = visualize_gt(exports, args.input)
for name, paths in results.items():
    for p in paths:
        print(p)
