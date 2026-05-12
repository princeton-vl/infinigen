#!/usr/bin/env -S uv run --no-sync python
import argparse
import json
from pathlib import Path

from infinigen_v2.exporters.util.format import ExportType
from infinigen_v2.exporters.visualize_gt import visualize_gt

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
