# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from pathlib import Path

from infinigen2.exporters.util.format import ExportType
from infinigen2.exporters.visualize_gt_boxes import visualize_object_boxes
from infinigen2.exporters.visualize_gt_passes import (
    VISUALIZATION_FUNCS,
    visualize_any_frametype,
)

__all__ = [
    "visualize_gt",
]

logger = logging.getLogger(__name__)


def _merge_exports(
    exports: list[dict[ExportType, list[Path]]],
) -> dict[ExportType, list[Path]]:
    merged: dict[ExportType, list[Path]] = {}
    for export in exports:
        for export_type, paths in export.items():
            merged.setdefault(export_type, []).extend(paths)
    return merged


def _visualize_object_boxes_if_present(
    exports: dict[ExportType, list[Path]], output_folder: Path
) -> list[Path]:
    required = (ExportType.OBJECT_DATA, ExportType.CAMERA, ExportType.IMAGE)
    missing = [t for t in required if t not in exports]
    if missing:
        logger.info(f"Skipping 3D box overlay, no {[t.value for t in missing]}")
        return []

    camera_paths = exports[ExportType.CAMERA]
    if len(camera_paths) > 1:
        logger.warning(
            f"Expected one camera file per call, got {len(camera_paths)=}; using the first"
        )

    tables = exports.get(ExportType.OBJECT_INDEX_TABLE, [])
    return visualize_object_boxes(
        exports[ExportType.IMAGE],
        exports[ExportType.OBJECT_DATA][0],
        camera_paths[0],
        output_folder / "bbox3d",
        table_json=tables[0] if tables else None,
    )


def visualize_gt(
    exports: dict[ExportType, list[Path]] | list[dict[ExportType, list[Path]]],
    output_folder: Path,
) -> dict[ExportType, list[Path]]:
    if isinstance(exports, list):
        exports = _merge_exports(exports)

    all_vis_paths = []
    for export_type, frames in exports.items():
        if export_type not in VISUALIZATION_FUNCS:
            continue
        all_vis_paths.extend(
            visualize_any_frametype(export_type, frames, output_folder)
        )

    all_vis_paths.extend(_visualize_object_boxes_if_present(exports, output_folder))
    return {ExportType.VISUALIZATIONS: all_vis_paths}
