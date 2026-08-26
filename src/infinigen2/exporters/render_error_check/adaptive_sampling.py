# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import re
from pathlib import Path

import bpy
import numpy as np

from infinigen2 import context
from infinigen2.exporters.util.blender_render import load_single_channel

logger = logging.getLogger(__name__)


class AdaptiveSamplingError(RuntimeError):
    pass


SAMPLE_COUNT_SLOT = "sample_count_"

SAMPLE_PROGRESS_RE = re.compile(r"Fra:(\d+).*\| Sample (\d+)/(\d+)")

SAMPLE_COUNT_AT_CAP = 1.0 - 1e-6


def configure_sample_count_output(
    view_layer: bpy.types.ViewLayer, folder: Path
) -> None:
    view_layer.cycles.pass_debug_sample_count = True
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    render_layers = next(
        (n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers"), None
    )
    if render_layers is None:
        render_layers = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(folder)
    out.format.file_format = "OPEN_EXR"
    out.format.color_mode = "RGB"
    out.format.color_depth = "32"
    out.file_slots[0].path = SAMPLE_COUNT_SLOT
    tree.links.new(render_layers.outputs["Debug Sample Count"], out.inputs[0])


def _frames_at_sample_cap(render_log: str, max_samples: int) -> set[int]:
    frames = set()
    for m in SAMPLE_PROGRESS_RE.finditer(render_log):
        frame, sample, total = (int(g) for g in m.groups())
        if total == max_samples and sample >= max_samples:
            frames.add(frame)
    return frames


def unconverged_fraction(sample_count_path: Path) -> float:
    counts = load_single_channel(sample_count_path)
    return float(np.mean(counts >= SAMPLE_COUNT_AT_CAP))


def _cleanup_sample_counts(folder: Path) -> None:
    for path in folder.glob(f"{SAMPLE_COUNT_SLOT}*.exr"):
        path.unlink()


def assert_adaptive_sampling_converged(
    render_log: str,
    sample_count_folder: Path,
    max_samples: int,
    fail_fraction: float,
) -> dict[int, float]:
    fractions = {}
    for frame in sorted(_frames_at_sample_cap(render_log, max_samples)):
        path = sample_count_folder / f"{SAMPLE_COUNT_SLOT}{frame:04d}.exr"
        if not path.exists():
            logger.warning(f"Missing sample count pass {path}, skipping frame")
            continue
        fractions[frame] = unconverged_fraction(path)
    _cleanup_sample_counts(sample_count_folder)

    noisy = {f: frac for f, frac in fractions.items() if frac > 0}
    if not noisy:
        return fractions
    msg = (
        f"Adaptive sampling hit the {max_samples=} cap before reaching the noise "
        f"threshold; unconverged pixel fraction per frame: "
        f"{ {f: round(v, 4) for f, v in noisy.items()} }"
    )
    if max(noisy.values()) > fail_fraction:
        mode = context.globals.error_mode_unconverged_samples
        context.raise_or_warn(mode, AdaptiveSamplingError(msg), logger)
    else:
        logger.warning(msg)
    return fractions
