# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from infinigen2 import context

logger = logging.getLogger(__name__)


class FrameCheckError(RuntimeError):
    pass


def frame_mean_pixel(path: Path) -> float:
    img = np.asarray(iio.imread(path))
    arr = img.astype(np.float64)
    if np.issubdtype(img.dtype, np.integer):
        arr = arr / np.iinfo(img.dtype).max
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return float(arr.mean())


def assert_frames_not_black(
    paths: list[Path], mean_pixel_thresh: float | None
) -> dict[str, float]:
    if mean_pixel_thresh is None:
        return {}
    if context.globals.error_mode_black_frame == "ignore":
        return {}
    dark = {}
    for path in paths:
        mean = frame_mean_pixel(path)
        if mean <= mean_pixel_thresh:
            dark[path.name] = round(mean, 5)
    if dark:
        error = FrameCheckError(
            f"rendered frames are ~black (mean pixel <= {mean_pixel_thresh}): {dark}"
        )
        context.raise_or_warn(context.globals.error_mode_black_frame, error, logger)
    return dark
