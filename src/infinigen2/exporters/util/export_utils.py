# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

# ruff: noqa: E402
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "load_exr",
]


def load_exr(path: Path) -> np.ndarray:
    assert Path(path).exists() and Path(path).suffix == ".exr", path
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"Failed to read EXR: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
