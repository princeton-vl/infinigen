# ruff: noqa: E402
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pathlib import Path

import cv2
import numpy as np


def load_exr(path: Path) -> np.ndarray:
    assert Path(path).exists() and Path(path).suffix == ".exr", path
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"Failed to read EXR: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
