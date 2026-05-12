# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from copy import copy
from pathlib import Path

SUFFIX_ORDERING = ["cam_rig", "resample", "frame", "subcam"]


def get_suffix(indices):
    suffix = ""

    if indices is None:
        return suffix

    indices = copy(indices)

    for key in SUFFIX_ORDERING:
        val = indices.get(key, 0)
        if key == "frame" and isinstance(val, int):
            suffix += "_" + f"{val:04d}"
        else:
            suffix += "_" + str(val)

    return suffix


def parse_suffix(s):
    if isinstance(s, Path):
        s = s.name

    if "." in s:
        s = s[: s.index(".")]

    s = s.strip("_")

    s_parts = s.split("_")
    if len(s_parts) > len(SUFFIX_ORDERING) + 1:
        raise ValueError(f"Couldnt parse {s=} with {len(s_parts)=}")

    if len(s_parts) == len(SUFFIX_ORDERING) + 1:
        s_parts = s_parts[1:]  # discard leading filename / description etc

    if len(s_parts) != len(SUFFIX_ORDERING):
        return None

    return {SUFFIX_ORDERING[i]: int(s_parts[i]) for i in range(len(s_parts))}
