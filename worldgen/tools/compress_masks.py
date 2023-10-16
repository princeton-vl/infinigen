# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

import argparse
from pathlib import Path

import numpy as np


def show(x):
    return f"({x.shape} {x.dtype} {x.max()})"

def compress(arr):
    H, W, *_ = arr.shape
    vals, indices = np.unique(np.squeeze(arr.reshape((H * W, -1))), return_inverse=True, axis=0)
    max_ind = (vals.shape[0] - 1)
    if max_ind < 2**8:
        indices = indices.astype(np.uint8)
    elif max_ind < 2**16:
        indices = indices.astype(np.uint16)
    else:
        indices = indices.astype(np.uint32)
    return dict(vals=vals, indices=indices, shape=np.asarray(arr.shape))

def recover(d):
    return d['vals'][d['indices']].reshape(d['shape'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_frames_dir", type=Path)
    args = parser.parse_args()
    assert args.target_frames_dir.exists()
    assert args.target_frames_dir.name.startswith("frames_")
    for file_path in args.target_frames_dir.glob("*.npy"):
        arr = np.load(file_path)
        if np.issubdtype(arr.dtype, np.integer) and (arr.size > 1000):
            d = compress(arr)
            assert show(arr) == show(recover(d))
            np.savez(file_path.with_suffix(".npz"), **d)
            print(f"{file_path} -> {file_path.with_suffix('.npz')}")
            file_path.unlink()