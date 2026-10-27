# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Large sequential reads that pull cold files on a network file system into the
page cache ahead of readers that issue small requests (Blender, OpenCV, pickle)."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

CHUNK = 64 << 20


def _read_through(path, chunk=CHUNK):
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            data = f.read(chunk)
            if not data:
                return n
            n += len(data)


def prefetch_files(paths, threads=4, min_size=1 << 20):
    """Read every file in `paths` (directories recursively) once with large requests,
    `threads` files at a time, skipping files under `min_size` bytes. Returns the
    number of bytes read."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            files.append(p)
    files = [f for f in files if f.stat().st_size >= min_size]
    if not files:
        return 0
    t0 = time.perf_counter()
    if threads > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            total = sum(ex.map(_read_through, files))
    else:
        total = sum(_read_through(f) for f in files)
    dt = time.perf_counter() - t0
    logger.info(
        f"prefetched {len(files)} files, {total / 1e6:.0f} MB in {dt:.1f}s "
        f"({total / 1e6 / max(dt, 1e-6):.0f} MB/s)"
    )
    return total
