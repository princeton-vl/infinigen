# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - original Infinigen task execution (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/execute_tasks.py)
# - David Yan, Jia Deng: optimized implementation

import logging

import bpy

from infinigen.perf import options
from infinigen.perf.core.util.prefetch import prefetch_files

logger = logging.getLogger("infinigen.core.execute_tasks")


def _open_scene(path):
    # Blender reads a cold .blend with many small requests; one pass of large
    # reads first puts the file in the page cache.
    if options.prefetch_input_scene():
        try:
            prefetch_files([path])
        except Exception as e:  # never let the warm-up stop the pipeline
            logger.warning(f"prefetch of the input scene skipped: {e}")
    bpy.ops.wm.open_mainfile(filepath=str(path))


def _save_scene(path):
    bpy.ops.wm.save_mainfile(
        filepath=str(path), compress=options.compress_output_blend()
    )
