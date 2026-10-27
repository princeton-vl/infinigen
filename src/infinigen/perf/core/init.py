# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - original Infinigen Blender initialization (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/init.py)
# - David Yan, Jia Deng: optimized implementation

import logging
import os

from infinigen.core import init as _default
from infinigen.perf import options

logger = logging.getLogger(_default.__name__)


def configure_cycles_devices(use_gpu=True):
    # OptiX caches the kernels Cycles compiles in the directory named by
    # OPTIX_CACHE_PATH; a persistent one lets later renders skip compiling
    # them. Only compiled code is cached, so renders are unchanged.
    if use_gpu and "OPTIX_CACHE_PATH" not in os.environ:
        directory = options.optix_kernel_cache()
        if directory is not None:
            try:
                os.makedirs(directory, exist_ok=True)
                os.environ["OPTIX_CACHE_PATH"] = directory
                logger.info(f"OptiX kernel cache in {directory}")
            except OSError as e:  # never let the cache stop a render
                logger.warning(f"OptiX kernel cache left to the driver: {e}")
    return _default.configure_cycles_devices.default(use_gpu=use_gpu)
