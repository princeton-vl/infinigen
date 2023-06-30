# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Date Signed: June 5 2023

import psutil
import os
import gin
from util.logging import Timer as oTimer


def report_memory():
    process = psutil.Process(os.getpid())
    print(f"memory usage: {process.memory_info().rss}")


@gin.configurable("TerrainTimer")
class Timer(oTimer):
    def __init__(self, desc, verbose):
        super().__init__(desc)
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        if self.verbose:
            super().__exit__(exc_type, exc_val, traceback)
            report_memory()
