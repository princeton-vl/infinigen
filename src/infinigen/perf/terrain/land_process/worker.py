# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Land-process worker entry point, `python -m ...worker JOBFILE`. Kept out of `pool`
so that `-m` does not re-execute the pool module the terrain already imported."""

import sys

from infinigen.perf.terrain.land_process.pool import worker_main

if __name__ == "__main__":
    worker_main(sys.argv[1])
