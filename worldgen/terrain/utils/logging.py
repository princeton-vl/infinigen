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
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            super().__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        if self.verbose:
            super().__exit__(exc_type, exc_val, traceback)
            report_memory()
