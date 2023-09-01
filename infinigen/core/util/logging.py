# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Lahav Lipson: logging formats, timer format
# - Alex Raistrick: Timer
# - Alejandro Newell: Suppress


import os, sys
from datetime import datetime
from pathlib import Path
import logging
import uuid

import bpy
import gin

@gin.configurable
class Timer:

    def __init__(self, desc, disable_timer=False, logger=None):
        self.disable_timer = disable_timer
        if self.disable_timer:    
            return
        self.name = f'[{desc}]'
        if logger is None:
            logger = logging.getLogger("infinigen.times")
        self.logger = logger

    def __enter__(self):
        if self.disable_timer:
            return
        self.start = datetime.now()
        self.logger.info(f'{self.name}')

    def __exit__(self, exc_type, exc_val, traceback):
        if self.disable_timer:
            return
        self.end = datetime.now()
        self.duration = self.end - self.start # timedelta
        if exc_type is None:
            self.logger.info(f'{self.name} finished in {str(self.duration)}')
        else:
            self.logger.info(f'{self.name} failed with {exc_type}')

class Suppress():

    def __enter__(self, logfile=os.devnull):
        open(logfile, 'w').close()
        self.old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    def __exit__(self, type, value, traceback):
        os.close(1)
        os.dup(self.old)
        os.close(self.old)

class LogLevel():

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.orig_level = None

    def __enter__(self):
        self.orig_level = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, *_):
        self.logger.setLevel(self.orig_level)

def save_polycounts(file):
    for col in bpy.data.collections:
        polycount = sum(len(obj.data.polygons) for obj in col.all_objects if (obj.type == "MESH" and obj.data is not None))
        file.write(f"{col.name}: {polycount:,}\n")
    for stat in bpy.context.scene.statistics(bpy.context.view_layer).split(' | ')[2:]:
        file.write(stat)

@gin.configurable
def create_text_file(log_dir, filename, text=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / filename).touch()
    if text is not None:
        (log_dir / filename).write_text(text)
