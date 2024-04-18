# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from pathlib import Path
import logging
import psutil
import os

import numpy as np
import pandas as pd

from contextlib import nullcontext

from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.logging import Timer
from infinigen.core.util.blender import GarbageCollect, count_instance, count_objects

logger = logging.getLogger(__name__)

class RandomStageExecutor:

    def __init__(self, scene_seed, output_folder: Path, params):
        self.scene_seed = scene_seed
        self.output_folder = output_folder
        self.params = params

        self.results = []

    def _should_run_stage(self, name, use_chance, prereq):
        if prereq is not None:
            try:
                e = next(e for e in self.results if e['name'] == prereq)
            except StopIteration:
                raise ValueError(f'{self} could not find matching name for {prereq=}')
            if not e['ran']:
                logger.info(f'Skipping run_stage({name}...) due to unmet {prereq=}')
                return
        with FixedSeed(int_hash((self.scene_seed, name, 0))):
            if use_chance and np.random.uniform() > self.params[f'{name}_chance']:
                logger.debug(f'Not running {name} due to random chance')
                return False
            if not use_chance and not self.params.get(f'{name}_enabled', True):
                logger.debug(f'Not running {name} due to manually set not enabled')
                return False      
        return True
    
    def save_results(self, path):
        pd.DataFrame.from_records(self.results).to_csv(path)

    def run_stage(
        self, name, fn, *args, 
        use_chance=True, gc=True, default=None, 
        prereq=None, **kwargs):

        mem_usage = psutil.Process(os.getpid()).memory_info().rss
            
        will_run = self._should_run_stage(name, use_chance, prereq)
        
        if not will_run:
            self.results.append({'name': name, 'ran': will_run, 'mem_at_finish': mem_usage, 'obj_count': count_objects(),\
                'instance_count': count_instance()})
            return default

        gc_context = GarbageCollect() if gc else nullcontext()

        seed = self.params.get(f'{name}_seed')
        if seed is None:
            seed = int_hash((self.scene_seed, name))
        logger.debug(f'run_stage({name=}) using {seed=}')
        
        with FixedSeed(seed):
            with Timer(name), gc_context:
                ret = fn(*args, **kwargs)
                mem_usage = psutil.Process(os.getpid()).memory_info().rss
                self.results.append({'name': name, 'ran': will_run, 'mem_at_finish': mem_usage, 'obj_count': count_objects(),\
                'instance_count': count_instance()})
                return ret
