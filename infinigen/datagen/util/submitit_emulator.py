# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Lahav Lipson - LocalJob
# - Alex Raistrick - Local queue handler
# - David Yan - Bugfix


import re
import time
import sys
from pathlib import Path
from dataclasses import dataclass
import os
from functools import partial
import itertools
import copy
import random
import logging

import subprocess
from multiprocessing import Process
import threading
import submitit
import gin

import numpy as np
from shutil import which

CUDA_VARNAME = "CUDA_VISIBLE_DEVICES"

@dataclass
class LocalJob:

    job_id: int
    process: Process
    finalized: bool = False

    def status(self):

        if self.finalized:
            return "COMPLETED"

        if self.process is None:
            return "PENDING"

        exit_code = self.process.exitcode
        if exit_code is None:
            return "RUNNING"

        # process has returned but the object is still there, we can clean it up
        assert not self.finalized
        self.finalized = True
        self.process = None

        if exit_code == 0:
            return "COMPLETED"
        return "FAILED"

    def kill(self):
        if self.process is None:
            return
        self.process.kill()

def get_fake_job_id():
    # Lahav assures me these will never conflict
    return np.random.randint(int(1e10), int(1e11))

def job_wrapper(func, inner_args, inner_kwargs, stdout_file: Path, stderr_file: Path, cuda_devices=None):

    with stdout_file.open('w') as stdout, stderr_file.open('w') as stderr:
        sys.stdout = stdout
        sys.stderr = stderr
        if cuda_devices is not None:
            os.environ[CUDA_VARNAME] = ','.join([str(i) for i in cuda_devices])
        else:
            os.environ[CUDA_VARNAME] = ''
        return func(*inner_args, **inner_kwargs)

def launch_local(func, args, kwargs, job_id, log_folder, name, cuda_devices=None):
    
    stderr_file = log_folder / f"{job_id}_0_log.err"
    stdout_file = log_folder / f"{job_id}_0_log.out"
    with stdout_file.open('w') as f:
        f.write(f"{func} {args}\n")

    kwargs = dict(
        func=func,
        inner_args=args,
        inner_kwargs=kwargs,
        stdout_file=stdout_file, 
        stderr_file=stderr_file, 
        cuda_devices=cuda_devices
    )
    proc = Process(target=job_wrapper, kwargs=kwargs, name=name)
    proc.start()

    return proc

class ImmediateLocalExecutor:

    def __init__(self, folder: str):
        self.log_folder = Path(folder).resolve()
        self.log_folder.mkdir(exist_ok=True)
        self.parameters = {}

    def update_parameters(self, **parameters):
        self.parameters.update(parameters)
    
    def submit(self, func, *args, **kwargs):
        job_id = get_fake_job_id()
        name = self.parameters.get('name', None)
        proc = launch_local(func, args, kwargs, job_id, 
            log_folder=self.log_folder, name=name)
        return LocalJob(job_id=job_id, process=proc)

@gin.configurable
class LocalScheduleHandler:

    _inst = None

    @classmethod
    def instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self, jobs_per_gpu=1, use_gpu=True):
        self.queue = []
        self.jobs_per_gpu = jobs_per_gpu
        self.use_gpu = use_gpu

    def enqueue(self, func, args, kwargs, params, log_folder):

        job = LocalJob(job_id=get_fake_job_id(), process=None)
        job_rec = dict(
            func=func, args=args, kwargs=kwargs,
            params=params, 
            job=job, log_folder=log_folder,
            gpu_assignment=None
        )
        
        self.queue.append(job_rec)
        return job

    @gin.configurable
    def total_resources(self):

        resources = {}

        if self.use_gpu:
            if which('/bin/nvidia-smi') is not None:
                result = subprocess.check_output('/bin/nvidia-smi -L'.split()).decode()                                            
                gpus_uuids = set(i for i in range(len(result.splitlines())))

                if CUDA_VARNAME in os.environ:
                    visible = [int(s.strip()) for s in os.environ[CUDA_VARNAME].split(',')]
                    gpus_uuids = gpus_uuids.intersection(visible)
                    print(f"Restricting to {gpus_uuids=} due to toplevel {CUDA_VARNAME} setting")

                resources['gpus'] = set(itertools.product(range(len(gpus_uuids)), range(self.jobs_per_gpu)))
            else:
                resources['gpus'] = {'0'}
            
        return resources

    def resources_available(self, total):

        resources = copy.copy(total)

        for job_rec in self.queue:
            if job_rec['job'].status() != 'RUNNING':
                continue
            if (g := job_rec['gpu_assignment']) is not None:
                resources['gpus'] -= g
        return resources
        
    def poll(self):

        total = self.total_resources()
        available = self.resources_available(total)

        for job_rec in self.queue:
            if job_rec['job'].status() != 'PENDING':
                continue
            self.attempt_dispatch_job(job_rec, available, total)
            
    def dispatch(self, job_rec, resources):

        gpu_assignment = resources.get("gpus", None)
        if gpu_assignment is None:
            gpu_idxs = None
        else:
            gpu_idxs = [g[0] for g in gpu_assignment]

        job_rec['job'].process = launch_local(
            func=job_rec["func"],  args=job_rec["args"], kwargs=job_rec["kwargs"], 
            job_id=job_rec["job"].job_id, log_folder=job_rec["log_folder"], 
            name=job_rec["params"].get("name", None), 
            cuda_devices=gpu_idxs
        )
        job_rec['gpu_assignment'] = gpu_assignment
   
    def attempt_dispatch_job(self, job_rec, available, total, select_gpus='first'):
        
        n_gpus = job_rec['params'].get('gpus', 0) or 0
        
        if n_gpus == 0 or not self.use_gpu:
            return self.dispatch(job_rec, resources={})
        
        if n_gpus <= len(available['gpus']):
            if select_gpus == 'first':
                gpus = set(itertools.islice(list(available['gpus']), n_gpus))
            elif select_gpus == 'random':
                gpus = set(np.random.choice(list(available['gpus']), n_gpus))
            else:
                raise ValueError(f'Unrecognized {select_gpus=}')
            available['gpus'] -= gpus
            return self.dispatch(job_rec, resources={'gpus': gpus})

class ScheduledLocalExecutor:

    def __init__(self, folder: str):
        self.log_folder = Path(folder)
        self.log_folder.mkdir(exist_ok=True)
        self.parameters = {}

    def update_parameters(self, **parameters):
        self.parameters.update(parameters)

    def submit(self, func, *args, **kwargs):
        return LocalScheduleHandler.instance().enqueue(
            func, args, kwargs, params=self.parameters, log_folder=self.log_folder)

"""
key: pid
value: command
"""
def get_all_processes():
    psef_regex = re.compile(" *([0-9]+) +(.*)").fullmatch
    psef_out = subprocess.check_output("ps -e -o pid,cmd --no-headers".split()).decode()
    groups = (psef_regex(l).groups() for l in psef_out.splitlines())
    return {int(pid):cmd for pid, cmd in groups}