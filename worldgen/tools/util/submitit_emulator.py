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
from pyadl import ADLManager

import numpy as np
from shutil import which

GPU_VISIBIITY_ENVVAR_NAMES = {
    'NVIDIA': 'CUDA_VISIBLE_DEVICES',
    'AMD': 'HIP_VISIBLE_DEVICES'
}

NVIDIA_SMI_PATH = '/bin/nvidia-smi'

def get_hardware_gpus(gpu_type):
        if gpu_type is None:
            return {'0'}
        elif gpu_type == 'NVIDIA':
            if which(NVIDIA_SMI_PATH) is None:
                raise ValueError(f'Attempted to use {gpu_type=} but could not find {NVIDIA_SMI_PATH}')
            result = subprocess.check_output(f'{NVIDIA_SMI_PATH} -L'.split()).decode()                                            
            return set(i for i in range(len(result.splitlines())))
        elif gpu_type == 'AMD':
            return set(i for i, _ in enumerate(ADLManager.getInstance().getDevices()))
        else:
            raise ValueError(f'Unrecognized {gpu_type=}')

def get_visibile_gpus(gpu_type):

    gpus_uuids = get_hardware_gpus()

    envvar = GPU_VISIBIITY_ENVVAR_NAMES[gpu_type]
    if envvar in os.environ:
        visible = [int(s.strip()) for s in os.environ[envvar].split(',')]
        gpus_uuids = gpus_uuids.intersection(visible)
        logging.warning(f"Restricting to {gpus_uuids=} due to toplevel {envvar} setting")

    return gpus_uuids

def set_gpu_visibility(gpu_type, devices=None):
    
    if devices is None:
        varstr = ''
    else:
        varstr = ','.join([str(i) for i in devices])

    envvar = GPU_VISIBIITY_ENVVAR_NAMES.get(gpu_type)
    if envvar is not None:
        os.environ[envvar] = varstr

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

def job_wrapper(func, inner_args, inner_kwargs, stdout_file: Path, stderr_file: Path, gpu_devices=None, gpu_type=None):

    with stdout_file.open('w') as stdout, stderr_file.open('w') as stderr:
        sys.stdout = stdout
        sys.stderr = stderr
        set_gpu_visibility(gpu_type, gpu_devices)
        return func(*inner_args, **inner_kwargs)

def launch_local(func, args, kwargs, job_id, log_folder, name, gpu_devices=None, gpu_type=None):
    
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
        gpu_devices=gpu_devices,
        gpu_type=gpu_type
    )
    proc = Process(target=job_wrapper, kwargs=kwargs, name=name)
    proc.start()

    return proc

class ImmediateLocalExecutor:

    def __init__(self, folder: str):
        self.log_folder = Path(folder).resolve()
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

    def __init__(self, jobs_per_gpu=1, gpu_type='NVIDIA'):
        self.queue = []
        self.jobs_per_gpu = jobs_per_gpu
        self.gpu_type = gpu_type

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

        if self.gpu_type is not None:
            gpu_uuids = get_visibile_gpus(self.gpu_type)
            if len(gpu_uuids) == 0:
                gpu_uuids = {'0'}
            resources['gpus'] = set(itertools.product(range(len(gpu_uuids)), range(self.jobs_per_gpu)))
            
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
            func=job_rec["func"],  
            args=job_rec["args"], 
            kwargs=job_rec["kwargs"], 
            job_id=job_rec["job"].job_id, 
            log_folder=job_rec["log_folder"], 
            name=job_rec["params"].get("name", None), 
            gpu_devices=gpu_idxs, 
            gpu_type=self.gpu_type
        )
        job_rec['gpu_assignment'] = gpu_assignment
   
    def attempt_dispatch_job(self, job_rec, available, total, select_gpus='first'):
        
        n_gpus = job_rec['params'].get('gpus', 0) or 0
        
        if n_gpus == 0 or self.gpu_type is None:
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

if __name__ == "__main__":

    test = 'upload'

    if test == 'render':

        random_command = "../blender/blender --background -noaudio --python generate.py -- --seed 56823 --task coarse -p main.output_folder=outputs/test2 move_camera.stereo_baseline=0.15 LOG_DIR=logs"
        executor = ScheduledLocalExecutor(folder="test_log")
        executor.update_parameters(gpus=1)

        import submitit
        newjob = executor.submit(submitit.helpers.CommandFunction(random_command.split()))

    elif test == 'upload':

        executor = ScheduledLocalExecutor(folder="test_log")
        from manage_datagen_jobs import upload_folder
        newjob = executor.submit(upload_folder, 'test_log', 'dev')

    for i in range(3):
        LocalScheduleHandler.instance().poll()
        time.sleep(3)
        print(newjob.status())
        
    processes = get_all_processes()
    for pid, cmd in processes.items():
        if cmd.startswith('blender'):
            print(pid, cmd)
    print(newjob)
    newjob.kill()