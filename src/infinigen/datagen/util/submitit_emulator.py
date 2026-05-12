# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson - LocalJob
# - Alex Raistrick - Local queue handler
# - David Yan - Bugfix


import copy
import itertools
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from shutil import which

import gin
import numpy as np

logger = logging.getLogger(__name__)

CUDA_VARNAME = "CUDA_VISIBLE_DEVICES"
NVIDIA_SMI_PATH = "/bin/nvidia-smi"


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


class FileTee:
    """
    Wrap a file-like object in order to write all its output to a stream aswell (usually sys.stdout or sys.stderr)
    """

    def __init__(self, inner, stream):
        self.inner = inner
        self.stream = stream

    def write(self, data):
        self.inner.write(data)
        self.stream.write(data)

    def flush(self):
        self.inner.flush()
        self.stream.flush()

    def close(self):
        self.inner.close()
        self.stream.close()

    def fileno(self):
        return self.inner.fileno()


def get_fake_job_id():
    # Lahav assures me these will never conflict
    return np.random.randint(int(1e10), int(1e11))


def job_wrapper(
    command: list[str],
    stdout_file: Path,
    stderr_file: Path,
    cuda_devices=None,
    stdout_passthrough: bool = False,
):
    with stdout_file.open("w") as stdout, stderr_file.open("w") as stderr:
        if stdout_passthrough:
            # TODO: send output to BOTH the file and the console
            stdout = None
            stderr = None

        if cuda_devices is not None:
            env = os.environ.copy()
            env[CUDA_VARNAME] = ",".join([str(i) for i in cuda_devices])
        else:
            env = None

        subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            shell=False,
            check=False,  # dont throw CalledProcessError
            env=env,
        )


def launch_local(
    command: str,
    job_id: str,
    log_folder: Path,
    name: str,
    cuda_devices=None,
    stdout_passthrough: bool = False,
):
    stderr_file = log_folder / f"{job_id}_0_log.err"
    stdout_file = log_folder / f"{job_id}_0_log.out"

    with stdout_file.open("w") as f:
        f.write(f"{command}\n")
    stderr_file.touch()

    kwargs = dict(
        command=command,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        cuda_devices=cuda_devices,
        stdout_passthrough=stdout_passthrough,
    )
    proc = Process(target=job_wrapper, kwargs=kwargs, name=name)
    proc.start()

    return proc


class ImmediateLocalExecutor:
    def __init__(self, folder: str | None, stdout_passthrough: bool = False):
        if folder is None:
            self.log_folder = None
        else:
            self.log_folder = Path(folder).resolve()
            self.log_folder.mkdir(exist_ok=True)

        self.stdout_passthrough = stdout_passthrough

        self.parameters = {}

    def update_parameters(self, **parameters):
        self.parameters.update(parameters)

    def submit(self, command: str):
        job_id = get_fake_job_id()
        name = self.parameters.get("name", None)
        proc = launch_local(
            command,
            job_id,
            log_folder=self.log_folder,
            name=name,
            stdout_passthrough=self.stdout_passthrough,
        )
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

    def enqueue(
        self, command: str, params: dict, log_folder: Path, stdout_passthrough: bool
    ):
        job = LocalJob(job_id=get_fake_job_id(), process=None)
        job_rec = dict(
            command=command,
            params=params,
            job=job,
            log_folder=log_folder,
            gpu_assignment=None,
            stdout_passthrough=stdout_passthrough,
        )

        # matches behavior of submitit (?) - user code expects to be able to set up its
        #   symlinks for these logfiles right at job queue time
        stderr_file = log_folder / f"{job.job_id}_0_log.err"
        stdout_file = log_folder / f"{job.job_id}_0_log.out"
        stderr_file.touch()
        stdout_file.touch()

        self.queue.append(job_rec)
        return job

    @gin.configurable
    def total_resources(self) -> set:
        resources = {}

        if self.use_gpu:
            if which(NVIDIA_SMI_PATH) is None:
                raise ValueError(
                    f"LocalScheduleHandler.use_gpu=True yet could not find {NVIDIA_SMI_PATH}, "
                    "please use --pipeline_overrides LocalScheduleHandler.use_gpu=False if your machine does not have a supported GPU"
                )

            result = subprocess.check_output(f"{NVIDIA_SMI_PATH} -L".split()).decode()
            gpus_uuids = set(i for i in range(len(result.splitlines())))

            if CUDA_VARNAME in os.environ:
                visible = [int(s.strip()) for s in os.environ[CUDA_VARNAME].split(",")]
                gpus_uuids = gpus_uuids.intersection(visible)
                logger.debug(
                    f"Restricting to {gpus_uuids=} due to toplevel {CUDA_VARNAME} setting"
                )

            resources["gpus"] = set(
                itertools.product(gpus_uuids, range(self.jobs_per_gpu))
            )

        return resources

    def resources_available(self, total) -> set:
        resources = copy.copy(total)

        for job_rec in self.queue:
            if job_rec["job"].status() != "RUNNING":
                continue
            if (g := job_rec["gpu_assignment"]) is not None:
                resources["gpus"] -= g

        return resources

    def poll(self):
        total = self.total_resources()
        available = self.resources_available(total)
        logger.debug(f"Checked resources, {total=} {available=}")

        for job_rec in self.queue:
            if job_rec["job"].status() != "PENDING":
                continue
            self.attempt_dispatch_job(job_rec, available, total)

    def dispatch(self, job_rec, resources):
        gpu_assignment = resources.get("gpus", None)
        if gpu_assignment is None:
            gpu_idxs = None
        else:
            gpu_idxs = [g[0] for g in gpu_assignment]

        job_rec["job"].process = launch_local(
            command=job_rec["command"],
            job_id=job_rec["job"].job_id,
            log_folder=job_rec["log_folder"],
            name=job_rec["params"].get("name", None),
            cuda_devices=gpu_idxs,
            stdout_passthrough=job_rec["stdout_passthrough"],
        )
        job_rec["gpu_assignment"] = gpu_assignment

    def attempt_dispatch_job(
        self, job_rec, available: set, total: set, select_gpus="first"
    ):
        n_gpus = job_rec["params"].get("gpus", 0) or 0

        if n_gpus == 0 or not self.use_gpu:
            return self.dispatch(job_rec, resources={})

        if n_gpus <= len(available["gpus"]):
            if select_gpus == "first":
                gpus = set(itertools.islice(list(available["gpus"]), n_gpus))
            elif select_gpus == "random":
                gpus = set(np.random.choice(list(available["gpus"]), n_gpus))
            else:
                raise ValueError(f"Unrecognized {select_gpus=}")
            available["gpus"] -= gpus
            return self.dispatch(job_rec, resources={"gpus": gpus})


class ScheduledLocalExecutor:
    def __init__(self, folder: str, stdout_passthrough: bool = False):
        self.log_folder = Path(folder)
        self.log_folder.mkdir(exist_ok=True)

        self.stdout_passthrough = stdout_passthrough

        self.parameters = {}

    def update_parameters(self, **parameters):
        self.parameters.update(parameters)

    def submit(self, command):
        return LocalScheduleHandler.instance().enqueue(
            command,
            params=self.parameters,
            log_folder=self.log_folder,
            stdout_passthrough=self.stdout_passthrough,
        )


"""
key: pid
value: command
"""


def get_all_processes():
    psef_regex = re.compile(" *([0-9]+) +(.*)").fullmatch
    psef_out = subprocess.check_output("ps -e -o pid,cmd --no-headers".split()).decode()
    groups = (psef_regex(l).groups() for l in psef_out.splitlines())
    return {int(pid): cmd for pid, cmd in groups}
