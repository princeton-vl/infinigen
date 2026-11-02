# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Run planned land-process steps in worker processes, several tiles at once.

Each step has its gin parameters and seed resolved where it was planned and only
touches its tile folder, so a worker needs no gin or scene state and writes the
same files the step writes inline.
"""

import logging
import os
import pickle
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import gin

from infinigen.perf.terrain.land_process import planning

logger = logging.getLogger(__name__)

WORKER_MODULE = "infinigen.perf.terrain.land_process.worker"
JOB_FILE = "land_process.job"
LOG_FILE = "land_process.log"


class _Job:
    def __init__(self, folder, steps):
        self.folder = Path(folder)
        self.steps = steps
        self.proc = None
        self.log = None
        self.started = None
        self.finished = False
        self.markers = []


class LandProcessPool:
    def __init__(self, max_workers):
        self.max_workers = max(1, int(max_workers))
        self.jobs = {}

    def has_job(self, folder):
        return Path(folder) in self.jobs

    def submit(self, folder, steps):
        steps = [step for step in steps if step is not None]
        if not steps:
            return
        folder = Path(folder)
        if folder in self.jobs:
            raise ValueError(f"land processes already submitted for {folder}")
        self.jobs[folder] = _Job(folder, steps)
        self._pump()

    def add_finish_marker(self, folder, marker):
        job = self.jobs[Path(folder)]
        if job.finished:
            Path(marker).touch()
        else:
            job.markers.append(Path(marker))

    def wait(self):
        while any(not job.finished for job in self.jobs.values()):
            self._pump()
            time.sleep(0.2)

    def terminate(self):
        for job in self.jobs.values():
            if job.proc is not None and not job.finished:
                job.proc.kill()
                job.proc.wait()
                job.finished = True
                job.log.close()

    def _n_running(self):
        return sum(
            1 for job in self.jobs.values() if job.proc is not None and not job.finished
        )

    def _pump(self):
        self._reap()
        for job in self.jobs.values():
            if (
                job.proc is None
                and not job.finished
                and self._n_running() < self.max_workers
            ):
                self._start(job)

    def _start(self, job):
        job.folder.mkdir(parents=True, exist_ok=True)
        jobfile = job.folder / JOB_FILE
        with open(jobfile, "wb") as f:
            pickle.dump(job.steps, f)
        job.log = open(job.folder / LOG_FILE, "w")
        env = dict(os.environ)
        # one thread per worker, so the workers do not oversubscribe the CPUs
        for var in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            env[var] = "1"
        # the cave SDF's KD-tree queries may use this worker's share of the CPUs
        env["INFINIGEN_KDTREE_THREADS"] = str(
            max(1, _usable_cpus() // (self._n_running() + 1))
        )
        job.started = time.time()
        job.proc = subprocess.Popen(
            [sys.executable, "-m", WORKER_MODULE, str(jobfile)],
            stdout=job.log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logger.info(
            f"land processes for {job.folder} started in worker pid {job.proc.pid}"
        )

    def _reap(self):
        for job in self.jobs.values():
            if job.proc is None or job.finished or job.proc.poll() is None:
                continue
            job.finished = True
            job.log.close()
            elapsed = time.time() - job.started
            if job.proc.returncode != 0:
                log_path = job.folder / LOG_FILE
                tail = "".join(open(log_path).readlines()[-30:])
                raise RuntimeError(
                    f"land processes for {job.folder} failed with exit code "
                    f"{job.proc.returncode} after {elapsed:.1f}s; see {log_path}\n{tail}"
                )
            for marker in job.markers:
                marker.touch()
            logger.info(f"land processes for {job.folder} finished in {elapsed:.1f}s")


class _PoolContext:
    def __init__(self, pool):
        self.pool = pool
        self.previous = None

    def __enter__(self):
        self.previous = planning._active_pool
        planning._active_pool = self.pool
        return self.pool

    def __exit__(self, exc_type, exc, tb):
        planning._active_pool = self.previous
        if exc_type is None:
            self.pool.wait()
        else:
            self.pool.terminate()
        return False


def _usable_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _memory_limit_bytes():
    """Memory limit from Slurm, else this process's tightest cgroup limit, or None."""
    env = os.environ
    try:
        if env.get("SLURM_MEM_PER_NODE"):
            return int(env["SLURM_MEM_PER_NODE"]) << 20
        if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_ON_NODE"):
            return (
                int(env["SLURM_MEM_PER_CPU"]) * int(env["SLURM_CPUS_ON_NODE"])
            ) << 20
    except ValueError:
        pass
    limit = None
    try:
        cg = [line for line in open("/proc/self/cgroup") if line.startswith("0::")]
        if cg:
            path = Path("/sys/fs/cgroup" + cg[0].strip().split("::", 1)[1])
            while str(path) != "/sys/fs/cgroup" and path != path.parent:
                for name in ("memory.high", "memory.max"):
                    p = path / name
                    if p.exists():
                        v = p.read_text().strip()
                        if v != "max":
                            limit = int(v) if limit is None else min(limit, int(v))
                path = path.parent
        mem = [line for line in open("/proc/self/cgroup") if ":memory:" in line]
        if mem:
            p = (
                Path("/sys/fs/cgroup/memory" + mem[0].strip().split(":", 2)[2])
                / "memory.limit_in_bytes"
            )
            if p.exists():
                v = int(p.read_text())
                if v < 1 << 60:
                    limit = v if limit is None else min(limit, v)
    except (OSError, ValueError):
        pass
    return limit


def _rss_bytes():
    try:
        status = [
            line for line in open("/proc/self/status") if line.startswith("VmRSS")
        ]
        return int(status[0].split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return 0


@gin.configurable
def land_process_pool(
    enabled=False, max_workers=None, worker_mem_gb=3.0, main_reserve_gb=2.0
):
    """Run land-process steps submitted inside this context in worker processes and
    wait for them on exit; a failed worker raises RuntimeError. Disabled (the default)
    or with one worker, steps run inline. `max_workers` defaults to the usable CPUs,
    capped so `worker_mem_gb` per worker, the main process's size and `main_reserve_gb`
    fit any memory limit, which only changes how many tiles run at once."""
    if not enabled:
        return nullcontext()
    if max_workers is None:
        max_workers = _usable_cpus()
        limit = _memory_limit_bytes()
        if limit is not None and worker_mem_gb > 0:
            free = limit - _rss_bytes() - main_reserve_gb * 1e9
            cap = max(1, int(free // (worker_mem_gb * 1e9)))
            if cap < max_workers:
                logger.info(
                    f"land process pool: {cap} concurrent workers (memory limit "
                    f"{limit / 1e9:.1f} GB, {worker_mem_gb} GB per worker) instead of {max_workers}"
                )
                max_workers = cap
    if max_workers <= 1:
        return nullcontext()
    return _PoolContext(LandProcessPool(max_workers))


def worker_main(jobfile):
    """Run the steps pickled in `jobfile` in this process."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    import cv2

    cv2.setNumThreads(1)
    logger.info(f"worker pid {os.getpid()} running {jobfile}")
    with open(jobfile, "rb") as f:
        steps = pickle.load(f)
    planning.run_steps(steps)
