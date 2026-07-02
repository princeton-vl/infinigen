# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import getpass
import os
import platform
import socket
import subprocess


def _detect_gpus() -> list[str]:
    gpus = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus += [line.strip() for line in result.stdout.strip().splitlines()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines()[1:]:
                if line.strip():
                    gpus.append(line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("Chipset Model:"):
                        gpus.append(line.split(":", 1)[1].strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return gpus


def get_hardware_info() -> dict:
    info = {}

    info["username"] = getpass.getuser()
    info["hostname"] = socket.gethostname()
    info["platform"] = platform.platform()
    info["cpu"] = platform.processor() or platform.machine()

    gpus = _detect_gpus()
    if gpus:
        info["gpus_all"] = gpus

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        info["CUDA_VISIBLE_DEVICES"] = cuda_visible

    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    if rocr_visible is not None:
        info["ROCR_VISIBLE_DEVICES"] = rocr_visible

    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    if slurm_vars:
        info["slurm"] = slurm_vars

    return info
