# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - Hei Law: initial version

import subprocess
import time
from copy import copy
from pathlib import Path

import gin
import submitit

from infinigen.datagen.util.submitit_emulator import LocalJob
from infinigen.tools.suffixes import SUFFIX_ORDERING, get_suffix, parse_suffix

class JobState:
    NotQueued = "notqueued"
    Queued = "queued"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "crashed"
    Cancelled = "cancelled"

class SceneState:
    NotDone = "notdone"
    Done = "done"
    Crashed = "crashed"

CONCLUDED_JOBSTATES = {JobState.Succeeded, JobState.Failed, JobState.Cancelled}
JOB_OBJ_SUCCEEDED = 'MARK_AS_SUCCEEDED'


# Will throw exception if the scene was not found. Sometimes this happens if the scene was queued very very recently
# Keys: JobID ArrayJobID User Group State Clustername Ncpus Nnodes Ntasks Reqmem PerNode Cput Walltime Mem ExitStatus
@gin.configurable
def seff(job_obj, retry_on_error=True):
    scene_id = job_obj.job_id
    assert scene_id.isdigit()
    while True:
        try:
            seff_out = subprocess.check_output(f"/usr/bin/seff -d {scene_id}".split()).decode()
            lines = seff_out.splitlines()
            return dict(zip(lines[0].split(' ')[2:], lines[1].split(' ')[2:]))["State"]
        except Exception as e:
            if not retry_on_error:
                raise e
            time.sleep(1)

def get_scene_state(scene: dict, taskname: str, scene_folder: Path):

    if not scene.get(f'{taskname}_submitted', False):
        return JobState.NotQueued
    elif scene.get(f'{taskname}_crash_recorded', False):
        return JobState.Failed
    elif scene.get(f'{taskname}_force_cancelled', False):
        return JobState.Cancelled
        
    #if scene['all_done']:
    #    return JobState.Succeeded # TODO Hacky / incorrect for nonfatal

    job_obj = scene[f'{taskname}_job_obj']
    
    # for when both local and slurm scenes are being mixed
    if isinstance(job_obj, str):
        assert job_obj == JOB_OBJ_SUCCEEDED
        return JobState.Succeeded
    elif isinstance(job_obj, LocalJob):
        res = job_obj.status()
    elif isinstance(job_obj, submitit.Job):
        res = seff(job_obj)
    else:
        raise TypeError(f'Unrecognized {job_obj=}')

    # map from submitit's scene state strings to our JobState enum
    if res in {"PENDING", "REQUEUED"}:
        return JobState.Queued
    elif res == 'RUNNING':
        return JobState.Running
    elif not (scene_folder/"logs"/f"FINISH_{taskname}").exists():
        return JobState.Failed
    
    return JobState.Succeeded

def cancel_job(job_obj):
    
    if isinstance(job_obj, str):
        assert job_obj == JOB_OBJ_SUCCEEDED
        return JobState.Succeeded
    elif isinstance(job_obj, LocalJob):
        job_obj.kill()
    elif isinstance(job_obj, submitit.Job):
        # TODO: does submitit have a cancel?
        subprocess.check_call(['/usr/bin/scancel', str(job_obj.job_id)])
    else:
        raise TypeError(f'Unrecognized {job_obj=}')