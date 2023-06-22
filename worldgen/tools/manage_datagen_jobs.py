# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - Hei Law: initial version
# Date Signed: May 2 2023

import argparse
import logging
import os
import re
import random
import gin
import subprocess
import time
import sys
import time
import math
import itertools
from uuid import uuid4
from enum import Enum
from copy import copy

from functools import partial, cache
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from shutil import which, rmtree, copyfile

from tqdm import tqdm

import numpy as np
import submitit
import wandb
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tools.util.show_gpu_table import nodes_with_gpus
from tools.util.cleanup import cleanup
from tools.util.submitit_emulator import ScheduledLocalExecutor, ImmediateLocalExecutor, LocalScheduleHandler, LocalJob

from tools.util import upload_util
from tools.util.upload_util import upload_job_folder # for pickle not to freak out

class JobState:
    NotQueued = "notqueued"
    Queued = "queued"
    Running = "running"
    Succeeded = "succeeded"
    Failed = "crashed"

class SceneState:
    NotDone = "notdone"
    Done = "done"
    Crashed = "crashed"

CONCLUDED_STATES = {JobState.Succeeded, JobState.Failed}

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
        except:
            if not retry_on_error:
                raise
            time.sleep(1)

def get_scene_state(scene_dict, taskname, scene_folder):

    if not scene_dict.get(f'{taskname}_submitted', False):
        return JobState.NotQueued
    if scene_dict.get(f'{taskname}_crash_recorded', False):
        return JobState.Failed

    job_obj = scene_dict[f'{taskname}_job_obj']
    
    # for when both local and slurm scenes are being mixed
    if isinstance(job_obj, str):
        assert job_obj == 'MARK_AS_SUCCEEDED'
        return JobState.Succeeded
    elif isinstance(job_obj, LocalJob):
        res = job_obj.status()
    else:
        res = seff(job_obj)

    # map from submitit's scene state strings to our JobState enum
    if res in {"PENDING", "REQUEUED"}:
        return JobState.Queued
    elif res == 'RUNNING':
        return JobState.Running
    elif not (scene_folder/"logs"/f"FINISH_{taskname}").exists():
        return JobState.Failed
    
    return JobState.Succeeded

def seed_generator():
    seed_int = np.random.randint(np.iinfo(np.int32).max)
    return hex(seed_int).removeprefix('0x')

@gin.configurable
def get_cmd(seed, task, configs, taskname, output_folder, driver_script='generate.py', input_folder=None, niceness=None):
    
    if isinstance(task, list):
        task = " ".join(task)

    cmd = ''
    if niceness is not None:
        cmd += f'nice -n {niceness} '
    cmd += f'{BLENDER_PATH} --background -y -noaudio --python {driver_script} -- '
    if input_folder is not None:
        cmd += '--input_folder ' + str(input_folder) + ' '
    if output_folder is not None:
        cmd += '--output_folder ' + str(output_folder) + ' '
    cmd += f'--seed {seed} --task {task} --task_uniqname {taskname} '
    if len(configs) != 0:
        cmd += f'-g {" ".join(configs)} ' 
    cmd += '-p'
    
    return cmd.split()

@gin.configurable
def get_slurm_banned_nodes(config_path=None):
    if config_path is None:
        return []
    with Path(config_path).open('r') as f:
        return list(f.read().split())
    
def get_suffix(indices):

    suffix = ''

    if indices is None:
        return suffix
    
    indices = copy(indices)

    for key in ['cam_rig', 'resample', 'frame', 'subcam']:
        val = indices.get(key, 0)
        suffix += '_' + (f'{val}' if key != 'frame' else f'{val:04d}')

    return suffix

@gin.configurable
def slurm_submit_cmd(cmd, folder, name, mem_gb=None, cpus=None, gpus=0, hours=1, slurm_account=None, slurm_exclude: list = None, **_):

    executor = submitit.AutoExecutor(folder=(folder / "logs"))
    executor.update_parameters(
        mem_gb=mem_gb,
        name=name,
        cpus_per_task=cpus,
        timeout_min=60*hours,
    )
    
    if slurm_exclude is not None:
        executor.update_parameters(slurm_exclude=','.join(slurm_exclude))
    
    if gpus > 0:
        executor.update_parameters(gpus_per_node=gpus)
    if slurm_account is not None:
        executor.update_parameters(slurm_account=slurm_account)

    while True:
        try:
            if callable(cmd[0]):
                func, *arg = cmd
                return executor.submit(func, *arg)
            render_fn = submitit.helpers.CommandFunction(cmd)
            return executor.submit(render_fn)
        except submitit.core.utils.FailedJobError as e:
            current_time_str = datetime.now().strftime("%m/%d %I:%M%p")
            print(f"[{current_time_str}] Job submission failed with error:\n{e}")
            time.sleep(60)

@gin.configurable
def local_submit_cmd(cmd, folder, name, use_scheduler=False, **kwargs):
    
    ExecutorClass = ScheduledLocalExecutor if use_scheduler else ImmediateLocalExecutor
    executor = ExecutorClass(folder=(folder / "logs"))
    executor.update_parameters(name=name, **kwargs)
    if callable(cmd[0]):
        func, *arg = cmd
        return executor.submit(func, *arg)
    else:
        func = submitit.helpers.CommandFunction(cmd)
        return executor.submit(func)

@gin.configurable
def queue_upload(folder, submit_cmd, name, taskname, dir_prefix_len=0, method='rclone', seed=None, **kwargs):
    func = partial(upload_job_folder, dir_prefix_len=dir_prefix_len, method=method)
    res = submit_cmd((func, folder, taskname), folder, name, **kwargs)
    return res, None

@gin.configurable
def queue_coarse(
    folder,
    submit_cmd,
    name,
    seed,
    configs,
    taskname=None,
    exclude_gpus=[],
    overrides=[],
    input_indices=None, output_indices=None,
    **kwargs
):
    """
    Generating the coarse scene
    """

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f'{folder}/coarse{output_suffix}')

    cmd = get_cmd(seed, 'coarse', configs, taskname, output_folder=output_folder) + f'''
        LOG_DIR='{folder / "logs"}'
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('w') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")
    (folder / "run_pipeline.sh").chmod(0o774)

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        gpus=0,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        **kwargs
    )
    return res, output_folder

@gin.configurable
def queue_populate(
    submit_cmd,
    folder,
    name,
    seed,
    configs,
    taskname=None,
    overrides=[],
    input_indices=None, output_indices=None,
    **kwargs,
):
    """
    Generating the fine scene
    """

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    input_folder = folder/f'coarse{input_suffix}'
    output_folder = input_folder

    cmd = get_cmd(seed, 'populate', configs, taskname, 
                  input_folder=input_folder, 
                  output_folder=output_folder) + f'''
        LOG_DIR='{folder / "logs"}'
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        gpus=0,
        slurm_exclude=get_slurm_banned_nodes(),
        **kwargs
    )
    return res, output_folder

@gin.configurable
def queue_fine_terrain(
    submit_cmd,
    folder,
    name,
    seed,
    configs,
    gpus=0,
    taskname=None,
    exclude_gpus=[],
    overrides=[],
    input_indices=None, output_indices=None,
    **kwargs
):
    """
    Generating the fine scene
    """

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f'{folder}/fine{output_suffix}')

    enable_gpu_in_terrain = "Terrain.device='cuda'" if gpus > 0 else ""
    cmd = get_cmd(seed, 'fine_terrain', configs, taskname,
                  input_folder=f'{folder}/coarse{input_suffix}',
                  output_folder=output_folder) + f'''
        LOG_DIR='{folder / "logs"}'
        {enable_gpu_in_terrain}
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        gpus=gpus,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        **kwargs
    )
    return res, output_folder

@gin.configurable
def queue_combined(
    submit_cmd,
    folder,
    name,
    seed,
    configs,
    taskname=None,
    mem_gb=None,
    exclude_gpus=[],
    cpus=None,
    gpus=0,
    hours=None,
    slurm_account=None,
    overrides=[],
    include_coarse=True,
    input_indices=None, output_indices=None,
):
    
    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    tasks = 'populate fine_terrain'

    if include_coarse:
        tasks = 'coarse ' + tasks

    output_folder = Path(f'{folder}/fine{output_suffix}')

    enable_gpu_in_terrain = "Terrain.device='cuda'" if gpus > 0 else ""
    cmd = get_cmd(seed, tasks, configs, taskname, 
                  input_folder=f'{folder}/coarse{input_suffix}' if not include_coarse else None,
                  output_folder=output_folder) + f'''
        LOG_DIR='{folder / "logs"}'
        {enable_gpu_in_terrain}
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        mem_gb=mem_gb,
        folder=folder,
        name=name,
        cpus=cpus,
        gpus=gpus,
        hours=hours,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        slurm_account=slurm_account,
    )
    return res, output_folder

@gin.configurable
def queue_render(
    submit_cmd,
    folder,
    name,
    seed,
    render_type,
    configs,
    taskname=None,
    overrides=[],
    exclude_gpus=[],
    input_indices=None, output_indices=None,
    **submit_kwargs
):

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f'{folder}/frames{output_suffix}')

    cmd = get_cmd(seed, "render", configs, taskname,
                  input_folder=f'{folder}/fine{input_suffix}',
                  output_folder=f'{output_folder}') + f'''
        render.render_image_func=@{render_type}/render_image
        LOG_DIR='{folder / "logs"}'
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        **submit_kwargs,
    )
    return res, output_folder

@gin.configurable
def queue_mesh_save(
    submit_cmd,
    folder,
    name,
    seed,
    configs,
    taskname=None,
    overrides=[],
    exclude_gpus=[],
    input_indices=None, output_indices=None,
    reuse_subcams=True,
    **submit_kwargs
):

    if (output_indices['subcam'] > 0) and reuse_subcams:
        return "MARK_AS_SUCCEEDED", None

    input_suffix = get_suffix(input_indices)

    output_suffix = get_suffix(output_indices)

    output_folder = Path(f'{folder}/savemesh{output_suffix}')

    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = get_cmd(seed, "mesh_save", configs, taskname,
                  input_folder=f'{folder}/fine{input_suffix}',
                  output_folder=f'{folder}/savemesh{output_suffix}') + f'''
        LOG_DIR='{folder / "logs"}'
    '''.split("\n") + overrides

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        **submit_kwargs,
    )
    return res, output_folder

@gin.configurable
def queue_opengl(
    submit_cmd,
    folder,
    name,
    seed,
    configs,
    taskname=None,
    overrides=[],
    exclude_gpus=[],
    input_indices=None, output_indices=None,
    reuse_subcams=True,
    **submit_kwargs
):

    if (output_indices['subcam'] > 0) and reuse_subcams:
        return "MARK_AS_SUCCEEDED", None

    output_suffix = get_suffix(output_indices)

    tmp_script = Path(folder) / "tmp" / f"opengl_{uuid4().hex}.sh"
    tmp_script.parent.mkdir(exist_ok=True)
    print(f"Creating {tmp_script}")

    process_mesh_path = Path("../process_mesh/build/process_mesh").resolve()
    input_folder = Path(folder)/f'savemesh{output_suffix}' # OUTPUT SUFFIX IS CORRECT HERE. I know its weird. But input suffix really means 'prev tier of the pipeline
    output_folder = Path(folder) / f"frames{output_suffix}"
    output_folder.mkdir(exist_ok=True)
    assert input_folder.exists(), input_folder
    assert isinstance(overrides, list) and ("\n" not in ' '.join(overrides))

    start_frame, end_frame = output_indices['frame'], output_indices['last_cam_frame']
    with tmp_script.open('w') as f:
        f.write("set -e\n") # Necessary to detect if the script fails
        for frame_idx in range(start_frame, end_frame + 1):
            line = (
                f"{process_mesh_path} --width 1920 --height 1080 -in {input_folder} "
                f"--frame {frame_idx} -out {output_folder}\n"
            )
            line = re.sub("( \([A-Za-z0-9]+\))", "", line)
            f.write(line)
        f.write(f"touch {folder}/logs/FINISH_{taskname}")

    cmd = f"bash {tmp_script}".split()

    with (folder / "run_pipeline.sh").open('a') as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus) + get_slurm_banned_nodes(),
        **submit_kwargs,
    )
    return res, output_folder

@gin.configurable
def init_db(args, inorder_seeds=False, enumerate_scenetypes=[None]):

    n_scenes = args.num_scenes

    scenes = []

    if args.use_existing:
        for seed_folder in args.output_folder.iterdir():
            
            if not seed_folder.is_dir():
                continue
            if not (seed_folder/'logs').exists():
                logging.warning(f'Skipping {seed_folder=} due to missing "logs" subdirectory')
                continue

            n_scenes -= 1

            scene_dict = {'seed': seed_folder.name, 'all_done': SceneState.NotDone}

            finish_key = 'FINISH_'
            for finish_file_name in (seed_folder/'logs').glob(finish_key + '*'):
                taskname = os.path.basename(finish_file_name)[len(finish_key):]
                print(f'Marking {seed_folder.name=} {taskname=} as completed')
                scene_dict[f'{taskname}_submitted'] = True
                scene_dict[f'{taskname}_job_obj'] = 'MARK_AS_SUCCEEDED'

            scenes.append(scene_dict)
    elif args.specific_seed is not None and len(args.specific_seed):
        return [{"seed": s, "all_done": SceneState.NotDone} for s in args.specific_seed]
    
    if n_scenes > 0:
        for scenetype in enumerate_scenetypes:
            for i in range(n_scenes//len(enumerate_scenetypes)):
                seed = i if inorder_seeds else seed_generator()
                configs = []
                if scenetype is not None:
                    configs.append(scenetype)
                    seed = f'{scenetype}_{i}'
                scene = {"all_done": SceneState.NotDone, "seed": seed, 'scene_configs': configs}
                print(f'Added scene {seed}')
                scenes.append(scene)
    return scenes

def update_symlink(scene_folder, scenes):
    for new_name, scene in scenes:
        std_out = scene_folder / "logs" / f"{scene.job_id}_0_log.out"
        to = scene_folder / "logs" / f"{new_name}.out"
        if os.path.islink(to):
            os.unlink(to)
            os.unlink(scene_folder / "logs" / f"{new_name}.err")
        os.symlink(std_out.resolve(), to)
        os.symlink(std_out.with_suffix('.err').resolve(), scene_folder / "logs" / f"{new_name}.err")

def get_disk_usage(folder):
    out = subprocess.check_output(f"df -h {folder.resolve()}".replace(" (Princeton)", "").split()).decode()
    return int(re.compile("[\s\S]* ([0-9]+)% [\s\S]*").fullmatch(out).group(1)) / 100

def make_html_page(output_path, scenes, frame, camera_pair_id, **kwargs):
    env = Environment(
        loader=FileSystemLoader("tools"),
        autoescape=select_autoescape(),
    )

    template = env.get_template("template.html")
    seeds = [scene['seed'] for scene in scenes]
    html  = template.render(
        seeds=seeds,
        **kwargs,
        frame=frame,
        camera_pair_id=camera_pair_id,
    )

    with output_path.open('a') as f:
        f.write(html)

def run_task(queue_func, scene_folder, scene_dict, taskname):

    assert scene_folder.parent.exists(), scene_folder
    scene_folder.mkdir(exist_ok=True)
    stage_scene_name = f"{scene_folder.parent.stem}_{scene_folder.stem}_{taskname}"
    assert not scene_dict.get(f'{taskname}_submitted', False)

    seed = scene_dict['seed']

    logging.info(f"{seed} - Submitting {taskname} scene")
    job_obj, output_folder = queue_func(
        folder=scene_folder,
        name=stage_scene_name,
        seed=seed,
        taskname=taskname
    )
    scene_dict[f'{taskname}_job_obj'] = job_obj
    scene_dict[f'{taskname}_output_folder'] = output_folder
    scene_dict[f'{taskname}_submitted'] = 1  # marked as submitted
    update_symlink(scene_folder, [(taskname, job_obj)])

def check_and_perform_cleanup(args, seed, crashed):
    scene_folder = args.output_folder/seed
    if args.cleanup == 'all' or (args.cleanup == 'except_crashed' and not crashed):
        logging.info(f"{seed} - Removing entirety of {scene_folder}")
        rmtree(scene_folder)
    elif args.cleanup == 'big_files' or (args.cleanup == 'except_crashed' and crashed):
        logging.info(f"{seed} - Cleaning up any large files")
        cleanup(scene_folder, verbose=False)
    
    if args.remove_write:
        subprocess.check_output(f"chmod -R a-w {scene_folder}".split())

def iterate_sequential_tasks(task_list, get_task_state, overrides, configs, input_indices=None, output_indices=None):

    if len(task_list) == 0:
        return JobState.Succeeded

    prev_state = JobState.Succeeded
    assert task_list[0].get('condition', 'prev_succeeded') == 'prev_succeeded'

    for i, task_spec in enumerate(task_list):
        
        # check that we should actually run this step, according to its condition
        cond = task_spec.get('condition', 'prev_succeeded')
        if cond == 'prev_succeeded' and prev_state != JobState.Succeeded:
            return
        elif cond == 'prev_failed' and prev_state != JobState.Failed:
            continue # we wont run this scene, but skipping doesnt count as crashing
        elif cond == 'prev_redundant':
            pass # any outcome is fine

        # determine whether the current step failing would be catastrophic
        fatal = (
            i + 1 >= len(task_list) or
            task_list[i + 1].get('condition', 'prev_succeeded') != 'prev_failed'
        )
            
        queue_func = partial(task_spec['func'], overrides=overrides, configs=configs, 
            input_indices=input_indices, output_indices=output_indices)

        taskname = task_spec['name'] + get_suffix(output_indices)
        state = get_task_state(taskname=taskname)
        yield state, taskname, queue_func, fatal

        prev_state = state

@gin.configurable
def iterate_scene_tasks(
    scene_dict, args,
    monitor_all, # if True, enumerate scenes that we might have launched earlier, even if we wouldnt launch them now (due to crashes etc)

    # provided by gin
    global_tasks, view_dependent_tasks, camera_dependent_tasks, 
    frame_range, cam_id_ranges, num_resamples=1, render_frame_range=None,
    view_block_size=1, # how many frames should share each `view_dependent_task`
    cam_block_size=None, # how many frames should share each `camera_dependent_task`
    cleanup_viewdep=False, # cleanup the results of `view_dependent_tasks` once each view iter is done?
    viewdep_paralell=True, # can we work on multiple view depenendent tasks (usually `fine`) in paralell?
    camdep_paralell=True # can we work on multiple camera dependent tasks (usually render/gt) in paralell?
):
    
    '''
    This function is a generator which yields all scenes we might want to consider monitoring or running for a particular scene

    It `yield`s the available scenes, regardless of whether they are already running etc
    '''

    for task in global_tasks + view_dependent_tasks + camera_dependent_tasks:
        if '_' in task['name']:
            raise ValueError(f'{task=} with {task["name"]=} is invalid, must not contain underscores')

    if cam_block_size is None:
        cam_block_size = view_block_size
    
    if cam_id_ranges[0] <= 0 or cam_id_ranges[1] <= 0:
        raise ValueError(f'{cam_id_ranges=} is invalid, both num. rigs and num subcams must be >= 1 or no work is done')
    assert view_block_size >= 1
    assert cam_block_size >= 1
    if cam_block_size > view_block_size:
        cam_block_size = view_block_size
    seed = scene_dict['seed']

    scene_folder = args.output_folder/seed
    get_task_state = partial(get_scene_state, scene_dict=scene_dict, scene_folder=scene_folder)

    global_overrides = [f'execute_tasks.frame_range={repr(list(frame_range))}', f'execute_tasks.camera_id=[0, 0]']
    global_configs = args.configs + scene_dict.get('scene_configs', [])
    global_iter = iterate_sequential_tasks(global_tasks, get_task_state,
        overrides=args.override+global_overrides, configs=global_configs)

    for state, *rest in global_iter:
        yield state, *rest
    if not state == JobState.Succeeded:
        logging.debug(f'{seed=} waiting on global')
        return

    view_range = render_frame_range if render_frame_range is not None else frame_range
    view_frames = range(view_range[0], view_range[1] + 1, view_block_size) # blender frame_range is inclusive, but python's range is end-exclusive 
    resamples = range(num_resamples)
    cam_rigs = range(cam_id_ranges[0])
    subcams = range(cam_id_ranges[1])

    running_views = 0
    for cam_rig, view_frame in itertools.product(cam_rigs, view_frames):

        view_frame_range = [view_frame, min(frame_range[1], view_frame + view_block_size - 1)] # blender frame_end is INCLUSIVE
        view_overrides = [
            f'execute_tasks.frame_range=[{view_frame_range[0]},{view_frame_range[1]}]',
            f'execute_tasks.camera_id=[{cam_rig},{0}]'
        ]

        view_idxs = dict(cam_rig=cam_rig, frame=view_frame)
        view_tasks_iter = iterate_sequential_tasks(
            view_dependent_tasks, get_task_state,
            overrides=args.override+view_overrides, 
            configs=global_configs, output_indices=view_idxs
        )
        for state, *rest in view_tasks_iter:
            yield state, *rest
        if state not in CONCLUDED_STATES:
            if viewdep_paralell:
                running_views += 1
                logging.debug(f'{seed=} {cam_rig,view_frame=} waiting on viewdep')
                continue
            else:
                return 
        elif state == JobState.Failed and not monitor_all:
            return

        running_blocks = 0
        for subcam, resample_idx in itertools.product(subcams, resamples):
            for cam_frame in range(view_frame_range[0], view_frame_range[1] + 1, cam_block_size):
                cam_frame_range = [cam_frame, min(view_frame_range[1], cam_frame + cam_block_size - 1)] # blender frame_end is INCLUSIVE
                cam_overrides = [
                    f'execute_tasks.frame_range=[{cam_frame_range[0]},{cam_frame_range[1]}]',
                    f'execute_tasks.camera_id=[{cam_rig},{subcam}]',
                    f'execute_tasks.resample_idx={resample_idx}'
                ]

                camdep_indices = dict(
                    cam_rig=cam_rig, frame=cam_frame, subcam=subcam, resample=resample_idx,
                    view_first_frame=view_frame_range[0], last_view_frame=view_frame_range[1], last_cam_frame=cam_frame_range[1] # this line explicitly used by most jobs
                ) 
                camera_dep_iter = iterate_sequential_tasks(
                    camera_dependent_tasks, get_task_state,
                    overrides=args.override+cam_overrides, configs=global_configs,
                    input_indices=view_idxs if len(view_dependent_tasks) else None,
                    output_indices=camdep_indices)
                for state, *rest in camera_dep_iter:
                    yield state, *rest
                if state not in CONCLUDED_STATES:
                    if camdep_paralell:
                        running_blocks += 1
                        logging.debug(f'{seed=} {cam_rig,cam_frame=} waiting on viewdep')
                        continue
                    else:
                        return
                elif state == JobState.Failed and not monitor_all:
                    return

        if running_blocks > 0:
            running_views += 1
            continue

        key = f'viewdep_{cam_rig}_{view_frame}_cleaned'
        if cleanup_viewdep and args.cleanup != 'none' and not scene_dict.get(key, False):
            for stage_rec in view_dependent_tasks:
                taskname = stage_rec['name']
                path = scene_dict[f'{taskname}_output_folder']
                print(f'Cleaning {path} for {taskname}')
                if path == scene_folder:
                    print(f'Skipping {path}')
                    continue
                if path is not None and path.exists():
                    cleanup(path)
            scene_dict[key] = True

    if running_views > 0:
        return

    # Upload
    if args.upload:
        state = get_task_state(taskname='upload')
        yield state, 'upload', queue_upload, True
        if state != JobState.Succeeded:
            return
        
    if scene_dict['all_done'] != SceneState.NotDone:
        return

    # Cleanup
    with (args.output_folder / "finished_seeds.txt").open('a') as f:
        f.write(f"{seed}\n")
    scene_dict['all_done'] = SceneState.Done
    check_and_perform_cleanup(args, seed, crashed=False)

def infer_crash_reason(stdout_file, stderr_file: Path):

    if not stderr_file.exists():
        return f'{stderr_file} not found'
    
    try:
        error_log = stderr_file.read_text()
    except UnicodeDecodeError:
        return f"failed to parse log file {stderr_file}"

    if "System is out of GPU memory" in error_log:
        return "Out of GPU memory"
    elif "this scene is timed-out" in error_log:
        return "Timed out"
    elif "<Signals.SIGKILL: 9>" in error_log:
        return "SIGKILL: 9 (out-of-memory, probably)"
    elif "SIGCONT" in error_log:
        return "SIGCONT (timeout?)"
    elif "srun: error" in error_log:
        return "srun error"

    if not stdout_file.exists():
        return f'{stdout_file} not found'
    if not stderr_file.exists():
        return f'{stderr_file} not found'

    output_text = f"{stdout_file.read_text()}\n{stderr_file.read_text()}\n"
    matches = re.findall("(Error:[^\n]+)\n", output_text)

    if len(matches):
        return ','.join([m.strip() for m in matches if ('Error: Not freed memory blocks' not in m)])
    else:
        return "unknown cause" 

def record_crashed_seed(crashed_seed, crash_stage, f, fatal=True):
    time_str = datetime.now().strftime("%m/%d %I:%M%p")
    stdout_file = args.output_folder / crashed_seed / "logs" / f"{crash_stage}.out"
    stderr_file = args.output_folder / crashed_seed / "logs" / f"{crash_stage}.err"
    scene_id, *_ = stderr_file.resolve().stem.split('_')
    node_of_scene = ""
    if which('sacct'):
        try:
            node_of_scene, *rest  = subprocess.check_output(f"{which('sacct')} -j {scene_id} --format Node --noheader".split()).decode().split()
        except Exception as e:
            logging.warning(f'sacct threw {e}')
            return
    reason = infer_crash_reason(stdout_file, stderr_file)
    text = f"{crashed_seed} {crash_stage} {scene_id} {node_of_scene} {reason} {fatal=} {time_str}\n"
    print('Crashed: ' + text)
    f.write(text)

    return reason
    
def write_html_summary(all_scenes, output_folder, max_size=5000):

    names = [("index" if (idx == 0) else f"index_{idx}") for idx in range(0, len(all_scenes), max_size)]
    for name, idx in zip(names, range(0, len(all_scenes), max_size)):
        html_path = output_folder / f"{name}.html"
        if not html_path.exists():
            make_html_page(html_path, all_scenes[idx:idx+max_size], frame=100,
            camera_pair_id=0, samples=[f"resmpl{i}" for i in range(5)], pages=names,
        )

def stats_summary(stats):
    stats = {k: v for k, v in stats.items() if not k.startswith(JobState.NotQueued)}
    lemmatized = set(l.split('_')[0] for l in stats.keys())
    stats = {l: sum(v for k, v in stats.items() if k.startswith(l)) for l in lemmatized}
    for p in set(k.split('/')[0] for k in stats.keys()):
        stats[f'{p}/total'] = sum(v for k, v in stats.items() if k.startswith(p))
    return stats

@gin.configurable
def manage_datagen_jobs(all_scenes, elapsed, num_concurrent, sleep_threshold=0.95):

    if LocalScheduleHandler._inst is not None:
        LocalScheduleHandler.instance().poll()

    curr_concurrent_max = math.floor(1 + num_concurrent * elapsed / args.warmup_sec) if elapsed < args.warmup_sec else num_concurrent

    # Check results / current state of scenes we have already launched 
    stats = defaultdict(int)
    for scene in all_scenes:
        scene['num_running'], scene['num_done'] = 0, 0
        any_fatal = False
        for state, taskname, _, fatal in iterate_scene_tasks(scene, args, monitor_all=True):
            stats[f'{state}/{taskname}'] += 1
            scene['num_done'] += state in CONCLUDED_STATES
            scene['num_running'] += state not in CONCLUDED_STATES
            if state == JobState.Failed:
                if not scene.get(f'{taskname}_crash_recorded', False):
                    scene[f'{taskname}_crash_recorded'] = True
                    with (args.output_folder / "crashed.txt").open('a') as f:
                        record_crashed_seed(scene['seed'], taskname, f, fatal=fatal)
                if fatal:
                    any_fatal = True

        if scene['num_running'] == 0 and any_fatal and scene['all_done'] == SceneState.NotDone:
            scene['all_done'] = SceneState.Crashed    
            with (args.output_folder / "crashed.txt").open('a') as f:
                check_and_perform_cleanup(args, scene['seed'], crashed=True)

    # Report stats, with sums by prefix, and extra info
    stats = stats_summary(stats)
    stats['disk_usage'] = get_disk_usage(args.output_folder)
    stats['concurrent_max'] = curr_concurrent_max
    wandb.log(stats)
    for k,v in sorted(stats.items()):
        print(f"{k.ljust(30)} : {v}")
    print("-" * 60)

    # Dont launch new scenes if disk is getting full
    if stats['disk_usage'] > sleep_threshold:
        print(f"{args.output_folder} is too full ({get_disk_usage(args.output_folder)}%). Sleeping.")
        wandb.alert(title='Disk full', text=f'Sleeping due to full disk at {args.output_folder=}', wait_duration=3*60*60)
        time.sleep(60)
        return

    # Launch new scenes to bring the current load back up to `curr_concurrent_max`
    scenes = [j for j in all_scenes if (j["all_done"] == SceneState.NotDone)]
    scenes = sorted(scenes, key=lambda s: s['num_running'] + s['num_done'], reverse=True) # greedily try to finish nearly-done videos asap
    to_be_launched = curr_concurrent_max - stats.get(f'{JobState.Running}/all', 0) - stats.get(f'{JobState.Queued}/all', 0)
    if to_be_launched <= 0:
        return
    for scene in scenes[:curr_concurrent_max]:
        for state, taskname, queue_func, _ in iterate_scene_tasks(scene, args, monitor_all=False):
            if state != JobState.NotQueued:
                continue
            to_be_launched -= 1
            run_task(queue_func, args.output_folder / str(scene['seed']), scene, taskname)
            if to_be_launched == 0:
                break
        if to_be_launched == 0:
            break

@gin.configurable
def main(args, shuffle=True, wandb_project='render_beta'):

    os.umask(0o007)

    all_scenes = init_db(args)
    scene_name = args.output_folder.parts[-1]

    write_html_summary(all_scenes, args.output_folder) if args.cleanup != 'all' else None
    wandb.init(name=scene_name, config=vars(args), project=wandb_project, mode=args.wandb_mode)

    logging.basicConfig(
        filename=str(args.output_folder / "jobs.log"),
        level=args.loglevel,
        format='[%(asctime)s]: %(message)s',
    )

    start_time = datetime.now()

    scenes = [j for j in all_scenes if j['all_done'] == SceneState.NotDone]
    if shuffle:
        np.random.shuffle(scenes)
    else:
        scenes = sorted(scenes, key=lambda j: j['seed'])
    while any(j['all_done'] == SceneState.NotDone for j in all_scenes):
        now = datetime.now()
        print(f'{args.output_folder} {start_time.strftime("%m/%d %I:%M%p")} -> {now.strftime("%m/%d %I:%M%p")}')
        logging.info('=' * 80)
        manage_datagen_jobs(scenes, elapsed=(now-start_time).seconds)
        logging.info("-" * 80)
        time.sleep(4)

def test_upload(args):

    from_folder = args.output_folder/f'test_upload_{args.output_folder.name}'
    from_folder.mkdir(parents=True, exist_ok=True)
    (from_folder/'test_file.txt').touch()

    upload_util.upload_folder(from_folder, Path('infinigen/test_upload/'))
    rmtree(from_folder)
    
if __name__ == "__main__":
    assert Path('.').resolve().parts[-1] == 'worldgen'

    slurm_available = (which("sbatch") is not None)
    parser = argparse.ArgumentParser() # to guarantee that the render scenes finish, try render_image.time_limit=2000
    parser.add_argument('--blender_path', type=str, default=None)
    parser.add_argument('-o', '--output_folder', type=Path, required=True)
    parser.add_argument('--num_scenes', type=int, default=1)
    parser.add_argument('--meta_seed', type=int, default=None)
    parser.add_argument('--specific_seed', default=None, nargs='+', help="It will cast to an int if it's a number, otherwise str")
    parser.add_argument('--use_existing', action='store_true')

    parser.add_argument('--warmup_sec', type=float, default=0)
    parser.add_argument('--cleanup', type=str, choices=['all', 'big_files', 'none', 'except_crashed'], default='none')
    parser.add_argument('--remove_write', action='store_true')
    parser.add_argument('--upload', action='store_true')
    
    parser.add_argument('--configs', nargs='*', default=[])
    parser.add_argument('-p', '--override', nargs='+', type=str, default=[], help="gin-config settings to override. FYI all scenes will use the same overrides.")

    parser.add_argument('--wandb_mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--pipeline_configs', type=str, nargs='+', default=["allcs" if slurm_available else "local"])
    parser.add_argument('--pipeline_overrides', nargs='+', type=str, default=[], help="pipeline gin-config settings to override")
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument( '-v', '--verbose', action="store_const", dest="loglevel", const=logging.INFO)
    args = parser.parse_args()

    envvar = 'INFINIGEN_ASSET_FOLDER'

    if not args.upload and args.cleanup == 'all':
        raise ValueError(f'Pipeline is configured {args.upload=} {args.cleanup=} --- no output would be preserved')

    global BLENDER_PATH
    if args.blender_path is None:
        if 'BLENDER' in os.environ:
            BLENDER_PATH = os.environ['BLENDER']
        else:
            BLENDER_PATH = '../blender/blender' # assuming we run from infinigen/worldgen
    else:
        BLENDER_PATH = args.blender_path
    if not os.path.exists(BLENDER_PATH):
        raise ValueError(f'Couldnt not find {BLENDER_PATH=}, make sure --blender_path or $BLENDER is specified')

    assert args.specific_seed is None or args.num_scenes == 1

    if args.output_folder.exists() and not args.use_existing:
        raise FileExistsError(f'--output_folder {args.output_folder} already exists! Quitting to avoid overwrite. Please delete it, or specify a new --output_folder')

    args.output_folder.mkdir(parents=True, exist_ok=args.use_existing)

    if args.meta_seed is not None:
        random.seed(args.meta_seed)
        np.random.seed(args.meta_seed)

    find_gin = lambda n: os.path.join("tools", "pipeline_configs", f"{n}.gin")
    configs = [find_gin(n) for n in args.pipeline_configs]
    for c in configs:
        assert os.path.exists(c), c
    bindings = args.pipeline_overrides
    gin.parse_config_files_and_bindings(configs, bindings=bindings)

    main(args)
