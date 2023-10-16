# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - Hei Law: initial version


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
import importlib

from uuid import uuid4
from enum import Enum
from copy import copy
from ast import literal_eval

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from shutil import which

import pandas as pd
import numpy as np
import submitit
from jinja2 import Environment, FileSystemLoader, select_autoescape

ORIG_SYS_PATH = list(sys.path) # Make a new instance of sys.path
import infinigen.core.init
BPY_SYS_PATH = list(sys.path) # Make instance of `bpy`'s modified sys.path

from infinigen.datagen.monitor_tasks import iterate_scene_tasks, on_scene_termination
from infinigen.datagen.util import upload_util
from infinigen.datagen.states import (
    JobState, 
    SceneState, 
    CONCLUDED_JOBSTATES, 
    JOB_OBJ_SUCCEEDED, 
    cancel_job
)
from infinigen.datagen.util.submitit_emulator import (
    ScheduledLocalExecutor, 
    ImmediateLocalExecutor, 
    LocalScheduleHandler
)

from infinigen.datagen import job_funcs 
from infinigen.datagen.job_funcs import (
    # referenced by name via gin configs
    queue_coarse,
    queue_combined,
    queue_fine_terrain,
    queue_mesh_save,
    queue_opengl,
    queue_populate,
    queue_render,
    queue_upload
)

logger = logging.getLogger(__name__)

wandb = None # will be imported and initialized ONLY if installed and enabled

# used only if enabled in gin configs
PARTITION_ENVVAR = 'INFINIGEN_SLURMPARTITION' 
EXCLUDE_FILE_ENVVAR = 'INFINIGEN_SLURM_EXCLUDENODES_LIST'
NUM_CONCURRENT_ENVVAR = 'INFINIGEN_NUMCONCURRENT_TARGET'

def node_from_slurm_jobid(scene_id):

    if not which('sacct'):
        return None
    
    try:
        node_of_scene, *rest  = subprocess.check_output(f"{which('sacct')} -j {scene_id} --format Node --noheader".split()).decode().split()
        return node_of_scene
    except Exception as e:
        logger.warning(f'sacct threw {e}')
        return None

def seed_generator():
    seed_int = np.random.randint(np.iinfo(np.int32).max)
    return hex(seed_int).removeprefix('0x')

@gin.configurable
def get_slurm_banned_nodes(config_path=None):
    if config_path == f'ENVVAR_{EXCLUDE_FILE_ENVVAR}':
        config_path = os.environ.get(EXCLUDE_FILE_ENVVAR)
    if config_path is None:
        return []
    with Path(config_path).open('r') as f:
        return list(f.read().split())

@gin.configurable
def slurm_submit_cmd(
    cmd, 
    folder, 
    name, 
    mem_gb=None, 
    cpus=None, 
    gpus=0, 
    hours=1, 
    slurm_account=None, 
    slurm_exclude: list = None, 
    slurm_niceness=None,
    **_
):

    executor = submitit.AutoExecutor(folder=(folder / "logs"))
    executor.update_parameters(
        mem_gb=mem_gb,
        name=name,
        cpus_per_task=cpus,
        timeout_min=60*hours,
    )
    
    exclude = get_slurm_banned_nodes()
    if slurm_exclude is not None:
        exclude += slurm_exclude
    if len(exclude):
        executor.update_parameters(slurm_exclude=','.join(exclude))
    
    if gpus > 0:
        executor.update_parameters(gpus_per_node=gpus)
    if slurm_account is not None:

        if slurm_account == f'ENVVAR_{PARTITION_ENVVAR}':
            slurm_account = os.environ.get(PARTITION_ENVVAR)
            if slurm_account is None:
                logger.warning(f'{PARTITION_ENVVAR=} was not set, using no slurm account')

        if isinstance(slurm_account, list):
            slurm_account = np.random.choice(slurm_account)

        executor.update_parameters(slurm_account=slurm_account)

    slurm_additional_params = {}

    if slurm_niceness is not None:
        slurm_additional_params['nice'] = slurm_niceness

    executor.update_parameters(slurm_additional_parameters=slurm_additional_params)

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

def init_db_from_existing(output_folder: Path):

    # TODO in future: directly use existing_db (with some cleanup / checking).

    db_path = output_folder/'scenes_db.csv'
    if not db_path.exists():
        raise ValueError(f'Recieved --use_existing but {db_path=} did not exist')
    existing_db = pd.read_csv(db_path, converters={"configs": literal_eval})

    def init_scene(seed_folder):
        if not seed_folder.is_dir():
            return None
        if not (seed_folder/'logs').exists():
            logger.warning(f'Skipping {seed_folder=} due to missing "logs" subdirectory')
            return None

        scene_dict = {
            'seed': seed_folder.name, 
            'all_done': SceneState.NotDone,
        }

        if 'configs' in existing_db.columns:
            mask = existing_db["seed"].astype(str) == seed_folder.name
            configs = existing_db.loc[mask, "configs"].iloc[0]
            scene_dict['configs']: list(configs)

        finish_key = 'FINISH_'
        for finish_file_name in (seed_folder/'logs').glob(finish_key + '*'):
            taskname = os.path.basename(finish_file_name)[len(finish_key):]
            logger.info(f'Marking {seed_folder.name=} {taskname=} as completed')
            scene_dict[f'{taskname}_submitted'] = True
            scene_dict[f'{taskname}_job_obj'] = JOB_OBJ_SUCCEEDED

        return scene_dict

    return [init_scene(seed_folder) for seed_folder in output_folder.iterdir()]

@gin.configurable
def sample_scene_spec(i, seed_range=None, config_distribution=None, config_sample_mode='random'):

    if seed_range is None:
        seed = seed_generator()
    else:
        start, end = seed_range
        if i > end - start:
            return None
        seed = hex(start + i).removeprefix('0x')

    if config_distribution is None:
        configs = []
    elif config_sample_mode == 'random':
        configs_options, weights = zip(*config_distribution) # list of rows to list per column
        ps = np.array(weights) / sum(weights)
        configs = np.random.choice(configs_options, p=ps)
    elif config_sample_mode == 'roundrobin':
        configs_options, weights = zip(*config_distribution) # list of rows to list per column
        if not all(isinstance(w, int) for w in weights):
            raise ValueError(f'{config_sample_mode=} expects integer scene counts as weights but got {weights=} with non-integer values')
        idx = np.argmin(i % sum(weights) + 1 > np.cumsum(weights))
        configs = configs_options[idx]
    else:
        raise ValueError(f'Unrecognized {config_sample_mode=}')
    
    if isinstance(configs, str) and " " in configs:
        configs = configs.split(" ")
    if not isinstance(configs, list):
        configs = [configs]

    return {
        "all_done": SceneState.NotDone, 
        "seed": seed, 
        'configs': configs
    }

@gin.configurable
def init_db(args):

    if args.use_existing:
        scenes = init_db_from_existing(args.output_folder)
    elif args.specific_seed is not None:
        scenes = [{"seed": s, "all_done": SceneState.NotDone} for s in args.specific_seed]
    else:
        scenes = [sample_scene_spec(i) for i in range(args.num_scenes)]    

    scenes = [s for s in scenes if s is not None]

    if len(scenes) < args.num_scenes:
        logger.warning(f'Initialized only {len(scenes)=} despite {args.num_scenes=}. Likely due to --use_existing, --specific_seed or seed_range.')

    return scenes

def update_symlink(scene_folder, scenes):
    for new_name, scene in scenes:

        if scene == JOB_OBJ_SUCCEEDED:
            continue
        elif isinstance(scene, str):
            raise ValueError(f'Failed due to {scene=}')

        to = scene_folder / "logs" / f"{new_name}.out"
        std_out = scene_folder / "logs" / f"{scene.job_id}_0_log.out"

        if os.path.islink(to):
            os.unlink(to)
            os.unlink(scene_folder / "logs" / f"{new_name}.err")
        os.symlink(std_out.resolve(), to)
        os.symlink(std_out.with_suffix('.err').resolve(), scene_folder / "logs" / f"{new_name}.err")

def get_disk_usage(folder):
    out = subprocess.check_output(f"df -h {folder.resolve()}".replace(" (Princeton)", "").split()).decode()
    return int(re.compile("[\s\S]* ([0-9]+)% [\s\S]*").fullmatch(out).group(1)) / 100

def make_html_page(output_path, scenes, frame, camera_pair_id, **kwargs):

    template_path = infinigen.core.init.repo_root()/"infinigen/datagen/util"
    assert template_path.exists(), template_path
    env = Environment(
        loader=FileSystemLoader(template_path),
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

@gin.configurable
def run_task(
    queue_func, 
    scene_folder, 
    scene_dict, 
    taskname, 
    dryrun=False
):

    assert scene_folder.parent.exists(), scene_folder
    scene_folder.mkdir(exist_ok=True)

    scene_folder = scene_folder.resolve()

    stage_scene_name = f"{scene_folder.parent.stem}_{scene_folder.stem}_{taskname}"
    assert not scene_dict.get(f'{taskname}_submitted', False)

    if dryrun:
        scene_dict[f'{taskname}_job_obj'] = JOB_OBJ_SUCCEEDED
        scene_dict[f'{taskname}_submitted'] = 1
        return

    job_obj, output_folder = queue_func(
        seed=scene_dict['seed'],
        folder=scene_folder,
        name=stage_scene_name,
        taskname=taskname
    )
    scene_dict[f'{taskname}_job_obj'] = job_obj
    scene_dict[f'{taskname}_output_folder'] = output_folder
    scene_dict[f'{taskname}_submitted'] = 1  # marked as submitted
    update_symlink(scene_folder, [(taskname, job_obj)])


def infer_crash_reason(stdout_file, stderr_file: Path):

    if not stderr_file.exists():
        return f'{stderr_file} not found'
    
    try:
        error_log = stderr_file.read_text()
    except UnicodeDecodeError:
        return f"failed to parse log file {stderr_file}"

    if "System is out of GPU memory" in error_log:
        return "Out of GPU memory"
    elif "this scene is timed-out" in error_log or 'DUE TO TIME LIMIT' in error_log:
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

    ignore_errors = [
        'Error: Not freed memory blocks',
    ]

    matches = [m for m in matches if not any(w in m for w in ignore_errors)]

    if len(matches):
        return ','.join(matches)
    else:
        return f"Could not summarize cause, check {stderr_file}" 

def record_crashed_seed(scene, taskname, f, fatal=True):

    seed = scene['seed']

    stdout_file = args.output_folder / seed / "logs" / f"{taskname}.out"
    stderr_file = args.output_folder / seed / "logs" / f"{taskname}.err"

    scene_id, *_ = stderr_file.resolve().stem.split('_')
    node = node_from_slurm_jobid(scene_id)
    time_str = datetime.now().strftime("%m/%d %I:%M%p")
        
    reason = infer_crash_reason(stdout_file, stderr_file)
    text = f"{time_str} {str(stderr_file)} {reason=} {node=} {fatal=}\n"
    print('Crashed: ' + text)

    scene[f'{taskname}_crash_recorded'] = True

    return reason
    
def write_html_summary(all_scenes, output_folder, max_size=5000):

    names = [
        "index" if (idx == 0) else f"index_{idx}"
        for idx in range(0, len(all_scenes), max_size)
    ]
    for name, idx in zip(names, range(0, len(all_scenes), max_size)):
        html_path = output_folder / f"{name}.html"
        if not html_path.exists():
            make_html_page(html_path, all_scenes[idx:idx+max_size], frame=100,
            camera_pair_id=0, samples=[f"resmpl{i}" for i in range(5)], pages=names,
        )
            
def monitor_existing_jobs(all_scenes, aggressive_cancel_on_crash=False):

    state_counts = defaultdict(int)

    for scene in all_scenes:

        seed = scene['seed']
        scene['num_running'], scene['num_done'] = 0, 0
        any_fatal = False

        for state, taskname, _, fatal in iterate_scene_tasks(scene, args, monitor_all=True):
            
            if state == JobState.NotQueued:
                continue

            taskname_stem = taskname.split('_')[0]
            state_counts[(state, taskname_stem)] += 1
            scene['num_done'] += state in CONCLUDED_JOBSTATES
            scene['num_running'] += state not in CONCLUDED_JOBSTATES
            
            if state == JobState.Failed:
                if not scene.get(f'{taskname}_crash_recorded', False):
                    logging.info(f'{seed} - recording crash for {taskname}')
                    with (args.output_folder / "crash_summaries.txt").open('a') as f:
                        record_crashed_seed(scene, taskname, f, fatal=fatal)
                if fatal:
                    any_fatal = True

        if any_fatal:
            logging.info(f'{seed} - recording fatally crashed')
            scene['any_fatal_crash'] = True

        if aggressive_cancel_on_crash and any_fatal:
            suffix = 'job_obj'
            to_cancel = [k for k in scene.keys() if k.endswith(suffix)]
            for k in to_cancel:
                cancel_key = k.replace(suffix, 'force_cancelled')
                if scene.get(cancel_key, False):
                    continue
                logging.info(f'{seed} - cancelling {k} due to fatal crash')
                scene[cancel_key] = True
                cancel_job(scene[k])
                
        if (
            any_fatal and
            scene['num_running'] == 0 and
            scene['all_done'] == SceneState.NotDone
        ):
            logging.info(f'{seed} - processing scene termination due to fatal crash')
            on_scene_termination(args, scene, crashed=True)


    return state_counts

def stats_summary(state_counts):

    uniq_states = set(s for (s, _) in state_counts.keys())
    def get_count(state): 
        return sum(v for (s, _), v in state_counts.items() if s == state)
    totals = {s: get_count(s) for s in uniq_states}

    stats = {f'{s}/{t}': v for (s, t), v in state_counts.items()}        

    return stats, totals

@gin.configurable
def jobs_to_launch_next(
    scenes: list[dict],
    state_counts: dict[tuple[str, str], int],
    greedy=True, 

    # following kwargs are designed to help minimize over-eager starting new scenes, 
    # or limit paralellism to help greedily finish scenes / lower overall latency.
    # warning: may reduce throughput, especially if not using warmup_sec, or cluster capacity varies
    max_queued_task: int = None,
    max_queued_total: int = None,
    max_stuck_at_task: int = None
):
    
    def is_candidate_for_launch(scene):
        return (
            scene['all_done'] == SceneState.NotDone and
            not scene.get('any_fatal_crash', False)
        )
    scenes = [s for s in scenes if is_candidate_for_launch(s)]

    def inflight(s):
        return s['num_running'] + s['num_done']
    if greedy:
        scenes = sorted(copy(scenes), key=inflight, reverse=True)

    started_counts = np.array([inflight(s) for s in scenes])
    started_uniq, curr_per_started = np.unique(started_counts, return_counts=True)
    started_uniq = list(started_uniq)

    logging.debug(f'Pipeline state: {list(zip(started_uniq, curr_per_started))}')

    total_queued = sum(
        v for (s, _), v in state_counts.items() 
        if s == JobState.Queued
    )
        
    for scene in scenes:

        seed = scene['seed']
        
        started_if_launch = inflight(scene) + 1
        stuck_at_next = (
            curr_per_started[started_uniq.index(started_if_launch)] 
            if started_if_launch in started_uniq else 0
        )

        if (
            max_stuck_at_task is not None and 
            stuck_at_next >= max_stuck_at_task
        ):
            logging.info(
                f"{seed} - Not launching due to {stuck_at_next=} >"
                f" {max_stuck_at_task} for {started_if_launch=}"
            )
            continue

        for rec in iterate_scene_tasks(scene, args, monitor_all=False):
            state, taskname, queue_func, _ = rec

            if state != JobState.NotQueued:
                continue

            queued_key = (JobState.Queued, taskname.split('_')[0])
            queued = state_counts.get(queued_key, 0)
            if max_queued_task is not None and queued >= max_queued_task:
                logging.info(f"{seed} - Not launching due to {queued=} > {max_queued_task} for {taskname}")
                continue
            if max_queued_total is not None and total_queued >= max_queued_total:
                logging.info(f"{seed} - Not launching due to {total_queued=} > {max_queued_total} for {taskname}")
                continue

            yield scene, taskname, queue_func

            state_counts[queued_key] += 1
            total_queued += 1

def compute_control_state(args, totals, elapsed, num_concurrent):

    if num_concurrent == f'ENVVAR_{NUM_CONCURRENT_ENVVAR}':
        num_concurrent = int(os.environ[NUM_CONCURRENT_ENVVAR])

    control_state = {}
    control_state['n_in_flight'] = totals.get(JobState.Running, 0) + totals.get(JobState.Queued, 0)
    control_state['disk_usage'] = get_disk_usage(args.output_folder)

    warmup_pct = min(elapsed / args.warmup_sec, 1) if args.warmup_sec > 0 else 1
    control_state['curr_concurrent_max'] = math.ceil(warmup_pct * num_concurrent)

    if control_state['n_in_flight'] > control_state['curr_concurrent_max']:
        raise ValueError(
            f"manage_datagen_jobs observed {control_state['n_in_flight']=},"
            f" which exceeds allowed {control_state['curr_concurrent_max']=}"
        )
    control_state['try_to_launch'] = max(control_state['curr_concurrent_max'] - control_state['n_in_flight'], 0)

    return control_state

def record_states(stats, totals, control_state):
    
    pretty_stats = copy(stats)
    pretty_stats.update({f'control_state/{k}': v for k, v in control_state.items()})
    pretty_stats.update({f'{k}/total': v for k, v in totals.items()})

    if wandb is not None:
        wandb.log(pretty_stats)
    print("=" * 60)
    for k, v in sorted(pretty_stats.items()):
        print(f"{k.ljust(30)} : {v}")
    print("-" * 60)

@gin.configurable
def manage_datagen_jobs(all_scenes, elapsed, num_concurrent, disk_sleep_threshold=0.95):

    if LocalScheduleHandler._inst is not None:
        sys.path = ORIG_SYS_PATH #hacky workaround because bpy module breaks with multiprocessing
        LocalScheduleHandler.instance().poll()
        sys.path = BPY_SYS_PATH

    state_counts = monitor_existing_jobs(all_scenes)
    stats, totals = stats_summary(state_counts)
    control_state = compute_control_state(args, totals, elapsed, num_concurrent)

    new_jobs = jobs_to_launch_next(all_scenes, state_counts)
    new_jobs = list(itertools.islice(new_jobs, control_state['try_to_launch']))
    control_state['will_launch'] = len(new_jobs) # may be less due to jobs_to_launch optional kwargs, or running out of num_jobs

    pd.DataFrame.from_records(all_scenes).to_csv(args.output_folder/'scenes_db.csv')
    record_states(stats, totals, control_state)

    # Dont launch new scenes if disk is getting full
    if control_state['disk_usage'] > disk_sleep_threshold:
        message = f"{args.output_folder} is full ({100*control_state['disk_usage']}%). Sleeping."
        print(message)
        if wandb is not None:
            wandb.alert(title=f'{args.output_folder} full', text=message, wait_duration=3*60*60)
        time.sleep(60)
        return

    for scene, taskname, queue_func in new_jobs:    
        logger.info(f"{scene['seed']} - running {taskname}")
        run_task(queue_func, args.output_folder / str(scene['seed']), scene, taskname)

@gin.configurable
def main(args, shuffle=True, wandb_project='render', upload_commandfile_method=None):

    command_path = args.output_folder/'datagen_command.sh'
    with command_path.open('w') as f:
        f.write(' '.join(sys.argv))
    if upload_commandfile_method is not None:
        upload = upload_util.get_upload_func(upload_commandfile_method)
        upload(command_path, upload_util.get_upload_destfolder(args.output_folder))

    all_scenes = init_db(args)
    scene_name = args.output_folder.parts[-1]

    if args.cleanup != all:
        write_html_summary(all_scenes, args.output_folder)

    if args.wandb_mode != 'disabled':
        global wandb
        wandb = importlib.import_module('wandb')

    if wandb is not None:
        wandb.init(
            name=scene_name, 
            config=vars(args), 
            project=wandb_project, 
            mode=args.wandb_mode
        )

    logging.basicConfig(
        filename=str(args.output_folder / "jobs.log"),
        level=args.loglevel,
        format='[%(asctime)s]: %(message)s',
    )

    print(f'Using {get_slurm_banned_nodes()=}')

    if shuffle:
        np.random.shuffle(all_scenes)
    else:
        all_scenes = sorted(all_scenes, key=lambda j: j['seed'])

    start_time = datetime.now()
    while any(j['all_done'] == SceneState.NotDone for j in all_scenes):
        now = datetime.now()
        print(f'{args.output_folder} {start_time.strftime("%m/%d %I:%M%p")} -> {now.strftime("%m/%d %I:%M%p")}')
        manage_datagen_jobs(all_scenes, elapsed=(now-start_time).total_seconds())
        time.sleep(2)

if __name__ == "__main__":
    
    os.umask(0o007)

    slurm_available = (which("sbatch") is not None)
    parser = argparse.ArgumentParser() # to guarantee that the render scenes finish, try render_image.time_limit=2000
    parser.add_argument(
        '-o', 
        '--output_folder', 
        type=Path, 
        required=True
    )
    parser.add_argument(
        '--num_scenes', 
        type=int, 
        default=1,
        help="Number of scenes to attempt before terminating"
    )
    parser.add_argument(
        '--meta_seed', 
        type=int, 
        default=None,
        help="What seed should be used to determine the random seeds of each scene? "
        "Leave as None unless deliberately replicating past runs"
    )
    parser.add_argument(
        '--specific_seed', 
        default=None, 
        nargs='+', 
        help="The default, None, will choose a random seed per scene. Otherwise, all "
        "scenes will have the specified seed. Interpreted as an integer if possible."
    )
    parser.add_argument(
        '--use_existing', 
        action='store_true',
        help="If set, then assume output_folder is an existing folder from a "
        "terminated run, and make a best-possible-effort to resume from where "
        "it left off"
    )
    parser.add_argument(
        '--warmup_sec', 
        type=float, 
        default=0,
        help="Perform a staggered start over the specified period, so that jobs dont "
        "sync up or all write to disk at similar times."
    )
    parser.add_argument(
        '--cleanup', 
        type=str, 
        choices=['all', 'big_files', 'none', 'except_logs', 'except_crashed'], 
        default='none',
        help="What files should be cleaned up by the manager as it runs?"
    )
    parser.add_argument(
        '--configs', 
        nargs='*', 
        default=[],
        help="List of gin config names to pass through to all underlying "
        "scene generation jobs."
    )
    parser.add_argument(
        '-p', 
        '--overrides', 
        nargs='+', 
        type=str, 
        default=[], 
        help="List of gin overrides to pass through to all underlying "
        "scene generation jobs"
    )
    parser.add_argument(
        '--wandb_mode', 
        type=str, 
        default='disabled', 
        choices=['online', 'offline', 'disabled'],
        help="Mode kwarg for wandb.init(). Set up wandb before use."
    )
    parser.add_argument(
        '--pipeline_configs', 
        type=str, 
        nargs='+',
        help="List of gin config names from tools/pipeline_configs "
        "to configure this execution"
    )
    parser.add_argument(
        '--pipeline_overrides', 
        nargs='+', 
        type=str, 
        default=[], 
        help="List of gin overrides to configure this execution",
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('-d', '--debug', action="store_const", 
                        dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument( '-v', '--verbose', action="store_const", 
                        dest="loglevel", const=logging.INFO)
    args = parser.parse_args()

    using_upload = any('upload' in x for x in args.pipeline_configs)

    if not using_upload and args.cleanup in ['except_logs', 'except_crashed', 'all']:
        raise ValueError(
            f'Pipeline is configured with {args.cleanup=}'
            ' yet {args.upload=}! No output would be preserved!'
        )
    if using_upload and args.cleanup == 'none':
        logging.warning(
            f'Upload performs some cleanup, so combining upload.gin with '
            '--cleanup none will not result in ALL files being preserved'
        )
    
    assert args.specific_seed is None or args.num_scenes == 1

    overwrite_ok = args.use_existing or args.overwrite
    if args.output_folder.exists() and not overwrite_ok:
        raise FileExistsError(
            f'--output_folder {args.output_folder} already exists! Please delete it,'
            ' specify a different --output_folder, or use --overwrite'
        )
    args.output_folder.mkdir(parents=True, exist_ok=overwrite_ok)

    if args.meta_seed is not None:
        random.seed(args.meta_seed)
        np.random.seed(args.meta_seed)

    infinigen.core.init.apply_gin_configs(
        configs_folder=Path('infinigen/datagen/configs'),
        configs=args.pipeline_configs,
        overrides=args.pipeline_overrides,
        mandatory_folders=[
            'infinigen/datagen/configs/compute_platform',
            'infinigen/datagen/configs/data_schema'
        ]
    )

    main(args)
