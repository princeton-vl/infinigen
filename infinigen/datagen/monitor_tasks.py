# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - Hei Law: initial version

import itertools
from functools import partial
import logging
from shutil import rmtree
import subprocess

import gin

from infinigen.datagen.util.cleanup import cleanup
from infinigen.datagen.util import upload_util
from infinigen.datagen.states import (
    JobState, 
    SceneState, 
    CONCLUDED_JOBSTATES, 
    get_scene_state, 
    get_suffix,
)

logger = logging.getLogger(__name__)

def iterate_sequential_tasks(
    task_list, 
    get_task_state, 
    overrides, 
    configs, 
    input_indices=None, 
    output_indices=None
):

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

def apply_cleanup_options(args, seed, crashed, scene_folder):

    if args.cleanup == 'all' or (args.cleanup == 'except_crashed' and not crashed):
        logger.info(f"{seed} - Removing entirety of {scene_folder}")
        rmtree(scene_folder)
    elif args.cleanup == 'big_files':
        logger.info(f"{seed} - Cleaning up any large files")
        cleanup(scene_folder, verbose=False)
    elif args.cleanup == 'except_logs':
        logger.info(f"{seed} - Cleaning up everything except logs")
        for f in scene_folder.iterdir():
            if f.name == 'logs':
                continue
            if f.is_dir():
                rmtree(f)
            else:
                f.unlink()
    elif args.cleanup == 'none':
        pass
    else:
        raise ValueError(f'Unrecognized {args.cleanup=} {crashed=}')

@gin.configurable
def on_scene_termination(
    args, 
    scene: dict, 
    crashed: bool, 
    enforce_upload_manifest=False,
    remove_write_permission=False # safeguard finished data against accidental meddling
):
    seed = scene['seed']

    if crashed:
        with (args.output_folder / "crashed_seeds.txt").open('a') as f:
            f.write(f"{seed}\n")
        scene['all_done'] = SceneState.Crashed    
    else:
        with (args.output_folder / "finished_seeds.txt").open('a') as f:
            f.write(f"{seed}\n")
        scene['all_done'] = SceneState.Done

    scene_folder = args.output_folder/seed
    apply_cleanup_options(args, seed, crashed, scene_folder)
            
    if scene_folder.exists() and (
        remove_write_permission is True or
        (remove_write_permission == 'except_crashed' and not crashed)
    ):
        subprocess.check_output(f"chmod -R a-w {scene_folder}".split())

    if enforce_upload_manifest:
        scene_folder = args.output_folder/scene['seed']
        upload_util.check_files_covered(scene_folder, upload_util.UPLOAD_MANIFEST)
        

def check_intermediate_cleanup(args, scene, idxs, stagetype_name, tasklist):

    raise NotImplementedError # todo fix

    idxs_str = '_'.join(idxs.values())
    key = f'{stagetype_name}_{idxs_str}_cleaned'
    if (
        args.cleanup != 'none' and
        not scene.get(key, False)
    ):
        for stage_rec in tasklist:
            taskname = stage_rec['name']
            path = scene[f'{taskname}_output_folder']
            print(f'Doing end-of-{stagetype_name} cleanup for {path} for {taskname}')
            if path is not None and path.exists():
                rmtree(path)
        scene[key] = True

@gin.configurable
def iterate_scene_tasks(
    scene_dict, 
    args,

    # if True, enumerate scenes that we might have launched earlier, 
    # even if we wouldnt launch them now (due to crashes etc)
    monitor_all, 

    # provided by gin
    global_tasks, 
    view_dependent_tasks, 
    camera_dependent_tasks, 

    frame_range, 
    cam_id_ranges, 
    num_resamples=1, 
    render_frame_range=None,
    finalize_tasks = [],
    view_block_size=1, # how many frames should share each `view_dependent_task`
    cam_block_size=None, # how many frames should share each `camera_dependent_task`
    #cleanup_viewdep=False, # TODO fix. Should cleanup the results of `view_dependent_tasks` once each view iter is done?
    viewdep_paralell=True, # can we work on multiple view depenendent tasks (usually `fine`) in paralell?
    camdep_paralell=True # can we work on multiple camera dependent tasks (usually render/gt) in paralell?
):
    
    '''
    This function is a generator which yields all scenes we might want to consider 
    monitoring or running for a particular scene

    It `yield`s the available scenes, regardless of whether they are already running etc
    '''

    for task in global_tasks + view_dependent_tasks + camera_dependent_tasks:
        if '_' in task['name']:
            raise ValueError(f'{task=} with {task["name"]=} is invalid, must not contain underscores')

    if cam_block_size is None:
        cam_block_size = view_block_size
    
    if cam_id_ranges[0] <= 0 or cam_id_ranges[1] <= 0:
        raise ValueError(
            f'{cam_id_ranges=} is invalid, both num. rigs and '
            'num subcams must be >= 1 or no work is done'
        )
    assert view_block_size >= 1
    assert cam_block_size >= 1
    if cam_block_size > view_block_size:
        cam_block_size = view_block_size
    seed = scene_dict['seed']

    scene_folder = args.output_folder/seed
    get_task_state = partial(get_scene_state, scene=scene_dict, scene_folder=scene_folder)

    global_overrides = [
        f'execute_tasks.frame_range={repr(list(frame_range))}', 
        'execute_tasks.camera_id=[0, 0]'
    ]
    global_configs = scene_dict.get('configs', []) + args.configs
    global_iter = iterate_sequential_tasks(
        global_tasks, 
        get_task_state,
        overrides=args.overrides+global_overrides, 
        configs=global_configs
    )

    for state, *rest in global_iter:
        yield state, *rest
    if not state == JobState.Succeeded:
        return

     # blender frame_range is inclusive, but python's range is end-exclusive 
    view_range = render_frame_range if render_frame_range is not None else frame_range
    view_frames = range(view_range[0], view_range[1] + 1, view_block_size)
    resamples = range(num_resamples)
    cam_rigs = range(cam_id_ranges[0])
    subcams = range(cam_id_ranges[1])

    running_views = 0
    for cam_rig, view_frame in itertools.product(cam_rigs, view_frames):

        view_frame_range = [view_frame, min(frame_range[1], view_frame + view_block_size - 1)] 
        view_overrides = [
            f'execute_tasks.frame_range=[{view_frame_range[0]},{view_frame_range[1]}]',
            f'execute_tasks.camera_id=[{cam_rig},{0}]'
        ]

        view_idxs = dict(cam_rig=cam_rig, frame=view_frame)
        view_tasks_iter = iterate_sequential_tasks(
            view_dependent_tasks, get_task_state,
            overrides=args.overrides+view_overrides, 
            configs=global_configs, output_indices=view_idxs
        )
        for state, *rest in view_tasks_iter:
            yield state, *rest
        if state not in CONCLUDED_JOBSTATES:
            if viewdep_paralell:
                running_views += 1
                continue
            else:
                return 
        elif state == JobState.Failed and not monitor_all:
            return

        running_blocks = 0
        for subcam, resample_idx in itertools.product(subcams, resamples):
            for cam_frame in range(
                view_frame_range[0], 
                view_frame_range[1] + 1, 
                cam_block_size
            ):
                
                cam_frame_range = [
                    cam_frame, 
                    min(view_frame_range[1], cam_frame + cam_block_size - 1)
                ] # blender frame_end is INCLUSIVE
                cam_overrides = [
                    f'execute_tasks.frame_range=[{cam_frame_range[0]},{cam_frame_range[1]}]',
                    f'execute_tasks.camera_id=[{cam_rig},{subcam}]',
                    f'execute_tasks.resample_idx={resample_idx}'
                ]

                camdep_indices = dict(
                    cam_rig=cam_rig, 
                    frame=cam_frame, 
                    subcam=subcam, 
                    resample=resample_idx,
                )
                # extra semi-redundant info needed for openglgt mostly
                extra_indices = dict(
                    view_first_frame=view_frame_range[0], 
                    last_view_frame=view_frame_range[1], 
                    last_cam_frame=cam_frame_range[1] 
                ) 
                camera_dep_iter = iterate_sequential_tasks(
                    camera_dependent_tasks, 
                    get_task_state,
                    overrides=args.overrides+cam_overrides, 
                    configs=global_configs,
                    input_indices=view_idxs if len(view_dependent_tasks) else None,
                    output_indices={**camdep_indices, **extra_indices}
                )

                for state, *rest in camera_dep_iter:
                    yield state, *rest
                if state not in CONCLUDED_JOBSTATES:
                    if camdep_paralell:
                        running_blocks += 1
                        continue
                    else:
                        return
                elif state == JobState.Failed and not monitor_all:
                    return

        if running_blocks > 0:
            running_views += 1
            continue

    if running_views > 0:
        return

    finalize_iter = iterate_sequential_tasks(
        finalize_tasks, 
        get_task_state,
        overrides=args.overrides+global_overrides, 
        configs=global_configs
    )
    for state, *rest in finalize_iter:
        yield state, *rest
    if not state == JobState.Succeeded:
        return
    
    if scene_dict['all_done'] == SceneState.NotDone:
        on_scene_termination(args, scene_dict, crashed=False)
