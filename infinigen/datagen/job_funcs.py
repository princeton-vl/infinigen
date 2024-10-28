# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alex Raistrick: refactor, local rendering, video rendering
# - Lahav Lipson: stereo version, local rendering
# - David Yan: export integration
# - Hei Law: initial version


import logging
import re
import sys
from pathlib import Path
from shutil import copytree
from uuid import uuid4

import gin

import infinigen
from infinigen.datagen.util.show_gpu_table import nodes_with_gpus
from infinigen.datagen.util.upload_util import get_commit_hash
from infinigen.tools.suffixes import get_suffix

from . import states

logger = logging.getLogger(__name__)

UPLOAD_UTIL_PATH = (
    infinigen.repo_root() / "infinigen" / "datagen" / "util" / "upload_util.py"
)
assert UPLOAD_UTIL_PATH.exists(), f"{UPLOAD_UTIL_PATH=} does not exist"

CUSTOMGT_PATH = Path(__file__).parent / "customgt" / "build" / "customgt"
if not CUSTOMGT_PATH.exists():
    logger.warning(
        f"{CUSTOMGT_PATH=} does not exist, if opengl_gt is enabled it will fail"
    )


@gin.configurable
def get_cmd(
    seed,
    task,
    configs,
    taskname,
    output_folder,
    driver_script="infinigen_examples.generate_nature",  # replace with a regular path to a .py, or another installed module
    input_folder=None,
    process_niceness=None,
):
    if isinstance(task, list):
        task = " ".join(task)

    cmd = ""
    if process_niceness is not None:
        cmd += f"nice -n {process_niceness} "
    cmd += f"{sys.executable} "

    if driver_script.endswith(".py"):
        cmd += driver_script + " "
    else:
        cmd += "-m " + driver_script + " "

    # No longer supported using pip bpy
    # if blender_thread_limit is not None:
    #    cmd += f'--threads {blender_thread_limit} '

    cmd += "-- "

    if input_folder is not None:
        cmd += "--input_folder " + str(input_folder) + " "
    if output_folder is not None:
        cmd += "--output_folder " + str(output_folder) + " "
    cmd += f"--seed {seed} --task {task} --task_uniqname {taskname} "
    if len(configs) != 0:
        cmd += f'-g {" ".join(configs)} '
    cmd += "-p"

    return cmd.split()


@gin.configurable
def queue_upload(
    folder,
    submit_cmd,
    name,
    taskname,
    dir_prefix_len=0,
    method="rclone",
    seed=None,
    **kwargs,
):
    modulepath = str(
        UPLOAD_UTIL_PATH.with_suffix("").relative_to(infinigen.repo_root())
    ).replace("/", ".")

    cmd = (
        f"{sys.executable} -m {modulepath} "
        "--parent_folder " + str(folder) + " "
        "--task_uniqname " + taskname + " "
        f"--dir_prefix_len {dir_prefix_len} "
        f"--method {method}"
    ).split()

    res = submit_cmd(cmd, folder, name, **kwargs)
    return res, None


@gin.configurable
def queue_export(
    folder,
    submit_cmd,
    name,
    seed,
    configs,
    taskname=None,
    exclude_gpus=[],
    overrides=[],
    input_indices=None,
    output_indices=None,
    **kwargs,
):
    input_suffix = get_suffix(input_indices)
    input_folder_priority_options = [
        f"fine{input_suffix}",
        "fine",
        f"coarse{input_suffix}",
        "coarse",
    ]

    for option in input_folder_priority_options:
        input_folder = f"{folder}/{option}"
        if (Path(input_folder) / "scene.blend").exists():
            break
    else:
        logger.warning(
            f"No scene.blend found in {input_folder} for any of {input_folder_priority_options}"
        )

    cmd = (
        get_cmd(
            seed,
            "export",
            configs,
            taskname,
            output_folder=f"{folder}/frames",
            input_folder=input_folder,
        )
        + f"""
        LOG_DIR='{folder / "logs"}'
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(cmd, folder=folder, name=name, gpus=0, **kwargs)
    return res, folder


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
    input_indices=None,
    output_indices=None,
    **kwargs,
):
    """
    Generating the coarse scene
    """

    get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f"{folder}/coarse{output_suffix}")

    cmd = (
        get_cmd(seed, "coarse", configs, taskname, output_folder=output_folder)
        + f"""
        LOG_DIR='{folder / "logs"}'
    """.split("\n")
        + overrides
    )

    commit = get_commit_hash()
    with (folder / "run_pipeline.sh").open("w") as f:
        f.write(f"# git checkout {commit}\n\n")
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")
    (folder / "run_pipeline.sh").chmod(0o774)

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        gpus=0,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
        **kwargs,
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
    input_prefix="fine",
    exclude_gpus=[],
    overrides=[],
    input_indices=None,
    output_indices=None,
    **kwargs,
):
    """
    Generating the fine scene
    """

    input_suffix = get_suffix(input_indices)
    get_suffix(output_indices)

    input_folder = folder / f"{input_prefix}{input_suffix}"
    output_folder = input_folder

    cmd = (
        get_cmd(
            seed,
            "populate",
            configs,
            taskname,
            input_folder=input_folder,
            output_folder=output_folder,
        )
        + f"""
        LOG_DIR='{folder / "logs"}'
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        gpus=0,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
        **kwargs,
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
    input_indices=None,
    output_indices=None,
    **kwargs,
):
    """
    Generating the fine scene
    """

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f"{folder}/fine{output_suffix}")

    enable_gpu_in_terrain = "Terrain.device='cuda'" if gpus > 0 else ""
    cmd = (
        get_cmd(
            seed,
            "fine_terrain",
            configs,
            taskname,
            input_folder=f"{folder}/coarse{input_suffix}",
            output_folder=output_folder,
        )
        + f"""
        LOG_DIR='{folder / "logs"}'
        {enable_gpu_in_terrain}
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        gpus=gpus,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
        **kwargs,
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
    exclude_gpus=[],
    gpus=0,
    overrides=[],
    include_coarse=True,
    input_indices=None,
    output_indices=None,
    **kwargs,
):
    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    tasks = "populate fine_terrain"

    if include_coarse:
        tasks = "coarse " + tasks

    output_folder = Path(f"{folder}/fine{output_suffix}")

    enable_gpu_in_terrain = "Terrain.device='cuda'" if gpus > 0 else ""
    cmd = (
        get_cmd(
            seed,
            tasks,
            configs,
            taskname,
            input_folder=f"{folder}/coarse{input_suffix}"
            if not include_coarse
            else None,
            output_folder=output_folder,
        )
        + f"""
        LOG_DIR='{folder / "logs"}'
        {enable_gpu_in_terrain}
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        gpus=gpus,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
        **kwargs,
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
    input_indices=None,
    output_indices=None,
    **submit_kwargs,
):
    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f"{folder}/frames{output_suffix}")

    input_folder_priority_options = [
        f"fine{input_suffix}",
        "fine",
        f"coarse{input_suffix}",
        "coarse",
    ]

    for option in input_folder_priority_options:
        input_folder = f"{folder}/{option}"
        if (Path(input_folder) / "scene.blend").exists():
            break
    else:
        logger.warning(
            f"No scene.blend found in {input_folder} for any of {input_folder_priority_options}"
        )

    cmd = (
        get_cmd(
            seed,
            "render",
            configs,
            taskname,
            input_folder=input_folder,
            output_folder=f"{output_folder}",
        )
        + f"""
        render.render_image_func=@{render_type}/render_image
        LOG_DIR='{folder / "logs"}'
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
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
    input_indices=None,
    output_indices=None,
    reuse_subcams=True,
    **submit_kwargs,
):
    if (output_indices["subcam"] > 0) and reuse_subcams:
        return states.JOB_OBJ_SUCCEEDED, None

    input_suffix = get_suffix(input_indices)
    output_suffix = get_suffix(output_indices)

    output_folder = Path(f"{folder}/savemesh{output_suffix}")

    output_folder.mkdir(parents=True, exist_ok=True)

    input_folder_priority_options = [
        f"fine{input_suffix}",
        "fine",
        f"coarse{input_suffix}",
        "coarse",
    ]

    for option in input_folder_priority_options:
        input_folder = f"{folder}/{option}"
        if (Path(input_folder) / "scene.blend").exists():
            break
    else:
        raise ValueError(
            f"No scene.blend found in {input_folder} for any of {input_folder_priority_options}"
        )

    cmd = (
        get_cmd(
            seed,
            "mesh_save",
            configs,
            taskname,
            input_folder=input_folder,
            output_folder=f"{folder}/savemesh{output_suffix}",
        )
        + f"""
        LOG_DIR='{folder / "logs"}'
    """.split("\n")
        + overrides
    )

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
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
    input_indices=None,
    output_indices=None,
    reuse_subcams=True,
    gt_testing=False,
    **submit_kwargs,
):
    if (output_indices["subcam"] > 0) and reuse_subcams:
        return states.JOB_OBJ_SUCCEEDED, None

    output_suffix = get_suffix(output_indices)

    input_folder = (
        Path(folder) / f"savemesh{output_suffix}"
    )  # OUTPUT SUFFIX IS CORRECT HERE. I know its weird. But input suffix really means 'prev tier of the pipeline
    # dst_output_indices = dict(output_indices)
    start_frame, end_frame = output_indices["frame"], output_indices["last_cam_frame"]

    key = "execute_tasks.point_trajectory_src_frame="
    point_trajectory_src_frame = None
    for item in overrides:
        if item.startswith(key):
            point_trajectory_src_frame = int(item[len(key) :])
    assert point_trajectory_src_frame is not None

    if gt_testing:
        copy_folder = Path(folder) / f"frames{output_suffix}"
        output_folder = Path(folder) / f"opengl_frames{output_suffix}"
        copytree(copy_folder, output_folder, dirs_exist_ok=True)
    else:
        output_folder = Path(folder) / f"frames{output_suffix}"
        output_folder.mkdir(exist_ok=True)

    assert isinstance(overrides, list) and ("\n" not in " ".join(overrides))

    tmp_script = Path(folder) / "tmp" / f"opengl_{uuid4().hex}.sh"
    tmp_script.parent.mkdir(exist_ok=True)
    with tmp_script.open("w") as f:
        lines = [
            "set -e",
            f"""
                if [ ! -d "{str(input_folder)}" ]; then
                exit 1
                fi
            """,
        ]
        lines.append(
            f"{sys.executable} {infinigen.repo_root()/'infinigen/tools/process_static_meshes.py'} {input_folder} {point_trajectory_src_frame}"
        )
        lines += [
            f"{CUSTOMGT_PATH} --input_dir {input_folder} --dst_input_dir {input_folder} "
            f"--frame {frame_idx} --dst_frame {frame_idx+1} --output_dir {output_folder} "
            for frame_idx in range(start_frame, end_frame + 1)
        ]
        # point trajectory
        lines += [
            f"{CUSTOMGT_PATH} --input_dir {input_folder} --dst_input_dir {input_folder} "
            f"--frame {point_trajectory_src_frame} --dst_frame {frame_idx} --flow_only 1 --flow_type 2 --output_dir {output_folder} "
            for frame_idx in range(start_frame, end_frame + 1)
        ]

        # depth of block end frame
        lines += [
            f"{CUSTOMGT_PATH} --input_dir {input_folder} --dst_input_dir {input_folder} "
            f"--frame {end_frame+1} --dst_frame {end_frame+1} --depth_only 1 --output_dir {output_folder} "
        ]
        # depth of point trajectory source frame
        lines += [
            f"{CUSTOMGT_PATH} --input_dir {input_folder} --dst_input_dir {input_folder} "
            f"--frame {point_trajectory_src_frame} --dst_frame {point_trajectory_src_frame} --depth_only 1 --output_dir {output_folder} "
        ]

        lines.append(
            f"{sys.executable} {infinigen.repo_root()/'infinigen/tools/compress_masks.py'} {output_folder}"
        )
        lines.append(
            f"{sys.executable} {infinigen.repo_root()/'infinigen/tools/compute_occlusion_masks.py'} {output_folder} {point_trajectory_src_frame}"
        )

        lines.append(
            f"{sys.executable} {infinigen.repo_root()/'infinigen/tools/compress_masks.py'} {output_folder}"
        )

        lines.append(
            f'{sys.executable} -c "from infinigen.tools.datarelease_toolkit import reorganize_old_framesfolder; '
            f'reorganize_old_framesfolder({repr(str(output_folder))})"'
        )
        lines.append(f"touch {folder}/logs/FINISH_{taskname}")

        for line in lines:
            line = re.sub("( \([A-Za-z0-9]+\))", "", line)
            f.write(line + "\n")

    cmd = f"bash {tmp_script}".split()

    with (folder / "run_pipeline.sh").open("a") as f:
        f.write(f"{' '.join(' '.join(cmd).split())}\n\n")

    res = submit_cmd(
        cmd,
        folder=folder,
        name=name,
        slurm_exclude=nodes_with_gpus(*exclude_gpus),
        **submit_kwargs,
    )
    return res, output_folder
