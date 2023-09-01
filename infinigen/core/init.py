# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
import bpy

import argparse
import ast
import os
import random
import sys
import cProfile
import shutil
from pathlib import Path
import logging
from functools import partial
import pprint
from collections import defaultdict

# ruff: noqa: F402
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # This must be done BEFORE import cv2. 
# See https://github.com/opencv/opencv/issues/21326#issuecomment-1008517425

import gin
import numpy as np
from numpy.random import randint

from infinigen.core.util.math import int_hash
from infinigen.core.util.organization import Task
from infinigen.core.util.logging import Suppress, LogLevel

logger = logging.getLogger(__name__)

def parse_args_blender(parser):
    if '--' in sys.argv:
        # Running using a blender commandline python. 
        # args before '--' are intended for blender not infinigen
        argvs = sys.argv[sys.argv.index('--')+1:]
        return parser.parse_args(argvs)
    else:
        return parser.parse_args()
    

def parse_seed(seed, task=None):

    if seed is None:
        if task is not None and Task.Coarse not in task:
            raise ValueError(
                'Running tasks on an already generated scene, you need to specify --seed or results will'
                ' not be view-consistent')
        return randint(1e7), 'chosen at random'

    # WARNING: Do not add support for decimal numbers here, it will cause ambiguity, as some hex numbers are valid decimals

    try:
        return int(seed, 16), 'parsed as hexadecimal'
    except ValueError:
        pass

    return int_hash(seed), 'hashed string to integer'

def apply_scene_seed(seed, task=None):
    scene_seed, reason = parse_seed(seed, task)
    logger.info(f'Converted {seed=} to {scene_seed=}, {reason}')
    gin.constant('OVERALL_SEED', scene_seed)
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    return scene_seed

def sanitize_override(override: list):

    if (
        ('=' in override) and 
        not any((c in override) for c in "\"'[]")
    ):
        k, v = override.split('=')
        try:
            ast.literal_eval(v)
        except (ValueError, SyntaxError):
            if "@" not in v:
                override = f'{k}="{v}"'

    return override

def repo_root():
    return Path(__file__).parent.parent.parent

def contains_any_stem(filenames, folder):
    if not folder.exists():
        return False
    names = [p.stem for p in folder.iterdir()]
    return any(s.stem in names or s.name in names for s in map(Path, filenames))

def mandatory_config_dir_satisfied(mandatory_folder, root, configs):
    mandatory_folder = Path(mandatory_folder)
    mandatory_folder_rel = root/mandatory_folder

    if not (mandatory_folder.exists() or mandatory_folder_rel.exists()):
        raise FileNotFoundError(f'Could not find {mandatory_folder} or {mandatory_folder_rel}')
    
    return (
        contains_any_stem(configs, mandatory_folder) or
        contains_any_stem(configs, mandatory_folder_rel)
    )

@gin.configurable
def apply_gin_configs(
    configs_folder: Path,
    configs: list[str] = None,
    overrides: list[str] = None, 
    skip_unknown=False, 
    mandatory_folders: Path = None,
):
    
    if configs is None:
        configs = []
    if overrides is None:
        overrides = []
    if mandatory_folders is None:
        mandatory_folders = []
    configs_folder = Path(configs_folder)

    root = repo_root()

    configs_folder_rel = root/configs_folder
    if configs_folder_rel.exists():
        configs_folder = configs_folder_rel
        gin.add_config_file_search_path(configs_folder)
    elif configs_folder.exists():
        gin.add_config_file_search_path(configs_folder)
    else:
        raise FileNotFoundError(f'Couldnt find {configs_folder} or {configs_folder_rel}')

    for p in mandatory_folders:
        if not mandatory_config_dir_satisfied(p, root, configs):
            raise ValueError(
                f"Please load one or more config from {p} to avoid unexpected behavior."
            )
        
    search_paths = [configs_folder, root, Path('.')]

    def find_config(p):
        p = Path(p)
        try:
            return next(
                file
                for folder in search_paths
                for file in folder.glob('**/*.gin')
                if file.stem == p.stem
            )
        except StopIteration:
            raise FileNotFoundError(f'Could not find {p} or {p.stem} in any of {search_paths}')
            
    configs = [find_config(g) for g in ['base.gin'] + configs]
    overrides = [sanitize_override(o) for o in overrides]

    with LogLevel(logger=logging.getLogger(), level=logging.CRITICAL):
        gin.parse_config_files_and_bindings(
            configs, 
            bindings=overrides, 
            skip_unknown=skip_unknown
        )

def import_addons(names):
    for name in names:
        try:
            with Suppress():
                bpy.ops.preferences.addon_enable(module=name)
        except Exception:
            logger.warning(f'Could not load addon "{name}"')

def configure_blender():
    bpy.context.preferences.system.scrollback = 0 
    bpy.context.preferences.edit.undo_steps = 0
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    bpy.context.scene.cycles.volume_step_rate = 0.1
    bpy.context.scene.cycles.volume_preview_step_rate = 0.1
    bpy.context.scene.cycles.volume_max_steps = 32

    import_addons(['ant_landscape', 'real_snow'])

    