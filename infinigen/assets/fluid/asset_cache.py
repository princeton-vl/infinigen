# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import argparse
import os
import sys
from pathlib import Path
from mathutils import Vector
import importlib
from collections import defaultdict


import bpy
import gin
import numpy as np
import json

from infinigen.assets.fluid.fluid import (
    find_available_cache,
    set_obj_on_fire,
    fire_smoke_ground_truth,
)

import time
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
import logging

logger = logging.getLogger(__name__)

RAND_SEED_MAX = 1e5
ASSET_ENV_VAR = "ASSET_PATH"
SPECIES_MAX = 20
I_MAX = 20

@gin.configurable
class FireCachingSystem:
    def __init__(self, asset_folder = None, create=False, max_fire_assets = 3, max_per_kind = 1) -> None:
        if asset_folder == None:
            raise ValueError("asset_folder not set for Fire")
        
        cache_folder = os.path.join(asset_folder, "Fire")

        if not os.path.exists(cache_folder):
            if create:
                os.mkdir(cache_folder)
            else:
                raise ValueError(f"Could not find user-specified {cache_folder=}")

        self.cache_folder = cache_folder
        self.n_placed = defaultdict(int)
        self.max_fire_assets = max_fire_assets
        self.max_per_kind = max_per_kind
        logger.info(f"Fire cache folder is {self.cache_folder}")

    def get_cached_species(self, factory_class):
        factory_name = factory_class.__name__
        species = []
        factory_dir = os.path.join(self.cache_folder, factory_name)
        for sim_folder in os.listdir(factory_dir):
            config_file = os.path.join(factory_dir, sim_folder, "config.json")
            if not os.path.isfile(config_file):
                continue
            with open(config_file, "r") as f:
                config = json.load(f)
                s = config["species"]
                species.append(s)
        return species

    def create_cached_assets(self, factory_class, args):
        factory_name = factory_class.__name__
        factory = None
        for subdir in os.listdir("assets"):
            with gin.unlock_config():
                module = importlib.import_module(f'assets.{subdir.split(".")[0]}')
            if hasattr(module, factory_name):
                factory = getattr(module, factory_name)
                break
        if factory is None:
            raise ModuleNotFoundError(f"{factory_name} not Found.")

        butil.clear_scene(keep=["Camera"])
        factory_dir = os.path.join(self.cache_folder, factory_name)
        sim_num = find_available_cache(factory_dir)
        sim_folder = os.path.join(factory_dir, sim_num)
        Path(sim_folder).mkdir(parents=True, exist_ok=True)

        config = {"factory": factory_name, "cache_folder": self.cache_folder}
        species = np.random.randint(SPECIES_MAX)
        config["species"] = species
        f = factory(species)
        i = np.random.randint(I_MAX)
        config["i"] = i
        obj = f.spawn_asset(i)
        f.finalize_assets(obj)
        if factory_name in ["CachedRealisticTreeFactory"]:
            resolution = args.resolution if args.resolution else 300
            dom = set_obj_on_fire(
                obj,
                args.start_frame,
                resolution=resolution,
                simulation_duration=args.simulation_duration,
                noise_scale=2,
                add_turbulence=True,
                adaptive_domain=False,
                output_folder=sim_folder,
                estimate_domain=args.estimate_domain,
                dissolve_speed=args.dissolve_speed,
                dom_scale=args.dom_scale,
            )
        else:
            resolution = args.resolution if args.resolution else 200
            dom = set_obj_on_fire(
                obj,
                args.start_frame,
                resolution=resolution,
                simulation_duration=args.simulation_duration,
                noise_scale=3,
                add_turbulence=True,
                adaptive_domain=False,
                output_folder=sim_folder,
                estimate_domain=args.estimate_domain,
                dissolve_speed=args.dissolve_speed,
                dom_scale=args.dom_scale,
            )
        dom.name = f"sd_{sim_num}"
        config["obj_loc"] = (obj.location[0], obj.location[1], obj.location[2])
        config["dom_loc"] = (dom.location[0], dom.location[1], dom.location[2])
        config["obj_rot"] = (
            obj.rotation_euler[0],
            obj.rotation_euler[1],
            obj.rotation_euler[2],
        )
        config["dom_rot"] = (
            dom.rotation_euler[0],
            dom.rotation_euler[1],
            dom.rotation_euler[2],
        )

        with open(os.path.join(sim_folder, "config.json"), "w") as file:
            json.dump(config, file)
        bpy.ops.wm.save_mainfile(filepath=str(Path(sim_folder) / "simulation.blend"))

    def find_i_list(self, factory):
        factory_name = factory.__class__.__name__
        factory_dir = os.path.join(self.cache_folder, factory_name)
        i_list = []
        for sim_folder in os.listdir(factory_dir):
            full_sim_folder = os.path.join(factory_dir, sim_folder)
            config_file = os.path.join(factory_dir, sim_folder, "config.json")
            if (
                not os.path.isfile(os.path.join(full_sim_folder, "simulation.blend"))
            ) or (not os.path.isfile(config_file)):
                continue
            with open(config_file, "r") as f:
                config = json.load(f)
                s = config["species"]
                i = config["i"]
                if factory.factory_seed == s:
                    i_list.append((i, full_sim_folder, sim_folder))
        return i_list

    def read_config(self, full_sim_folder):
        config_file = os.path.join(full_sim_folder, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
            return config

    def link_fire(self, full_sim_folder, sim_folder, obj, factory):
        logger.info("importing fire")
        blendfile = os.path.join(full_sim_folder, "simulation.blend")
        section = "\\Object\\"
        object = f"sd_{sim_folder}"

        filepath = blendfile + section + object
        directory = blendfile + section
        filename = object

        old_set = set(bpy.data.objects[:])
        bpy.ops.wm.append(filepath=filepath, filename=filename, directory=directory)
        new_set = set(bpy.data.objects[:]) - old_set

        dom = None
        for new_obj in new_set:
            if new_obj["fire_system_type"] == "domain":
                dom = new_obj

        assert dom["fire_system_type"] == "domain"

        config = self.read_config(full_sim_folder)

        dom.location = (
            obj.parent.location + Vector(config["dom_loc"]) - Vector(config["obj_loc"])
        )
        dom.rotation_euler = obj.parent.rotation_euler

        ######should be used if no opengl gt########
        # gt_mesh, vol = fire_smoke_ground_truth(dom)

        # gt_mesh.hide_viewport = True
        # gt_mesh.hide_render = True
        # vol.hide_viewport = True
        # vol.hide_render = True

        self.n_placed[factory.__class__.__name__] += 1

        return dom


