# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma: original Infinigen terrain (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/terrain/core.py)
# - David Yan, Jia Deng: optimized implementation

import os
from ctypes import c_int32
from pathlib import Path

import numpy as np
from numpy import ascontiguousarray as AC

from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import fluid as fluid_materials
from infinigen.assets.materials import snow as snow_materials
from infinigen.core.tagging import tag_object, tag_system
from infinigen.core.util.logging import Timer
from infinigen.core.util.organization import (
    Attributes,
    ElementTag,
    Materials,
    Tags,
    TerrainNames,
)
from infinigen.perf import options
from infinigen.perf.core.util.prefetch import prefetch_files
from infinigen.terrain import core as _default
from infinigen.terrain.core import ASSET_ENV_VAR, logger, process_surface_input
from infinigen.terrain.scene import scene, transfer_scene_info
from infinigen.terrain.utils import Mesh, load_cdll


class Terrain(_default.Terrain):
    def __init__(
        self,
        seed,
        task,
        asset_folder,
        asset_version,
        on_the_fly_asset_folder="",
        device="cpu",
        main_terrain=TerrainNames.OpaqueTerrain,
        under_water=False,
        atmosphere=None,
        beach=None,
        eroded=None,
        ground_collection=None,
        lava=None,
        liquid_collection=None,
        mountain_collection=None,
        rock_collection=None,
        snow=None,
        min_distance=1,
        height_offset=0,
        whole_bbox=None,
        populated_bounds=(-75, 75, -75, 75, -25, 55),
        bounds=(-500, 500, -500, 500, -500, 500),
    ):
        dll = load_cdll(
            str(
                Path(__file__).parent.parent.parent.resolve()
                / "terrain"
                / "lib"
                / "cpu"
                / "elements"
                / "waterbody.so"
            )
        )
        func = dll.get_version
        func.argtypes = []
        func.restype = c_int32
        terrain_element_version = func()
        assert terrain_element_version == 1
        self.seed = seed
        self.device = device
        self.main_terrain = main_terrain
        self.under_water = under_water
        self.min_distance = min_distance
        self.populated_bounds = populated_bounds
        self.bounds = bounds

        self.surface_registry = {
            "atmosphere": process_surface_input(
                atmosphere, default=[(fluid_materials.AtmosphereLightHaze, 1)]
            ),
            "beach": process_surface_input(beach, default=material_assignments.beach),
            "eroded": process_surface_input(
                eroded, default=material_assignments.eroded
            ),
            "ground_collection": process_surface_input(
                ground_collection, default=material_assignments.ground
            ),
            "lava": process_surface_input(lava, default=[(fluid_materials.Lava, 1)]),
            "liquid_collection": process_surface_input(
                liquid_collection, default=material_assignments.liquid
            ),
            "mountain_collection": process_surface_input(
                mountain_collection, default=material_assignments.mountain
            ),
            "rock_collection": process_surface_input(
                rock_collection, default=material_assignments.rock
            ),
            "snow": process_surface_input(snow, default=[(snow_materials.Snow, 1)]),
        }

        if _default.Terrain.instance is not None:
            self.__dict__ = _default.Terrain.instance.__dict__.copy()
            return

        with Timer("Create terrain"):
            if asset_folder is None:
                if ASSET_ENV_VAR not in os.environ:
                    raise ValueError(
                        f"Terrain recieved {asset_folder=} yet {ASSET_ENV_VAR} was not set"
                    )
                asset_folder = os.environ[ASSET_ENV_VAR]

            if asset_folder != "":
                if not os.path.exists(asset_folder):
                    raise ValueError(
                        f"Could not find non-empty user-specified {asset_folder=}"
                    )
                asset_path = Path(asset_folder) / asset_version
                if not asset_path.exists():
                    raise ValueError(
                        f"{asset_folder=} did not contain {asset_version=}, please download it"
                    )
                logger.info(
                    f"Terrain using pre-generated {asset_path=} and on the fly {on_the_fly_asset_folder=}"
                )
            else:
                logger.info(f"Terrain using only on the fly {on_the_fly_asset_folder=}")
                asset_path = Path("")

            self.on_the_fly_asset_folder = Path(on_the_fly_asset_folder)
            self.reused_asset_folder = asset_path

            if options.prefetch_input_scene() and self.on_the_fly_asset_folder.is_dir():
                # the element loaders read the tile heightmaps and cave grids
                # with small requests; warm the page cache with large ones
                prefetch_files([self.on_the_fly_asset_folder])

            self.elements, scene_infos = scene(
                seed, Path(on_the_fly_asset_folder), asset_path, device
            )
            self.elements_list = list(self.elements.values())
            logger.info(
                f"Terrain elements: {[x.__class__.name for x in self.elements_list]}"
            )
            transfer_scene_info(self, scene_infos)
            _default.Terrain.instance = self

        for e in self.elements:
            self.elements[e].height_offset = height_offset
            self.elements[e].whole_bbox = whole_bbox

    def tag_terrain(self, obj):
        if len(obj.data.vertices) == 0:
            return

        # only the faces are used below (facewise reductions); skip reading
        # every point attribute of the object
        mesh = Mesh(obj=obj, mesh_only=True)
        first_time = 1
        # initialize with element tag
        element_tag = np.zeros(len(obj.data.vertices), dtype=np.int32)
        obj.data.attributes[Attributes.ElementTag].data.foreach_get(
            "value", element_tag
        )
        element_tag_f = mesh.facewise_intmax(element_tag)

        for i in range(ElementTag.total_cnt):
            mask_i = element_tag_f == i
            if mask_i.any():
                obj.data.attributes.new(
                    name=f"TAG_{ElementTag.map[i]}", type="FLOAT", domain="FACE"
                )
                obj.data.attributes[f"TAG_{ElementTag.map[i]}"].data.foreach_set(
                    "value", AC(mask_i.astype(np.float32))
                )
                if first_time:
                    # "landscape" is a collective name for terrain and water
                    tag_object(obj, Tags.Landscape)
                    first_time = 0
                else:
                    tag_object(obj)
        obj.data.attributes.remove(obj.data.attributes[Attributes.ElementTag])

        tag_thresholds = [
            (Tags.Cave, 0.5, 0),
            (Tags.LiquidCovered, 0.5, 1),
            (Materials.Eroded, 0.1, 0),
            (Materials.Lava, 0.1, 0),
            (Materials.Snow, 0.1, 0),
            (Tags.UpsidedownMountainsLowerPart, 0.5, 1),
            (Materials.Beach, 0.5, 0),
            (Tags.OutOfView, 0.5, 1),
        ]

        for tag_name, threshold, to_remove in tag_thresholds:
            if tag_name in obj.data.attributes.keys():
                tag = np.zeros(len(obj.data.vertices), dtype=np.float32)
                obj.data.attributes[tag_name].data.foreach_get("value", tag)
                tag_f = mesh.facewise_mean(tag)
                tag_f = tag_f > threshold
                if to_remove:
                    obj.data.attributes.remove(obj.data.attributes[tag_name])
                if tag_f.any():
                    obj.data.attributes.new(
                        name=f"TAG_{tag_name}", type="FLOAT", domain="FACE"
                    )
                    obj.data.attributes[f"TAG_{tag_name}"].data.foreach_set(
                        "value", AC(tag_f.astype(np.float32))
                    )
                    tag_object(obj)

        self.tag_dict = tag_system.tag_dict
