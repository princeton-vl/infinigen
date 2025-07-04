# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import logging
import os
from ctypes import c_int32
from pathlib import Path

import bpy
import gin
import numpy as np
from mathutils.bvhtree import BVHTree
from numpy import ascontiguousarray as AC

import infinigen
from infinigen.assets.composition import material_assignments
from infinigen.assets.materials import (
    fluid as fluid_materials,
)
from infinigen.assets.materials import (
    snow as snow_materials,
)
from infinigen.core.tagging import tag_object, tag_system
from infinigen.core.util.blender import SelectObjects, delete
from infinigen.core.util.logging import Timer
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.core.util.organization import (
    Assets,
    Attributes,
    ElementNames,
    ElementTag,
    Materials,
    SelectionCriterions,
    SurfaceTypes,
    Tags,
    TerrainNames,
    Transparency,
)
from infinigen.core.util.random import weighted_sample
from infinigen.core.util.test_utils import import_item
from infinigen.OcMesher.ocmesher import OcMesher as UntexturedOcMesher
from infinigen.OcMesher.ocmesher import __version__ as ocmesher_version
from infinigen.terrain.assets.ocean import ocean_asset
from infinigen.terrain.mesher import (
    OpaqueSphericalMesher,
    TransparentSphericalMesher,
    UniformMesher,
)
from infinigen.terrain.scene import scene, transfer_scene_info
from infinigen.terrain.surface_kernel.core import SurfaceKernel
from infinigen.terrain.utils import (
    AttributeType,
    FieldsType,
    Mesh,
    Vars,
    get_caminfo,
    load_cdll,
    move_modifier,
    write_attributes,
)

ocmesher_version_expected = "2.0"
if ocmesher_version != ocmesher_version_expected:
    raise ValueError(
        f"User has installed {ocmesher_version=} which is not for {infinigen.__version__=}, we expected {ocmesher_version_expected=}, you may need to re-run installation / recompile the codebase"
    )

logger = logging.getLogger(__name__)

fine_suffix = "_fine"
hidden_in_viewport = [ElementNames.Atmosphere]
ASSET_ENV_VAR = "INFINIGEN_ASSET_FOLDER"


@gin.configurable
def get_surface_type(surface, degrade_sdf_to_displacement=True):
    if not degrade_sdf_to_displacement:
        return surface.type
    else:
        if surface.type == SurfaceTypes.SDFPerturb:
            return SurfaceTypes.Displacement
        return surface.type


def process_surface_input(_input, default):
    if _input is None:
        return default
    if isinstance(_input, str):
        # e.g. 'ground' will return material_assignments.ground
        return getattr(material_assignments, _input)
    if isinstance(_input, (list, tuple)):
        # e.g. [('soil.Soil', 1)] will return [(Soil, 1)]
        return [(import_item(k), float(v)) for k, v in _input]


class OcMesher(UntexturedOcMesher):
    def __init__(self, cameras, bounds, **kwargs):
        UntexturedOcMesher.__init__(self, get_caminfo(cameras)[0], bounds, **kwargs)

    def __call__(self, kernels):
        sdf_kernels = [(lambda x, k0=k: k0(x)[Vars.SDF]) for k in kernels]
        meshes, in_view_tags = UntexturedOcMesher.__call__(self, sdf_kernels)
        with Timer("compute attributes"):
            write_attributes(kernels, None, meshes)
            for mesh, tag in zip(meshes, in_view_tags):
                mesh.vertex_attributes[Tags.OutOfView] = (~tag).astype(np.int32)
        with Timer("concat meshes"):
            mesh = Mesh.cat(meshes)
        return mesh


class CollectiveOcMesher(UntexturedOcMesher):
    def __init__(self, cameras, bounds, **kwargs):
        UntexturedOcMesher.__init__(self, get_caminfo(cameras)[0], bounds, **kwargs)

    def __call__(self, kernels):
        sdf_kernels = [
            lambda x: np.stack([k(x)[Vars.SDF] for k in kernels], -1).min(axis=-1)
        ]
        mesh, in_view_tag = UntexturedOcMesher.__call__(self, sdf_kernels)
        mesh = mesh[0]
        with Timer("compute attributes"):
            write_attributes(kernels, mesh, [])
            mesh.vertex_attributes[Tags.OutOfView] = (~in_view_tag[0]).astype(np.int32)
        mesh = Mesh(mesh=mesh)
        return mesh


@gin.configurable
class Terrain:
    instance = None

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
                Path(__file__).parent.resolve()
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

        if Terrain.instance is not None:
            self.__dict__ = Terrain.instance.__dict__.copy()
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

            self.elements, scene_infos = scene(
                seed, Path(on_the_fly_asset_folder), asset_path, device
            )
            self.elements_list = list(self.elements.values())
            logger.info(
                f"Terrain elements: {[x.__class__.name for x in self.elements_list]}"
            )
            transfer_scene_info(self, scene_infos)
            Terrain.instance = self

        for e in self.elements:
            self.elements[e].height_offset = height_offset
            self.elements[e].whole_bbox = whole_bbox

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if hasattr(self, "elements"):
            for e in self.elements:
                self.elements[e].cleanup()

    def export(
        self,
        mesher_backend="SphericalMesher",
        cameras=None,
        main_terrain_only=False,
        remove_redundant_attrs=True,
    ):
        meshes_dict = {}
        attributes_dict = {}
        if not main_terrain_only or TerrainNames.OpaqueTerrain == self.main_terrain:
            opaque_elements = [
                element
                for element in self.elements_list
                if element.transparency == Transparency.Opaque
            ]
            if opaque_elements != []:
                attributes_dict[TerrainNames.OpaqueTerrain] = set()
                if mesher_backend == "SphericalMesher":
                    mesher = OpaqueSphericalMesher(cameras, self.bounds)
                elif mesher_backend == "OcMesher":
                    mesher = OcMesher(cameras, self.bounds)
                elif mesher_backend == "UniformMesher":
                    mesher = UniformMesher(self.populated_bounds)
                else:
                    raise ValueError("unrecognized mesher_backend")
                with Timer(f"meshing {TerrainNames.OpaqueTerrain}"):
                    mesh = mesher([element for element in opaque_elements])
                    meshes_dict[TerrainNames.OpaqueTerrain] = mesh
                for element in opaque_elements:
                    attributes_dict[TerrainNames.OpaqueTerrain].update(
                        element.attributes
                    )

        individual_transparent_elements = [
            element
            for element in self.elements_list
            if element.transparency == Transparency.IndividualTransparent
        ]
        for element in individual_transparent_elements:
            if not main_terrain_only or element.__class__.name == self.main_terrain:
                if mesher_backend in ["SphericalMesher", "OcMesher"]:
                    special_args = {}
                    if element.__class__.name == ElementNames.Atmosphere:
                        special_args["pixels_per_cube"] = 100
                        special_args["inv_scale"] = 1
                    if mesher_backend == "SphericalMesher":
                        mesher = TransparentSphericalMesher(
                            cameras, self.bounds, **special_args
                        )
                    else:
                        mesher = OcMesher(
                            cameras,
                            self.bounds,
                            simplify_occluded=False,
                            **special_args,
                        )
                elif mesher_backend == "UniformMesher":
                    mesher = UniformMesher(self.populated_bounds, enclosed=True)
                else:
                    raise ValueError("unrecognized mesher_backend")
                with Timer(f"meshing {element.__class__.name}"):
                    mesh = mesher([element])
                    meshes_dict[element.__class__.name] = mesh
                attributes_dict[element.__class__.name] = element.attributes

        if (
            not main_terrain_only
            or TerrainNames.CollectiveTransparentTerrain == self.main_terrain
        ):
            collective_transparent_elements = [
                element
                for element in self.elements_list
                if element.transparency == Transparency.CollectiveTransparent
            ]
            if collective_transparent_elements != []:
                attributes_dict[TerrainNames.CollectiveTransparentTerrain] = set()
                if mesher_backend == "SphericalMesher":
                    mesher = TransparentSphericalMesher(cameras, self.bounds)
                elif mesher_backend == "OcMesher":
                    mesher = CollectiveOcMesher(
                        cameras, self.bounds, simplify_occluded=False
                    )
                elif mesher_backend == "UniformMesher":
                    mesher = UniformMesher(self.populated_bounds)
                else:
                    raise ValueError("unrecognized mesher_backend")
                with Timer(f"meshing {TerrainNames.CollectiveTransparentTerrain}"):
                    mesh = mesher(
                        [element for element in collective_transparent_elements]
                    )
                    meshes_dict[TerrainNames.CollectiveTransparentTerrain] = mesh
                for element in collective_transparent_elements:
                    attributes_dict[TerrainNames.CollectiveTransparentTerrain].update(
                        element.attributes
                    )

        if main_terrain_only or cameras is not None:
            for mesh_name in meshes_dict:
                mesh_name_unapplied = mesh_name
                if mesh_name + "_unapplied" in bpy.data.objects.keys():
                    mesh_name_unapplied = mesh_name + "_unapplied"

                for attribute in sorted(attributes_dict[mesh_name]):
                    surface = self.surfaces[attribute]
                    if get_surface_type(surface) == SurfaceTypes.Displacement:
                        assert (
                            surface.mod_name
                            in bpy.data.objects[mesh_name_unapplied].modifiers
                        ), "please make sure you include one of the scene config in your configs and the same in all tasks"
                        surface_kernel = SurfaceKernel(
                            surface.name,
                            attribute,
                            bpy.data.objects[mesh_name_unapplied].modifiers[
                                surface.mod_name
                            ],
                            self.device,
                        )
                        surface_kernel(meshes_dict[mesh_name])

                meshes_dict[mesh_name].blender_displacements = []
                for attribute in sorted(attributes_dict[mesh_name]):
                    surface = self.surfaces[attribute]
                    if get_surface_type(surface) == SurfaceTypes.BlenderDisplacement:
                        meshes_dict[mesh_name].blender_displacements.append(
                            surface.mod_name
                        )

        if cameras is not None:
            if remove_redundant_attrs:
                for mesh_name in meshes_dict:
                    if len(attributes_dict[mesh_name]) == 1:
                        meshes_dict[mesh_name].vertex_attributes.pop(
                            list(attributes_dict[mesh_name])[0]
                        )
        else:
            self.bounding_box = (
                np.array(self.populated_bounds)[::2],
                np.array(self.populated_bounds)[1::2],
            )

        return meshes_dict, attributes_dict

    def sample_surface_templates(self):
        with FixedSeed(int_hash(["terrain surface", self.seed])):
            self.surfaces = {}
            for element in self.elements_list:
                for attribute in element.attributes:
                    if attribute not in self.surfaces:
                        surf = weighted_sample(self.surface_registry[attribute])
                        self.surfaces[attribute] = surf()
                        logger.info(f"{attribute=} will use material {surf.__name__}")

    def apply_surface_templates(self, attributes_dict):
        for mesh_name in attributes_dict:
            for attribute in sorted(attributes_dict[mesh_name]):
                with FixedSeed(
                    int_hash(
                        [
                            "terrain surface instantiate",
                            self.seed,
                            self.surfaces[attribute].__class__.__name__,
                        ]
                    )
                ):
                    if len(attributes_dict[mesh_name]) == 1:
                        self.surfaces[attribute].apply(
                            bpy.data.objects[mesh_name],
                            selection=None,
                            ocean_folder=self.on_the_fly_asset_folder / Assets.Ocean,
                        )
                    else:
                        self.surfaces[attribute].apply(
                            bpy.data.objects[mesh_name], selection=attribute
                        )

    def surfaces_into_sdf(self):
        for element in self.elements_list:
            if element.transparency == Transparency.Opaque:
                corresponding_mesh = TerrainNames.OpaqueTerrain
            elif element.transparency == Transparency.CollectiveTransparent:
                corresponding_mesh = TerrainNames.CollectiveTransparentTerrain
            else:
                corresponding_mesh = element.__class__.name
            mesh_name_unapplied = corresponding_mesh
            if corresponding_mesh + "_unapplied" in bpy.data.objects.keys():
                mesh_name_unapplied = corresponding_mesh + "_unapplied"
            corresponding_mesh = bpy.data.objects[mesh_name_unapplied]
            for attribute in element.attributes:
                surface = self.surfaces[attribute]
                if get_surface_type(surface) == SurfaceTypes.SDFPerturb:
                    assert (
                        surface.mod_name in corresponding_mesh.modifiers
                    ), f"{surface.mod_name} not in {corresponding_mesh.modifiers.keys()} please make sure you include one of the scene config in your configs and the same in all tasks"
                    element.displacement.append(
                        SurfaceKernel(
                            surface.name,
                            attribute,
                            corresponding_mesh.modifiers[surface.mod_name],
                            self.device,
                        )
                    )

    @gin.configurable
    def coarse_terrain(self):
        coarse_meshes, attributes_dict = self.export(mesher_backend="UniformMesher")
        terrain_objs = {}
        for name in coarse_meshes:
            obj = coarse_meshes[name].export_blender(name)
            if name != self.main_terrain:
                terrain_objs[name] = obj
            if name in hidden_in_viewport:
                obj.hide_viewport = True
        self.sample_surface_templates()
        self.apply_surface_templates(attributes_dict)
        self.surfaces_into_sdf()

        # do second time to avoid surface application difference resulting in floating rocks
        coarse_meshes, _ = self.export(
            main_terrain_only=True, mesher_backend="UniformMesher"
        )
        main_mesh = coarse_meshes[self.main_terrain]

        # WaterCovered annotation
        if ElementNames.Liquid in self.elements:
            main_mesh.vertex_attributes[Tags.LiquidCovered] = (
                self.elements[ElementNames.Liquid](main_mesh.vertices, sdf_only=1)[
                    Vars.SDF
                ]
                < 0
            ).astype(np.float32)
        main_unapplied = bpy.data.objects[self.main_terrain]
        main_unapplied.name = self.main_terrain + "_unapplied"
        main_unapplied.hide_render = True
        main_unapplied.hide_viewport = True
        terrain_objs[self.main_terrain] = main_obj = main_mesh.export_blender(
            self.main_terrain
        )
        mat = main_unapplied.data.materials[0]
        main_obj.data.materials.append(mat)

        self.terrain_objs = terrain_objs
        for name in self.terrain_objs:
            if name not in hidden_in_viewport:
                self.tag_terrain(self.terrain_objs[name])
        return main_obj

    @gin.configurable
    def fine_terrain(
        self,
        output_folder,
        cameras,
        optimize_terrain_diskusage=True,
        mesher_backend="SphericalMesher",
    ):
        # redo sampling to achieve attribute -> surface correspondance
        self.sample_surface_templates()
        if (self.on_the_fly_asset_folder / Assets.Ocean).exists():
            with FixedSeed(int_hash(["Ocean", self.seed])):
                ocean_asset(
                    output_folder / Assets.Ocean,
                    bpy.context.scene.frame_start,
                    bpy.context.scene.frame_end,
                    link_folder=self.on_the_fly_asset_folder / Assets.Ocean,
                )
        self.surfaces_into_sdf()
        fine_meshes, _ = self.export(mesher_backend=mesher_backend, cameras=cameras)
        for mesh_name in fine_meshes:
            obj = fine_meshes[mesh_name].export_blender(mesh_name + "_fine")
            if mesh_name not in hidden_in_viewport:
                self.tag_terrain(obj)
            if not optimize_terrain_diskusage:
                object_to_copy_from = bpy.data.objects[mesh_name]
                self.copy_materials_and_displacements(
                    mesh_name,
                    obj,
                    object_to_copy_from,
                    fine_meshes[mesh_name].blender_displacements,
                )
            else:
                Mesh(obj=obj).save(output_folder / f"{mesh_name}.glb")
                np.save(
                    output_folder / f"{mesh_name}.b_displacement",
                    fine_meshes[mesh_name].blender_displacements,
                )
                delete(obj)

    def copy_materials_and_displacements(
        self, mesh_name, object_to_copy_to, object_to_copy_from, displacements
    ):
        mat = object_to_copy_from.data.materials[0]
        object_to_copy_to.data.materials.append(mat)
        mesh_name_unapplied = mesh_name
        if mesh_name + "_unapplied" in bpy.data.objects.keys():
            mesh_name_unapplied = mesh_name + "_unapplied"
        for mod_name in displacements:
            move_modifier(
                object_to_copy_to,
                bpy.data.objects[mesh_name_unapplied].modifiers[mod_name],
            )
        object_to_copy_from.hide_render = True
        object_to_copy_from.hide_viewport = True
        if mesh_name in hidden_in_viewport:
            object_to_copy_to.hide_viewport = True

    def load_glb(self, output_folder):
        for mesh_name in os.listdir(output_folder):
            if not mesh_name.endswith(".glb"):
                continue
            mesh_name = mesh_name[:-4]
            object_to_copy_to = Mesh(
                path=output_folder / f"{mesh_name}.glb"
            ).export_blender(mesh_name + "_fine")
            object_to_copy_from = bpy.data.objects[mesh_name]
            displacements = np.load(output_folder / f"{mesh_name}.b_displacement.npy")
            self.copy_materials_and_displacements(
                mesh_name, object_to_copy_to, object_to_copy_from, displacements
            )

    def compute_camera_space_sdf(self, XYZ):
        sdf = np.ones(len(XYZ), dtype=np.float32) * 1e9
        for element in self.elements_list:
            if element.__class__.name == ElementNames.Atmosphere:
                continue
            element_sdf = element(XYZ, sdf_only=1)["sdf"]
            if self.under_water and element.__class__.name == ElementNames.Liquid:
                element_sdf *= -1
                element_sdf -= self.min_distance
            sdf = np.minimum(sdf, element_sdf)

        return sdf

    def get_bounding_box(self):
        min_gen, max_gen = self.bounding_box
        if self.under_water:
            max_gen[2] = min(max_gen[2], self.water_plane - self.min_distance)
        else:
            min_gen[2] = max(min_gen[2], self.water_plane + self.min_distance)
        return min_gen, max_gen

    @gin.configurable
    def build_terrain_bvh_and_attrs(
        self,
        terrain_tags_queries,
        avoid_border=False,
        looking_at_center_region_of_size=None,
    ):
        exclude_list = [ElementNames.Atmosphere, ElementNames.Clouds]
        terrain_objs = [t for t in self.terrain_objs if t not in exclude_list]

        for mesh in terrain_objs:
            with SelectObjects(bpy.data.objects[mesh]):
                bpy.ops.object.duplicate(linked=0, mode="TRANSLATION")
        for i, mesh in enumerate(terrain_objs):
            with SelectObjects(bpy.data.objects[f"{mesh}.001"]):
                for m in bpy.data.objects[f"{mesh}.001"].modifiers:
                    bpy.ops.object.modifier_apply(modifier=m.name)

        far_ocean = (
            self.under_water
            and self.surfaces[Materials.LiquidCollection].info["is_ocean"]
        )
        if far_ocean:
            obj = bpy.data.objects[f"{ElementNames.Liquid}.001"]
            obj.data.attributes.new(
                name="vertexwise_min_dist", type=AttributeType.Float, domain="POINT"
            )
            obj.data.attributes["vertexwise_min_dist"].data.foreach_set(
                FieldsType.Value,
                np.zeros(len(obj.data.vertices), dtype=np.float32) + 20,
            )

        with SelectObjects(bpy.data.objects[f"{terrain_objs[0]}.001"]):
            for i, mesh in enumerate(terrain_objs):
                if i != 0:
                    bpy.data.objects[f"{mesh}.001"].select_set(True)
            bpy.ops.object.join()
            terrain_obj = bpy.context.view_layer.objects.active

        terrain_mesh = Mesh(obj=terrain_obj)
        camera_selection_answers = {}
        for q0 in terrain_tags_queries:
            if type(q0) is not tuple:
                q = (q0,)
            else:
                q = q0
            if q[0] in [SelectionCriterions.CloseUp]:
                continue
            if q[0] == SelectionCriterions.Altitude:
                min_altitude, max_altitude = q[1:3]
                altitude = terrain_mesh.vertices[:, 2]
                camera_selection_answers[q0] = terrain_mesh.facewise_mean(
                    (altitude > min_altitude) & (altitude < max_altitude)
                )
            else:
                camera_selection_answers[q0] = np.zeros(
                    len(terrain_mesh.faces), dtype=bool
                )
                for key in self.tag_dict:
                    if set(q).issubset(set(key.split("."))):
                        camera_selection_answers[q0] |= (
                            terrain_mesh.face_attributes["MaskTag"]
                            == self.tag_dict[key]
                        ).reshape(-1)
                camera_selection_answers[q0] = camera_selection_answers[q0].astype(
                    np.float64
                )

        if np.abs(np.asarray(terrain_obj.matrix_world) - np.eye(4)).max() > 1e-4:
            raise ValueError(
                f"Not all transformations on {terrain_obj.name} have been applied. This function won't work correctly."
            )

        if "vertexwise_min_dist" not in terrain_mesh.vertex_attributes:
            terrain_mesh.vertex_attributes["vertexwise_min_dist"] = np.zeros(
                (len(terrain_mesh.vertices), 1), dtype=np.float32
            )

        if avoid_border:
            min_gen, max_gen = self.bounding_box
            dist_to_bbox = np.zeros((len(terrain_mesh.vertices), 1)) + 1e9
            for i in range(3):
                dist_to_bbox[:, 0] = np.minimum(
                    dist_to_bbox[:, 0],
                    terrain_mesh.vertices[:, i] - min_gen[i],
                    max_gen[i] - terrain_mesh.vertices[:, i],
                )
            dist_to_bbox = np.maximum(dist_to_bbox, 0)
            terrain_mesh.vertex_attributes["vertexwise_min_dist"] = np.maximum(
                terrain_mesh.vertex_attributes["vertexwise_min_dist"],
                30 / (dist_to_bbox + 1e-9),
            )
        if looking_at_center_region_of_size is not None:
            center_region_dist = np.zeros((len(terrain_mesh.vertices), 1))
            for i in range(2):
                center_region_dist[
                    terrain_mesh.vertices[:, i] > looking_at_center_region_of_size / 2,
                    0,
                ] = 1e9
                center_region_dist[
                    terrain_mesh.vertices[:, i] < -looking_at_center_region_of_size / 2,
                    0,
                ] = 1e9
            terrain_mesh.vertex_attributes["vertexwise_min_dist"] = np.maximum(
                terrain_mesh.vertex_attributes["vertexwise_min_dist"],
                center_region_dist,
            )

        vertexwise_min_dist = terrain_mesh.facewise_mean(
            terrain_mesh.vertex_attributes["vertexwise_min_dist"].reshape(-1)
        )

        depsgraph = bpy.context.evaluated_depsgraph_get()
        scene_bvh = BVHTree.FromObject(terrain_obj, depsgraph)
        delete(terrain_obj)

        return scene_bvh, camera_selection_answers, vertexwise_min_dist

    def tag_terrain(self, obj):
        if len(obj.data.vertices) == 0:
            return

        mesh = Mesh(obj=obj)
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
