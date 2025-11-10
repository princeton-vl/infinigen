# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bpy
import coacd
import mathutils
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

import infinigen.core.sim.exporters.utils as exputils
from infinigen.core.sim.exporters.base import (
    PathItem,
    RigidBody,
    SimBuilder,
)
from infinigen.core.sim.kinematic_node import JointType, KinematicNode
from infinigen.core.sim.physics import joint_dynamics as jointdyna
from infinigen.core.sim.physics import material_physics as mtlphysics
from infinigen.core.util import blender as butil
from infinigen.tools.export import bake_object, skipBake, triangulate_mesh


class UnsupportedAxisError(Exception):
    pass


class USDBuilder(SimBuilder):
    def __init__(self, assets_dir):
        super().__init__(assets_dir)

        self.stage = self._initialize_usd()

        self.asset_freq = defaultdict(int)
        self.joint_freq = defaultdict(int)

        self.post_processing_collision_info = defaultdict(dict)
        self.exclude_links = set()

        self.link_count = 0

    @property
    def usd(self):
        """Returns the USD file as a string."""
        return self.stage.GetRootLayer().ExportToString()

    def _initialize_usd(self):
        """Initialize the USD."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        default_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/Asset")).GetPrim()
        UsdGeom.Scope.Define(stage, Sdf.Path("/Asset/joints"))
        UsdGeom.Scope.Define(stage, Sdf.Path("/Asset/Looks"))
        stage.SetDefaultPrim(default_prim)
        return stage

    def build(
        self,
        blend_obj: bpy.types.Object,
        kinematic_root: KinematicNode,
        sample_joint_params_fn: Callable,
        metadata: Dict,
        visual_only: bool = False,
        image_res: int = 512,
        extra_exclude: Optional[set] = None,
    ):
        super().build(blend_obj, metadata)

        # construct a skeleton for the rigid body
        root, _ = self._construct_rigid_body_skeleton(kinematic_root)
        root = self._wrap_in_body(root)
        self._simplify_skeleton(root)

        body_info = self._add_assets(root, visual_only, image_res)
        self._add_joints(root, body_info, sample_joint_params_fn, extra_exclude)

    def _add_assets(self, root: RigidBody, visual_only: bool, image_res: int):
        """Populates the USD with its xforms and assets."""
        body_usd_info = dict()

        for body in root:
            # create an xform for the body
            link_name = f"link_{self.link_count}" if self.link_count > 0 else "world"
            link_path = f"/Asset/{link_name}"
            xform = self._add_xform(usd_path=link_path)
            self.link_count += 1

            # add all the assets
            vismesh_refs = []
            colmesh_refs = []
            assets = []
            link_path = f"/Asset/{link_name}"
            for asset in body.assets:
                vismesh, colmeshes, asset = self._add_mesh(
                    asset.attribs, link_path, visual_only, image_res
                )
                if asset is not None:
                    vismesh_refs.append(vismesh)
                    colmesh_refs.append(colmeshes)
                    assets.append(asset)

            # set the xform position to be the center of the aabb of its assets
            aabb_center = exputils.get_aabb_center(assets)
            xform.AddTranslateOp().Set(Gf.Vec3d(*aabb_center))
            body_usd_info[body] = {"path": link_path, "center": aabb_center}

            # offset each individual geom
            for vismesh, colmeshes, asset in zip(vismesh_refs, colmesh_refs, assets):
                geom_center = exputils.get_aabb_center(asset)
                offset = geom_center - aabb_center
                vismesh.AddTranslateOp().Set(Gf.Vec3d(*offset))
                for colmesh in colmeshes:
                    colmesh.AddTranslateOp().Set(Gf.Vec3d(*offset))

                    verts = np.array(colmesh.GetPointsAttr().Get())
                    faces = np.array(colmesh.GetFaceVertexIndicesAttr().Get()).reshape(
                        -1, 3
                    )

                    # add information for post processing
                    self.post_processing_collision_info[link_name][
                        colmesh.GetPrim().GetName()
                    ] = {
                        "offset": geom_center,
                        "mesh": trimesh.Trimesh(
                            vertices=verts, faces=faces, process=False
                        ),
                        "ref": colmesh,
                    }

        return body_usd_info

    def _add_joints(
        self,
        root: RigidBody,
        body_info: Dict,
        sample_joint_params_fn: Callable,
        extra_exclude: Optional[set] = None,
    ):
        """Populates the USD with its joints."""

        # sample the physics distribution for the joints
        joint_params = sample_joint_params_fn()

        # add a fixed joint to the root link
        joint_prim = self.stage.DefinePrim("/Asset/root_joint", "PhysicsFixedJoint")
        joint = UsdPhysics.FixedJoint(joint_prim)
        joint.CreateBody1Rel().SetTargets([body_info[root]["path"]])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*body_info[root]["center"]))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        # add the top link as the articulation root
        UsdPhysics.ArticulationRootAPI.Apply(joint_prim)

        all_link_paths = set()
        filtered_pairs = defaultdict(set)
        if extra_exclude is not None:
            for e in extra_exclude:
                body1_path, body2_path = f"/Asset/{e[0]}", f"/Asset/{e[1]}"
                filtered_pairs[body1_path].add(body2_path)
                filtered_pairs[body2_path].add(body1_path)

        for body in root:
            for child_body, kinematic_nodes in body.children.items():
                if len(kinematic_nodes) > 1:
                    raise NotImplementedError(
                        "Multi jointed bodies not supported yet in USD exporter."
                    )
                node = kinematic_nodes[0]
                joint_name = self.metadata[node.idn]["joint label"]
                unique_joint_idx = self.joint_freq[joint_name]
                unique_joint_name = f"{joint_name}_{self.joint_freq[joint_name]}"
                unique_joint_name = exputils.clean_name(unique_joint_name)
                self.joint_freq[joint_name] += 1
                if node.joint_type == JointType.HINGE:
                    joint_prim = self.stage.DefinePrim(
                        f"/Asset/joints/{unique_joint_name}", "PhysicsRevoluteJoint"
                    )
                    joint = UsdPhysics.RevoluteJoint(joint_prim)
                if node.joint_type == JointType.SLIDING:
                    joint_prim = self.stage.DefinePrim(
                        f"/Asset/joints/{unique_joint_name}", "PhysicsPrismaticJoint"
                    )
                    joint = UsdPhysics.PrismaticJoint(joint_prim)

                joint_properties = jointdyna.get_joint_properties(
                    joint_name, joint_params
                )
                drive_api = UsdPhysics.DriveAPI.Get(
                    joint_prim, "angular"
                ) or UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive_api.CreateStiffnessAttr().Set(joint_properties["stiffness"])
                drive_api.CreateDampingAttr().Set(joint_properties["damping"])

                parent_path = body_info[body]["path"]
                child_path = body_info[child_body]["path"]
                all_link_paths.add(parent_path)
                all_link_paths.add(child_path)
                filtered_pairs[parent_path].add(child_path)
                filtered_pairs[child_path].add(parent_path)

                joint.CreateBody0Rel().SetTargets([parent_path])
                joint.CreateBody1Rel().SetTargets([child_path])

                R = exputils.get_coord_frame(
                    self.blend_obj,
                    node.idn,
                    unique_joint_idx,
                    body_info[child_body]["center"],
                )

                prefix = node.idn
                poschild, axis, range_min, range_max = exputils.get_joint_properties(
                    self.blend_obj, prefix
                )

                position = R @ poschild
                axis = R @ axis

                axis = axis / np.linalg.norm(axis)
                mapping = {
                    (1, 0, 0): ("X", False),
                    (-1, 0, 0): ("X", True),
                    (0, 1, 0): ("Y", False),
                    (0, -1, 0): ("Y", True),
                    (0, 0, 1): ("Z", False),
                    (0, 0, -1): ("Z", True),
                }

                axis_dim = None
                for vec, (dim, f) in mapping.items():
                    if np.allclose(axis, vec, atol=1e-2):
                        axis_dim, flip = dim, f

                if axis_dim is None:
                    raise UnsupportedAxisError(
                        f"USD exporter only supports axes along the X, Y, Z dimensions. Current axis: {axis}"
                    )

                joint.CreateAxisAttr().Set(axis_dim)

                quat = Gf.Quatf(1, 0, 0, 0)
                if flip:
                    if axis_dim == "X":
                        quat = Gf.Quatf(0, 1, 0, 0)
                    elif axis_dim == "Y":
                        quat = Gf.Quatf(0, 0, 1, 0)
                    elif axis_dim == "Z":
                        quat = Gf.Quatf(0, 0, 0, 1)

                posparent = (
                    body_info[child_body]["center"] - body_info[body]["center"]
                ) + position

                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*posparent))
                joint.CreateLocalRot0Attr().Set(quat)
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*position))
                joint.CreateLocalRot1Attr().Set(quat)

                if flip:
                    range_min, range_max = -range_max, -range_min

                # convert from radians to degrees
                multiplier = 180 / np.pi if node.joint_type == JointType.HINGE else 1

                if not (range_min == range_max == 0):
                    joint.CreateLowerLimitAttr().Set(range_min * multiplier)
                    joint.CreateUpperLimitAttr().Set(range_max * multiplier)

        for link_path in all_link_paths:
            link_prim = self.stage.GetPrimAtPath(link_path)
            UsdPhysics.FilteredPairsAPI.Apply(link_prim)
            rel = link_prim.CreateRelationship("physics:filteredPairs")
            ignore_paths = filtered_pairs[link_path]  # links to ignore

            rel.SetTargets(ignore_paths)
            for ignore_path in ignore_paths:
                self.exclude_links.add(
                    (
                        link_prim.GetName(),
                        self.stage.GetPrimAtPath(ignore_path).GetName(),
                    )
                )

    def _add_xform(self, usd_path: str):
        """Creates an xform."""
        xform = UsdGeom.Xform.Define(self.stage, usd_path)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        return xform

    def _add_mesh(
        self, attribs: List[PathItem], usd_path: str, visual_only: bool, image_res: int
    ):
        """Adds an asset along with its materials to the USD and returns the asset."""
        mesh_vert, mesh_face, mesh_facenum, labels, asset = self._get_geometry_info(
            attribs
        )
        if exputils.is_2d(asset):
            # ignore geometries that are effectively 2d
            # usually represented as single points
            return None, None, None

        if len(labels) == 0:
            asset_name = "geom"
        else:
            asset_name = "_".join(list(labels))
        asset_name = exputils.clean_name(asset_name)

        unique_name = f"{asset_name}_{self.asset_freq[asset_name]}"
        vis_path = str(usd_path) + "/visual"
        UsdGeom.Scope.Define(self.stage, Sdf.Path(vis_path))
        vismesh_path = vis_path + f"/{unique_name}"
        self.asset_freq[asset_name] += 1

        mat_physics = mtlphysics.get_material_properties(asset)

        vismesh = UsdGeom.Mesh.Define(self.stage, vismesh_path)

        # set the geometry for the vismesh
        vismesh.GetPointsAttr().Set(mesh_vert)
        vismesh.GetFaceVertexCountsAttr().Set(mesh_facenum)
        vismesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mesh_face))
        vismesh.GetPrim().GetAttribute("subdivisionScheme").Set("bilinear")

        self._bake_materials(asset, unique_name, image_res)

        # adding the texture uv coordinates
        uv_layer = asset.data.uv_layers.active
        has_material = True
        if uv_layer is None:
            has_material = False

        if has_material:
            uvs = [loop.uv[:] for loop in uv_layer.data]
            indices = []
            for polygon in asset.data.polygons:
                indices.extend(polygon.loop_indices)

            texCoords = UsdGeom.PrimvarsAPI(vismesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
            )
            texCoords.Set(uvs)
            texCoords.SetIndices(indices)

            # create the material
            mtl_path = Sdf.Path(f"/Asset/Looks/{unique_name}_mat")
            material = UsdShade.Material.Define(self.stage, mtl_path)
            mtl_prim = self.stage.GetPrimAtPath(mtl_path)
            mat_phys = UsdPhysics.MaterialAPI.Apply(mtl_prim)
            mat_phys.CreateDensityAttr().Set(mat_physics["density"])
            mat_phys.CreateDynamicFrictionAttr().Set(mat_physics["friction"])

            stInput = material.CreateInput(
                "frame:stPrimvarName", Sdf.ValueTypeNames.Token
            )
            stInput.Set("st")

            # create the pbr shader
            pbrShader = UsdShade.Shader.Define(self.stage, str(mtl_path) + "/PBRShader")
            pbrShader.CreateIdAttr("UsdPreviewSurface")
            material.CreateSurfaceOutput().ConnectToSource(
                pbrShader.ConnectableAPI(), "surface"
            )

            # create texture coordinate reader
            stReader = UsdShade.Shader.Define(self.stage, str(mtl_path) + "/stReader")
            stReader.CreateIdAttr("UsdPrimvarReader_float2")
            stReader.CreateInput("varname", Sdf.ValueTypeNames.String).ConnectToSource(
                stInput
            )
            stReader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            diffuse_file = Path(self.assets_dir) / f"{unique_name}_DIFFUSE.png"
            normal_file = Path(self.assets_dir) / f"{unique_name}_NORMAL.png"
            roughness_file = Path(self.assets_dir) / f"{unique_name}_ROUGHNESS.png"
            metallic_file = Path(self.assets_dir) / f"{unique_name}_METAL.png"
            transmission_file = (
                Path(self.assets_dir) / f"{unique_name}_TRANSMISSION.png"
            )
            self._add_texture_map(
                "diffuse", diffuse_file, mtl_path, pbrShader, stReader
            )
            self._add_texture_map("normal", normal_file, mtl_path, pbrShader, stReader)
            self._add_texture_map(
                "roughness", roughness_file, mtl_path, pbrShader, stReader
            )
            self._add_texture_map(
                "metallic", metallic_file, mtl_path, pbrShader, stReader
            )
            self._add_texture_map(
                "transmission", transmission_file, mtl_path, pbrShader, stReader
            )

            # now bind the material to the mesh
            vismesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
            UsdShade.MaterialBindingAPI(vismesh).Bind(material)

        colmeshes = []
        if not visual_only:
            # run convex decomposition to generate collision meshes
            col_count = 0

            # duplicate and split the object into parts
            asset.hide_viewport = False
            asset.select_set(True)
            clone = butil.deep_clone_obj(asset)
            parts = butil.split_object(clone)

            for part in parts:
                # convert Blender mesh to trimesh
                translation = mathutils.Vector(-exputils.get_aabb_center(asset))
                vertices = np.array(
                    [list(v.co + translation) for v in part.data.vertices]
                )
                faces = np.array(
                    [list(f.vertices) for f in part.data.loop_triangles]
                ).reshape(-1, 3)
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                tri_mesh.merge_vertices(merge_norm=True, merge_tex=True)
                trimesh.repair.fix_inversion(tri_mesh)

                # determine preprocessing mode
                preprocess_mode = "off"
                if not tri_mesh.is_volume:
                    print(
                        tri_mesh.is_watertight,
                        tri_mesh.is_winding_consistent,
                        np.isfinite(tri_mesh.center_mass).all(),
                        tri_mesh.volume > 0.0,
                    )
                    preprocess_mode = "on"
                    if len(tri_mesh.vertices) < 4:
                        logging.warning(
                            f"Mesh is not a volume. Only {len(tri_mesh.vertices)} vertices."
                        )

                # run CoACD for convex decomposition
                coacd_mesh = coacd.Mesh(tri_mesh.vertices, tri_mesh.faces)
                subparts = coacd.run_coacd(
                    mesh=coacd_mesh,
                    threshold=0.05,
                    max_convex_hull=-1,
                    preprocess_mode=preprocess_mode,
                    mcts_max_depth=3,
                )

                # define collision scope and make them invisible
                col_path = str(usd_path) + "/collision"
                col_scope = UsdGeom.Scope.Define(self.stage, Sdf.Path(col_path))
                col_scope.GetVisibilityAttr().Set("invisible")

                # create USD collision meshes
                for vs, fs in subparts:
                    colmesh_path = f"{col_path}/{unique_name}_col{col_count}"
                    colmesh = UsdGeom.Mesh.Define(self.stage, colmesh_path)
                    col_count += 1

                    colmesh.GetPointsAttr().Set(vs.flatten())
                    colmesh.GetFaceVertexCountsAttr().Set([3] * len(fs))
                    colmesh.GetFaceVertexIndicesAttr().Set(
                        Vt.IntArray(fs.flatten().tolist())
                    )
                    col_prim = colmesh.GetPrim()
                    UsdPhysics.CollisionAPI.Apply(col_prim)
                    UsdPhysics.MeshCollisionAPI.Apply(col_prim)

                    # explicitly enable collisions and approximation
                    col_prim.CreateAttribute(
                        "physics:approximation", Sdf.ValueTypeNames.Token, custom=True
                    ).Set("convexHull")
                    col_prim.CreateAttribute(
                        "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, custom=True
                    ).Set(True)

                    if has_material:
                        col_prim.ApplyAPI(UsdShade.MaterialBindingAPI)
                        UsdShade.MaterialBindingAPI(vismesh).Bind(material)

                    colmeshes.append(colmesh)

        return vismesh, colmeshes, asset

    def _bake_materials(self, asset: bpy.types.Object, name: str, image_res: int):
        # set the materials
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = 1  # choose render sample
        # Set the tile size
        bpy.context.scene.cycles.tile_x = image_res
        bpy.context.scene.cycles.tile_y = image_res

        if not skipBake(asset):
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            asset.hide_render = False
            asset.hide_viewport = False
            bake_object(asset, self.assets_dir, image_res, True, name)
            asset.hide_render = True
            asset.hide_viewport = True

    def _add_texture_map(
        self,
        texture_type: str,
        file: Path,
        mtl_path: Sdf.Path,
        pbr_shader: UsdShade.Shader,
        st_reader: UsdShade.Shader,
    ):
        if not file.exists():
            return

        type_to_path = {
            "diffuse": "/diffuseTexture",
            "normal": "/normalTexture",
            "roughness": "/roughnessTexture",
            "metallic": "/metallicTexture",
            "transmission": "/transmissionTexture",
        }

        type_to_input = {
            "diffuse": "diffuseColor",
            "normal": "normal",
            "roughness": "roughness",
            "metallic": "metallic",
            "transmission": "transmission",
        }

        type_to_output = {
            "diffuse": "rgb",
            "normal": "rgb",
            "roughness": "r",
            "metallic": "r",
            "transmission": "r",
        }

        texture_sampler = UsdShade.Shader.Define(
            self.stage, str(mtl_path) + type_to_path[texture_type]
        )
        texture_sampler.CreateIdAttr("UsdUVTexture")

        # relative path
        parts = file.parts
        index = parts.index("assets")
        file = Path(*parts[index:])

        texture_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            str(Path(*parts[index:]))
        )
        texture_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )
        texture_sampler.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture_sampler.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture_sampler.CreateOutput(
            type_to_output[texture_type], Sdf.ValueTypeNames.Float
        )
        pbr_shader.CreateInput(
            type_to_input[texture_type], Sdf.ValueTypeNames.Float
        ).ConnectToSource(
            texture_sampler.ConnectableAPI(), type_to_output[texture_type]
        )

    def _get_geometry_info(self, attribs):
        """Returns geometry and metadata of mesh given attributes."""
        attribs = exputils.attribs_to_tuples(attribs)
        extra_attribs = [("axis_group", 0)]
        asset = exputils.get_geometry_given_attribs(
            self.blend_obj, attribs, extra_attribs=extra_attribs
        )
        triangulate_mesh(asset)
        labels = self._get_labels(asset)
        # translate bounding box center to world origin
        translation = mathutils.Vector(-exputils.get_aabb_center(asset))
        vertices = [list(vertex.co + translation) for vertex in asset.data.vertices]
        faces = [list(triangle.vertices) for triangle in asset.data.loop_triangles]
        faces = list(itertools.chain.from_iterable(faces))
        facenum = [3 for _ in range(len(faces))]
        return vertices, faces, facenum, labels, asset


def export(
    blend_obj: bpy.types.Object,
    sim_blueprint: Dict,
    seed: int,
    sample_joint_params_fn: Callable,
    export_dir: Path = Path("./sim_exports/usd"),
    image_res: int = 512,
    visual_only: bool = True,
    file_extension: str = "usda",
    get_raw_output: bool = False,
    extra_exclude: Optional[set] = None,
    **kwargs,
):
    """Export function for the USD file format."""
    # parse the provided blueprint and set the object export directory
    asset_name, kinematic_root, metadata = exputils.parse_sim_blueprint(sim_blueprint)

    # create export directories
    obj_export_dir = export_dir / asset_name / str(seed)
    obj_assets_dir = obj_export_dir / "assets"
    obj_export_dir.mkdir(parents=True, exist_ok=True)
    obj_assets_dir.mkdir(parents=True, exist_ok=True)

    # build asset
    builder = USDBuilder(obj_assets_dir)
    builder.build(
        blend_obj=blend_obj,
        kinematic_root=kinematic_root,
        sample_joint_params_fn=sample_joint_params_fn,
        metadata=metadata,
        visual_only=visual_only,
        image_res=image_res,
        extra_exclude=extra_exclude,
    )

    if get_raw_output:
        return builder.stage.GetRootLayer().ExportToString(), metadata

    links = builder.post_processing_collision_info.keys()
    if extra_exclude is not None:
        for e in extra_exclude:
            if e[0] in links and e[1] in links:
                builder.exclude_links.add(e)

    # post process collision geometries
    exputils.post_process_collisions(
        builder.post_processing_collision_info, obj_assets_dir, builder.exclude_links
    )

    # export the USD file
    export_file = f"{obj_export_dir}/{asset_name}.{file_extension}"
    builder.stage.Export(export_file)

    metadata.update(builder.get_bounding_box_info())

    metadata_path = obj_export_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return Path(export_file), metadata_path
