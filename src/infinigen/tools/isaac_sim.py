# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan, Beining Han

# Acknowledgement: This file draws inspiration from https://docs.omniverse.nvidia.com/isaacsim/latest/index.html

import json

import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.kit import SimulationApp
from omni.kit.commands import execute as omni_exec
from pxr import Sdf, Usd, UsdGeom, UsdLux

# ruff: noqa: E402
enable_extension("omni.isaac.examples")
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.physx.scripts import utils

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(launch_config=CONFIG)


class RobotController(BaseController):
    def __init__(self):
        super().__init__(name="robot_controller")

    def forward(self):
        return ArticulationAction(joint_velocities=[2, 2])


class InfinigenIsaacScene(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.world = World(
            stage_units_in_meters=1.0, backend="numpy", physics_dt=1 / 400.0
        )
        self.world._physics_context.set_gravity(-9.8)
        self.scene = self.world.scene
        self._support = None
        self.setup_scene()

    def setup_scene(self):
        self._add_infinigen_scene()
        self._add_lighting()
        self._add_robot()

    def _add_infinigen_scene(self):
        create_prim(
            prim_path="/World/Support",
            usd_path=self.cfg.scene_path,
            semantic_label="scene",
        )
        self._support = XFormPrim(prim_path="/World/Support", name="Support")

        stage = omni.usd.get_context().get_stage()

        prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
        if self.cfg.json_path is None:
            for prim in prims:
                utils.setStaticCollider(prim)
            self.scene.add(self._support)
            return

        with open(self.cfg.json_path) as json_file:
            relations = json.load(json_file)["objs"]

        obj_to_target = {}
        for key, value in relations.items():
            obj = value.get("obj")
            if obj:
                obj_to_target[
                    obj.replace("(", "_").replace(")", "_").replace(".", "_")
                ] = key

        for prim in prims:
            prim_name = prim.GetName()
            target = obj_to_target.get(prim_name)

            if "SPLIT" in prim_name:
                do_not_cast_shadows = prim.CreateAttribute(
                    "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
                )
                do_not_cast_shadows.Set(True)

            if "terrain" in prim_name:
                continue

            if not target:
                utils.setStaticCollider(prim)
                continue

            if any(
                x["relation"]["relation_type"] == "StableAgainst"
                and "Subpart(wall)" in x["relation"].get("parent_tags")
                or "Subpart(ceiling)" in x["relation"].get("parent_tags")
                for x in relations[target]["relations"]
            ):
                utils.setStaticCollider(prim)
            else:
                utils.setRigidBody(prim, "convexDecomposition", False)

        self.scene.add(self._support)

    def _add_lighting(self):
        omni_exec(
            "CreatePrim",
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            select_new_prim=False,
            attributes={
                UsdLux.Tokens.inputsIntensity: 5000,
                UsdLux.Tokens.inputsColor: (0.7, 0.88, 1.0),
            },
            create_default_xform=True,
        )
        omni_exec(
            "CreatePrim",
            prim_path="/World/DistantLight",
            prim_type="DistantLight",
            select_new_prim=False,
            attributes={UsdLux.Tokens.inputsIntensity: 8000},
            create_default_xform=True,
        )

    def _get_camera_loc(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/Support/CameraRigs_0_0")
        xform = UsdGeom.Xformable(prim)
        transform_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform_matrix.ExtractTranslation()
        translation[2] = 0
        return translation, [1, 0, 0, 0]

    def _add_robot(self):
        robot_path = get_assets_root_path() + "/Isaac/Robots/Jetbot/jetbot.usd"
        init_pos, _ = self._get_camera_loc()
        init_pos[-1] += 0.3
        self.robot = self.scene.add(
            WheeledRobot(
                prim_path="/World/Robot",
                name="Robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=robot_path,
                position=init_pos,
            )
        )
        self.robot.set_local_scale(np.array([4, 4, 4]))
        self.controller = RobotController()
        self.world.reset()

    def apply_action(self):
        self.robot.apply_action(self.controller.forward())

    def reset(self):
        self.world.reset()

    def run(self):
        self.world.reset()
        while simulation_app.is_running():
            self.apply_action()
            self.world.step(render=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=str)
    parser.add_argument("--json-path", type=str)
    args = parser.parse_args()

    scene = InfinigenIsaacScene(args)

    scene.reset()
    scene.run()
    simulation_app.close()
