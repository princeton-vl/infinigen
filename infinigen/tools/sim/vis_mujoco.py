# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Max Gonzalez Saez-Diez: Primary author

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from infinigen.core.sim import sim_factory as sf


class MujocoAssetInitializer:
    def __init__(
        self,
        asset_name: str,
        seed: int = 42,
        output_dir: str = "/tmp",
        use_cached: bool = False,
        parent_alpha: float = 0.5,
        collision_mesh: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize MuJoCo asset visualizer with configuration parameters"""
        self.asset_name = asset_name
        self.seed = seed
        self.collision_mesh = collision_mesh
        self.num_frames = kwargs.get("num_frames", 150)
        self.fps = kwargs.get("fps", 30)
        self.add_ground = kwargs.get("add_ground", False)
        # resolution of rendered videos
        self.resolution = kwargs.get("resolution", "1920x1080")
        self.img_width = int(self.resolution.split("x")[0])
        self.img_height = int(self.resolution.split("x")[1])
        self.mujoco_dir = Path(output_dir)
        self.sim_asset_dir = (
            self.mujoco_dir / "sim_assets/mjcf" / asset_name / str(seed)
        )
        self.render_out_dir = self.mujoco_dir / "renders/mjcf" / asset_name / str(seed)

        # Clean directories if requested
        if kwargs.get("remove_existing", False):
            asset_sim_path = self.mujoco_dir / "sim_assets/mjcf" / asset_name
            asset_render_path = self.mujoco_dir / "renders/mjcf" / asset_name
            if asset_sim_path.exists():
                shutil.rmtree(asset_sim_path)
            if asset_render_path.exists():
                shutil.rmtree(asset_render_path)

        self.sim_asset_dir.mkdir(parents=True, exist_ok=True)
        self.render_out_dir.mkdir(parents=True, exist_ok=True)
        self.kwargs = kwargs
        self.parent_alpha = parent_alpha  # Transparency level for parent geometry
        self.margin = 1.5

        # Setup the asset, environment and joint information
        self.setup_asset(use_cached)
        self.setup_environment()
        self.setup_joints_data()

    def setup_asset(self, use_cached: bool) -> None:
        """Load or generate the MJCF asset file"""
        if (
            use_cached
            and not self.collision_mesh
            and self.sim_asset_dir.exists()
            and any(self.sim_asset_dir.glob("*.xml"))
        ):
            export_path = next((self.sim_asset_dir).glob("*.xml"))
            print(f"Using cached MJCF for {self.asset_name} at: {export_path}")
        else:
            try:
                (
                    export_path,
                    _,
                ) = sf.spawn_simready(
                    name=self.asset_name,
                    seed=int(self.seed),
                    exporter="mjcf",
                    export_dir=self.mujoco_dir / "sim_assets",
                    visual_only=(not self.collision_mesh),
                    **self.kwargs,
                )
                print(f"Generated MJCF for {self.asset_name} at: {export_path}")
            except Exception as e:
                print(f"Failed to generate MJCF for {self.asset_name}: {e}")
                raise

        # Read the XML file
        original = ET.parse(str(export_path))
        tree = ET.parse(str(export_path))
        root = tree.getroot()

        # Find or create asset element
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        # Create skybox texture
        texture = ET.SubElement(asset, "texture")
        texture.set("name", "skybox")
        texture.set("type", "skybox")
        texture.set("builtin", "gradient")
        texture.set("rgb1", "0.529 0.808 0.922")
        texture.set("width", "512")
        texture.set("height", "512")

        # Save modified XML
        tree.write(str(export_path), encoding="utf-8", xml_declaration=True)

        # Reload with MuJoCo
        self.spec = mujoco.MjSpec.from_file(str(export_path))

        # Save original XML
        original.write(str(export_path), encoding="utf-8", xml_declaration=True)

    def setup_environment(self) -> None:
        """Add lighting and ground plane to the scene"""
        # Add three-point lighting setup
        self.spec.worldbody.add_light(
            name="light1",
            castshadow=False,
            pos=[0, 0, 10],
            dir=[0, 0, -1],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
            ambient=[0.3, 0.3, 0.3],
        )
        self.spec.worldbody.add_light(
            name="light2",
            castshadow=False,
            pos=[10, 0, 0],
            dir=[-1, 0, 0],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
            ambient=[0.3, 0.3, 0.3],
        )
        self.spec.worldbody.add_light(
            name="light3",
            castshadow=False,
            pos=[0, 10, 0],
            dir=[0, -1, 0],
            diffuse=[0.8, 0.8, 0.8],
            specular=[0.1, 0.1, 0.1],
            ambient=[0.3, 0.3, 0.3],
        )

        # Add ground plane (if not removed)
        if self.add_ground:
            self.spec.worldbody.add_geom(
                name="ground",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[10, 10, 0.1],
                rgba=[0.3, 0.3, 0.3, 1.0],
                pos=[0, 0, -0.1],
            )

    def setup_joints_data(self) -> None:
        """Extract joint data and relationships between bodies"""
        temp_model = self.spec.compile()
        temp_data = mujoco.MjData(temp_model)
        self.joint_info = {}
        mujoco.mj_step(temp_model, temp_data)

        for j in range(temp_model.njnt):
            name = mujoco.mj_id2name(temp_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            jtype = temp_model.jnt_type[j]

            # Calculate joint axis in world coordinates
            local_axis = temp_model.jnt_axis[j]
            body_id = temp_model.jnt_bodyid[j]
            R_world_body = temp_data.xmat[body_id].reshape(3, 3)
            axis_world = R_world_body @ local_axis

            # Get parent-child body relationship
            child_body_id = temp_model.jnt_bodyid[j]
            parent_body_id = temp_model.body_parentid[child_body_id]

            # Find associated geometries
            child_geom_ids = []
            for g in range(temp_model.ngeom):
                if temp_model.geom_bodyid[g] == child_body_id:
                    child_geom_ids.append(g)

            parent_geom_ids = []
            for g in range(temp_model.ngeom):
                if temp_model.geom_bodyid[g] == parent_body_id:
                    parent_geom_ids.append(g)

            jnt_min = temp_model.jnt_range[j][0]
            jnt_max = temp_model.jnt_range[j][1]
            jnt_start_val = temp_data.qpos[j]

            if jtype == 3 and (jnt_min == jnt_max == 0):
                jnt_min = -np.pi
                jnt_max = np.pi
            elif jtype == 2 and (jnt_min == jnt_max == 0):
                jnt_min = -1
                jnt_max = 1

            # Store joint information
            self.joint_info[name] = {
                "joint_id": j,
                "type": jtype,
                "axis_world": axis_world,
                "child_body_ids": child_geom_ids,
                "parent_body_ids": parent_geom_ids,
                "jnt_init": jnt_start_val,
                "jnt_min": jnt_min,
                "jnt_max": jnt_max,
                "jnt_init_to_max": jnt_max - jnt_start_val,
                "jnt_init_to_min": jnt_start_val - jnt_min,
                "qpos_adr": temp_model.jnt_qposadr[j],
            }

        # Set up cameras for all joints
        self.setup_joint_cameras(temp_model, temp_data)

    def setup_joint_cameras(self, temp_model: Any, temp_data: Any) -> None:
        """Create cameras focused on each joint from multiple angles"""
        model = temp_model
        data = temp_data

        for joint_name, joint_info in self.joint_info.items():
            # Save original pose to restore after testing joint movement
            qpos_original = data.qpos.copy()

            # Collect bounds at both extremes of joint movement
            child_body_ids = joint_info["child_body_ids"]
            allb = []

            for child_body_id in child_body_ids:
                geom_name = model.geom(child_body_id).name

                # Sample at start position and at mid-movement
                for stage in ["min", "max"]:
                    if stage == "min":
                        qpos_value = joint_info["jnt_min"]
                    else:
                        qpos_value = joint_info["jnt_max"]

                    data.qpos[joint_info["qpos_adr"]] = qpos_value
                    mujoco.mj_forward(model, data)
                    bbmax, bbmin = self.get_geom_aabb_world(model, data, geom_name)

                    allb.append(bbmax)
                    allb.append(bbmin)

            # Restore original position
            data.qpos[:] = qpos_original
            mujoco.mj_forward(model, data)

            # Calculate joint bounding box
            joint_bbmax = np.max(np.array(allb), axis=0)
            joint_bbmin = np.min(np.array(allb), axis=0)
            joint_center = (joint_bbmax + joint_bbmin) / 2
            joint_half_dims = (joint_bbmax - joint_bbmin) / 2

            # Camera settings
            fovy = 45  # Vertical field of view
            fovx = (
                self.img_width / self.img_height
            ) * fovy  # Horizontal field of view based on aspect ratio

            # Add cameras from four standard views
            # Front view
            d_to_see_vertical = joint_half_dims[2] / np.tan(np.radians(fovy / 2))
            d_to_see_horizontal = joint_half_dims[1] / np.tan(np.radians(fovx / 2))
            d_to_see = (
                max(d_to_see_vertical, d_to_see_horizontal) * self.margin
            )  # margin

            self.spec.worldbody.add_camera(
                name=f"joint_{joint_name}_front_view",
                pos=[
                    joint_bbmax[0] + d_to_see,
                    joint_center[1],
                    joint_center[2],
                ],
                fovy=fovy,
                xyaxes=[0, 1, 0, 0, 0, 1],
            )

            # Top view
            d_to_see_vertical = joint_half_dims[0] / np.tan(np.radians(fovy / 2))
            d_to_see_horizontal = joint_half_dims[1] / np.tan(np.radians(fovx / 2))
            d_to_see = (
                max(d_to_see_vertical, d_to_see_horizontal) * self.margin
            )  # margin

            self.spec.worldbody.add_camera(
                name=f"joint_{joint_name}_top_view",
                pos=[joint_center[0], joint_center[1], joint_bbmax[2] + d_to_see],
                fovy=fovy,
                xyaxes=[1, 0, 0, 0, 1, 0],
            )

            # Left side view
            d_to_see_vertical = joint_half_dims[2] / np.tan(np.radians(fovy / 2))
            d_to_see_horizontal = joint_half_dims[0] / np.tan(np.radians(fovx / 2))
            d_to_see = (
                max(d_to_see_vertical, d_to_see_horizontal) * self.margin
            )  # margin

            self.spec.worldbody.add_camera(
                name=f"joint_{joint_name}_left_side_view",
                pos=[
                    joint_center[0],
                    joint_bbmin[1] - d_to_see,
                    joint_center[2],
                ],
                fovy=fovy,
                xyaxes=[1, 0, 0, 0, 0, 1],
            )

            # Right side view
            self.spec.worldbody.add_camera(
                name=f"joint_{joint_name}_right_side_view",
                pos=[
                    joint_center[0],
                    joint_bbmax[1] + d_to_see,
                    joint_center[2],
                ],
                fovy=fovy,
                xyaxes=[-1, 0, 0, 0, 0, 1],
            )

    def save_frames(self, out_file_path: Path, frames: List[np.ndarray]) -> None:
        """Save frames as an MP4 video file"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(out_file_path), fourcc, self.fps, (self.img_width, self.img_height)
        )

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {out_file_path}")

        for frame_rgb in frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()

        # Re-encode with ffmpeg for browser compatibility
        temp_path = out_file_path.with_suffix(".tmp.mp4")
        out_file_path.rename(temp_path)

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(temp_path),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_file_path),
            ],
            check=True,
            capture_output=True,
        )

        temp_path.unlink()

        print(
            f"Saved asset: {self.asset_name}, seed: {self.seed}, camera: {Path(out_file_path).stem} to {out_file_path}"
        )

    def get_geom_aabb_world(
        self, model: Any, data: Any, geom_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate world-space axis-aligned bounding box for a geometry"""
        # Get geom ID
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id == -1:
            raise ValueError(f"Geom '{geom_name}' not found in model.")

        # Transform local AABB to world coordinates
        R = data.geom_xmat[geom_id].reshape(3, 3)
        center_local = model.geom_aabb[geom_id, :3]
        half_sizes = model.geom_aabb[geom_id, 3:]
        center_world = data.geom_xpos[geom_id] + R @ center_local
        world_half = np.abs(R) @ half_sizes

        # Calculate min/max bounds
        world_min = center_world - world_half
        world_max = center_world + world_half
        return world_max, world_min

    def _update_sim(self, step: int, joint_info: Dict[str, Any]) -> None:
        """Update simulation state for current animation step"""
        angle = (step / (self.num_frames - 1)) * 2 * np.pi
        # Use sine wave to animate joint: 0 -> 1 -> 0 -> -1 -> 0
        if angle < np.pi:
            qpos_value = joint_info["jnt_init"] + joint_info[
                "jnt_init_to_max"
            ] * np.sin(angle)
        else:
            qpos_value = joint_info["jnt_init"] - joint_info[
                "jnt_init_to_min"
            ] * np.sin(angle - np.pi)
        self.data.qpos[joint_info["qpos_adr"]] = qpos_value
        mujoco.mj_forward(self.model, self.data)

    def _set_geom_visibility(self, joint_name: str) -> None:
        """Set parent/child geometry visibility for joint visualization"""
        if joint_name not in self.joint_info:
            return

        joint_info = self.joint_info[joint_name]
        child_body_ids = joint_info["child_body_ids"]
        parent_body_ids = joint_info["parent_body_ids"]

        # Store original rgba values to restore later
        self.original_geom_rgba = np.copy(self.model.geom_rgba)

        # Make all geometries invisible except ground
        for i in range(self.model.ngeom):
            if self.model.geom(i).name == "ground":
                continue
            else:
                self.model.geom_rgba[i, 3] = 0.1

        # Make parent geometries semi-transparent
        for i in parent_body_ids:
            self.model.geom_rgba[i, 3] = self.parent_alpha

        # Make child geometries fully visible
        for i in child_body_ids:
            self.model.geom_rgba[i, 3] = 1.0

    def _restore_geom_visibility(self) -> None:
        """Restore original geometry visibility after rendering"""
        if hasattr(self, "original_geom_rgba"):
            for i in range(self.model.ngeom):
                self.model.geom_rgba[i, 3] = self.original_geom_rgba[i, 3]

    def _render_single_joint_animation(
        self, joint_name: str, joint_info: Dict[str, Any], all_camera_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """Render animation frames for a single joint from all camera views"""
        frames = {cam_name: [] for cam_name in all_camera_names}
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.model.vis.global_.offwidth = self.img_width
        self.model.vis.global_.offheight = self.img_height

        renderer = mujoco.Renderer(
            self.model, width=self.img_width, height=self.img_height
        )

        # Set visibility to highlight this joint's parent/child
        self._set_geom_visibility(joint_name)

        # Render each frame of the animation
        for step in range(self.num_frames):
            self._update_sim(step, joint_info)

            # Capture from each camera view
            for camera_name in all_camera_names:
                renderer.update_scene(self.data, camera=camera_name)
                pixels = renderer.render()
                frames[camera_name].append(pixels)

        # Restore original visibility
        self._restore_geom_visibility()
        return frames

    def render_all_animations(self) -> None:
        """Render animations for all joints in the model"""
        for joint_name, joint_info in self.joint_info.items():
            # Get cameras specific to this joint
            joint_cameras = [
                cam.name
                for cam in self.spec.cameras
                if cam.name.startswith("joint_") and joint_name in cam.name
            ]

            # Render frames for each view
            frames = self._render_single_joint_animation(
                joint_name, joint_info, joint_cameras
            )

            # Save videos
            for camera_name, cam_frames in frames.items():
                cam_out_path = self.render_out_dir / f"{camera_name}.mp4"
                self.save_frames(cam_out_path, cam_frames)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Render MuJoCo asset animations")
    # parser.add_argument(
    #     "--asset_name", default="lamp", help="Name of the asset to render"
    # )
    # parser.add_argument("--seed", type=int, default=2, help="Random seed")
    # parser.add_argument(
    #     "--parent_alpha",
    #     type=float,
    #     default=0.3,
    #     help="Transparency level for parent geom (0-1)",
    # )
    # parser.add_argument(
    #     "--collision_mesh",
    #     action="store_true",
    #     help="Whether to load asset in visual-only mode",
    # )
    # args = parser.parse_args()

    asset_initializer = MujocoAssetInitializer(
        asset_name="door",
        seed=3,
        parent_alpha=0.3,
        collision_mesh=False,
        use_cached=True,
        add_ground=False,
        resolution="1080x1080",
    )
    asset_initializer.render_all_animations()
