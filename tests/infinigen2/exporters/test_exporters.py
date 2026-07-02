# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import inspect
from pathlib import Path

import bpy
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import procfunc as pf
import pytest

from infinigen.core.util import blender as butil
from infinigen2.exporters import render_cycles, render_eevee
from infinigen2.exporters.util.format import ExportType, RenderPass
from infinigen2.exporters.visualize_gt import visualize_flow
from infinigen2.util.camera_projection import get_3x4_RT_matrix_from_blender


def configure_cube_scene(big=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2) if not big else (20, 20, 20))

    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (6, -6, 6)
    camera.item().rotation_euler = np.deg2rad(
        (54, 0, 45)
    )  # aligns camera center on cube corner

    camera.item().keyframe_insert("location", frame=1)
    camera.item().location.x += 1
    camera.item().keyframe_insert("location", frame=2)

    light = pf.ops.primitives.point_lamp(energy=2000)
    light.item().location = camera.item().location

    return [cube, light], camera


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_cube_render_rgb(tmp_path, method, save_blend=True):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(ExportType.IMAGE, Path("rgb_%f.png"), np.dtype(np.uint8)),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 256),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)

    assert results[ExportType.IMAGE][0].exists()

    rgb = imageio.imread(results[ExportType.IMAGE][0])

    grayscale = rgb.mean(axis=2)

    # middle pixel should be brighter than corners
    center_y, center_x = rgb.shape[0] // 2, rgb.shape[1] // 2
    assert grayscale[center_y, center_x] > grayscale[0, 0]
    assert grayscale[center_y, center_x] > grayscale[0, 255]
    assert grayscale[center_y, center_x] > grayscale[255, 0]
    assert grayscale[center_y, center_x] > grayscale[255, 255]


def colorize_normals(surface_normals):
    assert surface_normals.max() < 1 + 1e-4
    assert surface_normals.min() > -1 - 1e-4
    norm = np.linalg.norm(surface_normals, axis=2)
    color = np.round((surface_normals + 1) * (255 / 2)).astype(np.uint8)
    color[norm < 1e-4] = 0
    return color


@pytest.mark.render_gpu
@pytest.mark.slow
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_cube_render_ground_truth_zoomed_in(tmp_path, method, save_blend=True):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(ExportType.DEPTH, Path("depth.npy"), np.dtype(np.float32)),
        RenderPass(ExportType.SURFACE_NORMAL, Path("normal.npy"), np.dtype(np.float32)),
        RenderPass(ExportType.OPTICAL_FLOW, Path("flow.npy"), np.dtype(np.float32)),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(256, 256),
        )
    elif method == "opengl":
        raise NotImplementedError("OpenGL ground truth is not implemented")
    else:
        raise ValueError(f"Unknown ground truth method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)

    depth = np.load(results[ExportType.DEPTH][0])
    normal = np.load(results[ExportType.SURFACE_NORMAL][0])
    # flow = np.load(results[ExportType.OPTICAL_FLOW][0])

    center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
    print(depth[center_y, center_x])

    # plt.imshow(colorize_normals(normal))
    # plt.colorbar()
    # plt.show()

    assert np.isclose(depth[center_y, center_x], 4 * np.sqrt(3), atol=0.1), depth[
        center_y, center_x
    ]

    # Pixels should get further away as we move out from the image center
    m = 15
    assert depth[128, 128] < depth[128, 128 - m]
    assert depth[128, 128] < depth[128 + m, 128]
    assert depth[128, 128] < depth[128, 128 + m]
    assert depth[128, 128] < depth[128 - m, 128 + m]
    assert depth[128, 128] < depth[128 + m, 128 - m]

    # Test surface normals in camera coordinates
    # Camera is looking down at cube corner, sees top face and two side faces

    # Sample points on different faces
    # Center region should be the top face
    top_face_normal = normal[128, 128]

    # Left side of image should show left face of cube
    left_face_normal = normal[128, 100]

    # Right side of image should show right face of cube
    right_face_normal = normal[128, 156]

    print(f"Top face normal: {top_face_normal}")
    print(f"Left face normal: {left_face_normal}")
    print(f"Right face normal: {right_face_normal}")

    # All normals should be unit vectors (normalized in postprocessing)
    assert np.isclose(np.linalg.norm(top_face_normal), 1.0, atol=0.01), (
        f"Top normal not unit: {np.linalg.norm(top_face_normal)}"
    )
    assert np.isclose(np.linalg.norm(left_face_normal), 1.0, atol=0.01), (
        f"Left normal not unit: {np.linalg.norm(left_face_normal)}"
    )
    assert np.isclose(np.linalg.norm(right_face_normal), 1.0, atol=0.01), (
        f"Right normal not unit: {np.linalg.norm(right_face_normal)}"
    )

    # Top face should point somewhat towards camera (negative Z component)
    # since camera is looking down at it
    assert top_face_normal[2] < 0, (
        f"Top face should point towards camera, got Z={top_face_normal[2]}"
    )

    # Left face should have negative X component (pointing left in camera coords)
    assert left_face_normal[0] < 0, (
        f"Left face should point left, got X={left_face_normal[0]}"
    )

    # Right face should have positive X component (pointing right in camera coords)
    assert right_face_normal[0] > 0, (
        f"Right face should point right, got X={right_face_normal[0]}"
    )

    # Test that normals change when camera pose changes (proves they're in camera coordinates)
    # Save the original camera pose
    original_location = camera.item().location.copy()
    original_rotation = camera.item().rotation_euler.copy()

    # Rotate camera slightly and render normals again
    camera.item().rotation_euler = np.deg2rad((54, 0, 55))  # Changed yaw by 5 degrees

    render_passes_2 = [
        RenderPass(
            ExportType.SURFACE_NORMAL, Path("normal2.npy"), np.dtype(np.float32)
        ),
    ]

    results_2 = render_cycles.render_cycles_ground_truth(
        objects=objects,
        camera=camera,
        output_folder=tmp_path,
        render_passes=render_passes_2,
        frame_start=1,
        frame_end=1,
        resolution=(256, 256),
    )

    normal_2 = np.load(results_2[ExportType.SURFACE_NORMAL][0])

    # plt.imshow(colorize_normals(normal_2))
    # plt.colorbar()
    # plt.show()

    # Restore original camera pose
    camera.item().location = original_location
    camera.item().rotation_euler = original_rotation

    # Compare normals at the same pixel locations
    # They should be different since the camera coordinate system changed
    center_normal_1 = normal[128, 128]
    center_normal_2 = normal_2[128, 128]

    left_normal_1 = normal[128, 100]
    left_normal_2 = normal_2[128, 100]

    print(f"Original center normal: {center_normal_1}")
    print(f"Rotated center normal: {center_normal_2}")
    print(f"Original left normal: {left_normal_1}")
    print(f"Rotated left normal: {left_normal_2}")

    # Normals should be different after camera rotation
    center_diff = np.linalg.norm(center_normal_1 - center_normal_2)
    left_diff = np.linalg.norm(left_normal_1 - left_normal_2)

    print(f"Center normal difference magnitude: {center_diff}")
    print(f"Left normal difference magnitude: {left_diff}")

    # If normals were in world coordinates, they wouldn't change when camera moves
    # Since they're in camera coordinates, they should change significantly
    assert center_diff > 0.1, (
        f"Center normals should change significantly with camera rotation, got diff={center_diff}"
    )
    assert left_diff > 0.1, (
        f"Left normals should change significantly with camera rotation, got diff={left_diff}"
    )


@pytest.mark.render_gpu
@pytest.mark.slow
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_cube_render_ground_truth_surface_normals(
    tmp_path, method, save_blend=True, plot=False
):
    objects, camera = configure_cube_scene(big=True)

    render_passes = [
        RenderPass(ExportType.SURFACE_NORMAL, Path("normal.npy"), np.dtype(np.float32)),
        RenderPass(
            ExportType.SURFACE_NORMAL_WORLD, Path("normal.npy"), np.dtype(np.float32)
        ),
    ]

    h, w = 256, 256

    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(w, h),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(w, h),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = (
            tmp_path / f"{inspect.currentframe().f_code.co_name}_{method}.blend"
        )
        butil.save_blend(blend_path)

    normals_world = np.load(results[ExportType.SURFACE_NORMAL_WORLD][0])
    print(normals_world.shape)
    print(f"normals_world[0, 0]: {normals_world[0, 0]}")
    print(f"normals_world[h-1, w-1]: {normals_world[h - 1, w - 1]}")
    print(f"normals_world[h-1, 0]: {normals_world[h - 1, 0]}")
    print(f"normals_world[0, w-1]: {normals_world[0, w - 1]}")

    # These hold when it is not wrt camera but wrt world
    assert np.allclose(normals_world[0, 0], np.array([1, 0, 0]), atol=1e-4)
    assert np.allclose(normals_world[0, 0], np.array([1, 0, 0]), atol=1e-4)
    assert np.allclose(normals_world[h - 1, w - 1], np.array([0, 0, 1]), atol=1e-4)
    assert np.allclose(normals_world[h - 1, 0], np.array([0, 0, 1]), atol=1e-4)
    assert np.allclose(normals_world[0, w - 1], np.array([0, -1, 0]), atol=1e-4)

    if plot:
        plt.imshow(colorize_normals(normals_world))
        plt.show()

    normals_camera = np.load(results[ExportType.SURFACE_NORMAL][0])
    print(normals_camera.shape)
    print(f"normals_camera[0, 0]: {normals_camera[0, 0]}")
    print(f"normals_camera[h-1, w-1]: {normals_camera[h - 1, w - 1]}")
    print(f"normals_camera[h-1, 0]: {normals_camera[h - 1, 0]}")
    print(f"normals_camera[0, w-1]: {normals_camera[0, w - 1]}")

    assert normals_camera[0, 0][0] > 0
    assert normals_camera[0, 0][1] > 0
    assert normals_camera[0, 0][2] < 0

    assert normals_camera[h - 1, w - 1][1] < 0
    assert normals_camera[h - 1, w - 1][2] < 0

    assert normals_camera[0, w - 1][0] < 0
    assert normals_camera[0, w - 1][1] > 0
    assert normals_camera[0, w - 1][2] < 0


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_cube_render_surface_normals_multiframe(tmp_path, method):
    """Camera-space normals must use the correct per-frame camera rotation."""
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cube = pf.ops.primitives.mesh_cube(scale=(20, 20, 20))
    camera = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (6, -6, 6)
    camera.item().rotation_euler = np.deg2rad((54, 0, 45))
    camera.item().keyframe_insert("rotation_euler", frame=1)
    camera.item().rotation_euler[2] += np.deg2rad(30)
    camera.item().keyframe_insert("rotation_euler", frame=2)

    light = pf.ops.primitives.point_lamp(energy=2000)
    light.item().location = camera.item().location
    objects = [cube, light]

    render_passes = [
        RenderPass(ExportType.SURFACE_NORMAL, Path("normal.npy"), np.dtype(np.float32)),
        RenderPass(
            ExportType.SURFACE_NORMAL_WORLD, Path("normal.npy"), np.dtype(np.float32)
        ),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    cam_paths = results[ExportType.SURFACE_NORMAL]
    world_paths = results[ExportType.SURFACE_NORMAL_WORLD]
    assert len(cam_paths) == 2
    assert len(world_paths) == 2

    for frame_idx, frame_num in enumerate([1, 2]):
        bpy.context.scene.frame_set(frame_num)
        bpy.context.view_layer.update()
        RT = np.array(get_3x4_RT_matrix_from_blender(camera))
        R = RT[:3, :3]

        normals_world = np.load(world_paths[frame_idx])
        normals_cam = np.load(cam_paths[frame_idx])

        # Manually rotate world normals to camera space
        valid = np.linalg.norm(normals_world, axis=2) > 0.5
        expected_cam = (R @ normals_world[valid].T).T

        actual_cam = normals_cam[valid]

        assert np.allclose(actual_cam, expected_cam, atol=0.02), (
            f"Frame {frame_num}: camera-space normals don't match "
            f"world normals rotated by the camera pose. "
            f"Max diff: {np.abs(actual_cam - expected_cam).max():.4f}"
        )


@pytest.mark.render_gpu
@pytest.mark.parametrize(
    "method",
    [
        "cycles",
        "eevee",
    ],
)
def test_cube_render_optical_flow(tmp_path, method, save_blend=True, plot=False):
    objects, camera = configure_cube_scene()

    print("Configuring optical flow render pass...")
    render_passes = [
        RenderPass(
            ExportType.OPTICAL_FLOW,
            Path("flow.npy"),
            np.dtype(np.float32),
        ),
    ]

    print(f"Starting {method} render...")
    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    print("Render completed!")
    print(f"Results: {results}")
    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)

    flow = np.load(results[ExportType.OPTICAL_FLOW][0])
    print(f"Flow shape: {flow.shape}")
    print(f"Flow dtype: {flow.dtype}")
    print(f"Flow min: {flow.min()}, max: {flow.max()}")
    print(f"Flow sample [0,0]: {flow[0, 0]}")
    print(f"Flow sample [center]: {flow[flow.shape[0] // 2, flow.shape[1] // 2]}")

    if plot:
        input_path = tmp_path / "flow.npy"
        output_path = tmp_path / "flow_color.png"
        flow_color = visualize_flow(input_path, output_path)
        plt.imshow(flow_color)
        plt.show()

    flow_uv = flow[..., :2]

    mag = np.linalg.norm(flow_uv, axis=2)
    mask = mag > 1e-3

    print(f"Mask any: {mask.any()}")
    print(f"Mask sum: {mask.sum()}")
    print(f"Mag min: {mag.min()}, max: {mag.max()}")

    assert mask.any()

    mean_u = flow_uv[..., 0][mask].mean()
    print(f"Mean u: {mean_u}")

    mean_v = flow_uv[..., 1][mask].mean()
    print(f"Mean v: {mean_v}")

    assert mean_u < 0
    assert mag[mask].mean() > 0.01

    assert mean_v < 0

    # Check that center pixel has more flow than other pixels
    center_y, center_x = flow.shape[0] // 2, flow.shape[1] // 2
    center_flow = flow[center_y, center_x]
    center_flow_mag = np.linalg.norm(center_flow)
    print(f"Center flow: {center_flow}")
    assert center_flow_mag > 0.01
    corner_flow = flow[0, 0]
    print(f"Corner flow: {corner_flow}")
    corner_flow_mag = np.linalg.norm(corner_flow)
    assert corner_flow_mag < center_flow_mag


@pytest.mark.render_gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    [
        "cycles",
        "eevee",
    ],
)
def test_cube_render_optical_flow_horizontal(tmp_path, method, save_blend=True):
    objects, camera = configure_cube_scene()

    camera.item().animation_data_clear()
    camera.item().location = (0, 0, 10)
    camera.item().rotation_euler = np.deg2rad((0, 0, 0))
    camera.item().keyframe_insert("location", frame=1)
    camera.item().location.x += 1
    camera.item().keyframe_insert("location", frame=2)

    render_passes = [
        RenderPass(
            ExportType.OPTICAL_FLOW,
            Path("flow.npy"),
            np.dtype(np.float32),
        ),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)

    flow = np.load(results[ExportType.OPTICAL_FLOW][0])
    flow_uv = flow[..., :2]

    mag = np.linalg.norm(flow_uv, axis=2)
    mask = mag > 1e-3

    assert mask.any()

    mean_u = flow_uv[..., 0][mask].mean()
    mean_v = flow_uv[..., 1][mask].mean()

    print(f"Mean u: {mean_u}")
    print(f"Mean v: {mean_v}")

    assert mean_u < 0
    assert np.abs(mean_v) < 0.2 * np.abs(mean_u)


@pytest.mark.render_gpu
@pytest.mark.parametrize(
    "method",
    [
        "cycles",
        "eevee",
    ],
)
def test_cube_render_optical_flow_object_moving(
    tmp_path, method, save_blend=True, plot=False
):
    objects, camera = configure_cube_scene()

    cube = objects[0]

    camera.item().animation_data_clear()

    camera.item().location = (0, 0, 10)
    camera.item().rotation_euler = np.deg2rad((0, 0, 0))

    cube.item().animation_data_clear()
    cube.item().location = (0, 0, 0)
    cube.item().keyframe_insert("location", frame=1)
    cube.item().location.x += 1
    cube.item().keyframe_insert("location", frame=2)

    render_passes = [
        RenderPass(ExportType.OPTICAL_FLOW, Path("flow.npy"), np.dtype(np.float32)),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee_ground_truth(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=2,
            resolution=(256, 256),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        butil.save_blend(blend_path)

    flow = np.load(results[ExportType.OPTICAL_FLOW][0])
    if plot:
        input_path = tmp_path / "flow.npy"
        output_path = tmp_path / "flow_color.png"
        flow_color = visualize_flow(input_path, output_path)
        plt.imshow(flow_color)
        plt.show()

    flow_uv = flow[..., :2]

    mag = np.linalg.norm(flow_uv, axis=2)
    mask = mag > 1e-3

    assert mask.any()

    mean_u = flow_uv[..., 0][mask].mean()
    mean_v = flow_uv[..., 1][mask].mean()

    assert mean_u > 0
    assert np.isclose(mean_v, 0.0)
    assert np.abs(mean_v) < 0.2 * np.abs(mean_u)
