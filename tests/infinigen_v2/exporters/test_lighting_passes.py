import inspect
from pathlib import Path

import bpy
import imageio.v2 as imageio
import numpy as np
import procfunc as pf
import pytest

from infinigen_v2.exporters import render_cycles, render_eevee
from infinigen_v2.exporters.util.format import ExportType, RenderPass

from .test_exporters import (
    configure_cube_scene,
)


def configure_emission_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    cube = pf.ops.primitives.mesh_cube(scale=(2, 2, 2))

    # Create a pure emission red material
    emission = pf.nodes.shader.emission(color=(1, 0, 0, 1), strength=1.0)
    pf.ops.object.set_material(cube, surface=emission)

    # Remove all lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Black background
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0, 0, 0, 1)
        bg.inputs[1].default_value = 0.0

    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (6, -6, 6)
    camera.item().rotation_euler = np.deg2rad((54, 0, 45))
    bpy.context.scene.camera = camera.item()

    return [cube], camera


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def simple_test_cube_render_lighting_splits(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(
            ExportType.DIFFUSE_DIRECT,
            Path("diffdir_%f.png"),
            np.dtype(np.uint8),
        ),
        RenderPass(
            ExportType.EMISSION,
            Path("emit_%f.png"),
            np.dtype(np.uint8),
        ),
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
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    for rp in render_passes:
        assert results[rp.type][0].exists()
        img = imageio.imread(results[rp.type][0])
        assert img.mean() > 0

    print(results[ExportType.DIFFUSE_DIRECT][0])
    print(results[ExportType.EMISSION][0])

    # Check emission is zero everywhere
    emit = imageio.imread(results[ExportType.EMISSION][0])
    assert emit.max() == 0, "Emission should be zero everywhere"

    # Check diffuse direct is non-zero at the center pixel
    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])
    center_y, center_x = diffdir.shape[0] // 2, diffdir.shape[1] // 2
    assert diffdir[center_y, center_x] > 0, (
        "Diffuse direct should be non-zero at the center"
    )


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_emission_pass_matches_rgb(tmp_path, method, save_blend=False):
    objects, camera = configure_emission_scene()

    render_passes = [
        RenderPass(ExportType.IMAGE, Path("rgb_%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.EMISSION, Path("emit_%f.png"), np.dtype(np.uint8)),
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"
        pf.ops.file.save_blend(blend_path)

    rgb = imageio.imread(results[ExportType.IMAGE][0])
    emit = imageio.imread(results[ExportType.EMISSION][0])
    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])

    # Emission should match RGB closely
    if method == "cycles":
        assert np.allclose(rgb, emit, atol=1), (
            "Emission pass should match combined RGB when only emission present"
        )
    # Eevee gives different values at the boundary, so we check the center patch
    elif method == "eevee":
        center_y, center_x = rgb.shape[0] // 2, rgb.shape[1] // 2
        assert np.allclose(
            rgb[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10],
            emit[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10],
            atol=1,
        ), "Emission pass should match combined RGB when only emission present"

    # Diffuse direct should be near zero everywhere
    assert diffdir.max() <= 1, (
        "Diffuse direct should be zero or near-zero for purely emissive material"
    )


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_cube_render_lighting_splits_extended(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.GLOSSY_DIRECT, Path("glossdir_%f.png"), np.dtype(np.uint8)
        ),
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
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])
    glossdir = imageio.imread(results[ExportType.GLOSSY_DIRECT][0])

    center_y, center_x = diffdir.shape[0] // 2, diffdir.shape[1] // 2

    assert diffdir.mean() > 0
    assert glossdir.mean() > 0
    assert diffdir[center_y, center_x].mean() > 0


def configure_environment_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bg.inputs[1].default_value = 1.0
    camera: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    camera.item().location = (0, 0, 0)
    camera.item().rotation_euler = np.deg2rad((0, 0, 0))
    bpy.context.scene.camera = camera.item()
    return [], camera


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles"])
def test_emission_scene_all_lighting_passes(tmp_path, method, save_blend=False):
    objects, camera = configure_emission_scene()

    render_passes = [
        RenderPass(ExportType.IMAGE, Path("rgb_%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.EMISSION, Path("emit_%f.png"), np.dtype(np.uint8)),
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.DIFFUSE_INDIRECT, Path("diffind_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.GLOSSY_DIRECT, Path("glossdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.GLOSSY_INDIRECT, Path("glossind_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.TRANSMISSION_DIRECT, Path("transdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.TRANSMISSION_INDIRECT,
            Path("transind_%f.png"),
            np.dtype(np.uint8),
        ),
        RenderPass(ExportType.ENVIRONMENT, Path("env_%f.png"), np.dtype(np.uint8)),
    ]

    if method == "cycles":
        results = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    rgb = imageio.imread(results[ExportType.IMAGE][0])
    emit = imageio.imread(results[ExportType.EMISSION][0])

    assert np.allclose(rgb, emit, atol=1)

    zero_passes = [
        ExportType.DIFFUSE_DIRECT,
        ExportType.DIFFUSE_INDIRECT,
        ExportType.GLOSSY_DIRECT,
        ExportType.GLOSSY_INDIRECT,
        ExportType.TRANSMISSION_DIRECT,
        ExportType.TRANSMISSION_INDIRECT,
        ExportType.ENVIRONMENT,
    ]
    for p in zero_passes:
        img = imageio.imread(results[p][0])
        assert img.max() <= 1


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_environment_pass_matches_rgb(tmp_path, method, save_blend=False):
    objects, camera = configure_environment_scene()

    render_passes = [
        RenderPass(ExportType.IMAGE, Path("rgb_%f.png"), np.dtype(np.uint8)),
        RenderPass(ExportType.ENVIRONMENT, Path("env_%f.png"), np.dtype(np.uint8)),
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
    ]

    assert len(render_passes) > 0, f"No render passes in {render_passes=}"

    if method == "cycles":
        results = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if results is None:
        raise ValueError(f"No results for {method=} with {render_passes=}")

    if save_blend:
        frame = inspect.currentframe().f_code.co_name  # type: ignore
        blend_path = tmp_path / f"{frame}.blend"
        pf.ops.file.save_blend(blend_path)

    rgb = imageio.imread(results[ExportType.IMAGE][0])
    env = imageio.imread(results[ExportType.ENVIRONMENT][0])

    assert np.allclose(rgb, env, atol=1)

    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])
    assert diffdir.max() <= 1


def create_diffuse_material(color=(1, 0, 0, 1)):
    bsdf = pf.nodes.shader.diffuse_bsdf(color=color, roughness=0.0)
    return pf.Material(surface=bsdf)


def create_glossy_material(color=(1, 1, 1, 1)):
    bsdf = pf.nodes.shader.principled_bsdf(
        base_color=color,
        metallic=1.0,
        roughness=0.05,
    )
    return pf.Material(surface=bsdf)


def create_transmission_material(color=(1, 1, 1, 1)):
    bsdf = pf.nodes.shader.principled_bsdf(
        base_color=color,
        transmission_weight=1.0,
        roughness=0.0,
        ior=1.45,
    )
    return pf.Material(surface=bsdf)


def create_volume_material(density=0.1, anisotropy=0.0):
    vol = pf.nodes.shader.volume_principled(
        density=density,
        anisotropy=anisotropy,
    )
    return pf.Material(
        volume=vol,
    )


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_lighting_passes_diffuse_material(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    cube = objects[0]
    diffuse_mat = create_diffuse_material()
    pf.ops.object.set_material(cube, surface=diffuse_mat.surface)

    render_passes = [
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.GLOSSY_DIRECT, Path("glossdir_%f.png"), np.dtype(np.uint8)
        ),
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
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])
    glossdir = imageio.imread(results[ExportType.GLOSSY_DIRECT][0])

    diff_mean = diffdir.mean()
    gloss_mean = glossdir.mean()

    # Diffuse material should have notably higher diffuse contribution
    assert diff_mean > 2 * gloss_mean


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_lighting_passes_glossy_material(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    cube = objects[0]
    glossy_mat = create_glossy_material()
    pf.ops.object.set_material(cube, surface=glossy_mat.surface)

    render_passes = [
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.GLOSSY_DIRECT, Path("glossdir_%f.png"), np.dtype(np.uint8)
        ),
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
        pf.ops.file.save_blend(blend_path)

    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])
    glossdir = imageio.imread(results[ExportType.GLOSSY_DIRECT][0])

    diff_mean = diffdir.mean()
    gloss_mean = glossdir.mean()

    # Glossy material should have at least as much glossy contribution as diffuse
    assert gloss_mean >= 0.9 * diff_mean and gloss_mean > 0


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles"])
def test_lighting_passes_transmission_material(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    cube = objects[0]
    glass_mat = create_transmission_material()
    pf.ops.object.set_material(cube, surface=glass_mat.surface)

    render_passes = [
        RenderPass(
            ExportType.TRANSMISSION_DIRECT, Path("transdir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
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
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    transdir = imageio.imread(results[ExportType.TRANSMISSION_DIRECT][0])
    diffdir = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])

    trans_mean = transdir.mean()
    diff_mean = diffdir.mean()

    # Transmission material should have noticeable transmission component compared to diffuse
    assert trans_mean >= 0.5 * diff_mean and trans_mean > 0


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["eevee"])
def test_lighting_passes_volume_direct(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    cube = objects[0]
    vol_mat = create_volume_material(density=0.2, anisotropy=0.0)
    pf.ops.object.set_material(cube, volume=vol_mat.volume)

    render_passes = [
        RenderPass(
            ExportType.VOLUME_DIRECT, Path("volumedir_%f.png"), np.dtype(np.uint8)
        ),
        RenderPass(
            ExportType.DIFFUSE_DIRECT, Path("diffdir_%f.png"), np.dtype(np.uint8)
        ),
    ]

    if method == "eevee":
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
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    vol = imageio.imread(results[ExportType.VOLUME_DIRECT][0])
    diff = imageio.imread(results[ExportType.DIFFUSE_DIRECT][0])

    assert vol.mean() > 0
    assert diff.max() <= 250


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_lighting_passes_diffuse_color_albedo_invariance(
    tmp_path, method, save_blend=False
):
    objects, camera = configure_cube_scene()

    cube = objects[0]
    light = objects[1]

    green_mat = create_diffuse_material(color=(0, 1, 0, 1))
    pf.ops.object.set_material(cube, surface=green_mat.surface)

    render_passes_a = [
        RenderPass(
            ExportType.DIFFUSE_COLOR,
            Path("diffcol_a_%f.png"),
            np.dtype(np.uint8),
        ),
    ]

    if method == "cycles":
        results_a = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_a,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results_a = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_a,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    light.item().data.energy *= 10.0

    render_passes_b = [
        RenderPass(
            ExportType.DIFFUSE_COLOR,
            Path("diffcol_b_%f.png"),
            np.dtype(np.uint8),
        ),
    ]

    if method == "cycles":
        results_b = render_cycles.render_cycles(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_b,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results_b = render_eevee.render_eevee(
            objects=objects,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_b,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    img_a = imageio.imread(results_a[ExportType.DIFFUSE_COLOR][0])
    img_b = imageio.imread(results_b[ExportType.DIFFUSE_COLOR][0])

    cy, cx = img_a.shape[0] // 2, img_a.shape[1] // 2
    patch_a = img_a[cy - 3 : cy + 4, cx - 3 : cx + 4]
    patch_b = img_b[cy - 3 : cy + 4, cx - 3 : cx + 4]
    assert np.allclose(patch_a, patch_b, atol=1)

    # test that green is strictly more than red and blue
    assert (
        patch_a[:, :, 1].mean() > patch_a[:, :, 0].mean()
        and patch_a[:, :, 1].mean() > patch_a[:, :, 2].mean()
    )

    center = img_a[cy, cx]
    assert center[1] > center[0]
    assert center[1] > center[2]


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_lighting_passes_ambient_occlusion_suzanne_vs_cube(
    tmp_path, method, save_blend=False
):
    objects_cube, camera = configure_cube_scene()

    render_passes_cube = [
        RenderPass(
            ExportType.AMBIENT_OCCLUSION,
            Path("ao_cube_%f.png"),
            np.dtype(np.uint8),
        ),
    ]

    if method == "cycles":
        results_cube = render_cycles.render_cycles(
            objects=objects_cube,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_cube,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results_cube = render_eevee.render_eevee(
            objects=objects_cube,
            camera=camera,
            output_folder=tmp_path,
            render_passes=render_passes_cube,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    suzanne = pf.ops.primitives.mesh_monkey(scale=(2, 2, 2))

    cam2: pf.CameraObject = pf.ops.primitives.perspective_camera(
        focal_length_mm=50,
        sensor_width_mm=30,
        sensor_height_mm=30,
        clip_start=0.1,
        clip_end=1000,
    )
    cam2.item().location = (6, -6, 6)
    cam2.item().rotation_euler = np.deg2rad((54, 0, 45))
    bpy.context.scene.camera = cam2.item()

    light2 = pf.ops.primitives.point_lamp(energy=2000)
    light2.item().location = cam2.item().location

    render_passes_suzanne = [
        RenderPass(
            ExportType.AMBIENT_OCCLUSION,
            Path("ao_suzanne_%f.png"),
            np.dtype(np.uint8),
        ),
    ]

    if method == "cycles":
        results_suzanne = render_cycles.render_cycles(
            objects=[suzanne, light2],
            camera=cam2,
            output_folder=tmp_path,
            render_passes=render_passes_suzanne,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    elif method == "eevee":
        results_suzanne = render_eevee.render_eevee(
            objects=[suzanne, light2],
            camera=cam2,
            output_folder=tmp_path,
            render_passes=render_passes_suzanne,
            frame_start=1,
            frame_end=1,
            resolution=(128, 128),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if save_blend:
        blend_path = tmp_path / f"{inspect.currentframe().f_code.co_name}.blend"  # type: ignore
        pf.ops.file.save_blend(blend_path)

    ao_cube = imageio.imread(results_cube[ExportType.AMBIENT_OCCLUSION][0])
    ao_suzanne = imageio.imread(results_suzanne[ExportType.AMBIENT_OCCLUSION][0])

    print(ao_cube.shape, ao_suzanne.shape)
    print(ao_cube[64, 64])
    print(ao_suzanne[64, 64])
    nz_cube = ao_cube.mean(axis=2) > 0
    nz_suz = ao_suzanne.mean(axis=2) > 0

    mean_cube = ao_cube[nz_cube].mean()
    mean_suz = ao_suzanne[nz_suz].mean()
    print(mean_cube, mean_suz)

    assert mean_suz < mean_cube
