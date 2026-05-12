from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from infinigen.core.util import blender as butil
from infinigen_v2.exporters import render_cycles, render_eevee
from infinigen_v2.exporters.util.export_utils import load_exr
from infinigen_v2.exporters.util.format import ExportType, RenderPass

from .test_exporters import configure_cube_scene


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_image_denoised(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(
            ExportType.IMAGE,
            Path("Image/%c/image_%f.png"),
            np.uint8,
        ),
        RenderPass(
            ExportType.IMAGE_DENOISED,
            Path("ImageDenoised/%c/image_denoised_%f.png"),
            np.uint8,
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
        blend_path = tmp_path / "test_image_denoised.blend"
        butil.save_blend(blend_path)

    png_noisy = results[ExportType.IMAGE][0]
    png_denoised = results[ExportType.IMAGE_DENOISED][0]

    assert png_noisy.exists()
    assert png_denoised.exists()

    img_noisy = imageio.imread(png_noisy)
    img_denoised = imageio.imread(png_denoised)

    assert img_noisy.shape == (256, 256, 3)
    assert img_denoised.shape == (256, 256, 3)
    assert img_noisy.dtype == np.uint8
    assert img_denoised.dtype == np.uint8


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_image_hdr(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(
            ExportType.IMAGE_HDR,
            Path("ImageHDR/%c/image_hdr_%f.exr"),
            np.float32,
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
        blend_path = tmp_path / "test_image_hdr.blend"
        butil.save_blend(blend_path)

    exr_path = results[ExportType.IMAGE_HDR][0]
    assert exr_path.exists()

    hdr = load_exr(exr_path)
    assert hdr.shape == (256, 256, 3)
    assert np.isfinite(hdr).all()


@pytest.mark.render_gpu
@pytest.mark.parametrize("method", ["cycles", "eevee"])
def test_image_denoised_hdr(tmp_path, method, save_blend=False):
    objects, camera = configure_cube_scene()

    render_passes = [
        RenderPass(
            ExportType.IMAGE_DENOISED_HDR,
            Path("ImageDenoisedHDR/%c/image_denoised_hdr_%f.exr"),
            np.float32,
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
        blend_path = tmp_path / "test_image_denoised_hdr.blend"
        butil.save_blend(blend_path)

    exr_path = results[ExportType.IMAGE_DENOISED_HDR][0]
    assert exr_path.exists()

    hdr = load_exr(exr_path)
    assert hdr.shape == (256, 256, 3)
    assert np.isfinite(hdr).all()
