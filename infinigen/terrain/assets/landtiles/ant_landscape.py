# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


from pathlib import Path

import bpy
import cv2
import numpy as np

from infinigen.terrain.land_process.erosion import run_erosion
from infinigen.terrain.land_process.snowfall import run_snowfall
from infinigen.terrain.utils import smooth, random_int
from infinigen.core.util.organization import AssetFile, LandTile


def create(
    preset_name,
    subdivision_x,
    subdivision_y,
):
    def presets(**kwargs):
        bpy.ops.mesh.landscape_add(ant_terrain_name="Landscape", land_material="", water_material="", texture_block="", at_cursor=True, smooth_mesh=True, tri_face=False, sphere_mesh=False, subdivision_x=subdivision_x, subdivision_y=subdivision_y, mesh_size=2, mesh_size_x=2, mesh_size_y=2, random_seed=max(0, random_int()), water_plane=False, water_level=0.01, remove_double=False, show_main_settings=True, show_noise_settings=True, show_displace_settings=True, refresh=True, auto_refresh=True, **kwargs)

    if preset_name == LandTile.Canyon:
        strata = np.random.randint(6, 12)
        presets(noise_offset_x=0, noise_offset_y=-0.25, noise_offset_z=0, noise_size_x=1, noise_size_y=1.25, noise_size_z=1, noise_size=1.5, noise_type='marble_noise', basis_type='BLENDER', vl_basis_type='BLENDER', distortion=2, hard_noise='1', noise_depth=12, amplitude=0.5, frequency=2, dimension=1, lacunarity=2, offset=1, gain=1, marble_bias='0', marble_sharp='0', marble_shape='4', height=0.6, height_invert=False, height_offset=0, fx_mixfactor=0, fx_mix_mode='8', fx_type='20', fx_bias='0', fx_turb=0, fx_depth=3, fx_amplitude=0.5, fx_frequency=1.65, fx_size=1.5, fx_loc_x=3, fx_loc_y=2, fx_height=0.25, fx_invert=False, fx_offset=0.05, edge_falloff='2', falloff_x=4, falloff_y=4, edge_level=0.15, maximum=0.5, minimum=-0.2, vert_group="", strata=strata, strata_type='2')
    elif preset_name == LandTile.Canyons:
        strata = np.random.randint(2, 8)
        presets(noise_offset_x=0, noise_offset_y=0, noise_offset_z=0, noise_size_x=1, noise_size_y=1, noise_size_z=1, noise_size=0.5, noise_type='hetero_terrain', basis_type='PERLIN_NEW', vl_basis_type='CELLNOISE', distortion=1, hard_noise='0', noise_depth=8, amplitude=0.5, frequency=2, dimension=1.09, lacunarity=1.86, offset=0.77, gain=2, marble_bias='1', marble_sharp='0', marble_shape='7', height=0.5, height_invert=False, height_offset=-0, fx_mixfactor=0, fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=2, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=0.5, fx_invert=False, fx_offset=0, edge_falloff='3', falloff_x=8, falloff_y=8, edge_level=0, maximum=0.5, minimum=-0.5, vert_group="", strata=strata, strata_type='2')
    elif preset_name == LandTile.Cliff:
        presets(noise_offset_x=0, noise_offset_y=-0.88, noise_offset_z=3.72529e-09, noise_size_x=2, noise_size_y=2, noise_size_z=1, noise_size=1, noise_type='marble_noise', basis_type='VORONOI_F2F1', vl_basis_type='BLENDER', distortion=0.5, hard_noise='0', noise_depth=7, amplitude=0.5, frequency=2, dimension=1, lacunarity=2, offset=1, gain=1, marble_bias='0', marble_sharp='0', marble_shape='6', height=1.8, height_invert=False, height_offset=-0.15, fx_mixfactor=0, fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=2, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=0.5, fx_invert=False, fx_offset=0, edge_falloff='0', falloff_x=25, falloff_y=25, edge_level=0, maximum=1.25, minimum=0, vert_group="", strata=11, strata_type='0')
    elif preset_name == LandTile.Mesa:
        noise_size = np.random.uniform(0.5, 1)
        presets(noise_offset_x=0, noise_offset_y=0, noise_offset_z=0, noise_size_x=1, noise_size_y=1, noise_size_z=1, noise_size=noise_size, noise_type='shattered_hterrain', basis_type='VORONOI_F1', vl_basis_type='VORONOI_F2F1', distortion=1.15, hard_noise='1', noise_depth=8, amplitude=0.4, frequency=2, dimension=1, lacunarity=2, offset=1, gain=4, marble_bias='0', marble_sharp='0', marble_shape='0', height=0.5, height_invert=False, height_offset=0.2, fx_mixfactor=0, fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=1.5, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=0.5, fx_invert=False, fx_offset=0, edge_falloff='3', falloff_x=3, falloff_y=3, edge_level=0, maximum=0.25, minimum=0, vert_group="", strata=2.25, strata_type='2')
    elif preset_name == LandTile.River:
        presets(noise_offset_x=0, noise_offset_y=0, noise_offset_z=0, noise_size_x=1, noise_size_y=1, noise_size_z=1, noise_size=1, noise_type='marble_noise', basis_type='BLENDER', vl_basis_type='BLENDER', distortion=1, hard_noise='0', noise_depth=8, amplitude=0.5, frequency=2, dimension=1, lacunarity=2, offset=1, gain=1, marble_bias='2', marble_sharp='0', marble_shape='7', height=0.2, height_invert=False, height_offset=0, fx_mixfactor=0, fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=1.5, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=0.5, fx_invert=False, fx_offset=0, edge_falloff='0', falloff_x=40, falloff_y=40, edge_level=0, maximum=0.5, minimum=0, vert_group="", strata=1.25, strata_type='1')
    elif preset_name == LandTile.Volcano:
        presets(noise_offset_x=0, noise_offset_y=0, noise_offset_z=0, noise_size_x=1, noise_size_y=1, noise_size_z=1, noise_size=1, noise_type='marble_noise', basis_type='BLENDER', vl_basis_type='PERLIN_ORIGINAL', distortion=1.5, hard_noise='0', noise_depth=8, amplitude=0.5, frequency=1.8, dimension=1, lacunarity=2, offset=1, gain=2, marble_bias='2', marble_sharp='3', marble_shape='1', height=0.6, height_invert=False, height_offset=0, fx_mixfactor=0, fx_mix_mode='1', fx_type='14', fx_bias='0', fx_turb=0.5, fx_depth=2, fx_amplitude=0.38, fx_frequency=1.5, fx_size=1.15, fx_loc_x=-1, fx_loc_y=1, fx_height=0.5, fx_invert=False, fx_offset=0.06, edge_falloff='3', falloff_x=2, falloff_y=2, edge_level=0, maximum=1, minimum=-1, vert_group="", strata=5, strata_type='0')
    elif preset_name == LandTile.Mountain:
        presets(noise_offset_x=0, noise_offset_y=0, noise_offset_z=0, noise_size_x=1, noise_size_y=1, noise_size_z=1, noise_size=1, noise_type='hetero_terrain', basis_type='BLENDER', vl_basis_type='BLENDER', distortion=1, hard_noise='0', noise_depth=8, amplitude=0.5, frequency=2, dimension=1, lacunarity=2, offset=1, gain=1, marble_bias='0', marble_sharp='0', marble_shape='0', height=0.5, height_invert=False, height_offset=0, fx_mixfactor=0, fx_mix_mode='0', fx_type='0', fx_bias='0', fx_turb=0, fx_depth=0, fx_amplitude=0.5, fx_frequency=2, fx_size=1, fx_loc_x=0, fx_loc_y=0, fx_height=1, fx_invert=False, fx_offset=0, edge_falloff='3', falloff_x=4, falloff_y=4, edge_level=0, maximum=1, minimum=-1, vert_group="", strata=5, strata_type='0')


def ant_landscape_asset(
    folder,
    preset_name,
    tile_size,
    resolution,
    erosion=True,
    snowfall=True,
):
    Path(folder).mkdir(parents=True, exist_ok=True)
    N = 512
    create(preset_name, N, N)
    obj = bpy.context.active_object
    N = int(len(obj.data.vertices) ** 0.5)
    mverts_co = np.zeros((len(obj.data.vertices)*3), dtype=float)
    obj.data.vertices.foreach_get("co", mverts_co)
    mverts_co = mverts_co.reshape((N, N, 3))
    heightmap = cv2.resize(np.float32(mverts_co[..., -1]), (resolution, resolution)) * tile_size / 2
    if preset_name == LandTile.Mesa:
        heightmap *= 2
    heightmap = smooth(heightmap, 3)
    cv2.imwrite(str(folder/f'{AssetFile.Heightmap}.exr'), heightmap)
    bpy.data.objects.remove(obj, do_unlink=True)
    with open(folder/f'{AssetFile.TileSize}.txt', "w") as f:
        f.write(f"{tile_size}\n")
    if erosion: run_erosion(folder)
    if snowfall: run_snowfall(folder)
