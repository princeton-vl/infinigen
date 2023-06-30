import gin
from util.random import random_general as rg
@gin.configurable
def nishita_lighting(
    nw,
    cam,
    dust_density=("clip_gaussian", 1, 1, 0.1, 2),
    air_density=("clip_gaussian", 1, 0.2, 0.7, 1.3),
    sun_elevation=("spherical_sample", 10, None),
    dynamic=False,
    rising_angle=90,
    camera_based_rotation=None,
):
    sky_texture.sun_intensity = rg(sun_intensity)
    sky_texture.sun_elevation = np.radians(rg(sun_elevation))
    if camera_based_rotation is None:
        sky_texture.sun_rotation = np.random.uniform(0, 2 * math.pi)
    else:
        sky_texture.sun_rotation = 2 * math.pi - cam.parent.rotation_euler[2] + np.radians(camera_based_rotation)
    if dynamic:
        sky_texture.sun_rotation += (sky_texture.sun_elevation + np.radians(8)) / 2 * np.arctan(np.radians(rising_angle))
        sky_texture.keyframe_insert(data_path="sun_rotation", frame=bpy.context.scene.frame_end)
        sky_texture.sun_rotation -= (sky_texture.sun_elevation + np.radians(8)) * np.arctan(np.radians(rising_angle))
        sky_texture.keyframe_insert(data_path="sun_rotation", frame=bpy.context.scene.frame_start)

        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_end)
        sky_texture.sun_elevation = -np.radians(8)
        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_start)
        sky_texture.sun_elevation = -np.radians(5)
        sky_texture.keyframe_insert(data_path="sun_elevation", frame=bpy.context.scene.frame_start + 10)

    sky_texture.air_density =rg(air_density)
    sky_texture.dust_density = rg(dust_density)
def add_lighting(cam=None):
        surface = nishita_lighting(nw, cam)
@gin.configurable
    spot.data.energy = rg(energy)
    spot.data.spot_size = rg(spot_size)
