# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lingjie Mei: original Infinigen v1 TVFactory / MonitorFactory (infinigen/assets/objects/appliances/tv.py)
# - Alexander Raistrick: port to procfunc/v2

from math import pi, sqrt
from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.shaders.base_materials import metal_brushed, plastic
from infinigen2.shaders.masks import graphicdesign
from infinigen2.util import mesh as mesh_util
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "MonitorResult",
    "monitor",
    "monitor_body",
    "monitor_rand",
    "monitor_tv_rand",
    "pedestal_stand",
    "screen_dimensions_rand",
    "screen_emissive_material_rand",
    "screen_material_rand",
    "screen_off_material_rand",
]


class MonitorResult(NamedTuple):
    mesh: pf.MeshObject


def screen_dimensions_rand(rng: pf.RNG, diagonal: float | None = None) -> pf.Vector:
    """Active screen area in meters, ordered depth, width, height."""
    rng_diagonal, rng_params = rng.spawn(2)
    if diagonal is None:
        diagonal = pf.random.uniform(rng_diagonal, 21.5 * 0.0254, 34.0 * 0.0254)
    aspect_bias = pf.random.uniform(rng_params, 0.0, 1.0) ** 3
    aspect = 16.0 / 9.0 + (64.0 / 27.0 - 16.0 / 9.0) * aspect_bias
    depth = pf.random.uniform(rng_params, 0.008, 0.025)
    height = diagonal / sqrt(1.0 + aspect * aspect)
    return pf.Vector((depth, height * aspect, height))


@pf.nodes.node_function
def monitor_body(
    screen_dimensions: t.SocketOrVal[pf.Vector],
    bezel_width: t.SocketOrVal[float],
    chin_height: t.SocketOrVal[float],
    screen_bottom: t.SocketOrVal[float],
    screen_recess: t.SocketOrVal[float],
    housing_size: t.SocketOrVal[pf.Vector],
    housing_center_z: t.SocketOrVal[float],
    material: t.SocketOrVal[pf.Material],
    screen_material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    """Swept bezel frame around a recessed screen and chin, plus a back housing; bezel front at x=0."""
    depth = screen_dimensions.x
    inner_height = screen_dimensions.z + chin_height

    centerline = pf.nodes.geo.curve_quadrilateral(
        width=screen_dimensions.y + bezel_width, height=inner_height + bezel_width
    )
    # 90-degree miters project radial width by 1/sqrt(2); widen so rails stay bezel_width
    profile = pf.nodes.geo.curve_quadrilateral(width=bezel_width * 2**0.5, height=depth)
    frame = curve_to_mesh_with_uv(centerline, profile).mesh
    frame = pf.nodes.geo.transform(
        geometry=frame, translation=pf.nodes.math.combine_xyz(z=depth * -0.5)
    )
    back = mesh_util.box(
        size=pf.nodes.math.combine_xyz(
            screen_dimensions.y, inner_height, depth - screen_recess
        ),
        location=pf.nodes.math.combine_xyz(z=-screen_recess),
        anchor=(0.5, 0.5, 1.0),
    )
    shell = pf.nodes.geo.join_geometry([frame, back])
    shell = pf.nodes.geo.set_material(geometry=shell, material=material)

    grid = pf.nodes.geo.mesh_grid(
        size_x=screen_dimensions.y,
        size_y=screen_dimensions.z,
        vertices_x=2,
        vertices_y=2,
    )
    screen = pf.nodes.geo.store_named_attribute(
        geometry=grid.mesh,
        name="UVMap",
        value=pf.nodes.math.combine_xyz(screen_dimensions.y, screen_dimensions.z)
        * grid.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    screen = pf.nodes.geo.transform(
        geometry=screen,
        translation=pf.nodes.math.combine_xyz(
            y=chin_height * 0.5, z=0.0002 - screen_recess
        ),
    )
    screen = pf.nodes.geo.set_material(geometry=screen, material=screen_material)

    panel = pf.nodes.geo.store_named_attribute(
        geometry=pf.nodes.geo.join_geometry([shell, screen]),
        name="crease_edge",
        value=1.0,
        domain="EDGE",
    )
    panel = pf.nodes.geo.transform(
        geometry=panel,
        rotation=(pi / 2, 0.0, pi / 2),
        translation=pf.nodes.math.combine_xyz(
            z=screen_bottom + bezel_width + inner_height * 0.5
        ),
    )

    housing = mesh_util.box_with_support_loops(
        size=housing_size,
        location=pf.nodes.math.combine_xyz(0.001 - depth, 0.0, housing_center_z),
        anchor=(1.0, 0.5, 0.5),
        support_loop_offset=pf.nodes.math.combine_xyz(
            housing_size.x * 0.2, housing_size.x * 0.5, housing_size.x * 0.5
        ),
    )
    housing = pf.nodes.geo.set_shade_smooth(geometry=housing, shade_smooth=True)
    housing = pf.nodes.geo.set_material(geometry=housing, material=material)
    return pf.nodes.geo.join_geometry([panel, housing])


@pf.nodes.node_function
def pedestal_stand(
    neck_size: t.SocketOrVal[pf.Vector],
    neck_front_x: t.SocketOrVal[float],
    foot_size: t.SocketOrVal[pf.Vector],
    material: t.SocketOrVal[pf.Material],
) -> pf.ProcNode[pf.MeshObject]:
    """A rectangular base and upright rear stem, resting on z=0."""
    foot_thickness = foot_size.z
    foot = mesh_util.box_with_support_loops(
        size=foot_size,
        location=pf.nodes.math.combine_xyz(neck_front_x - neck_size.x * 0.5, 0.0, 0.0),
        anchor=(0.5, 0.5, 0.0),
        support_loop_offset=pf.nodes.math.combine_xyz(
            foot_thickness * 0.2, foot_thickness * 0.2, foot_thickness * 0.2
        ),
    )
    neck = mesh_util.box_with_support_loops(
        size=neck_size,
        location=pf.nodes.math.combine_xyz(neck_front_x, 0.0, foot_thickness * 0.5),
        anchor=(1.0, 0.5, 0.0),
        support_loop_offset=pf.nodes.math.combine_xyz(
            neck_size.x * 0.2, neck_size.x * 0.2, neck_size.x * 0.2
        ),
    )
    stand = pf.nodes.geo.join_geometry([neck, foot])
    stand = pf.nodes.geo.set_shade_smooth(geometry=stand, shade_smooth=True)
    return pf.nodes.geo.set_material(geometry=stand, material=material)


def screen_emissive_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    """A switched-on screen showing art content."""
    rng_content, rng_params = rng.spawn(2)
    content = graphicdesign.art_rand(rng_content, vector)
    roughness = pf.random.uniform(rng_params, 0.05, 0.3)
    emission_strength = pf.random.uniform(rng_params, 0.8, 2.5)
    surface = pf.nodes.shader.principled_bsdf(
        base_color=pf.Color((0.01, 0.01, 0.01)),
        roughness=roughness,
        emission_color=content,
        emission_strength=emission_strength,
    )
    return pf.Material(
        surface=surface, displacement=pf.nodes.math.constant((0.0, 0.0, 0.0))
    )


def screen_off_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    """A switched-off screen: near-black glossy glass that mirrors the room."""
    value = pf.random.uniform(rng, 0.003, 0.02)
    roughness = pf.random.uniform(rng, 0.01, 0.12)
    specular_ior_level = pf.random.uniform(rng, 0.5, 0.9)
    surface = pf.nodes.shader.principled_bsdf(
        base_color=pf.color.hsv_color(hue=0.0, saturation=0.0, value=value),
        roughness=roughness,
        specular_ior_level=specular_ior_level,
    )
    return pf.Material(
        surface=surface, displacement=pf.nodes.math.constant((0.0, 0.0, 0.0))
    )


def screen_material_rand(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
) -> pf.Material:
    rng_choice, rng_material = rng.spawn(2)
    material_fn = pf.control.choice(
        rng_choice,
        [(screen_emissive_material_rand, 2.0), (screen_off_material_rand, 1.0)],
    )
    return material_fn(rng_material, vector)


@pf.tracer.generator
def monitor(
    screen_dimensions: pf.Vector,
    bezel_width: float,
    chin_height: float,
    screen_bottom: float,
    screen_recess: float,
    housing_size: pf.Vector,
    housing_center_z: float,
    neck_size: pf.Vector,
    neck_front_x: float,
    foot_size: pf.Vector,
    material: pf.Material,
    screen_material: pf.Material,
    stand_material: pf.Material,
) -> MonitorResult:
    body = monitor_body(
        screen_dimensions=screen_dimensions,
        bezel_width=bezel_width,
        chin_height=chin_height,
        screen_bottom=screen_bottom,
        screen_recess=screen_recess,
        housing_size=housing_size,
        housing_center_z=housing_center_z,
        material=material,
        screen_material=screen_material,
    )
    stand = pedestal_stand(
        neck_size=neck_size,
        neck_front_x=neck_front_x,
        foot_size=foot_size,
        material=stand_material,
    )
    obj = pf.nodes.to_mesh_object(pf.nodes.geo.join_geometry([body, stand]))
    obj.item().name = monitor.__name__
    return MonitorResult(mesh=obj)


def monitor_rand(
    rng: pf.RNG,
    screen_dimensions: pf.Vector | None = None,
    material: pf.Material | None = None,
    screen_material: pf.Material | None = None,
    stand_material: pf.Material | None = None,
) -> MonitorResult:
    """A flat-panel display on a pedestal stand, screen facing +X, resting on z=0."""
    (
        rng_params,
        rng_dimensions,
        rng_material,
        rng_screen_material,
        rng_stand_choice,
        rng_stand_material,
        rng_stand,
    ) = rng.spawn(7)
    if screen_dimensions is None:
        screen_dimensions = screen_dimensions_rand(rng_dimensions)
    bezel_width = pf.random.uniform(rng_params, 0.002, 0.012)
    chin_height = 0.025 * pf.random.uniform(rng_params, 0.0, 1.0) ** 2
    screen_bottom = pf.random.uniform(rng_params, 0.04, 0.15)
    housing_depth = pf.random.uniform(rng_params, 0.015, 0.05)
    screen_recess = pf.random.uniform(rng_params, 0.0, 0.002)
    outer_width = screen_dimensions.y + 2.0 * bezel_width
    outer_height = screen_dimensions.z + 2.0 * bezel_width + chin_height
    housing_width = outer_width * pf.random.uniform(rng_params, 0.4, 0.85)
    housing_height = outer_height * pf.random.uniform(rng_params, 0.4, 0.75)
    housing_center_z = screen_bottom + outer_height * pf.random.uniform(
        rng_params, 0.4, 0.55
    )

    vector = pf.nodes.shader.coord().uv
    if material is None:
        material = plastic.plastic_grayscale_rand(rng_material, vector)
    if screen_material is None:
        tile_size = min(screen_dimensions.y, screen_dimensions.z)
        screen_vector = vector / pf.nodes.math.combine_xyz(tile_size, tile_size, 1.0)
        screen_material = screen_material_rand(rng_screen_material, screen_vector)
    stand_options = [
        (lambda _r, _v: material, 2.0),
        (metal_brushed.metal_brushed_linear_rand, 1.0),
    ]
    if stand_material is None:
        stand_fn = pf.control.choice(rng_stand_choice, stand_options)
        stand_material = stand_fn(rng_stand_material, vector)

    housing_size = pf.Vector((housing_depth, housing_width, housing_height))
    back_x = 0.001 - screen_dimensions.x - housing_depth
    stand_thickness = pf.random.uniform(rng_stand, 0.007, 0.012)
    foot_depth = (0.14 + outer_width * 0.1) * pf.random.uniform(rng_stand, 0.9, 1.1)
    foot_width = outer_width * pf.random.uniform(rng_stand, 0.30, 0.40)
    neck_depth = pf.random.uniform(rng_stand, 0.018, 0.028)
    neck_width = outer_width * pf.random.uniform(rng_stand, 0.06, 0.09)
    neck_size = pf.Vector(
        (neck_depth, neck_width, housing_center_z - stand_thickness * 0.5)
    )
    foot_size = pf.Vector((foot_depth, foot_width, stand_thickness))
    obj = monitor(
        screen_dimensions=screen_dimensions,
        bezel_width=bezel_width,
        chin_height=chin_height,
        screen_bottom=screen_bottom,
        screen_recess=screen_recess,
        housing_size=housing_size,
        housing_center_z=housing_center_z,
        neck_size=neck_size,
        neck_front_x=back_x + 0.005,
        foot_size=foot_size,
        material=material,
        screen_material=screen_material,
        stand_material=stand_material,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return MonitorResult(mesh=obj)


def monitor_tv_rand(rng: pf.RNG) -> MonitorResult:
    """A flat-panel TV on a pedestal stand, screen facing +X, resting on z=0."""
    rng_diagonal, rng_dimensions, rng_monitor = rng.spawn(3)
    diagonal = pf.random.clip_gaussian(rng_diagonal, 50.0, 20.0, 24.0, 98.0) * 0.0254
    screen_dimensions = screen_dimensions_rand(rng_dimensions, diagonal=diagonal)
    return monitor_rand(rng_monitor, screen_dimensions=screen_dimensions)
