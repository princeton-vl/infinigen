from typing import NamedTuple

import procfunc as pf
from procfunc.nodes import types as t

from infinigen_v2.generators.shaders.composites import tiles
from infinigen_v2.generators.shaders.materials import (
    brick_concrete,
    ceramic,
    glass_colored,
    granite,
    marble,
    metal_brushed,
    metal_hammered,
    terrazzo,
)


class VaseResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def star_profile(
    resolution: t.SocketOrVal[int],
    points: t.SocketOrVal[int],
    inner_radius: t.SocketOrVal[float],
) -> t.ProcNode[pf.CurveObject]:
    curve_star = pf.nodes.geo.curve_star(
        points=points, inner_radius=inner_radius, outer_radius=1.0
    )
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_star.curve, count=resolution
    )
    return resample_curve_count


@pf.nodes.node_function
def flip_index(
    v_resolution: t.SocketOrVal[int],
    u_resolution: t.SocketOrVal[int],
) -> t.ProcNode[float]:
    input_index = pf.nodes.geo.input_index()
    result_0_a_0 = input_index.astype(dtype=float) % v_resolution.astype(dtype=float)
    result_0_b = pf.nodes.math.floor(
        input_index.astype(dtype=float) / v_resolution.astype(dtype=float)
    )
    add = result_0_a_0 * u_resolution.astype(dtype=float) + result_0_b
    return add


class CylinderSideResult(NamedTuple):
    geometry: pf.ProcNode
    top: pf.ProcNode[bool]
    side: pf.ProcNode[bool]
    bottom: pf.ProcNode[bool]


@pf.nodes.node_function
def cylinder_side(
    u_resolution: t.SocketOrVal[int],
    v_resolution: t.SocketOrVal[int],
) -> CylinderSideResult:
    top_side_segments = v_resolution.astype(dtype=float) - 1.0
    cylinder = pf.nodes.geo.mesh_cylinder(
        vertices=u_resolution,
        side_segments=top_side_segments.astype(dtype=int),
    )
    geo = pf.nodes.geo.store_named_attribute(
        geometry=cylinder.mesh,
        name="UVMap",
        value=cylinder.uv_map,
        domain="CORNER",
        data_type="FLOAT2",
    )
    captured = pf.nodes.geo.capture_attribute(
        geometry=geo, domain="FACE", top=cylinder.bottom
    )
    return CylinderSideResult(
        geometry=captured.geometry,
        top=captured.top,
        side=cylinder.side,
        bottom=cylinder.top,
    )


class LoftingResult(NamedTuple):
    geometry: pf.ProcNode
    top: pf.ProcNode[bool]
    side: pf.ProcNode[bool]
    bottom: pf.ProcNode[bool]


@pf.nodes.node_function
def lofting(
    profile_curves: t.ProcNode[pf.CurveObject],
    u_resolution: t.SocketOrVal[int],
    v_resolution: t.SocketOrVal[int],
    use_nurb: t.SocketOrVal[bool],
) -> LoftingResult:
    cylinder_side_result = cylinder_side(
        u_resolution=u_resolution, v_resolution=v_resolution
    )
    input_index = pf.nodes.geo.input_index()
    field_on_domain = pf.nodes.geo.field_on_domain(value=input_index, domain="CURVE")
    instance_on_points_selection = pf.nodes.func.equal(a=field_on_domain, b=0)
    curve_line = pf.nodes.geo.curve_line(start=(0, 0, 0), end=(0, 0, 1))
    attribute_domain_size = pf.nodes.geo.attribute_domain_size(
        geometry=profile_curves, component="CURVE"
    )
    resample_curve_count = pf.nodes.geo.resample_curve_count(
        curve=curve_line,
        count=attribute_domain_size.spline_count,
    )
    instance_on_points = pf.nodes.geo.instance_on_points(
        points=profile_curves,
        instance=resample_curve_count,
        selection=instance_on_points_selection,
    )
    realize_instances = pf.nodes.geo.realize_instances(instance_on_points)
    input_position = pf.nodes.geo.input_position()
    flip_index_result = flip_index(
        v_resolution=attribute_domain_size.spline_count,
        u_resolution=u_resolution,
    )
    sample_index = pf.nodes.geo.sample_index(
        geometry=profile_curves,
        index=flip_index_result.astype(dtype=int),
        value=input_position,
    )
    set_position_1 = pf.nodes.geo.set_position(
        geometry=realize_instances, position=sample_index
    )
    curve_spline_type = pf.nodes.geo.curve_spline_type(
        curve=set_position_1, spline_type="CATMULL_ROM"
    )
    curve_spline_type_1 = pf.nodes.geo.curve_spline_type(
        curve=set_position_1, spline_type="NURBS"
    )
    resample_curve = pf.nodes.func.switch(
        switch=use_nurb, a=curve_spline_type, b=curve_spline_type_1
    )
    resample_curve_count_1 = pf.nodes.geo.resample_curve_count(
        curve=resample_curve, count=v_resolution
    )
    input_position_1 = pf.nodes.geo.input_position()
    flip_index_result_1 = flip_index(
        v_resolution=u_resolution, u_resolution=v_resolution
    )
    sample_index_1 = pf.nodes.geo.sample_index(
        geometry=resample_curve_count_1,
        index=flip_index_result_1.astype(dtype=int),
        value=input_position_1,
    )
    set_position = pf.nodes.geo.set_position(
        geometry=cylinder_side_result.geometry,
        position=sample_index_1,
    )
    return LoftingResult(
        geometry=set_position,
        top=cylinder_side_result.top,
        side=cylinder_side_result.side,
        bottom=cylinder_side_result.bottom,
    )


@pf.nodes.node_function
def vase_profile(
    profile_curve: t.ProcNode[pf.CurveObject],
    height: t.SocketOrVal[float],
    diameter: t.SocketOrVal[float],
    top_scale: t.SocketOrVal[float],
    neck_mid_position: t.SocketOrVal[float],
    neck_position: t.SocketOrVal[float],
    neck_scale: t.SocketOrVal[float],
    shoulder_position: t.SocketOrVal[float],
    shoulder_thickness: t.SocketOrVal[float],
    foot_scale: t.SocketOrVal[float],
    foot_height: t.SocketOrVal[float],
) -> t.ProcNode[pf.CurveObject]:
    transform_translation = pf.nodes.math.combine_xyz(z=height)
    transform_scale = top_scale * diameter
    transform = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_translation,
        scale=transform_scale.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )
    transform_a_a_2 = pf.nodes.math.clamp(1.0 - neck_position)
    transform_a_2 = pf.nodes.math.multiply_add(
        a=transform_a_a_2,
        b=neck_mid_position,
        addend=neck_position,
    )
    transform_1_translation = pf.nodes.math.combine_xyz(z=transform_a_2 * height)
    transform_numerator = neck_scale + top_scale
    transform_1_scale = diameter * (transform_numerator / 2.0)
    transform_1 = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_1_translation,
        scale=transform_1_scale.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )
    transform_2_translation = pf.nodes.math.combine_xyz(z=height * neck_position)
    transform_2_scale = diameter * neck_scale
    transform_2 = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_2_translation,
        scale=transform_2_scale.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )

    join_1 = pf.nodes.geo.join_geometry([transform, transform_1, transform_2])

    transform_a_a_1 = pf.nodes.math.map_range(
        value=shoulder_position,
        to_max=neck_position,
        to_min=foot_height,
    )
    transform_a_a_0 = (neck_position - foot_height) * shoulder_thickness
    transform_a_1 = pf.nodes.math.minimum(
        a=transform_a_a_1 + transform_a_a_0, b=neck_position
    )
    transform_3_translation = pf.nodes.math.combine_xyz(z=transform_a_1 * height)
    transform_3 = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_3_translation,
        scale=diameter.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )
    transform_a_0 = pf.nodes.math.maximum(
        a=transform_a_a_1 - transform_a_a_0, b=foot_height
    )
    transform_4_translation = pf.nodes.math.combine_xyz(z=transform_a_0 * height)
    transform_4 = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_4_translation,
        scale=diameter.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )

    join_2 = pf.nodes.geo.join_geometry([transform_3, transform_4])

    transform_5_translation = pf.nodes.math.combine_xyz(z=foot_height * height)
    transform_5_scale = diameter * foot_scale
    transform_5 = pf.nodes.geo.transform(
        geometry=profile_curve,
        translation=transform_5_translation,
        scale=transform_5_scale.astype(dtype=pf.Vector),
        rotation=(0, 0, 0),
    )
    transform_6 = pf.nodes.geo.transform(
        geometry=profile_curve,
        scale=transform_5_scale.astype(dtype=pf.Vector),
        translation=(0, 0, 0),
        rotation=(0, 0, 0),
    )

    join_3 = pf.nodes.geo.join_geometry([transform_5, transform_6])
    join = pf.nodes.geo.join_geometry([join_1, join_2, join_3])
    return join


@pf.nodes.node_function
def vase(
    u_resolution: t.SocketOrVal[int],
    v_resolution: t.SocketOrVal[int],
    height: t.SocketOrVal[float],
    diameter: t.SocketOrVal[float],
    profile_inner_radius: t.SocketOrVal[float],
    profile_star_points: t.SocketOrVal[int],
    top_scale: t.SocketOrVal[float],
    neck_mid_position: t.SocketOrVal[float],
    neck_position: t.SocketOrVal[float],
    neck_scale: t.SocketOrVal[float],
    shoulder_position: t.SocketOrVal[float],
    shoulder_thickness: t.SocketOrVal[float],
    foot_scale: t.SocketOrVal[float],
    foot_height: t.SocketOrVal[float],
) -> t.ProcNode[pf.MeshObject]:
    profile = star_profile(
        resolution=u_resolution,
        points=profile_star_points,
        inner_radius=profile_inner_radius,
    )

    profile_geo = vase_profile(
        profile_curve=profile,
        height=height,
        diameter=diameter,
        top_scale=top_scale,
        neck_mid_position=neck_mid_position,
        neck_position=neck_position,
        neck_scale=neck_scale,
        shoulder_position=shoulder_position,
        shoulder_thickness=shoulder_thickness,
        foot_scale=foot_scale,
        foot_height=foot_height,
    )

    lofting_result = lofting(
        profile_curves=profile_geo,
        u_resolution=u_resolution,
        v_resolution=v_resolution,
        use_nurb=False,
    )

    geo = pf.nodes.geo.delete_geometry(
        geometry=lofting_result.geometry,
        selection=lofting_result.bottom,
        domain="FACE",
    )

    return geo


def vase_material_distribution(rng: pf.RNG, vec) -> pf.Material:
    def vase_tile(rng, vector: pf.Vector) -> pf.Material:
        grout_displacement = pf.random.uniform(rng, 0, 0.002)
        grout = brick_concrete.brick_concrete_grout_distribution(
            rng, vector, displacement_additional_height=-grout_displacement
        )
        scale = pf.random.uniform(rng, 15, 25)
        return tiles.tile_indoor_wall_distribution(
            rng, vector, grout=grout, scale=scale
        )

    def vase_hammered(rng, _vector: pf.Vector) -> pf.Material:
        return metal_hammered.metal_hammered_distribution(
            rng, pf.nodes.shader.coord().generated
        )

    material_func = pf.control.choice(
        rng,
        [
            (ceramic.ceramic_distribution, 1.0),
            (metal_brushed.metal_brushed_linear_distribution, 1.0),
            (vase_hammered, 1.0),
            (vase_tile, 2.0),
            (terrazzo.terrazzo_multicolor_distribution, 2.0),
            (marble.marble_distribution, 2.0),
            (glass_colored.glass_colored_distribution, 1.0),
            (granite.granite_smooth_distribution, 1.0),
        ],
    )
    return material_func(rng, vec)


def vase_distribution(rng: pf.RNG) -> VaseResult:
    z = pf.random.uniform(rng, 0.17, 0.5)
    x = z * pf.random.uniform(rng, 0.3, 0.6)

    u_resolution = 64
    v_resolution = 64
    neck_scale = pf.random.uniform(rng, 0.2, 0.8)

    profile_inner_radius = pf.control.choice(
        rng, [(1.0, 0.5), (pf.random.uniform(rng, 0.8, 1.0), 0.5)]
    )
    profile_star_points = pf.random.randint(rng, 16, u_resolution // 2 + 1)
    top_scale = neck_scale * pf.random.uniform(rng, 0.8, 1.2)
    neck_mid_position = pf.random.uniform(rng, 0.7, 0.95)
    neck_position = 0.5 * neck_scale + 0.5 + pf.random.uniform(rng, -0.05, 0.05)
    shoulder_position = pf.random.uniform(rng, 0.3, 0.7)
    shoulder_thickness = pf.random.uniform(rng, 0.1, 0.25)
    foot_scale = pf.random.uniform(rng, 0.4, 0.6)
    foot_height = pf.random.uniform(rng, 0.01, 0.1)

    geo = vase(
        u_resolution=u_resolution,
        v_resolution=v_resolution,
        height=z,
        diameter=x,
        profile_inner_radius=profile_inner_radius,
        profile_star_points=profile_star_points,
        top_scale=top_scale,
        neck_mid_position=neck_mid_position,
        neck_position=neck_position,
        neck_scale=neck_scale,
        shoulder_position=shoulder_position,
        shoulder_thickness=shoulder_thickness,
        foot_scale=foot_scale,
        foot_height=foot_height,
    )

    uv = pf.nodes.shader.coord().uv
    mat_result = vase_material_distribution(rng, uv)
    geo = pf.nodes.geo.set_material(geo, mat_result)

    obj = pf.nodes.to_mesh_object(geo)

    thickness = pf.random.uniform(rng, 0.005, 0.01)
    pf.ops.modifier.solidify(obj, thickness=thickness)
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)

    return VaseResult(mesh=obj)
