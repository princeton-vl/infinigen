# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import procfunc as pf

from infinigen2.curves.skirting_board_profile import skirting_profile_rand
from infinigen2.shaders.functionality_lists import skirt_material_rand
from infinigen2.util.curve import curve_to_mesh_with_uv

__all__ = [
    "skirting_on_walls_rand",
    "skirting_rand",
]


def _skirting_path_curve(
    joined: pf.ProcNode,
    selection: pf.ProcNode,
    outward: pf.ProcNode,
    up_sign: float,
) -> pf.ProcNode:
    curve = pf.nodes.geo.mesh_to_curve(joined, selection=selection)
    # unify winding vs the wall normal: profile up for floor, mirrored down for ceiling
    chirality = pf.nodes.math.vector_cross_product(
        a=pf.nodes.geo.input_tangent(), b=outward
    ).z
    curve = pf.nodes.geo.reverse_curve(curve, selection=chirality * up_sign > 0.0)
    # fillet rounds corner junctions; resample drops degenerate T-junction points
    curve = pf.nodes.geo.fillet_curve_poly(
        curve, radius=0.02, limit_radius=True, count=2
    )
    curve = pf.nodes.geo.resample_curve_length(curve, length=0.04)
    return pf.nodes.geo.set_curve_normal(curve, normal=outward, mode="FREE")


@pf.tracer.grammar
def skirting_on_walls_rand(
    rng: pf.RNG,
    walls: list[pf.MeshObject],
    material: pf.Material,
    profile_curve: pf.CurveObject | None = None,
) -> list[pf.MeshObject]:
    rng_profile, rng_placement = rng.spawn(2)
    if profile_curve is None:
        profile_curve = skirting_profile_rand(rng_profile)
    profile_curve_geo = pf.nodes.geo.object_info(profile_curve).geometry
    profile_y = pf.nodes.geo.input_position().y
    profile_stat = pf.nodes.geo.attribute_statistic(
        geometry=profile_curve_geo, attribute=profile_y
    )
    profile_height = profile_stat.max - profile_stat.min
    # close the silhouette so fill_caps can seal the cut ends at door gaps
    profile_curve_geo = pf.nodes.geo.set_spline_cyclic(profile_curve_geo, cyclic=True)

    # follow the bottom/top boundary edges of the wall meshes
    wall_geos = [
        pf.nodes.geo.transform(
            pf.nodes.geo.object_info(w).geometry,
            translation=tuple(w.item().location),
            rotation=tuple(w.item().rotation_euler),
        )
        for w in walls
    ]
    joined = pf.nodes.geo.join_geometry(wall_geos)
    # weld coincident verts so boundary slivers collapse
    wall_merge_distance = 0.005
    joined = pf.nodes.geo.merge_by_distance(joined, distance=wall_merge_distance)

    # outward wall normal (swept depth is -normal): stable per-point across splits
    cap = pf.nodes.geo.capture_attribute(
        joined, domain="POINT", wall_outward=-pf.nodes.geo.input_normal()
    )
    joined = cap.geometry

    position_z = pf.nodes.geo.input_position().z
    z_stat = pf.nodes.geo.attribute_statistic(geometry=joined, attribute=position_z)
    # horizontal edges with both endpoints at the extreme
    edge_v = pf.nodes.geo.input_mesh_edge_vertices()
    floor_z = z_stat.min + 0.02
    ceil_z = z_stat.max - 0.02
    near_bottom = pf.nodes.func.boolean_and(
        a=edge_v.position_1.z < floor_z, b=edge_v.position_2.z < floor_z
    )
    near_top = pf.nodes.func.boolean_and(
        a=edge_v.position_1.z > ceil_z, b=edge_v.position_2.z > ceil_z
    )

    # drop short jamb-chamfer sliver edges that tilt the fill_caps door-gap ends
    long_edge = (
        pf.nodes.math.vector_distance(edge_v.position_1, edge_v.position_2) > 0.02
    )
    near_bottom = pf.nodes.func.boolean_and(a=near_bottom, b=long_edge)
    near_top = pf.nodes.func.boolean_and(a=near_top, b=long_edge)

    # drop floor edges where the wall does not back the skirting profile
    mid = (edge_v.position_1 + edge_v.position_2) * 0.5
    profile_top = pf.nodes.math.combine_xyz(
        x=mid.x, y=mid.y, z=z_stat.min + profile_height
    )
    wall_prox = pf.nodes.geo.proximity(geometry=joined, sample_position=profile_top)
    has_wall_backing = wall_prox.distance < wall_merge_distance
    near_bottom = pf.nodes.func.boolean_and(a=near_bottom, b=has_wall_backing)

    floor_curve_node = _skirting_path_curve(
        joined, near_bottom, cap.wall_outward, up_sign=1.0
    )
    ceiling_curve_node = _skirting_path_curve(
        joined, near_top, cap.wall_outward, up_sign=-1.0
    )

    geoms = pf.control.choice(
        rng_placement,
        [
            ([floor_curve_node], 3.0),
            ([ceiling_curve_node], 1.0),
            ([floor_curve_node, ceiling_curve_node], 2.0),
        ],
    )
    path_curves = pf.nodes.geo.join_geometry(geoms)

    skirt = curve_to_mesh_with_uv(path_curves, profile_curve_geo, fill_caps=True).mesh
    skirt = pf.nodes.geo.flip_faces(skirt)
    skirt = pf.nodes.to_mesh_object(skirt)
    skirt.item().name = "skirting"

    pf.ops.object.set_material(
        skirt,
        surface=material.surface,
        displacement=material.displacement,
    )

    return [skirt]


@pf.tracer.grammar
def skirting_rand(
    rng: pf.RNG,
    walls: list[pf.MeshObject],
) -> list[pf.MeshObject]:
    vec_wall = pf.nodes.shader.coord().uv
    rng_mat, rng_choice, rng_skirt = rng.spawn(3)
    skirt_mat = skirt_material_rand(rng_mat, vec_wall)
    skirt_option = pf.control.choice(
        rng_choice,
        [(skirting_on_walls_rand, 0.85), (lambda *_, **__: [], 0.15)],
    )
    skirts = skirt_option(
        rng_skirt,
        walls=walls,
        material=skirt_mat,
    )
    return skirts
