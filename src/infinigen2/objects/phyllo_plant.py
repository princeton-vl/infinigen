# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alejandro Newell: original Infinigen phyllotactic plant (https://github.com/princeton-vl/infinigen/blob/05a09759fe9478595a3323ec2d6e26ce3513223f/infinigen/assets/objects/trees/utils/geometrynodes.py)
# - Alexander Raistrick: refactor for Infinigen2

import math
from typing import NamedTuple

import numpy as np
import procfunc as pf
from procfunc.nodes import types as t

from infinigen2.objects import leaf, leaf_broadleaf
from infinigen2.shaders.base_materials import leaf as leaf_shader
from infinigen2.util import curve as curve_util

__all__ = [
    "PhylloResult",
    "PhylloPlantResult",
    "phyllo_points",
    "phyllo_dist",
    "phyllo_leaf_count",
    "phyllo_plant",
    "phyllo_plant_mesh",
    "plant_phyllo_basal_rand",
    "plant_phyllo_ascending_rand",
]


class PhylloResult(NamedTuple):
    points: pf.ProcNode
    rotation: pf.ProcNode[pf.Vector]
    scale: pf.ProcNode[float]


class PhylloPlantResult(NamedTuple):
    mesh: pf.MeshObject


@pf.nodes.node_function
def phyllo_points(
    count: t.SocketOrVal[int] = 50,
    max_radius: t.SocketOrVal[float] = 2.0,
    radius_exp: t.SocketOrVal[float] = 0.5,
    inner_pct: t.SocketOrVal[float] = 0.0,
    min_angle: t.SocketOrVal[float] = -0.5236,
    max_angle: t.SocketOrVal[float] = 0.7854,
    min_scale: t.SocketOrVal[float] = 0.3,
    max_scale: t.SocketOrVal[float] = 0.3,
    min_z: t.SocketOrVal[float] = 0.0,
    max_z: t.SocketOrVal[float] = 1.0,
    clamp_z: t.SocketOrVal[float] = 1.0,
    yaw_offset: t.SocketOrVal[float] = -1.5707964,
    seed: t.SocketOrVal[int] = 0,
) -> PhylloResult:
    line = pf.nodes.geo.mesh_line(
        start_location=(0.0, 0.0, 0.0), offset=(0.0, 0.0, 1.0), count=count
    )
    index = pf.nodes.geo.input_index()
    fraction = index.astype(dtype=float) / count.astype(dtype=float)
    azimuth = index.astype(dtype=float) * 2.3998
    radial_fraction = pf.nodes.math.map_range(
        value=fraction**radius_exp, to_min=inner_pct
    )
    radius = radial_fraction * max_radius
    height = pf.nodes.math.map_range(
        value=fraction, from_max=clamp_z, to_min=min_z, to_max=max_z
    )
    position = pf.nodes.math.combine_xyz(
        x=pf.nodes.math.cos(azimuth) * radius,
        y=pf.nodes.math.sin(azimuth) * radius,
        z=height,
    )
    points = pf.nodes.geo.set_position(geometry=line, position=position)
    statistics = pf.nodes.geo.attribute_statistic(
        geometry=line, attribute=radial_fraction
    )
    elevation = pf.nodes.math.map_range(
        value=radial_fraction,
        from_min=statistics.max,
        from_max=statistics.min,
        to_min=min_angle,
        to_max=max_angle,
    )
    roll = pf.nodes.func.random_value(min=-0.1, max=0.1, seed=seed)
    rotation = pf.nodes.math.combine_xyz(x=elevation, y=roll, z=azimuth + yaw_offset)
    scale = pf.nodes.func.random_value(min=min_scale, max=max_scale, seed=seed)
    return PhylloResult(points, rotation, scale)


@pf.nodes.node_function
def phyllo_dist(
    geometry: pf.ProcNode,
    count: t.SocketOrVal[int] = 50,
    max_radius: t.SocketOrVal[float] = 2.0,
    radius_exp: t.SocketOrVal[float] = 0.5,
    inner_pct: t.SocketOrVal[float] = 0.0,
    min_angle: t.SocketOrVal[float] = -0.5236,
    max_angle: t.SocketOrVal[float] = 0.7854,
    min_scale: t.SocketOrVal[float] = 0.3,
    max_scale: t.SocketOrVal[float] = 0.3,
    min_z: t.SocketOrVal[float] = 0.0,
    max_z: t.SocketOrVal[float] = 1.0,
    clamp_z: t.SocketOrVal[float] = 1.0,
    yaw_offset: t.SocketOrVal[float] = -1.5707964,
) -> pf.ProcNode[t.Instances]:
    layout = phyllo_points(
        count=count,
        max_radius=max_radius,
        radius_exp=radius_exp,
        inner_pct=inner_pct,
        min_angle=min_angle,
        max_angle=max_angle,
        min_scale=min_scale,
        max_scale=max_scale,
        min_z=min_z,
        max_z=max_z,
        clamp_z=clamp_z,
        yaw_offset=yaw_offset,
    )
    return pf.nodes.geo.instance_on_points(
        points=layout.points,
        instance=geometry,
        rotation=layout.rotation.astype(dtype=pf.Euler),
        scale=layout.scale.astype(dtype=pf.Vector),
    )


def _stem_curve(
    height: t.SocketOrVal[float],
    curve_x_degrees: t.SocketOrVal[float],
    curve_y_degrees: t.SocketOrVal[float],
    warble: t.SocketOrVal[float],
    handles: t.SocketOrVal[int],
    seed: t.SocketOrVal[int],
) -> pf.ProcNode:
    line = pf.nodes.geo.mesh_line(
        start_location=(0.0, 0.0, 0.0),
        offset=pf.nodes.math.combine_xyz(
            z=height / (handles.astype(dtype=float) - 1.0)
        ),
        count=handles,
    )
    fraction = pf.nodes.geo.input_position().z / height
    noise = pf.nodes.func.random_value(
        min=(-1.0, -1.0, 0.0),
        max=(1.0, 1.0, 0.0),
        id=pf.nodes.geo.input_index(),
        seed=seed,
    )
    bend = pf.nodes.math.combine_xyz(
        x=pf.nodes.math.tan(curve_y_degrees * math.pi / 180.0),
        y=-pf.nodes.math.tan(curve_x_degrees * math.pi / 180.0),
    )
    offset = bend * (height * fraction**2.0 * 0.5)
    offset = offset + noise * (
        height * warble * pf.nodes.math.sin(fraction * math.pi) ** 2.0
    )
    line = pf.nodes.geo.set_position(line, offset=offset)
    curve = pf.nodes.geo.mesh_to_curve(line)
    curve = pf.nodes.geo.curve_spline_type(curve, spline_type="BEZIER")
    curve = pf.nodes.geo.curve_set_handles(curve, handle_type="AUTO")
    return pf.nodes.geo.set_spline_resolution(curve, resolution=8)


@pf.nodes.node_function
def _warp_leaves(
    geometry: pf.ProcNode,
    local: t.SocketOrVal[pf.Vector],
    leaf_id: t.SocketOrVal[int],
    rotation: t.SocketOrVal[pf.Vector],
    scale: t.SocketOrVal[float],
    petiole: t.SocketOrVal[bool],
    width: t.SocketOrVal[float],
    leaf_size: t.SocketOrVal[float],
    bend: t.SocketOrVal[float],
    droop: t.SocketOrVal[float],
    twist_degrees: t.SocketOrVal[float],
    warble: t.SocketOrVal[float],
    seed: t.SocketOrVal[int],
) -> pf.ProcNode:
    coefficients = pf.nodes.func.random_value(
        min=(-1.0, -1.0, -1.0),
        max=(1.0, 1.0, 1.0),
        id=leaf_id,
        seed=seed,
    )
    fraction = pf.nodes.math.clamp(local.y, min=0.0, max=1.0)
    petiole = petiole.astype(dtype=float)
    twisted = pf.nodes.math.vector_rotate_axis_angle(
        local,
        axis=(0.0, 1.0, 0.0),
        angle=coefficients.y
        * twist_degrees
        * math.pi
        / 180.0
        * fraction
        * (1.0 - petiole),
    )
    warp = (
        pf.nodes.math.combine_xyz(
            x=coefficients.z * warble * pf.nodes.math.sin(fraction * math.pi * 2.0),
            z=coefficients.x * bend * fraction,
        )
        * fraction
    )
    delta = twisted + warp - local
    delta = delta * pf.nodes.math.combine_xyz(x=scale * width, y=scale, z=scale)
    delta = pf.nodes.math.vector_rotate_euler(delta, rotation=rotation)
    gravity = -droop * scale * leaf_size * fraction**2.0
    delta = delta + pf.nodes.math.combine_xyz(z=gravity)
    return pf.nodes.geo.set_position(geometry, offset=delta)


@pf.nodes.node_function
def _leaf_with_petiole(
    blade: t.SocketOrVal[pf.MeshObject],
    material: t.SocketOrVal[pf.Material],
    width: t.SocketOrVal[float],
    length_fraction: t.SocketOrVal[float],
    diameter_ratio: t.SocketOrVal[float],
) -> pf.ProcNode:
    direct = blade
    blade = pf.nodes.geo.transform(
        blade,
        translation=pf.nodes.math.combine_xyz(y=length_fraction),
        scale=pf.nodes.math.combine_xyz(
            x=1.0 - length_fraction, y=1.0 - length_fraction, z=1.0 - length_fraction
        ),
    )
    path = pf.nodes.geo.curve_line(
        start=(0.0, 0.0, 0.0),
        end=pf.nodes.math.combine_xyz(y=pf.nodes.math.maximum(length_fraction, 0.001)),
    )
    path = pf.nodes.geo.resample_curve_count(path, count=9)
    profile = pf.nodes.geo.curve_circle(resolution=12, radius=diameter_ratio * 0.5)
    tube = curve_util.curve_to_mesh_with_uv(path, profile, fill_caps=True).mesh
    tube = pf.nodes.geo.transform(
        tube, scale=pf.nodes.math.combine_xyz(x=1.0 / width, y=1.0, z=1.0)
    )
    end_distance = pf.nodes.math.minimum(
        pf.nodes.geo.input_position().y,
        length_fraction - pf.nodes.geo.input_position().y,
    )
    tube = pf.nodes.geo.store_named_attribute(
        tube,
        name="crease_edge",
        domain="EDGE",
        value=(end_distance < 0.00001).astype(dtype=float),
    )
    tube = pf.nodes.geo.set_material(tube, material=material)
    stalked = pf.nodes.geo.join_geometry([blade, tube])
    return pf.nodes.func.switch(switch=length_fraction > 0.0, a=direct, b=stalked)


@pf.nodes.node_function
def _plant_geometry(
    first: t.SocketOrVal[pf.MeshObject],
    stem_material: t.SocketOrVal[pf.Material],
    count: t.SocketOrVal[int],
    height: t.SocketOrVal[float],
    radius: t.SocketOrVal[float],
    leaf_length: t.SocketOrVal[float],
    leaf_size: t.SocketOrVal[float],
    leaf_width: t.SocketOrVal[float],
    petiole_length_fraction: t.SocketOrVal[float],
    petiole_diameter_ratio: t.SocketOrVal[float],
    density: t.SocketOrVal[float],
    target_count: t.SocketOrVal[float],
    sparse: t.SocketOrVal[float],
    count_override: t.SocketOrVal[bool],
    base_fraction: t.SocketOrVal[float],
    stem_curve_x_degrees: t.SocketOrVal[float],
    stem_curve_y_degrees: t.SocketOrVal[float],
    stem_warble: t.SocketOrVal[float],
    stem_handles: t.SocketOrVal[int],
    leaf_bend: t.SocketOrVal[float],
    leaf_droop: t.SocketOrVal[float],
    leaf_twist_degrees: t.SocketOrVal[float],
    leaf_warble: t.SocketOrVal[float],
    min_angle: t.SocketOrVal[float],
    max_angle: t.SocketOrVal[float],
    phase: t.SocketOrVal[float],
    seed: t.SocketOrVal[int],
) -> pf.ProcNode:
    layout = phyllo_points(
        count=count,
        max_radius=radius,
        inner_pct=0.8,
        min_angle=min_angle,
        max_angle=max_angle,
        min_scale=leaf_length * leaf_size * 0.55,
        max_scale=leaf_length * leaf_size * 1.45,
        min_z=height * base_fraction,
        max_z=height * 0.98,
        clamp_z=(count.astype(dtype=float) - 1.0) / count.astype(dtype=float),
        yaw_offset=phase - math.pi / 2,
        seed=seed,
    )
    points = pf.nodes.geo.transform(
        layout.points,
        rotation=pf.nodes.math.combine_xyz(z=phase).astype(dtype=pf.Euler),
    )
    stem_curve = _stem_curve(
        height,
        stem_curve_x_degrees,
        stem_curve_y_degrees,
        stem_warble,
        stem_handles,
        seed,
    )
    position = pf.nodes.geo.input_position()
    sampled = pf.nodes.geo.sample_curve(
        stem_curve, factor=position.z / height, data_type="FLOAT"
    )
    frame = pf.nodes.func.align_euler_to_vector(1.0, sampled.tangent, axis="Z")
    taper = 1.0 - 0.25 * position.z / height
    radial = pf.nodes.math.combine_xyz(x=position.x, y=position.y) * taper
    radial = pf.nodes.math.vector_rotate_euler(radial, rotation=frame)
    rotation = pf.nodes.func.rotate_euler(layout.rotation, rotate_by=frame)
    index = pf.nodes.geo.input_index()
    captured_points = pf.nodes.geo.capture_attribute(
        points,
        rotation=rotation,
        leaf_id=(index.astype(dtype=float) + 1.0).astype(dtype=int),
        scale=layout.scale,
    )
    points = pf.nodes.geo.set_position(
        captured_points.geometry, position=sampled.position + radial
    )
    prototype = _leaf_with_petiole(
        first,
        stem_material,
        leaf_width,
        petiole_length_fraction,
        petiole_diameter_ratio,
    )
    captured_prototype = pf.nodes.geo.capture_attribute(
        prototype,
        local=pf.nodes.geo.input_position(),
        petiole=pf.nodes.geo.material_selection(stem_material),
    )
    instances = pf.nodes.geo.instance_on_points(
        points=points,
        instance=captured_prototype.geometry,
        rotation=captured_points.rotation.astype(dtype=pf.Euler),
        scale=pf.nodes.math.combine_xyz(
            x=layout.scale * leaf_width, y=layout.scale, z=layout.scale
        ),
    )
    leaves = pf.nodes.geo.realize_instances(instances)
    leaves = _warp_leaves(
        leaves,
        captured_prototype.local,
        captured_points.leaf_id,
        captured_points.rotation,
        captured_points.scale,
        captured_prototype.petiole,
        leaf_width,
        leaf_size,
        leaf_bend,
        leaf_droop,
        leaf_twist_degrees,
        leaf_warble,
        seed,
    )
    stem_curve = pf.nodes.geo.resample_curve_count(
        stem_curve, count=(stem_handles.astype(dtype=float) * 4.0).astype(dtype=int)
    )
    stem_curve = pf.nodes.geo.set_curve_radius(
        stem_curve,
        radius=1.0 - 0.25 * pf.nodes.geo.spline_parameter().factor,
    )
    profile = pf.nodes.geo.curve_circle(
        resolution=12, radius=radius / math.cos(math.pi / 12)
    )
    stem_mesh = pf.nodes.geo.curve_to_mesh(
        stem_curve, profile_curve=profile, fill_caps=True
    )
    face = pf.nodes.geo.corners_of_face(pf.nodes.geo.input_index())
    endpoint = pf.nodes.geo.mesh_face_set_boundaries(
        face_group_id=(face.total > 4).astype(dtype=int)
    )
    stem_mesh = pf.nodes.geo.store_named_attribute(
        stem_mesh, name="crease_edge", domain="EDGE", value=endpoint.astype(dtype=float)
    )
    stem_mesh = pf.nodes.geo.set_material(stem_mesh, material=stem_material)
    return pf.nodes.geo.join_geometry([leaves, stem_mesh])


@pf.tracer.primitive(normalize=False)
def _plant_geometry_group(*args: object, **kwargs: object) -> pf.ProcNode:
    return _plant_geometry(*args, **kwargs)


@pf.tracer.primitive
def _normalize_leaf(obj: pf.MeshObject) -> pf.MeshObject:
    obj = pf.MeshObject(pf.ops.object.duplicate(obj).item())
    positions = pf.ops.attr.vertex_positions(obj)
    root = positions[positions[:, 1].argmin()]
    positions = positions - root
    positions = positions / positions[:, 1].max()
    pf.ops.attr.write_vertex_positions(obj, positions)
    edges = pf.ops.attr.edge_indices(obj)
    roots = np.isclose(positions[:, 1], 0.0)
    crease = np.all(roots[edges], axis=1).astype(np.float32)
    pf.ops.attr.write_attribute(obj, crease, "crease_edge", "EDGE", overwrite=True)
    pf.ops.attr.write_attribute(
        obj, roots.astype(np.float32), "crease_vert", "POINT", overwrite=True
    )
    return obj


def phyllo_plant(
    rng: pf.RNG,
    leaf_mesh: pf.MeshObject | None = None,
    stem_material: pf.Material | None = None,
    count: int = 24,
    height: float = 0.3,
    radius: float = 0.008,
    leaf_length: float = 0.16,
    min_angle: float = 0.2,
    max_angle: float = 0.9,
    phase: float = 0.0,
    leaf_size: float = 1.0,
    leaf_width: float = 1.0,
    petiole_length_fraction: float = 0.0,
    petiole_diameter_ratio: float = 0.03,
    density: float = 1.0,
    base_fraction: float = 0.0,
    target_count: float | None = None,
    sparse: float = 0.0,
    count_override: bool = False,
    stem_curve_x_degrees: float = 0.0,
    stem_curve_y_degrees: float = 0.0,
    stem_warble: float = 0.0,
    stem_handles: int = 8,
    leaf_bend: float = 0.0,
    leaf_droop: float = 0.0,
    leaf_twist_degrees: float = 0.0,
    leaf_warble: float = 0.0,
) -> PhylloPlantResult:
    rng_leaf, rng_geometry = rng.spawn(2)
    if leaf_mesh is None:
        leaf_mesh = leaf.leaf_simple_rand(rng_leaf).mesh
    if target_count is None:
        target_count = count
    if stem_material is None:
        stem_material = pf.Material(
            surface=pf.nodes.shader.principled_bsdf(
                base_color=(0.04, 0.10, 0.01, 1.0), roughness=0.65
            )
        )
    obj = phyllo_plant_mesh(
        leaf_mesh=leaf_mesh,
        stem_material=stem_material,
        seed=pf.random.randint(rng_geometry, 0, 10000),
        count=count,
        height=height,
        radius=radius,
        leaf_length=leaf_length,
        leaf_size=leaf_size,
        leaf_width=leaf_width,
        petiole_length_fraction=petiole_length_fraction,
        petiole_diameter_ratio=petiole_diameter_ratio,
        density=density,
        target_count=target_count,
        sparse=sparse,
        count_override=count_override,
        base_fraction=base_fraction,
        min_angle=min_angle,
        max_angle=max_angle,
        stem_curve_x_degrees=stem_curve_x_degrees,
        stem_curve_y_degrees=stem_curve_y_degrees,
        stem_warble=stem_warble,
        stem_handles=stem_handles,
        leaf_bend=leaf_bend,
        leaf_droop=leaf_droop,
        leaf_twist_degrees=leaf_twist_degrees,
        leaf_warble=leaf_warble,
        phase=phase,
    ).mesh
    pf.ops.modifier.subdivide_surface(obj, levels=2, _skip_apply=True)
    return PhylloPlantResult(obj)


@pf.tracer.generator
def phyllo_plant_mesh(
    leaf_mesh: pf.MeshObject,
    stem_material: pf.Material,
    seed: int,
    count: int = 24,
    height: float = 0.3,
    radius: float = 0.008,
    leaf_length: float = 0.16,
    min_angle: float = 0.2,
    max_angle: float = 0.9,
    phase: float = 0.0,
    leaf_size: float = 1.0,
    leaf_width: float = 1.0,
    petiole_length_fraction: float = 0.0,
    petiole_diameter_ratio: float = 0.03,
    density: float = 1.0,
    base_fraction: float = 0.0,
    target_count: float = 24.0,
    sparse: float = 0.0,
    count_override: bool = False,
    stem_curve_x_degrees: float = 0.0,
    stem_curve_y_degrees: float = 0.0,
    stem_warble: float = 0.0,
    stem_handles: int = 8,
    leaf_bend: float = 0.0,
    leaf_droop: float = 0.0,
    leaf_twist_degrees: float = 0.0,
    leaf_warble: float = 0.0,
) -> PhylloPlantResult:
    first = _normalize_leaf(leaf_mesh)
    geometry = _plant_geometry_group(
        first=first,
        stem_material=stem_material,
        count=count,
        height=height,
        radius=radius,
        leaf_length=leaf_length,
        leaf_size=leaf_size,
        leaf_width=leaf_width,
        petiole_length_fraction=petiole_length_fraction,
        petiole_diameter_ratio=petiole_diameter_ratio,
        density=density,
        target_count=target_count,
        sparse=sparse,
        count_override=count_override,
        base_fraction=base_fraction,
        min_angle=min_angle,
        max_angle=max_angle,
        stem_curve_x_degrees=stem_curve_x_degrees,
        stem_curve_y_degrees=stem_curve_y_degrees,
        stem_warble=stem_warble,
        stem_handles=stem_handles,
        leaf_bend=leaf_bend,
        leaf_droop=leaf_droop,
        leaf_twist_degrees=leaf_twist_degrees,
        leaf_warble=leaf_warble,
        phase=phase,
        seed=seed,
    )
    obj = pf.nodes.to_mesh_object(geometry)
    obj.item().name = phyllo_plant_mesh.__name__
    return PhylloPlantResult(obj)


@pf.tracer.primitive
def phyllo_leaf_count(
    count: float,
    leaf_size: float,
    leaf_width: float,
    density: float,
    sparse: float = 0.0,
) -> int:
    maximum = int(128 - 123 * sparse)
    return max(3, min(maximum, round(count * density / (leaf_size**2 * leaf_width))))


def _leaf_pattern_rand(
    rng_sparse: pf.RNG,
    rng_normal_size: pf.RNG,
    rng_large_size: pf.RNG,
    rng_size_choice: pf.RNG,
    rng_sparse_size: pf.RNG,
    plant_height: float | None,
    height: float,
    sparse: bool | None,
    leaf_size: float | None,
) -> tuple[float, float]:
    if plant_height is None:
        plant_height = height
    large_leaves = (plant_height >= 0.6) * 1.0
    if sparse is None:
        sparse_value = pf.control.choice(rng_sparse, [(0.0, 4.0), (1.0, 1.0)])
        sparse_value = sparse_value * large_leaves
    else:
        sparse_value = float(sparse)
    if leaf_size is None:
        normal_size = pf.random.uniform(rng_normal_size, 0.75, 1.4)
        regular_size = pf.control.choice(
            rng_size_choice,
            [
                (normal_size, 3.0),
                (pf.random.uniform(rng_large_size, 2.0, 2.6), 1.0),
            ],
        )
        regular_size = normal_size * (1.0 - large_leaves) + regular_size * large_leaves
        leaf_size = (
            regular_size * (1.0 - sparse_value)
            + pf.random.uniform(rng_sparse_size, 2.0, 3.0) * sparse_value
        )
    return leaf_size, sparse_value


def _phyllo_rand(
    rng: pf.RNG,
    target_count: float,
    height: float,
    radius: float,
    stem_reference_leaf_length: float,
    leaf_length: float,
    min_angle: float,
    max_angle: float,
    leaf_size: float | None,
    leaf_width: float | None,
    petiole_length_fraction: float | None,
    petiole_diameter_ratio: float,
    density: float | None,
    base_fraction: float | None,
    count: int | None,
    sparse: bool | None,
    stem_curve_x_degrees: float | None,
    stem_curve_y_degrees: float | None,
    stem_warble: float | None,
    stem_handles: int | None,
    leaf_bend: float | None,
    leaf_droop: float | None,
    plant_height: float | None,
    leaf_twist_degrees: float | None,
    leaf_warble: float | None,
) -> PhylloPlantResult:
    (
        rng_leaf_choice,
        rng_plant,
        rng_petiole,
        rng_sparse_choice,
        rng_normal_size,
        rng_large_size,
        rng_size_choice,
        rng_sparse_size,
        rng_width,
        rng_density,
        rng_sparse_target,
        rng_base,
        rng_curve_x,
        rng_curve_y,
        rng_stem_warble,
        rng_stem_handles,
        rng_leaf_bend,
        rng_leaf_droop,
        rng_leaf_twist,
        rng_leaf_warble,
        rng_leaf,
        rng_color,
        rng_material,
        rng_phase,
    ) = rng.spawn(24)
    if petiole_length_fraction is None:
        petiole_length_fraction = pf.random.uniform(rng_petiole, 0.0, 0.28)
    leaf_generator = pf.control.choice(
        rng_leaf_choice,
        [(leaf.leaf_simple_rand, 1.0), (leaf_broadleaf.leaf_broadleaf_rand, 1.0)],
    )
    pattern = _leaf_pattern_rand(
        rng_sparse_choice,
        rng_normal_size,
        rng_large_size,
        rng_size_choice,
        rng_sparse_size,
        plant_height,
        height,
        sparse,
        leaf_size,
    )
    leaf_size = pattern[0]
    sparse = pattern[1]
    radius = (
        radius
        * (
            (1.0 - petiole_length_fraction)
            * leaf_length
            * leaf_size
            / stem_reference_leaf_length
        )
        ** 0.5
    )
    if leaf_width is None:
        leaf_width = pf.random.uniform(rng_width, 0.5, 1.2)
    if density is None:
        density = pf.random.log_uniform(rng_density, 0.6, 1.6)
    sparse_target = (
        pf.random.randint(rng_sparse_target, 3, 6) * leaf_size**2 * leaf_width
    )
    target_count = target_count * (1.0 - sparse) + sparse_target * sparse
    count_override = count is not None
    if count is None:
        count = phyllo_leaf_count(target_count, leaf_size, leaf_width, density, sparse)
    if base_fraction is None:
        base_fraction = 0.15 * pf.random.uniform(rng_base, 0.0, 1.0) ** 3
    if stem_curve_x_degrees is None:
        stem_curve_x_degrees = pf.random.clip_gaussian(
            rng_curve_x, 0.0, 20.0, -55.0, 55.0
        )
    if stem_curve_y_degrees is None:
        stem_curve_y_degrees = pf.random.clip_gaussian(
            rng_curve_y, 0.0, 20.0, -55.0, 55.0
        )
    if stem_warble is None:
        stem_warble = pf.random.uniform(rng_stem_warble, 0.0, 0.025)
    if stem_handles is None:
        stem_handles = pf.random.randint(rng_stem_handles, 6, 11)
    if leaf_bend is None:
        leaf_bend = pf.random.uniform(rng_leaf_bend, 0.03, 0.22)
    if leaf_droop is None:
        leaf_droop = pf.random.uniform(rng_leaf_droop, 0.0, 0.9)
    if leaf_twist_degrees is None:
        leaf_twist_degrees = pf.random.uniform(rng_leaf_twist, 5.0, 35.0)
    if leaf_warble is None:
        leaf_warble = pf.random.uniform(rng_leaf_warble, 0.0, 0.035)
    base_color = leaf_shader.leaf_color_rand(rng_color)
    leaf_material = leaf_shader.leaf_rand(
        rng_material,
        vector=pf.nodes.shader.coord().uv,
        color=base_color,
    )
    leaf_mesh = leaf_generator(rng_leaf, material=leaf_material).mesh
    stem_material = pf.Material(
        surface=pf.nodes.shader.principled_bsdf(base_color=base_color, roughness=0.65)
    )
    result = phyllo_plant(
        rng_plant,
        leaf_mesh=leaf_mesh,
        stem_material=stem_material,
        count=count,
        height=height,
        radius=radius,
        leaf_length=leaf_length,
        min_angle=min_angle,
        max_angle=max_angle,
        phase=pf.random.uniform(rng_phase, 0.0, 2 * math.pi),
        leaf_size=leaf_size,
        leaf_width=leaf_width,
        petiole_length_fraction=petiole_length_fraction,
        petiole_diameter_ratio=petiole_diameter_ratio,
        density=density,
        target_count=target_count,
        sparse=sparse,
        count_override=count_override,
        base_fraction=base_fraction,
        stem_curve_x_degrees=stem_curve_x_degrees,
        stem_curve_y_degrees=stem_curve_y_degrees,
        stem_warble=stem_warble,
        stem_handles=stem_handles,
        leaf_bend=leaf_bend,
        leaf_droop=leaf_droop,
        leaf_twist_degrees=leaf_twist_degrees,
        leaf_warble=leaf_warble,
    )

    return result


def plant_phyllo_basal_rand(
    rng: pf.RNG,
    leaf_size: float | None = None,
    leaf_width: float | None = None,
    petiole_length_fraction: float | None = None,
    petiole_diameter_ratio: float = 0.03,
    density: float | None = None,
    base_fraction: float | None = None,
    count: int | None = None,
    sparse: bool | None = None,
    stem_curve_x_degrees: float | None = None,
    stem_curve_y_degrees: float | None = None,
    stem_warble: float | None = None,
    stem_handles: int | None = None,
    leaf_bend: float | None = None,
    leaf_droop: float | None = None,
    plant_height: float | None = None,
    leaf_twist_degrees: float | None = None,
    leaf_warble: float | None = None,
    stem_height: float | None = None,
    stem_radius: float | None = None,
    leaf_length: float | None = None,
    min_angle: float | None = None,
    max_angle: float | None = None,
) -> PhylloPlantResult:
    rng, rng_plant = rng.spawn(2)
    if stem_height is None:
        stem_height = pf.random.uniform(rng, 0.008, 0.04)
    if stem_radius is None:
        stem_radius = pf.random.uniform(rng, 0.0125, 0.019)
    if leaf_length is None:
        leaf_length = pf.random.uniform(rng, 0.09, 0.24)
    target_count = pf.random.randint(rng, 18, 49)
    if min_angle is None:
        min_angle = pf.random.uniform(rng, 0.12, 0.35)
    if max_angle is None:
        max_angle = pf.random.uniform(rng, 0.85, 1.35)
    return _phyllo_rand(
        rng_plant,
        target_count=target_count,
        height=stem_height,
        radius=stem_radius,
        stem_reference_leaf_length=0.33,
        leaf_length=leaf_length,
        min_angle=min_angle,
        max_angle=max_angle,
        leaf_size=leaf_size,
        leaf_width=leaf_width,
        petiole_length_fraction=petiole_length_fraction,
        petiole_diameter_ratio=petiole_diameter_ratio,
        density=density,
        base_fraction=base_fraction,
        count=count,
        sparse=sparse,
        stem_curve_x_degrees=stem_curve_x_degrees,
        stem_curve_y_degrees=stem_curve_y_degrees,
        stem_warble=stem_warble,
        stem_handles=stem_handles,
        leaf_bend=leaf_bend,
        leaf_droop=leaf_droop,
        plant_height=plant_height,
        leaf_twist_degrees=leaf_twist_degrees,
        leaf_warble=leaf_warble,
    )


def plant_phyllo_ascending_rand(
    rng: pf.RNG,
    leaf_size: float | None = None,
    leaf_width: float | None = None,
    petiole_length_fraction: float | None = None,
    petiole_diameter_ratio: float = 0.03,
    density: float | None = None,
    base_fraction: float | None = None,
    count: int | None = None,
    sparse: bool | None = None,
    stem_curve_x_degrees: float | None = None,
    stem_curve_y_degrees: float | None = None,
    stem_warble: float | None = None,
    stem_handles: int | None = None,
    leaf_bend: float | None = None,
    leaf_droop: float | None = None,
    plant_height: float | None = None,
    leaf_twist_degrees: float | None = None,
    leaf_warble: float | None = None,
    stem_height: float | None = None,
    stem_radius: float | None = None,
    leaf_length: float | None = None,
    min_angle: float | None = None,
    max_angle: float | None = None,
) -> PhylloPlantResult:
    rng, rng_plant = rng.spawn(2)
    if stem_height is None:
        stem_height = pf.random.log_uniform(rng, 0.08, 0.85)
    if stem_radius is None:
        stem_radius = pf.random.uniform(rng, 0.005, 0.0125)
    if leaf_length is None:
        leaf_length = pf.random.uniform(rng, 0.08, 0.23)
    target_count = pf.random.randint(rng, 9, 29) * (0.5 + stem_height / 0.4)
    if min_angle is None:
        min_angle = pf.random.uniform(rng, 0.12, 0.45)
    if max_angle is None:
        max_angle = pf.random.uniform(rng, 0.55, 1.1)
    return _phyllo_rand(
        rng_plant,
        target_count=target_count,
        height=stem_height,
        radius=stem_radius,
        stem_reference_leaf_length=0.25,
        leaf_length=leaf_length,
        min_angle=min_angle,
        max_angle=max_angle,
        leaf_size=leaf_size,
        leaf_width=leaf_width,
        petiole_length_fraction=petiole_length_fraction,
        petiole_diameter_ratio=petiole_diameter_ratio,
        density=density,
        base_fraction=base_fraction,
        count=count,
        sparse=sparse,
        stem_curve_x_degrees=stem_curve_x_degrees,
        stem_curve_y_degrees=stem_curve_y_degrees,
        stem_warble=stem_warble,
        stem_handles=stem_handles,
        leaf_bend=leaf_bend,
        leaf_droop=leaf_droop,
        plant_height=plant_height,
        leaf_twist_degrees=leaf_twist_degrees,
        leaf_warble=leaf_warble,
    )
