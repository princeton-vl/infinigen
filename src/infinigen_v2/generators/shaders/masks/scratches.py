import procfunc as pf
from procfunc.nodes import types as t


def scratches_distribution(
    rng: pf.RNG,
    vector: t.SocketOrVal[pf.Vector],
    displacement_a: t.SocketOrVal[pf.Vector] | None = None,
    displacement_b: t.SocketOrVal[pf.Vector] | None = None,
    height_threshold: t.SocketOrVal[float] = -1.0,
):
    angle1 = pf.random.uniform(rng, 10.0, 80.0)
    angle2 = pf.random.uniform(rng, -80.0, -10.0)
    scratch_scale = pf.random.log_uniform(rng, 5.0, 20.0)
    scratch_mask_ratio = pf.random.log_uniform(rng, 0.1, 0.9)
    scratch_mask_noise = pf.random.log_uniform(rng, 80.0, 100.0)
    scratch_detail = pf.random.uniform(rng, 10.0, 25.0)
    scratch_mask_detail = pf.random.uniform(rng, 0.5, 2.0)

    # scratch_depth = pf.random.log_uniform(rng, 0.1, 1.0)
    scratch_mask_result = scratches_mask(
        vector=vector,
        displacement_a=displacement_a,
        displacement_b=displacement_b,
        height_threshold=height_threshold,
        angle1=angle1,
        angle2=angle2,
        scratch_scale=scratch_scale,
        scratch_mask_ratio=scratch_mask_ratio,
        scratch_mask_noise=scratch_mask_noise,
        scratch_detail=scratch_detail,
        scratch_mask_detail=scratch_mask_detail,
    )
    return scratch_mask_result


@pf.nodes.node_function
def scratches_mask(
    vector: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    displacement_a: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    displacement_b: t.SocketOrVal[pf.Vector] = (0.0, 0.0, 0.0),
    height_threshold: t.SocketOrVal[float] = 0.0,
    angle1: t.SocketOrVal[float] = 45.0,
    angle2: t.SocketOrVal[float] = -20.0,
    scratch_scale: t.SocketOrVal[float] = 20.0,
    scratch_mask_ratio: t.SocketOrVal[float] = 0.8,
    scratch_mask_noise: t.SocketOrVal[float] = 90.0,
    scratch_detail: t.SocketOrVal[float] = 15.0,
    scratch_mask_detail: t.SocketOrVal[float] = 1.0,
):
    mapping_rotation_y = angle1
    noise_scale = scratch_scale
    mapping_1_rotation_y = angle2
    scratch_b_a = scratch_mask_ratio
    noise_2_scale = scratch_mask_noise
    noise_detail = scratch_detail
    noise_2_detail = scratch_mask_detail

    mapping_rotation = pf.nodes.func.combine_xyz(y=mapping_rotation_y)
    mapping = pf.nodes.shader.mapping(
        vector_type="TEXTURE",
        vector=vector,
        rotation=mapping_rotation,
        scale=(1.0, 500.0, 1.0),
    )

    noise = pf.nodes.shader.noise(
        vector=mapping,
        scale=noise_scale,
        detail=noise_detail,
        noise_dimensions="2D",
        noise_type="RIDGED_MULTIFRACTAL",
        roughness=0.47,
        lacunarity=5.2,
        offset=1.1,
        distortion=98.10,
    )

    mapping_1_rotation = pf.nodes.func.combine_xyz(y=mapping_1_rotation_y)
    mapping_1 = pf.nodes.shader.mapping(
        vector_type="TEXTURE",
        vector=vector,
        rotation=mapping_1_rotation,
        scale=(500.0, 1.0, 1.0),
    )

    noise_1 = pf.nodes.shader.noise(
        vector=mapping_1,
        scale=noise_scale,
        detail=noise_detail,
        noise_dimensions="2D",
        noise_type="RIDGED_MULTIFRACTAL",
        roughness=0.47,
        lacunarity=5.2,
        offset=1.1,
        distortion=98.10,
    )

    ramp_1 = pf.nodes.shader.color_ramp(
        fac=noise.fac,
        interpolation="LINEAR",
        points=[
            (0.153, (1.0, 1.0, 1.0, 1.0)),
            (0.2, (0.0, 0.0, 0.0, 1.0)),
        ],
    )

    ramp_2 = pf.nodes.shader.color_ramp(
        fac=noise_1.fac,
        interpolation="LINEAR",
        points=[
            (0.153, (1.0, 1.0, 1.0, 1.0)),
            (0.2, (0.0, 0.0, 0.0, 1.0)),
        ],
    )

    scratch_a = ramp_1.color.astype(dtype=float) + ramp_2.color.astype(dtype=float)

    mapping_2 = pf.nodes.shader.mapping(
        vector_type="TEXTURE",
        vector=vector,
        rotation=(0.1588, -0.5742, 0.192),
    )

    noise_2 = pf.nodes.shader.voronoi(
        vector=mapping_2,
        scale=noise_2_scale,
        detail=noise_2_detail,
        roughness=0.5,
        distance="EUCLIDEAN",
        feature="F1",
        voronoi_dimensions="2D",
    )
    color_ramp = pf.nodes.shader.color_ramp(
        fac=noise_2.color,
        interpolation="LINEAR",
        points=[
            (0.8, (0.0, 0.0, 0.0, 1.0)),
            (0.82, (1.0, 1.0, 1.0, 1.0)),
        ],
    )

    scratch_b = scratch_b_a * color_ramp.color.astype(dtype=float)

    # multiply scratch_b with scratch_a
    scratch_c = scratch_a * scratch_b
    return scratch_c > 0.001
