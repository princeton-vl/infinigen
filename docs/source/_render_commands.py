"""Derive the CLI command that reproduces a given example render.

Mirrors the per-category invocations in scripts/integration_v2/launch.sh,
dropping the run-specific --output flag.
"""

_TEMPLATES = {
    "Material": "infinigen2 {short} material_cube render_cycles --seed {seed} "
    "--passes rgb --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128",
    "Mask": "infinigen2 {short} material_plane_uv render_cycles --seed {seed} "
    "--passes rgb -r 384 384 -s 128",
    "Object": "infinigen2 {short} object_demo render_cycles --seed {seed} "
    "--passes rgb -r 512 512 -s 128",
    "Scene": "infinigen2 {short} render_cycles --seed {seed} "
    "--passes rgb -r 480 480 -s 256",
}


def replicate_command(category: str, name: str, seed: int) -> str | None:
    template = _TEMPLATES.get(category)
    if template is None:
        return None
    short = name.rsplit(".", 1)[-1]
    return template.format(short=short, seed=seed)


if __name__ == "__main__":
    print(
        replicate_command(
            "Material", "infinigen2.shaders.composites.bricks.bricks_rand", 0
        )
    )
    print(replicate_command("Mask", "infinigen2.shaders.masks.cracks.cracks_rand", 3))
    print(
        replicate_command(
            "Object", "infinigen2.objects.random_primitives.primitives_rand", 0
        )
    )
    print(replicate_command("Scene", "infinigen2.scenes.asset_demo.material_sphere", 5))
    print(
        replicate_command(
            "Exporter", "infinigen2.exporters.render_cycles.render_cycles", 0
        )
    )
