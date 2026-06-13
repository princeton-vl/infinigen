"""Biome presets: the only environment-specific layer.

The road-conforming core (xodr -> anchored terrain -> dist_to_road -> avoidance
-> road surface -> camera -> render) is biome-agnostic. A preset only changes:
  - terrain relief (amplitude / noise),
  - terrain colour + lighting,
  - which Infinigen assets get scattered (see vegetation.populate_biome).

Driven by `--biome` or spec.json["biome"].
"""

from __future__ import annotations

# canonical biome name -> spec.json / scene_type aliases
ALIASES = {
    "plain": "grassland",
    "grass": "grassland",
    "snowy_mountain": "snow",
    "snowy": "snow",
    "arctic": "snow",
    "snow_mountain": "snow",
}

BIOME_PRESETS = {
    "grassland": dict(
        amplitude=3.2, base_freq=0.012, octaves=6, transition=16.0,
        terrain_color=(0.20, 0.20, 0.10), sun_energy=3.0, sky_strength=1.0,
        scatters=["grass", "ground_cover", "pebbles", "trees"],
        tree_n=12, bush_n=36,
    ),
    "desert": dict(
        amplitude=5.5, base_freq=0.007, octaves=5, transition=20.0,
        terrain_color=(0.62, 0.48, 0.28), sun_energy=5.5, sky_strength=1.1,
        scatters=["cactus", "pebbles", "boulders_extra"],  # NO grass
        cactus_n=16, boulder_n=30, tree_n=0, bush_n=0,
    ),
    "mountain": dict(
        amplitude=20.0, base_freq=0.010, octaves=7, transition=16.0,
        terrain_color=(0.33, 0.30, 0.27), sun_energy=3.5, sky_strength=1.0,
        scatters=["pebbles", "boulders_extra", "trees", "grass_sparse"],
        boulder_n=28, tree_n=14, bush_n=8,
    ),
    "snow": dict(
        amplitude=14.0, base_freq=0.010, octaves=6, transition=16.0,
        terrain_color=(0.86, 0.88, 0.92), sun_energy=4.0, sky_strength=1.2,
        # NOTE: snow_layer.apply + bush/tree populate both hang on the large snow
        # terrain; alpine snowfield = white terrain material + rocks (realistic).
        scatters=["pebbles", "boulders_extra"],
        boulder_n=24, tree_n=0, bush_n=0,
    ),
    "forest": dict(
        amplitude=5.0, base_freq=0.012, octaves=6, transition=16.0,
        terrain_color=(0.12, 0.13, 0.06), sun_energy=2.2, sky_strength=0.8,
        scatters=["grass", "ground_cover", "fern", "pebbles", "trees_dense"],
        tree_n=22, bush_n=24,
    ),
}


def resolve(biome: str | None) -> tuple[str, dict]:
    name = (biome or "grassland").strip().lower()
    name = ALIASES.get(name, name)
    if name not in BIOME_PRESETS:
        name = "grassland"
    return name, BIOME_PRESETS[name]
