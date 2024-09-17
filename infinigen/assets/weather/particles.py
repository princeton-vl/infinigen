# Copyright (C) 2024, Princeton University.

# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Hei Law: Initial rain / snow implementation
# - Alexander Raistrick: Refactor

import logging
import typing

import bpy
from numpy.random import normal, uniform

from infinigen.core.generator import Generator
from infinigen.core.placement import AssetFactory, make_asset_collection, particles
from infinigen.core.util import butil
from infinigen.core.util.random import log_uniform

logger = logging.getLogger(__name__)


def spawn_emitter(follow_cam, mesh_type, size, offset, name=None):
    match mesh_type:
        case "plane":
            emitter = butil.spawn_plane(location=offset, size=size)
        case "cube":
            emitter = butil.spawn_cube(location=offset, size=size)
        case _:
            raise ValueError(f"Unknown mesh type {mesh_type}")

    butil.constrain_object(emitter, "COPY_LOCATION", use_offset=True, target=follow_cam)

    if name is None:
        name = follow_cam.name
    emitter.name = f"emitter({name=}, {mesh_type=})"

    emitter.hide_viewport = True
    emitter.hide_render = True  # will be undone if any particle systems are added

    butil.put_in_collection(emitter, butil.get_collection("particles"))

    return emitter


def rain_param_distribution():
    drops_per_sec_m2 = uniform(0.05, 1)
    velocity = uniform(9, 20)
    lifetime = 100

    return dict(
        mass=0.001,
        warmup_frames=100,
        density=drops_per_sec_m2 * lifetime,
        lifetime=lifetime,
        particle_size=uniform(0.01, 0.015),
        size_random=uniform(0.005, 0.01),
        normal_factor=-velocity,
        effect_gravity=0.0,
        use_die_on_collision=True,
    )


def falling_leaf_param_distribution():
    rate = uniform(0.001, 0.006)
    dur = max(bpy.context.scene.frame_end - bpy.context.scene.frame_start, 500)

    return dict(
        warmup_frames=1024,
        density=rate * dur,
        particle_size=normal(0.5, 0.15),
        size_random=uniform(0.1, 0.2),
        lifetime=dur,
        use_rotations=True,
        rotation_factor_random=1.0,
        use_die_on_collision=False,
        drag_factor=0.2,
        damping=0.3,
        mass=0.01,
        normal_factor=0.0,
        angular_velocity_mode="RAND",
        angular_velocity_factor=uniform(0, 3),
        use_dynamic_rotation=True,
    )


def floating_dust_param_distribution():
    return dict(
        mass=0.0001,
        count=int(7000 * uniform(0.5, 2)),
        lifetime=1000,
        warmup_frames=100,
        particle_size=0.001,
        size_random=uniform(0.7, 1.0),
        emit_from="VOLUME",
        damping=1.0,
        drag_factor=1.0,
        effect_gravity=uniform(0.3, 0.7),  # partially buoyant
    )


def marine_snow_param_distribution():
    return dict(
        mass=0.0001,
        count=int(10000 * uniform(0.5, 2)),
        lifetime=1000,
        warmup_frames=100,
        particle_size=0.005,
        size_random=uniform(0.7, 1.0),
        emit_from="VOLUME",
        brownian_factor=log_uniform(0.0002, 0.0005),
        damping=log_uniform(0.95, 0.98),
        drag_factor=uniform(0.85, 0.95),
        factor_random=uniform(0.1, 0.2),
        use_rotations=True,
        phase_factor_random=uniform(0.2, 0.5),
        use_dynamic_rotation=True,
        effect_gravity=uniform(0, 0.5),
    )


def snow_param_distribution():
    density = uniform(2, 26)

    return dict(
        mass=0.001,
        density=density,
        lifetime=2000,
        warmup_frames=1000,
        particle_size=0.003,
        emit_from="FACE",
        damping=1.0,
        drag_factor=1.0,
        use_rotations=True,
        use_die_on_collision=True,
    )


class FallingParticles(Generator):
    def __init__(
        self,
        particle_gen: AssetFactory,
        distribution: typing.Callable,
    ):
        self.particle_gen = particle_gen

        super().__init__(distribution)

    def generate(
        self,
        emitter: bpy.types.Object,
        collision: bpy.types.Collection = None,
    ):
        col = make_asset_collection(self.particle_gen, 5)

        emitter, system = particles.particle_system(
            emitter, col, self.params, collision
        )

        system.name = repr(self)
        system.settings.name = repr(self) + ".settings"
        emitter.hide_render = False

        logger.info(f"{self} baking particles")

        particles.bake(emitter, system)

        return emitter
