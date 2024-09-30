# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hei Law: initial system, particle settings
# - Alexander Raistrick: refactor, boids


import logging
import math
from copy import copy
from typing import Union

import bpy

from infinigen.core.util import blender as butil
from infinigen.core.util.logging import Suppress

logger = logging.getLogger(__name__)


def bake(emitter, system):
    logger.info(f"Baking particles for {emitter.name=}")

    hide_orig = emitter.hide_viewport
    emitter.hide_viewport = False

    with butil.SelectObjects(emitter):
        override = {
            "scene": bpy.context.scene,
            "active_object": emitter,
            "point_cache": system.point_cache,
        }
        with Suppress():
            bpy.context.scene.frame_end += 1
            with bpy.context.temp_override(**override):
                bpy.ops.ptcache.bake(bake=True)
            bpy.context.scene.frame_end -= 1

    emitter.hide_viewport = hide_orig


def configure_boids(system_config, settings):
    boids = system_config.boids
    boids.states[0].rule_fuzzy = settings.pop("rule_fuzzy", 0.5)

    if rules := settings.pop("rules", None):
        context = bpy.context.copy()
        context["particle_settings"] = system_config
        for _ in boids.states[0].rules.keys():
            with bpy.context.temp_override(**context):
                bpy.ops.boid.rule_del()
        for r in rules:
            with bpy.context.temp_override(**context):
                bpy.ops.boid.rule_add(type=r.pop("type"))
            for k, v in r.items():
                setattr(boids.states[0].rules[-1], k, v)
        assert len(boids.states[0].rules) == len(rules)

    if goal := settings.pop("goal_obj", None):
        try:
            goal_rule = next(r for r in boids.states[0].rules if r.type == "GOAL")
            goal_rule.object = goal
        except StopIteration:
            pass

    for k, v in settings.items():
        setattr(boids, k, v)


def as_particle_collection(subject, prefix="particleassets"):
    """
    Particle assets cannot have hide_render=True or they will be invisible,
    yet this is the default behavior for most asset collections
    """

    if subject.name.startswith(prefix):
        return subject

    subject.name = prefix + ":" + subject.name.split(":")[-1]
    for o in subject.objects:
        o.location.z -= 100
    subject.hide_viewport = True
    subject.hide_render = False

    return subject


def particle_system(
    emitter: bpy.types.Object,
    subject: Union[bpy.types.Object, bpy.types.Collection],
    settings: dict,
    collision_collection=None,
):
    """
    Generalized particle system.
    kwargs are passed through to particle_system.settings
    """

    if isinstance(subject, bpy.types.Collection):
        subject = as_particle_collection(subject)

    mod = emitter.modifiers.new(name="PARTICLE", type="PARTICLE_SYSTEM")
    system = mod.particle_system

    emitter.show_instancer_for_viewport = False
    emitter.show_instancer_for_render = False

    settings = copy(settings)

    if isinstance(subject, bpy.types.Object):
        system.settings.render_type = "OBJECT"
        system.settings.instance_object = subject
        objects = [subject]
    elif isinstance(subject, bpy.types.Collection):
        system.settings.render_type = "COLLECTION"
        system.settings.instance_collection = subject
        system.settings.use_collection_pick_random = True

        objects = list(subject.objects)
    else:
        raise ValueError(f"Unrecognized {type(subject)=}")

    butil.origin_set(objects, "ORIGIN_GEOMETRY", center="MEDIAN")

    dur = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    system.settings.frame_start = bpy.context.scene.frame_start - settings.pop(
        "warmup_frames", 0
    )
    system.settings.frame_end = (
        bpy.context.scene.frame_start
        + settings.pop("emit_duration", dur)
        + settings.pop("warmup_frames", 0)
    )

    if (g := settings.pop("effect_gravity", None)) is not None:
        system.settings.effector_weights.gravity = g

    if (d := settings.pop("density", None)) is not None:
        assert "count" not in settings
        measure = math.prod([v for v in emitter.dimensions if v != 0])
        system.settings.count = math.ceil(d * measure)

    if (b := settings.pop("boids_settings", None)) is not None:
        system.settings.physics_type = "BOIDS"
        configure_boids(system.settings, b)

    if collision_collection is not None:
        system.settings.collision_collection = collision_collection

    for k, v in settings.items():
        setattr(system.settings, k, v)

    return emitter, system
