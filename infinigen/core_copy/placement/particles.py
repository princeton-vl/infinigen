# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Hei Law: initial system, particle settings
# - Alexander Raistrick: refactor, boids


from typing import Union
import math
import logging
from copy import copy

import numpy as np
from numpy.random import uniform as U, normal as N, uniform
import bpy

from infinigen.core.util.logging import Suppress
from infinigen.core.util import blender as butil
from infinigen.core.util.random import log_uniform

logger = logging.getLogger(__name__)

def bake(emitter, system):

    logger.info(f'Baking particles for {emitter.name=}')

    with butil.SelectObjects(emitter):

        override = {
            'scene': bpy.context.scene,
            'active_object': emitter,
            'point_cache': system.point_cache,
        }
        with Suppress():
            bpy.context.scene.frame_end += 1
            bpy.ops.ptcache.bake(override, bake=True)
            bpy.context.scene.frame_end -= 1

def configure_boids(system_config, settings):
    boids = system_config.boids
    boids.states[0].rule_fuzzy = settings.pop('rule_fuzzy', 0.5)

    if rules := settings.pop('rules', None):
        context = bpy.context.copy()
        context['particle_settings'] = system_config
        for _ in boids.states[0].rules.keys():
            bpy.ops.boid.rule_del(context)
        for r in rules:
            bpy.ops.boid.rule_add(context, type=r.pop('type'))
            for k, v in r.items():
                setattr(boids.states[0].rules[-1], k, v)
        assert len(boids.states[0].rules) == len(rules)

    if goal := settings.pop('goal_obj', None):
        try:
            goal_rule = next(r for r in boids.states[0].rules if r.type == 'GOAL')
            goal_rule.object = goal
        except StopIteration:
            pass

    for k, v in settings.items():
        setattr(boids, k, v)

def as_particle_collection(subject, prefix='particleassets'):

    '''
    Particle assets cannot have hide_render=True or they will be invisible, 
    yet this is the default behavior for most asset collections
    '''

    if subject.name.startswith(prefix):
        return subject

    subject.name = prefix + ':' + subject.name.split(':')[-1]
    for o in subject.objects:
        o.location.z -= 100
    subject.hide_viewport = True
    subject.hide_render = False

    return subject


def particle_system(
    emitter: bpy.types.Object, 
    subject: Union[bpy.types.Object, bpy.types.Collection], 
    settings: dict, collision_collection=None,
):

    '''
    Generalized particle system.
    kwargs are passed through to particle_system.settings
    '''

    if isinstance(subject, bpy.types.Collection):
        subject = as_particle_collection(subject)

    emitter.name = f"particles:emitter({subject.name.split(':')[-1]})"

    mod = emitter.modifiers.new(name='PARTICLE', type='PARTICLE_SYSTEM')
    system = emitter.particle_systems[mod.name]

    emitter.show_instancer_for_viewport = False
    emitter.show_instancer_for_render = False

    settings = copy(settings)

    if isinstance(subject, bpy.types.Object):
        system.settings.render_type = 'OBJECT'
        system.settings.instance_object = subject
        objects = [subject]
    elif isinstance(subject, bpy.types.Collection):

        system.settings.render_type = 'COLLECTION'
        system.settings.instance_collection = subject
        system.settings.use_collection_pick_random=True

        objects = list(subject.objects)
    else:
        raise ValueError(f'Unrecognized {type(subject)=}')
    
    butil.origin_set(objects, "ORIGIN_GEOMETRY", center="MEDIAN")
    
    dur = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    system.settings.frame_start = bpy.context.scene.frame_start - settings.pop('warmup_frames', 0)
    system.settings.frame_end = bpy.context.scene.frame_start + settings.pop('emit_duration', dur) + settings.pop('warmup_frames', 0)

    if (g := settings.pop('effect_gravity', None)) is not None:
        system.settings.effector_weights.gravity = g

    if (d := settings.pop('density', None)) is not None:
        assert 'count' not in settings
        measure = math.prod([v for v in emitter.dimensions if v != 0])
        system.settings.count = math.ceil(d * measure)

    if (b := settings.pop('boids_settings', None)) is not None:
        system.settings.physics_type='BOIDS'
        configure_boids(system.settings, b)

    if collision_collection is not None:
        system.settings.collision_collection = collision_collection

    for k, v in settings.items():
        setattr(system.settings, k, v)

    return emitter, system

def falling_leaf_settings():

    rate = U(0.001, 0.006)
    dur = max(bpy.context.scene.frame_end - bpy.context.scene.frame_start, 500)

    return dict(
        warmup_frames=1024,
        density=rate * dur,
        particle_size=N(0.5, 0.15),
        size_random=U(0.1, 0.2),
        lifetime=dur,
        use_rotations=True,
        rotation_factor_random=1.0,
        use_die_on_collision=False,
        drag_factor=0.2,
        damping=0.3,
        mass=0.01,
        normal_factor=0.0,
        angular_velocity_mode='RAND',
        angular_velocity_factor=U(0, 3),
        use_dynamic_rotation=True
    )

def floating_dust_settings():

    return dict(
        mass=0.0001,
        count=int(7000*U(0.5, 2)),
        lifetime=1000,
        warmup_frames=100,
        particle_size=0.001,
        size_random=uniform(.7, 1.),
        emit_from='VOLUME',
        damping=1.0,
        drag_factor=1.0,
        effect_gravity=U(0.3, 0.7), # partially buoyant
    )

def marine_snow_setting():
    return dict(
        mass=0.0001,
        count=int(10000*U(0.5, 2)),
        lifetime=1000,
        warmup_frames=100,
        particle_size=0.005,
        size_random=uniform(.7, 1.),
        emit_from='VOLUME',
        brownian_factor=log_uniform(.0002, .0005),
        damping=log_uniform(.95, .98),
        drag_factor=uniform(.85, .95),
        factor_random=uniform(.1, .2),
        use_rotations=True,
        phase_factor_random=uniform(.2,.5),
        use_dynamic_rotation=True,
        effect_gravity=U(0, 0.5)
    )

def rain_settings():

    drops_per_sec_m2 = U(0.05, 1)
    velocity = U(9, 20)
    lifetime = 100

    return dict(
        mass=0.001,
        warmup_frames=100,
        density=drops_per_sec_m2*lifetime,
        lifetime=lifetime,
        particle_size=U(0.01, 0.015),
        size_random=U(0.005, 0.01),
        normal_factor=-velocity,
        effect_gravity=0.0,
        use_die_on_collision=True,
    )

def snow_settings():

    density = U(2, 26)

    return dict(
        mass=0.001,
        density=density,
        lifetime=2000,
        warmup_frames=1000,
        particle_size=0.003,
        emit_from='FACE',
        damping=1.0,
        drag_factor=1.0,
        use_rotations=True,
        use_die_on_collision=True,
    )