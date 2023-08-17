# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: FishSchoolFactory, basic version of FishFactory, anim & simulation
# - Mingzhe Wang: Fin placement


from collections import defaultdict

import bpy
import gin
import numpy as np
from numpy.random import uniform as U, normal as N, randint

import infinigen.assets.materials.scale
import infinigen.assets.materials.fishbody
from infinigen.assets.materials import fishfin, eyeball
from infinigen.core import surface
from infinigen.assets.materials.utils.surface_utils import sample_range

from infinigen.core.placement.factory import AssetFactory, make_asset_collection

from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util.genome import Joint
from infinigen.assets.creatures import parts
from infinigen.assets.creatures.util import creature, joining
from infinigen.assets.creatures.util import cloth_sim
from infinigen.assets.creatures.util.boid_swarm import BoidSwarmFactory

from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian, FixedSeed
from infinigen.assets.creatures.util.animation.driver_wiggle import animate_wiggle_bones
from infinigen.assets.creatures.util.creature_util import offset_center

from infinigen.assets.utils.tag import tag_object, tag_nodegroup

from infinigen.assets.materials import fish_eye_shader

def fin_params(scale=(1, 1, 1), dorsal=False):
    # scale = np.array((0.2, 1, 0.4)) * np.array((l / l_mean, 1, rad/r_mean)) * np.array(scale)
    noise = np.array(
        (clip_gaussian(1, 0.1, 0.8, 1.2), 1, 0.8 * clip_gaussian(1, 0.1, 0.8, 1.2)))
    scale *= noise
    scale = scale.astype(np.float32)
    if dorsal:
        #if U() < 0.8:
        # for dorsal fins, change the shape via RoundWeight
        RoundWeight = sample_range(0.8, 1)
        RoundingWeight = 1
        #else:
        #    RoundWeight = sample_range(0.4, 0.5)
        #    RoundingWeight = sample_range(0.04, 0.06)
        AffineZ = sample_range(0, 0.1)
        OffsetWeightZ = sample_range(0.6, 1)
        OffsetWeightY = 1
        Freq = U(100, 150)
    else:
        RoundWeight = 1
        RoundingWeight = sample_range(0.02, 0.07)
        AffineZ = sample_range(0.8, 1.2)
        OffsetWeightZ = sample_range(0.05, 0.2)
        OffsetWeightY = sample_range(0.2, 1)
        Freq = U(60, 80)

    return {
        'FinScale': scale,
        'RoundWeight': RoundWeight,
        'RoundingWeight': RoundingWeight,
        'AffineZ': AffineZ,
        'OffsetWeightZ': OffsetWeightZ,
        'OffsetWeightY': OffsetWeightY,
        'Freq': Freq
    }

def fish_postprocessing(body_parts, extras, params):
    
    get_extras = lambda k: [o for o in extras if k in o.name]
    main_template = surface.registry.sample_registry(params['surface_registry'])
    main_template.apply(body_parts + get_extras('BodyExtra'))

    mat = body_parts[0].active_material
    gold = (mat is not None and 'gold' in mat.name)
    body_parts[0].active_material.name.lower() or U() < 0.1
    fishfin.apply(get_extras('Fin'), shader_kwargs={'goldfish': gold })

    fish_eye_shader.apply(get_extras('Eyeball'))
    #eyeball.apply(get_extras('Eyeball'), shader_kwargs={"coord": "X"})

def fish_fin_cloth_sim_params():

    res = dict(
        compression_stiffness= 1200,
        tension_stiffness = 1200,
        shear_stiffness = 1200,
        bending_stiffness = 3000,

        tension_damping=100,
        compression_damping=100,
        shear_damping=100,
        bending_damping=100,

        air_damping = 5,
        mass = 0.3,
    )

    for k, v in res.items():
        res[k] = clip_gaussian(1, 0.2, 0.2, 3) * v

    return res

def fish_genome():

    temp_dict = defaultdict(lambda: 0.1, {'body_fish_eel': 0.01, 'body_fish_puffer': 0.001})
    body = genome.part(parts.generic_nurbs.NurbsBody(
        prefix='body_fish', tags=['body'], var=U(0.3, 1), 
        temperature=temp_dict, 
        shoulder_ik_ts=[0.0, 0.3, 0.6, 1.0], 
        n_bones=15,
        rig_reverse_skeleton=True
    ))

    if U() < 0.9:
        n_dorsal = 1 #if U() < 0.6 else randint(1, 4)
        coord = (U(0.3, 0.45), 1, 0.7)
        for i in range(n_dorsal):
            dorsal_fin = parts.ridged_fin.FishFin(fin_params((U(0.4, 0.6), 0.5, 0.2), dorsal=True), rig=False)
            genome.attach(genome.part(dorsal_fin), body, coord=coord, joint=Joint(rest=(0, -100, 0)))

    rot = lambda r: np.array((20, r, -205)) + N(0, 7, 3)
    
    if U() < 0.8:
        pectoral_fin = parts.ridged_fin.FishFin(fin_params((0.1, 0.5, 0.3)))
        coord = (U(0.65, 0.8), U(55, 65) / 180, .9)
        for side in [-1, 1]:
            genome.attach(genome.part(pectoral_fin), body, coord=coord, 
                joint=Joint(rest=rot(-13)), side=side)

    if U() < 0.8:
        pelvic_fin = parts.ridged_fin.FishFin(fin_params((0.08, 0.5, 0.25)))
        coord = (U(0.5, 0.65), U(8, 15)/180, .8)
        for side in [-1, 1]:
            genome.attach(genome.part(pelvic_fin), body, coord=coord, joint=Joint(rest=rot(28)), side=side)

    if U() < 0.8:
        hind_fin = parts.ridged_fin.FishFin(fin_params((0.1, 0.5, 0.3)))
        coord = (U(0.2, 0.3), N(36, 5)/180, .9)
        for side in [-1, 1]:
            genome.attach(genome.part(hind_fin), body, coord=coord, joint=Joint(rest=rot(28)), side=side)

    angle = U(140, 170)
    tail_fin = parts.ridged_fin.FishFin(fin_params((0.12, 0.5, 0.35)), rig=False)
    for vdir in [-1, 1]:
        genome.attach(genome.part(tail_fin), body, coord=(0.05, 0, 0), joint=Joint((0, -angle * vdir, 0)))
    
    eye_fac = parts.eye.MammalEye({'Eyelids': False, 'Radius': N(0.036, 0.01)})
    coord = (0.9, 0.6, 0.9)
    for side in [-1, 1]:
        genome.attach(genome.part(eye_fac), body, coord=coord, 
            joint=Joint(rest=(0,0,0)), side=side, rotation_basis='normal')

    if U() < 0:
        jaw = genome.part(parts.head.CarnivoreJaw({'length_rad1_rad2': (0.2, 0.1, 0.06)}))
        genome.attach(jaw, body, coord=(0.8, 0, 0.7), joint=Joint(rest=(0, U(-30, -80), 0)), rotation_basis="normal")

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            cloth=fish_fin_cloth_sim_params(),
            anim=fish_swim_params(),
            surface_registry=[
                (infinigen.assets.materials.fishbody, 3),
                #(infinigen.assets.materials.scale, 1),
            ]
        ) 
    )

def fish_swim_params():
    swim_freq = 3 * clip_gaussian(1, 0.3, 0.1, 2)
    swim_mag = N(20, 3)
    return dict(
        swim_mag=swim_mag,
        swim_freq=swim_freq,
        flipper_freq = 3 * clip_gaussian(1, 0.5, 0.1, 3) * swim_freq,
        flipper_mag = 0.35 * N(1, 0.1) * swim_mag,
        flipper_var = U(0, 0.2),
    )

def animate_fish_swim(arma, params):

    spine = [b for b in arma.pose.bones if 'Body' in b.name]
    fin_bones = [b for b in arma.pose.bones if 'extra_bone(Fin' in b.name]

    global_offset = U(0, 1000) # so swimming animations dont sync across fish
    animate_wiggle_bones(
        arma=arma, bones=spine, 
        off=global_offset,
        mag_deg=params['swim_mag'], freq=params['swim_freq'], wavelength=U(0.5, 2))
    v = params['flipper_var']
    for b in fin_bones:
        animate_wiggle_bones(
            arma=arma, bones=[b], off=global_offset+U(0, 1),
            mag_deg=params['flipper_mag']*N(1, v), 
            freq=params['flipper_mag']*N(1, v))

def simulate_fish_cloth(joined, extras, cloth_params, rigidity='cloth_pin_rigidity'):

    for e in [joined] + extras:
        assert e.type == 'MESH'
        if 'Fin' in e.name:
            assert rigidity in e.data.attributes
        else:
            surface.write_attribute(joined, lambda nw: 1, data_type='FLOAT', 
                                    name=rigidity, apply=True)
    joined = butil.join_objects([joined] + extras)

    cloth_sim.bake_cloth(joined, settings=cloth_params, 
                         attributes=dict(vertex_group_mass=rigidity))

    return joined

@gin.configurable
class FishFactory(AssetFactory):

    max_distance = 40

    def __init__(self, factory_seed=None, bvh=None, coarse=False, animation_mode=None, species_variety=None, **_):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode

        with FixedSeed(factory_seed):
            self.species_genome = fish_genome()
            self.species_variety = species_variety if species_variety is not None else clip_gaussian(0.2, 0.1, 0.05, 0.45)

    def create_asset(self, i, simulate=False, **kwargs):
        
        instance_genome = genome.interp_genome(self.species_genome, fish_genome(), self.species_variety)

        root, parts = creature.genome_to_creature(instance_genome, name=f'fish({self.factory_seed}, {i})')
        offset_center(root, x=True, z=False)

        # Force material consistency across a whole species of fish
        # TODO: Replace once Generator class is stnadardized
        def seeded_fish_postprocess(*args, **kwargs):
            with FixedSeed(self.factory_seed):
                fish_postprocessing(*args, **kwargs)

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root, parts, instance_genome, rigging=(self.animation_mode is not None), rig_before_subdiv=True,
            postprocess_func=seeded_fish_postprocess, adapt_mode='subdivide', **kwargs)
        if self.animation_mode is not None and arma is not None:
            if self.animation_mode == 'idle' or self.animation_mode == 'roam':
                animate_fish_swim(arma, instance_genome.postprocess_params['anim'])
            else:
                raise ValueError(f'Unrecognized {self.animation_mode=}')
            
        if simulate:
            joined = simulate_fish_cloth(joined, extras, instance_genome.postprocess_params['cloth'])
        else:
            joined = butil.join_objects([joined] + extras)
            joined.parent = root

        tag_object(root, 'fish')
            
        return root
    

class FishSchoolFactory(BoidSwarmFactory):

    @gin.configurable
    def fish_school_params(self):

        boids_settings = dict(
            use_flight = True,
            use_land = False,
            use_climb = False,

            rules = [
                dict(type='SEPARATE'),
                dict(type='GOAL'),
                dict(type='FLOCK'),
            ],

            air_speed_max = U(5, 10),
            air_acc_max = U(0.7, 1),
            air_personal_space = U(0.15, 2),
            bank = 0, # fish dont tip over / roll
            pitch = 0.4, #
            rule_fuzzy = U(0.6, 0.9)
        )

        return dict(      
            particle_size=U(0.3, 1),
            size_random=U(0.1, 0.7),

            use_rotation_instance=True,

            lifetime=bpy.context.scene.frame_end - bpy.context.scene.frame_start,
            warmup_frames=1, emit_duration=0, # all particles appear immediately
            emit_from='VOLUME',
            mass = 2,
            use_multiply_size_mass=True,
            effect_gravity=0,

            boids_settings=boids_settings
        )

    def __init__(self, factory_seed, bvh=None, coarse=False):
        with FixedSeed(factory_seed):
            settings = self.fish_school_params()
            col = make_asset_collection(FishFactory(factory_seed=randint(1e7), animation_mode='idle'), n=3)
        super().__init__(
            factory_seed, child_col=col, 
            collider_col=bpy.data.collections.get('colliders'),
            settings=settings, bvh=bvh,
            volume=("uniform", 3, 10), 
            coarse=coarse
        )

if __name__ == "__main__":
    import os
    for i in range(3):
        factory = FishFactory(i)
        root = factory.create_asset(i)
        root.location[0] = i * 3

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.abspath(os.curdir), "dev_fish5.blend"))