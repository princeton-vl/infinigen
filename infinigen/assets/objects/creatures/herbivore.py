# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from collections import defaultdict

import gin
import mathutils
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform as U

from infinigen.assets import materials
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.creatures import parts
from infinigen.assets.objects.creatures.util import cloth_sim, creature, genome, joining
from infinigen.assets.objects.creatures.util import hair as creature_hair
from infinigen.assets.objects.creatures.util.animation import idle, run_cycle
from infinigen.assets.objects.creatures.util.creature_util import offset_center
from infinigen.assets.objects.creatures.util.genome import Joint
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, clip_gaussian
from infinigen.core.util.random import weighted_sample


def herbivore_hair():
    mat_roughness = U(0.5, 0.9)

    puff = U(0.14, 0.4)
    length = clip_gaussian(0.035, 0.03, 0.01, 0.1)

    return {
        "density": 500000,
        "clump_n": np.random.randint(10, 300),
        "avoid_features_dist": 0.06,
        "grooming": {
            "Length MinMaxScale": np.array(
                (length, length * U(1.5, 4), U(15, 60)), dtype=np.float32
            ),
            "Puff MinMaxScale": np.array(
                (puff, U(0.5, 1.3), U(15, 60)), dtype=np.float32
            ),
            "Combing": U(0.5, 1),
            "Strand Random Mag": U(0, 0.003) if U() < 0.5 else 0,
            "Strand Perlin Mag": U(0, 0.006),
            "Strand Perlin Scale": U(15, 45),
            "Tuft Spread": N(0.06, 0.025),
            "Tuft Clumping": U(0.7, 0.95),
            "Root Radius": 0.0025,
            "Post Clump Noise Mag": 0.001 * N(1, 0.15),
            "Hair Length Pct Min": U(0.5, 0.9),
        },
        "material": {
            "Roughness": mat_roughness,
            "Radial Roughness": mat_roughness + N(0, 0.07),
            "Random Roughness": 0,
            "IOR": 1.55,
        },
    }


def herbivore_genome():
    temp_dict = defaultdict(
        lambda: 0.2, {"body_herbivore_giraffe": 0.02, "body_herbivore_llama": 0.1}
    )
    body = genome.part(
        parts.generic_nurbs.NurbsBody(
            prefix="body_herbivore", tags=["body"], var=1, temperature=temp_dict
        )
    )

    neck_t = 0.67
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    splay = clip_gaussian(130, 7, 90, 130) / 180
    shoulder_t = clip_gaussian(0.1, 0.05, 0.05, 0.2)
    params = {
        "length_rad1_rad2": np.array((1.8, 0.1, 0.05)) * N(1, (0.1, 0.05, 0.05), 3)
    }

    leg_rest = (0, 90, 0)  # (0, 90, 0)
    foot_rest = (0, -90, 0)
    foot_fac = parts.hoof.HoofAnkle()
    claw_fac = parts.hoof.HoofClaw()
    backleg_fac = parts.leg.QuadrupedBackLeg(params=params)
    frontleg_fac = parts.leg.QuadrupedFrontLeg(params=params)

    if U() < 0.15:
        lenscale = U(1, 1.3)
        backleg_fac.params["length_rad1_rad2"][0] *= lenscale
        frontleg_fac.params["length_rad1_rad2"][0] *= lenscale

    for side in [-1, 1]:
        # foot = genome.part(claw_fac)
        foot = genome.attach(
            genome.part(claw_fac),
            genome.part(foot_fac),
            coord=(0.7, -1, 0),
            joint=Joint(rest=(0, 90, 0)),
            rotation_basis="global",
        )
        back_leg = genome.attach(
            foot,
            genome.part(backleg_fac),
            coord=(0.95, 1, 0.2),
            joint=Joint(rest=foot_rest),
            rotation_basis="global",
        )
        genome.attach(
            back_leg,
            body,
            coord=(shoulder_t, splay, 1),
            joint=Joint(rest=leg_rest, bounds=shoulder_bounds),
            rotation_basis="global",
            side=side,
        )

    for side in [-1, 1]:
        # foot = genome.part(claw_fac)
        foot = genome.attach(
            genome.part(claw_fac),
            genome.part(foot_fac),
            coord=(0.7, 1, 0),
            joint=Joint(rest=(0, 90, 0)),
            rotation_basis="normal",
        )
        front_leg = genome.attach(
            foot,
            genome.part(frontleg_fac),
            coord=(0.95, 0, 0.5),
            joint=Joint(rest=(0, -70, 0)),
        )
        genome.attach(
            front_leg,
            body,
            coord=(neck_t - shoulder_t, splay + 0 / 180, 0.9),
            joint=Joint(rest=leg_rest),
            rotation_basis="global",
            side=side,
        )

    temp_dict = defaultdict(lambda: 0.2, {"body_herbivore_giraffe": 0.02})
    head_fac = parts.generic_nurbs.NurbsHead(
        prefix="head_herbivore", tags=["head"], var=0.5, temperature=temp_dict
    )
    head = genome.part(head_fac)

    eye_fac = parts.eye.MammalEye({"Radius": N(0.035, 0.01)})
    eye_t, splay = U(0.34, 0.45), U(80, 140) / 180
    r = U(0.7, 0.9)
    rot = np.array([0, 0, 0])
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(
            eye,
            head,
            coord=(eye_t, splay, r),
            joint=Joint(rest=rot),
            rotation_basis="normal",
            side=side,
        )

    jaw = genome.part(
        parts.head.CarnivoreJaw(
            {
                "length_rad1_rad2": (0.6 * head_fac.params["length"], 0.12, 0.08),
                "Canine Length": 0,
            }
        )
    )
    genome.attach(
        jaw,
        head,
        coord=(0.25 * N(1, 0.1), 0, 0.35 * N(1, 0.1)),
        joint=Joint(rest=(0, 10 * N(1, 0.1), 0)),
    )

    if U() < 0.7:
        nose = genome.part(parts.head_detail.CatNose())
        genome.attach(nose, head, coord=(0.95, 1, 0.45), joint=Joint(rest=(0, 20, 0)))

    t, splay = U(0.15, eye_t - 0.07), N(125, 15) / 180
    ear_fac = parts.head_detail.CatEar({})
    ear_fac.params["length_rad1_rad2"] *= N(1.2, 0.1, 3)
    rot = np.array([0, -10, -23]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        ear = genome.part(ear_fac)
        genome.attach(
            ear,
            head,
            coord=(t, splay, 1),
            joint=Joint(rest=rot),
            rotation_basis="normal",
            side=side,
        )

    if U() < 0.7:
        horn_fac = parts.horn.Horn()
        horn_fac.params["length"] *= U(0.1, 2)
        horn_fac.params["rad1"] *= U(0.07, 1.5)
        horn_fac.params["rad2"] *= U(0.07, 1.5)
        t, splay = U(0.25, t), U(splay + 20 / 180, 130 / 180)
        rot = np.array([U(-40, 0), 0, N(120, 10)])
        for side in [-1, 1]:
            horn = genome.part(horn_fac)
            genome.attach(
                horn,
                head,
                coord=(t, splay, 0.5),
                joint=Joint(rest=rot),
                rotation_basis="global",
                side=side,
            )
    elif U() < 0:
        horn_fac = parts.horn.Horn()
        horn_fac.params["length"] *= U(0.3, 1)
        horn_fac.params["rotation_x"] = 0
        horn = genome.part(horn_fac)
        genome.attach(
            horn,
            head,
            coord=(U(0.3, 0.9), 1, 0.6),
            joint=Joint(rest=(0, -90, -90)),
            rotation_basis="global",
        )

    genome.attach(head, body, coord=(0.97, 0, 0.2), joint=Joint(rest=(0, 20, 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(),
            hair=herbivore_hair(),
        ),
    )


@gin.configurable
class HerbivoreFactory(AssetFactory):
    max_distance = 40

    def __init__(
        self,
        factory_seed=None,
        bvh: mathutils.bvhtree.BVHTree = None,
        coarse: bool = False,
        animation_mode: str = None,
        hair: bool = True,
        clothsim_skin: bool = False,
        **kwargs,
    ):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode
        self.hair = hair
        self.clothsim_skin = clothsim_skin

        if self.hair and (self.animation_mode is not None or self.clothsim_skin):
            raise NotImplementedError(
                "Dynamic hair is not yet fully working. "
                "Please disable either hair or both of animation/clothsim"
            )

        with FixedSeed(self.factory_seed):
            body_material_fac = weighted_sample(material_assignments.herbivore)
            self.body_material = body_material_fac()
            self.tongue_material = materials.creature.Tongue()
            self.teeth_material = materials.creature.Bone()
            self.nose_material = materials.creature.Nose()

    def create_placeholder(self, **kwargs):
        return butil.spawn_cube(size=4)

    def apply_materials(self, root):
        self.body_material.apply(
            joining.get_parts(root, True) + joining.get_parts(root, False, "BodyExtra")
        )
        self.body_material.apply(joining.get_parts(root, False, "Tongue"))

        # TODO move these into the individual part generators
        self.tongue_material.apply(
            joining.get_parts(root, False, "Teeth")
            + joining.get_parts(root, False, "Claws")
        )
        self.teeth_material.apply(
            joining.get_parts(root, False, "Eyeball"), shader_kwargs={"coord": "X"}
        )
        self.nose_material.apply(joining.get_parts(root, False, "Nose"))

    def create_asset(self, i, placeholder, **kwargs):
        genome = herbivore_genome()
        root, parts = creature.genome_to_creature(
            genome, name=f"herbivore({self.factory_seed}, {i})"
        )
        # tag_object(root, 'herbivore')
        offset_center(root)

        dynamic = self.animation_mode is not None

        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            rigging=dynamic,
            postprocess_func=self.apply_materials,
            **kwargs,
        )

        butil.parent_to(root, placeholder, no_inverse=True)

        if self.hair:
            creature_hair.configure_hair(
                joined, root, genome.postprocess_params["hair"]
            )
        if dynamic:
            if self.animation_mode == "run":
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == "idle":
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets)
            else:
                raise ValueError(f"Unrecognized mode {self.animation_mode=}")
        if self.clothsim_skin:
            rigidity = surface.write_vertex_group(
                joined, cloth_sim.local_pos_rigity_mask, apply=True
            )
            cloth_sim.bake_cloth(
                joined,
                genome.postprocess_params["skin"],
                attributes=dict(vertex_group_mass=rigidity),
            )

        return root
