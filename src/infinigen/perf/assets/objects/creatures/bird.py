# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: regular bird and hair parameters in the original Infinigen bird (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/creatures/bird.py)
# - Beining Han: flying bird in the original Infinigen bird (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/creatures/bird.py)
# - David Yan, Jia Deng: optimized implementation

from numpy.random import uniform as U

from infinigen.assets.objects.creatures.bird import BirdFactory as _DefaultBirdFactory
from infinigen.assets.objects.creatures.bird import (
    FlyingBirdFactory as _DefaultFlyingBirdFactory,
)
from infinigen.assets.objects.creatures.bird import (
    apply_bird_materials,
    duck_genome,
    flying_bird_genome,
)
from infinigen.assets.objects.creatures.util import creature, joining
from infinigen.assets.objects.creatures.util import hair as creature_hair
from infinigen.assets.objects.creatures.util.animation import idle, run_cycle
from infinigen.assets.objects.creatures.util.animation.driver_wiggle import (
    animate_wiggle_bones,
)
from infinigen.assets.objects.creatures.util.creature_util import offset_center
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.perf.assets.objects.creatures.util import mesh_join


class BirdFactory(_DefaultBirdFactory):
    def create_asset(self, i, placeholder, hair=True, **kwargs):
        dynamic = self.animation_mode is not None

        genome = duck_genome(mode=self.animation_mode)
        root, parts = creature.genome_to_creature(
            genome, name=f"bird({self.factory_seed}, {i})"
        )
        tag_object(root, "bird")
        offset_center(root)
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            rigging=dynamic,
            postprocess_func=lambda root: apply_bird_materials(self, root),
            **kwargs,
        )

        joined_extras = mesh_join.join_objects_exact(extras)
        joined_extras.parent = joined

        butil.parent_to(root, placeholder, no_inverse=True)

        if hair:
            creature_hair.configure_hair(
                joined, root, genome.postprocess_params["hair"]
            )
        if dynamic:
            if self.animation_mode == "run":
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == "idle":
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets, wing_mag=U(0, 0.3))
            elif self.animation_mode == "swim":
                spine = [b for b in arma.pose.bones if "Body" in b.name]
                tail = [b for b in arma.pose.bones if "Tail" in b.name]
                animate_wiggle_bones(
                    arma=arma, bones=tail, mag_deg=U(0, 30), freq=U(0.5, 2)
                )
            else:
                raise ValueError(f"Unrecognized mode {self.animation_mode=}")
        return root


class FlyingBirdFactory(_DefaultFlyingBirdFactory):
    max_expected_radius = 1

    max_distance = 40

    def create_asset(self, i, placeholder, hair=True, animate=False, **kwargs):
        genome = flying_bird_genome(self.animation_mode)
        root, parts = creature.genome_to_creature(
            genome, name=f"flying_bird({self.factory_seed}, {i})"
        )
        joined, extras, arma, ik_targets = joining.join_and_rig_parts(
            root,
            parts,
            genome,
            rigging=self.animation_mode is not None,
            postprocess_func=lambda root: apply_bird_materials(self, root),
            **kwargs,
        )

        joined_extras = mesh_join.join_objects_exact(extras)
        joined_extras.parent = joined

        if hair:
            creature_hair.configure_hair(
                joined, root, genome.postprocess_params["hair"]
            )
        if self.animation_mode is not None:
            if self.animation_mode == "idle":
                idle.idle_body_noise_drivers(
                    ik_targets, body_mag=0.0, foot_motion_chance=1.0, head_benddown=0
                )
            else:
                raise ValueError(f"Unrecognized {self.animation_mode=}")

        return root
