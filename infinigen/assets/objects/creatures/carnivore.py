# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import gin
import mathutils
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform as U
from pydantic import BaseModel, ConfigDict, Field

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
from infinigen.core.placement.parameters import AssetParameters, ParameterizedAssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import clip_gaussian
from infinigen.core.util.random import weighted_sample


class CarnivoreHairParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    density: int = 500000
    clump_n: Annotated[int, Field(ge=5, le=69)]
    avoid_features_dist: float = 0.01
    grooming: dict[str, Any]
    material: dict[str, float]


class CarnivoreCreatureParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hair: CarnivoreHairParams
    body_theta_scale: Annotated[float, Field(ge=0.7, le=1.3)]
    use_carnivore_head: bool
    head_length_rad1_rad2: tuple[float, float, float]
    jaw_coord: tuple[float, float, float]
    jaw_rest_y: Annotated[float, Field(ge=10.0, le=35.0)]
    nurbs_head_var: Annotated[float, Field(ge=0.7, le=1.3)]
    eye_radius: Annotated[float, Field(ge=0.0, le=0.06)]
    eye_t: Annotated[float, Field(ge=0.61, le=0.64)]
    eye_splay: Annotated[float, Field(ge=90.0, le=140.0)]
    eye_r: Annotated[float, Field(ge=0.8, le=0.9)]
    nose_coord: tuple[float, float, float]
    ear_t: Annotated[float, Field(ge=0.19, le=0.47)]
    ear_splay: Annotated[float, Field(ge=100.0, le=150.0)]
    ear_rot: tuple[float, float, float]
    neck_coord: Annotated[float, Field(ge=0.94, le=1.0)]
    neck_rest_y: Annotated[float, Field(ge=5.0, le=35.0)]
    shoulder_splay: Annotated[float, Field(ge=90.0, le=130.0)]
    shoulder_t: Annotated[float, Field(ge=0.08, le=0.12)]
    leg_length_rad1_rad2: tuple[float, float, float]


def sample_creature_params() -> CarnivoreCreatureParams:
    mat_roughness = U(0.4, 0.7)
    length = clip_gaussian(0.022, 0.03, 0.01, 0.1)
    puff = U(0.14, 0.4)
    hair = CarnivoreHairParams(
        clump_n=int(np.random.randint(5, 70)),
        grooming={
            "Length MinMaxScale": np.array(
                (length, length * N(2, 0.5), U(15, 60)), dtype=np.float32
            ),
            "Puff MinMaxScale": np.array(
                (puff, puff * N(3, 0.5), U(15, 60)), dtype=np.float32
            ),
            "Combing": U(0.7, 1),
            "Strand Random Mag": 0.0,
            "Strand Perlin Mag": U(0, 0.006),
            "Strand Perlin Scale": U(15, 45),
            "Tuft Spread": N(0.01, 0.002),
            "Tuft Clumping": U(0.2, 0.8),
            "Root Radius": 0.001,
            "Post Clump Noise Mag": 0.0005 * N(1, 0.15),
            "Hair Length Pct Min": U(0.5, 0.9),
        },
        material={
            "Roughness": mat_roughness,
            "Radial Roughness": mat_roughness + N(0, 0.07),
            "Random Roughness": 0,
            "IOR": 1.55,
        },
    )
    use_carnivore_head = U() < 0.5
    if use_carnivore_head:
        head_length_rad1_rad2 = tuple(
            (np.array((0.36, 0.20, 0.18)) * N(1, 0.1, 3)).tolist()
        )
        jaw_coord = (0.2 * N(1, 0.1), 0.0, 0.35 * N(1, 0.1))
    else:
        headl = N(1, 0.1)
        head_length_rad1_rad2 = tuple(
            (np.array((headl, 0.20, 0.18)) * N(1, 0.1, 3)).tolist()
        )
        jaw_coord = (0.12, 0.0, 0.3 * N(1, 0.1))
    return CarnivoreCreatureParams(
        hair=hair,
        body_theta_scale=float(N(1, 0.1)),
        use_carnivore_head=use_carnivore_head,
        head_length_rad1_rad2=head_length_rad1_rad2,
        jaw_coord=jaw_coord,
        jaw_rest_y=float(U(10, 35)),
        nurbs_head_var=float(N(1, 0.1)),
        eye_radius=float(N(0.027, 0.009)),
        eye_t=float(U(0.61, 0.64)),
        eye_splay=float(U(90, 140)),
        eye_r=float(U(0.8, 0.9)),
        nose_coord=(float(U(0.9, 0.96)), 1.0, float(U(0.5, 0.7))),
        ear_t=float(N(0.33, 0.07)),
        ear_splay=float(U(100, 150)),
        ear_rot=tuple((np.array([-20, -10, -23]) + N(0, 4, 3)).tolist()),
        neck_coord=float(N(0.97, 0.01)),
        neck_rest_y=float(N(20, 5)),
        shoulder_splay=float(clip_gaussian(130, 7, 90, 130)),
        shoulder_t=float(clip_gaussian(0.12, 0.05, 0.08, 0.12)),
        leg_length_rad1_rad2=tuple(
            (np.array((1.6, 0.1, 0.05)) * N(1, (0.15, 0.05, 0.05), 3)).tolist()
        ),
    )


def tiger_hair_params(params: CarnivoreCreatureParams) -> dict[str, Any]:
    hair = params.hair
    return {
        "density": hair.density,
        "clump_n": hair.clump_n,
        "avoid_features_dist": hair.avoid_features_dist,
        "grooming": hair.grooming,
        "material": hair.material,
    }


def tiger_skin_sim_params():
    return {
        "bending_stiffness_max": 450.0,
        "compression_stiffness_max": 80.0,
        "goal_spring": 0.8,
        "pin_stiffness": 1,
        "shear_stiffness": 15.0,
        "shear_stiffness_max": 80.0,
        "tension_stiffness_max": 80.0,
        "uniform_pressure_force": 5.0,
        "use_pressure": True,
    }


def tiger_genome(params: CarnivoreCreatureParams):
    p = params
    body_fac = parts.generic_nurbs.NurbsBody(
        prefix="body_feline", tags=["body"], var=0.7, temperature=0.2
    )
    body_fac.params["thetas"][-3] *= p.body_theta_scale
    body = genome.part(body_fac)

    tail = genome.part(parts.tail.Tail())
    genome.attach(tail, body, coord=(0.07, 1, 1), joint=Joint(rest=(N(0, 10), 180, 0)))

    head_length_rad1_rad2 = np.array(p.head_length_rad1_rad2)
    if p.use_carnivore_head:
        head_fac = parts.head.CarnivoreHead({"length_rad1_rad2": head_length_rad1_rad2})
        head = genome.part(head_fac)

        jaw_pct = np.array((1.05, 0.55, 0.5))
        jaw = genome.part(
            parts.head.CarnivoreJaw(
                {"length_rad1_rad2": head_length_rad1_rad2 * jaw_pct}
            )
        )
        genome.attach(
            jaw,
            head,
            coord=p.jaw_coord,
            joint=Joint(rest=(0, p.jaw_rest_y, 0), pose=(0, 0, 0)),
        )

    else:
        head_fac = parts.generic_nurbs.NurbsHead(
            prefix="head_carnivore", tags=["head"], var=p.nurbs_head_var
        )
        head = genome.part(head_fac)

        jaw_pct = np.array((0.7, 0.55, 0.5))
        jaw = genome.part(
            parts.head.CarnivoreJaw(
                {"length_rad1_rad2": head_length_rad1_rad2 * jaw_pct}
            )
        )
        genome.attach(
            jaw,
            head,
            coord=p.jaw_coord,
            joint=Joint(rest=(0, p.jaw_rest_y, 0), pose=(0, 0, 0)),
        )

        eye_fac = parts.eye.MammalEye({"Radius": p.eye_radius})
        splay = p.eye_splay / 180
        rot = np.array([0, 0, 0])
        for side in [-1, 1]:
            eye = genome.part(eye_fac)
            genome.attach(
                eye,
                head,
                coord=(p.eye_t, splay, p.eye_r),
                joint=Joint(rest=rot),
                rotation_basis="normal",
                side=side,
            )

    nose = genome.part(parts.head_detail.CatNose())
    genome.attach(
        nose, head, coord=p.nose_coord, joint=Joint(rest=(0, 20, 0))
    )

    ear_fac = parts.head_detail.CatEar()
    rot = np.array(p.ear_rot)
    for side in [-1, 1]:
        ear = genome.part(ear_fac)
        genome.attach(
            ear,
            head,
            coord=(p.ear_t, p.ear_splay / 180, 1),
            joint=Joint(rest=rot),
            rotation_basis="normal",
            side=side,
        )

    neck_t = 0.7
    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    splay = p.shoulder_splay / 180
    leg_params = {"length_rad1_rad2": np.array(p.leg_length_rad1_rad2)}

    foot_fac = parts.foot.Foot()
    backleg_fac = parts.leg.QuadrupedBackLeg(params=leg_params)
    for side in [-1, 1]:
        back_leg = genome.attach(
            genome.part(foot_fac),
            genome.part(backleg_fac),
            coord=(0.9, 0, 0),
            joint=Joint(rest=(0, 0, 0)),
        )
        genome.attach(
            back_leg,
            body,
            coord=(p.shoulder_t, splay, 1.2),
            joint=Joint(rest=(0, 90, 0), bounds=shoulder_bounds),
            rotation_basis="global",
            side=side,
        )

    frontleg_fac = parts.leg.QuadrupedFrontLeg(params=leg_params)
    for side in [-1, 1]:
        front_leg = genome.attach(
            genome.part(foot_fac),
            genome.part(frontleg_fac),
            coord=(0.9, 0, 0),
            joint=Joint(rest=(0, 0, 0)),
        )
        genome.attach(
            front_leg,
            body,
            coord=(neck_t - p.shoulder_t, splay, 0.8),
            joint=Joint(rest=(0, 90, 0)),
            rotation_basis="global",
            side=side,
        )

    genome.attach(
        head,
        body,
        coord=(p.neck_coord, 0, 0),
        joint=Joint(rest=(0, p.neck_rest_y, 0)),
        rotation_basis="global",
    )

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            hair=tiger_hair_params(p),
            skin=tiger_skin_sim_params(),
        ),
    )


class CarnivoreParameters(AssetParameters):
    creature: CarnivoreCreatureParams
    body_material: Any = Field(json_schema_extra={"editable": False})
    tongue_material: Any = Field(json_schema_extra={"editable": False})
    teeth_material: Any = Field(json_schema_extra={"editable": False})
    nose_material: Any = Field(json_schema_extra={"editable": False})


@gin.configurable
class CarnivoreFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = CarnivoreParameters

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
        self.bvh = bvh
        self.animation_mode = animation_mode
        self.hair = hair
        self.clothsim_skin = clothsim_skin
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> CarnivoreParameters:
        return CarnivoreParameters(
            seed=seed,
            creature=sample_creature_params(),
            body_material=weighted_sample(material_assignments.carnivore)(),
            tongue_material=materials.creature.Tongue(),
            teeth_material=materials.creature.Bone(),
            nose_material=materials.creature.Nose(),
        )

    def apply_parameters(
        self, params: CarnivoreParameters, *, spawn_scope: bool = True
    ) -> None:
        if self.hair and (self.animation_mode is not None or self.clothsim_skin):
            raise NotImplementedError(
                "Dynamic hair is not yet fully working. "
                "Please disable either hair or both of animation/clothsim"
            )
        self.creature_params = params.creature
        self.body_material = params.body_material
        self.tongue_material = params.tongue_material
        self.teeth_material = params.teeth_material
        self.nose_material = params.nose_material
        self._use_fixed_spawn_draws = spawn_scope

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
        genome = tiger_genome(self.creature_params)
        root, parts = creature.genome_to_creature(
            genome, name=f"carnivore({self.factory_seed}, {i})"
        )
        # tag_object(root, 'carnivore')
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
                joined, root, genome.postprocess_params["hair"], is_dynamic=dynamic
            )

        if dynamic:
            if self.animation_mode == "run":
                run_cycle.animate_run(root, arma, ik_targets)
            elif self.animation_mode == "idle":
                idle.snap_iks_to_floor(ik_targets, self.bvh)
                idle.idle_body_noise_drivers(ik_targets)
            elif self.animation_mode == "tpose":
                pass
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
