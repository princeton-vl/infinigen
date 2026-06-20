# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: regular bird, hair params
# - Beining Han: adapt to create flying bird

from __future__ import annotations

from typing import Annotated, Any, ClassVar

import bpy
import gin
import numpy as np
from numpy.random import normal as N
from numpy.random import uniform as U
from pydantic import Field

import infinigen.assets.materials.creature as creature_materials
from infinigen.assets.composition import material_assignments
from infinigen.assets.objects.creatures import parts
from infinigen.assets.objects.creatures.util import creature, genome, joining
from infinigen.assets.objects.creatures.util import hair as creature_hair
from infinigen.assets.objects.creatures.util.animation import idle, run_cycle
from infinigen.assets.objects.creatures.util.animation.driver_wiggle import (
    animate_wiggle_bones,
)
from infinigen.assets.objects.creatures.util.creature_util import offset_center
from infinigen.assets.objects.creatures.util.genome import Joint
from infinigen.core.placement import animation_policy
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.parameters import (
    AssetParameters,
    LegacyBridgeParameters,
    ParameterizedAssetFactory,
    apply_bridge_parameters,
    legacy_init_to_parameters,
)
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, clip_gaussian
from infinigen.core.util.random import random_general as rg
from infinigen.core.util.random import weighted_sample


class BirdParameters(AssetParameters):
    Claw_Curl_Deg: Annotated[
        float, Field(ge=-0.2, le=2.2, json_schema_extra={"editable": True})
    ] = 1.0
    Extension: Annotated[
        float, Field(ge=0.01, le=0.1, json_schema_extra={"editable": True})
    ] = 0.05
    Eyelid_Fullness: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    Eyelid_Thickness_Ratio: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ] = 1.0
    Feather_Length_Rad1_Rad2: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    Feather_Rot_Rand_Bounds: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    Feather_length_rad1_rad2: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    N_Feathers: Annotated[
        float, Field(ge=7.0, le=25.0, json_schema_extra={"editable": True})
    ] = 16.0
    Num_Toes: Annotated[
        float, Field(ge=1.0, le=7.0, json_schema_extra={"editable": True})
    ] = 4.0
    Radius: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ] = 1.0
    Radius_1: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    Toe_Rotate: Annotated[
        float, Field(ge=0.25, le=1.15, json_schema_extra={"editable": True})
    ] = 0.7
    Toe_Splay: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    Toebean_Radius: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    Wing_Shape_Sculpting: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ] = 0.8
    bird_79: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    aspect: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ] = 1.0
    aspect_1: Annotated[
        float, Field(ge=0.25, le=0.55, json_schema_extra={"editable": True})
    ] = 0.4
    body_fac: Annotated[
        float, Field(ge=0.3, le=1.0, json_schema_extra={"editable": True})
    ] = 0.65
    eye_fac: Annotated[
        float, Field(ge=0.015, le=0.045, json_schema_extra={"editable": True})
    ] = 0.03
    foot_fac: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_1: Annotated[
        float, Field(ge=0.19, le=0.61, json_schema_extra={"editable": True})
    ] = 0.4
    foot_fac_2: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_3: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_4: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_5: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_6: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    foot_fac_7: Annotated[
        float, Field(ge=0.4, le=1.6, json_schema_extra={"editable": True})
    ] = 1.0
    fullness: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    fullness_1: Annotated[
        float, Field(ge=3.7, le=4.3, json_schema_extra={"editable": True})
    ] = 4.0
    grooming: Annotated[
        float, Field(ge=0.6, le=1.0, json_schema_extra={"editable": True})
    ] = 0.8
    grooming_1: Annotated[
        float, Field(ge=0.0, le=0.003, json_schema_extra={"editable": True})
    ] = 0.0015
    grooming_2: Annotated[
        float, Field(ge=0.5, le=1.0, json_schema_extra={"editable": True})
    ] = 0.75
    grooming_3: Annotated[
        float, Field(ge=0.5, le=0.9, json_schema_extra={"editable": True})
    ] = 0.7
    leg_fac: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ] = 1.0
    leg_fac_1: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    leg_fac_2: Annotated[
        float, Field(ge=0.7, le=1.3, json_schema_extra={"editable": True})
    ] = 1.0
    length: Annotated[
        float, Field(ge=0.03, le=0.06, json_schema_extra={"editable": True})
    ] = 0.045
    length_rad1_rad2: Annotated[
        float, Field(ge=0.4, le=2.0, json_schema_extra={"editable": True})
    ] = 1.2
    length_rad1_rad2_1: Annotated[
        float, Field(ge=0.85, le=1.15, json_schema_extra={"editable": True})
    ] = 1.0
    material: Annotated[
        float, Field(ge=0.0, le=0.4, json_schema_extra={"editable": True})
    ] = 0.2
    material_1: Annotated[
        float, Field(ge=0.1, le=0.3, json_schema_extra={"editable": True})
    ] = 0.2
    material_2: Annotated[
        float, Field(ge=0.0, le=0.2, json_schema_extra={"editable": True})
    ] = 0.1
    puff: Annotated[
        float, Field(ge=0.03, le=0.2, json_schema_extra={"editable": True})
    ] = 0.1
    r: Annotated[int, Field(ge=0, le=60301, json_schema_extra={"editable": True})] = 0
    wing_len: Annotated[
        float, Field(ge=0.5, le=2.5, json_schema_extra={"editable": True})
    ] = 1.2
    idle_wing_mag: Annotated[
        float, Field(ge=0.0, le=0.3, json_schema_extra={"editable": True})
    ] = 0.15
    body_material: Any = Field(json_schema_extra={"editable": False})
    tongue_material: Any = Field(json_schema_extra={"editable": False})
    bone_material: Any = Field(json_schema_extra={"editable": False})
    eyeball_material: Any = Field(json_schema_extra={"editable": False})
    beak_material: Any = Field(json_schema_extra={"editable": False})


def _sample_bird_spawn_parameters() -> dict[str, Any]:
    return {
        "Claw_Curl_Deg": N(1.0, 0.4),
        "Extension": U(0.01, 0.1),
        "Eyelid_Fullness": N(1.0, 0.1),
        "Eyelid_Thickness_Ratio": N(1.0, 0.05),
        "Feather_Length_Rad1_Rad2": N(1.0, 0.1),
        "Feather_Rot_Rand_Bounds": N(1.0, 0.1),
        "Feather_length_rad1_rad2": N(1.0, 0.2),
        "N_Feathers": N(16.0, 3.0),
        "Num_Toes": N(4.0, 1.0),
        "Radius": N(1.0, 0.05),
        "Radius_1": N(1.0, 0.1),
        "Toe_Rotate": N(0.7, 0.15),
        "Toe_Splay": N(1.0, 0.2),
        "Toebean_Radius": N(1.0, 0.2),
        "Wing_Shape_Sculpting": U(0.6, 1.0),
        "bird_79": N(1.0, 0.1),
        "aspect": N(1.0, 0.05),
        "aspect_1": N(0.4, 0.05),
        "body_fac": U(0.3, 1.0),
        "eye_fac": N(0.03, 0.005),
        "foot_fac": N(1.0, 0.1),
        "foot_fac_1": N(0.4, 0.07),
        "foot_fac_2": N(1.0, 0.1),
        "foot_fac_3": N(1.0, 0.2),
        "foot_fac_4": N(1.0, 0.1),
        "foot_fac_5": N(1.0, 0.2),
        "foot_fac_6": N(1.0, 0.1),
        "foot_fac_7": N(1.0, 0.2),
        "fullness": N(1.0, 0.1),
        "fullness_1": N(4.0, 0.1),
        "grooming": U(0.6, 1.0),
        "grooming_1": U(0.0, 0.003),
        "grooming_2": U(0.5, 1.0),
        "grooming_3": U(0.5, 0.9),
        "leg_fac": N(1.0, 0.05),
        "leg_fac_1": N(1.0, 0.1),
        "leg_fac_2": N(1.0, 0.1),
        "length": U(0.03, 0.06),
        "length_rad1_rad2": clip_gaussian(1.2, 0.7, 0.5, 2.5),
        "length_rad1_rad2_1": N(1.0, 0.05),
        "material": U(0.0, 0.4),
        "material_1": U(0.1, 0.3),
        "material_2": U(0.0, 0.2),
        "puff": U(0.03, 0.2),
        "r": int(np.random.randint(0, 60302)),
        "wing_len": clip_gaussian(1.2, 0.7, 0.5, 2.5),
        "idle_wing_mag": U(0.0, 0.3),
    }


def bird_hair_params(flying=True, p: BirdParameters | None = None):
    length = p.length if p is not None else (U(0.01, 0.025) if flying else U(0.03, 0.06))
    puff = p.puff if p is not None else U(0.03, 0.2)
    grooming = p.grooming if p is not None else U(0.6, 1.0)
    grooming_1 = p.grooming_1 if p is not None else U(0.0, 0.003)
    grooming_2 = p.grooming_2 if p is not None else U(0.5, 1.0)
    grooming_3 = p.grooming_3 if p is not None else U(0.5, 0.9)
    material = p.material if p is not None else U(0.0, 0.4)
    material_1 = p.material_1 if p is not None else U(0.1, 0.3)
    material_2 = p.material_2 if p is not None else U(0.0, 0.2)

    return {
        "density": 70000,
        "clump_n": 10,
        "avoid_features_dist": 0.02,
        "grooming": {
            "Length MinMaxScale": np.array(
                (length, length * N(2, 0.5), U(15, 60)), dtype=np.float32
            ),
            "Puff MinMaxScale": np.array(
                (puff, puff * N(1.5, 0.5), U(15, 60)), dtype=np.float32
            ),
            "Combing": grooming,
            "Strand Random Mag": 0.0,
            "Strand Perlin Mag": grooming_1,
            "Strand Perlin Scale": 30.0,
            "Tuft Spread": 0.01,
            "Tuft Clumping": grooming_2,
            "Root Radius": 0.006,
            "Post Clump Noise Mag": 0.001,
            "Hair Length Pct Min": grooming_3,
        },
        "material": {
            "Roughness": material,
            "Radial Roughness": material_1,
            "Random Roughness": material_2,
            "IOR": 1.55,
        },
    }


def duck_genome(mode, p: BirdParameters | None = None):
    body_lrr = (
        np.array((0.85, 0.25, 0.38)) * p.bird_79 * np.array([p.aspect, p.aspect_1, p.aspect_1])
        if p is not None
        else np.array((0.85, 0.25, 0.38)) * N(1, 0.2) * N(1, 0.2, 3)
    )
    body_fac = parts.generic_nurbs.NurbsBody(
        prefix="body_bird",
        tags=["body", "rigid"],
        var=p.body_fac if p is not None else U(0.3, 1),
    )
    body = genome.part(body_fac)
    body_length = body_fac.params["length"][0]

    tail = genome.part(parts.wings.BirdTail())
    genome.attach(
        tail, body, coord=(0.2, 1, 0.5), joint=Joint(rest=(0, 170 * N(1, 0.1), 0))
    )

    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    foot_fac = parts.foot.Foot(
        {
            "length_rad1_rad2": np.array((body_length * 0.1, 0.025, 0.04))
            * (p.length_rad1_rad2_1 if p is not None else N(1, 0.1))
            * (np.array([p.foot_fac, p.foot_fac, p.foot_fac]) if p is not None else N(1, 0.1, 3)),
            "Toe Length Rad1 Rad2": np.array((body_length * (p.foot_fac_1 if p is not None else N(0.4, 0.07)), 0.03, 0.02))
            * (p.foot_fac_2 if p is not None else N(1, 0.1))
            * (np.array([p.foot_fac_3, p.foot_fac_3, p.foot_fac_3]) if p is not None else N(1, 0.1, 3)),
            "Toe Splay": (p.fullness_1 if p is not None else 35 * N(1, 0.2)),
            "Toebean Radius": 0.03 * (p.Toebean_Radius if p is not None else N(1, 0.1)),
            "Toe Rotate": (0.0, -1.57, 0.0),
            "Claw Curl Deg": (p.fullness_1 if p is not None else 12 * N(1, 0.2)),
            "Claw Pct Length Rad1 Rad2": np.array((0.13, 0.64, 0.05))
            * (p.foot_fac_4 if p is not None else N(1, 0.1))
            * (np.array([p.foot_fac_5, p.foot_fac_5, p.foot_fac_5]) if p is not None else N(1, 0.1, 3)),
            "Thumb Pct": np.array((0.61, 1.17, 1.5))
            * (p.foot_fac_6 if p is not None else N(1, 0.1))
            * (np.array([p.foot_fac_7, p.foot_fac_7, p.foot_fac_7]) if p is not None else N(1, 0.1, 3)),
            "Toe Curl Scalar": 0.34 * (p.Claw_Curl_Deg if p is not None else N(1, 0.2)),
        },
        bald=True,
    )

    leg_fac = parts.leg.BirdLeg(
        {
            "length_rad1_rad2": (
                body_length * 0.5 * (p.leg_fac if p is not None else N(1, 0.05)),
                0.09 * (p.leg_fac_1 if p is not None else N(1, 0.1)),
                0.06 * (p.leg_fac_2 if p is not None else N(1, 0.1)),
            )
        }
    )
    leg_coord = (N(0.5, 0.05), N(0.7, 0.05), N(0.95, 0.05))
    for side in [-1, 1]:
        leg = genome.attach(
            genome.part(foot_fac),
            genome.part(leg_fac),
            coord=(0.9, 0, 0),
            joint=Joint(rest=(0, 0, 0)),
        )
        genome.attach(
            leg,
            body,
            coord=leg_coord,
            joint=Joint(rest=(0, 90, 0), bounds=shoulder_bounds),
            side=side,
        )

    extension = (
        p.Extension
        if p is not None
        else (U(0.7, 1) if mode == "flying" else U(0.01, 0.1))
    )
    wing_len = (
        body_length * 0.5 * p.length_rad1_rad2
        if p is not None
        else body_length * 0.5 * clip_gaussian(1.2, 0.7, 0.5, 2.5)
    )
    wing_fac = parts.wings.BirdWing(
        {
            "length_rad1_rad2": np.array(
                (
                    wing_len,
                    0.1 * (p.fullness if p is not None else N(1, 0.1)),
                    0.02 * (p.Radius_1 if p is not None else N(1, 0.2)),
                )
            ),
            "Extension": extension,
        }
    )

    wing_coord = (N(0.7, 0.02), 110 / 180 * N(1, 0.1), 0.95)
    if wing_fac.params["Extension"] > 0.5:
        wing_rot = (90, 0, 90)
    else:
        wing_rot = (90, 40, 90)
    for side in [-1, 1]:
        wing = genome.part(wing_fac)
        genome.attach(
            wing, body, coord=wing_coord, joint=Joint(rest=wing_rot), side=side
        )

    head_fac = parts.head.BirdHead()
    head = genome.part(head_fac)

    beak = genome.part(parts.beak.BirdBeak())
    genome.attach(beak, head, coord=(0.75, 0, 0.5), joint=Joint(rest=(0, 0, 0)))

    eye_fac = parts.eye.MammalEye(
        {"Radius": p.eye_fac if p is not None else N(0.03, 0.005)}
    )
    t, splay = U(0.6, 0.85), U(80, 110) / 180
    r = 0.85
    rot = np.array([0, 0, 90]) * N(1, 0.1, 3)
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(
            eye,
            head,
            coord=(t, splay, r),
            joint=Joint(rest=(0, 0, 0)),
            rotation_basis="normal",
            side=side,
        )

    genome.attach(head, body, coord=(1, 0, 0), joint=Joint(rest=(0, 0, 0)))

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(),
            hair=bird_hair_params(flying=False, p=p),
        ),
    )


def _sample_flying_bird_spawn_parameters() -> dict[str, Any]:
    return {
        "body_lrr_scale": N(1.0, 0.05, size=(3,)).tolist(),
        "tail_coord_x": U(0.08, 0.15),
        "tail_joint_y": N(1, 0.1),
        "foot_length_scale": N(1, 0.1, 3).tolist(),
        "toe_length": N(0.4, 0.02),
        "toe_len_scale": N(1, 0.1),
        "toe_len_scale3": N(1, 0.1, 3).tolist(),
        "toe_splay_factor": N(1, 0.2),
        "toe_rotate": N(0.55, 0.1),
        "toebean": N(1, 0.1),
        "claw_curl": N(1, 0.2),
        "claw_pct_scale": N(0.5, 0.05),
        "claw_pct_scale3": N(1, 0.1, 3).tolist(),
        "thumb_scale": N(1, 0.1),
        "thumb_scale3": N(1, 0.1, 3).tolist(),
        "toe_curl": N(1, 0.2),
        "leg_length": N(1, 0.05),
        "leg_rad1": N(1, 0.1),
        "leg_rad2": N(1, 0.1),
        "thigh": N(1, 0.1, 3).tolist(),
        "shin": N(1, 0.1, 3).tolist(),
        "leg_coord_x": N(0.5, 0.05),
        "leg_coord_y": N(0.2, 0.04),
        "leg_coord_z": N(0.8, 0.05),
        "leg_joint": U(135, 175),
        "extension": U(0.8, 1),
        "wing_len_factor": clip_gaussian(1.0, 0.2, 0.6, 1.5),
        "wing_width": U(0.08, 0.15),
        "wing_rad": N(1, 0.2),
        "feather_density": U(25, 40),
        "wing_coord_x": N(0.68, 0.02),
        "wing_coord_y": N(1, 0.1),
        "eye_radius": N(0.02, 0.005),
        "eye_t": U(0.7, 0.85),
        "eye_splay": U(80, 110) / 180,
        "eye_rot": N(1, 0.1, 3).tolist(),
        "head_coord_x": U(0.84, 0.85),
        "head_coord_z": U(1.05, 1.15),
        "head_joint": N(18, 5),
        "length": U(0.01, 0.025),
        "puff": U(0.03, 0.2),
        "grooming": U(0.6, 1.0),
        "grooming_1": U(0.0, 0.003),
        "grooming_2": U(0.5, 1.0),
        "grooming_3": U(0.5, 0.9),
        "material": U(0.0, 0.4),
        "material_1": U(0.1, 0.3),
        "material_2": U(0.0, 0.2),
        "hair_length_scale": N(2, 0.5),
        "hair_length_pct": U(15, 60),
        "hair_puff_scale": N(1.5, 0.5),
        "hair_puff_pct": U(15, 60),
    }


def _flying_bird_hair_params(p: FlyingBirdParameters | None = None):
    length = p.length if p is not None else U(0.01, 0.025)
    puff = p.puff if p is not None else U(0.03, 0.2)
    grooming = p.grooming if p is not None else U(0.6, 1.0)
    grooming_1 = p.grooming_1 if p is not None else U(0.0, 0.003)
    grooming_2 = p.grooming_2 if p is not None else U(0.5, 1.0)
    grooming_3 = p.grooming_3 if p is not None else U(0.5, 0.9)
    material = p.material if p is not None else U(0.0, 0.4)
    material_1 = p.material_1 if p is not None else U(0.1, 0.3)
    material_2 = p.material_2 if p is not None else U(0.0, 0.2)
    hair_length_scale = (
        p.hair_length_scale if p is not None else N(2, 0.5)
    )
    hair_length_pct = p.hair_length_pct if p is not None else U(15, 60)
    hair_puff_scale = p.hair_puff_scale if p is not None else N(1.5, 0.5)
    hair_puff_pct = p.hair_puff_pct if p is not None else U(15, 60)

    return {
        "density": 70000,
        "clump_n": 10,
        "avoid_features_dist": 0.02,
        "grooming": {
            "Length MinMaxScale": np.array(
                (length, length * hair_length_scale, hair_length_pct),
                dtype=np.float32,
            ),
            "Puff MinMaxScale": np.array(
                (puff, puff * hair_puff_scale, hair_puff_pct), dtype=np.float32
            ),
            "Combing": grooming,
            "Strand Random Mag": 0.0,
            "Strand Perlin Mag": grooming_1,
            "Strand Perlin Scale": 30.0,
            "Tuft Spread": 0.01,
            "Tuft Clumping": grooming_2,
            "Root Radius": 0.006,
            "Post Clump Noise Mag": 0.001,
            "Hair Length Pct Min": grooming_3,
        },
        "material": {
            "Roughness": material,
            "Radial Roughness": material_1,
            "Random Roughness": material_2,
            "IOR": 1.55,
        },
    }


def flying_bird_genome(mode, p: FlyingBirdParameters | None = None):
    body_lrr_scale = (
        np.array(p.body_lrr_scale)
        if p is not None
        else N(1.0, 0.05, size=(3,))
    )
    body_lrr = np.array((0.95, 0.13, 0.18)) * body_lrr_scale
    body = genome.part(parts.body.BirdBody({"length_rad1_rad2": body_lrr}))
    body_length = body_lrr[0]

    tail = genome.part(parts.wings.FlyingBirdTail())
    genome.attach(
        tail,
        body,
        coord=(
            p.tail_coord_x if p is not None else U(0.08, 0.15),
            1,
            0.5,
        ),
        joint=Joint(
            rest=(
                0,
                180 * (p.tail_joint_y if p is not None else N(1, 0.1)),
                0,
            )
        ),
    )

    shoulder_bounds = np.array([[-20, -20, -20], [20, 20, 20]])
    foot_length_scale = (
        np.array(p.foot_length_scale) if p is not None else N(1, 0.1, 3)
    )
    toe_length = p.toe_length if p is not None else N(0.4, 0.02)
    toe_len_scale = p.toe_len_scale if p is not None else N(1, 0.1)
    toe_len_scale3 = (
        np.array(p.toe_len_scale3) if p is not None else N(1, 0.1, 3)
    )
    foot_fac = parts.foot.Foot(
        {
            "length_rad1_rad2": np.array((body_length * 0.2, 0.01, 0.02))
            * foot_length_scale,
            "Toe Length Rad1 Rad2": np.array((body_length * toe_length, 0.02, 0.01))
            * toe_len_scale
            * toe_len_scale3,
            "Toe Splay": 8 * (p.toe_splay_factor if p is not None else N(1, 0.2)),
            "Toe Rotate": (
                0.0,
                -(p.toe_rotate if p is not None else N(0.55, 0.1)),
                0.0,
            ),
            "Toebean Radius": 0.01 * (p.toebean if p is not None else N(1, 0.1)),
            "Claw Curl Deg": 12 * (p.claw_curl if p is not None else N(1, 0.2)),
            "Claw Pct Length Rad1 Rad2": np.array((0.13, 0.64, 0.05))
            * (p.claw_pct_scale if p is not None else N(0.5, 0.05))
            * (
                np.array(p.claw_pct_scale3)
                if p is not None
                else N(1, 0.1, 3)
            ),
            "Thumb Pct": np.array((0.4, 0.5, 0.75))
            * (p.thumb_scale if p is not None else N(1, 0.1))
            * (np.array(p.thumb_scale3) if p is not None else N(1, 0.1, 3)),
            "Toe Curl Scalar": 0.34 * (p.toe_curl if p is not None else N(1, 0.2)),
        },
        bald=True,
    )

    leg_fac = parts.leg.BirdLeg(
        {
            "length_rad1_rad2": (
                body_length
                * 0.5
                * (p.leg_length if p is not None else N(1, 0.05)),
                0.04 * (p.leg_rad1 if p is not None else N(1, 0.1)),
                0.02 * (p.leg_rad2 if p is not None else N(1, 0.1)),
            ),
            "Thigh Rad1 Rad2 Fullness": np.array((0.12, 0.04, 1.26))
            * (np.array(p.thigh) if p is not None else N(1, 0.1, 3)),
            "Shin Rad1 Rad2 Fullness": np.array((0.1, 0.04, 5.0))
            * (np.array(p.shin) if p is not None else N(1, 0.1, 3)),
        }
    )
    leg_coord = (
        p.leg_coord_x if p is not None else N(0.5, 0.05),
        p.leg_coord_y if p is not None else N(0.2, 0.04),
        p.leg_coord_z if p is not None else N(0.8, 0.05),
    )
    for side in [-1, 1]:
        leg = genome.attach(
            genome.part(foot_fac),
            genome.part(leg_fac),
            coord=(0.9, 0, 0),
            joint=Joint(rest=(0, 0, 0)),
        )
        genome.attach(
            leg,
            body,
            coord=leg_coord,
            joint=Joint(
                rest=(0, p.leg_joint if p is not None else U(135, 175), 0),
                bounds=shoulder_bounds,
            ),
            side=side,
        )

    extension = p.extension if p is not None else U(0.8, 1)
    wing_len = (
        body_length
        * (p.wing_len_factor if p is not None else clip_gaussian(1.0, 0.2, 0.6, 1.5))
        * 0.8
    )
    wing_fac = parts.wings.FlyingBirdWing(
        {
            "length_rad1_rad2": np.array(
                (
                    wing_len,
                    p.wing_width if p is not None else U(0.08, 0.15),
                    0.02 * (p.wing_rad if p is not None else N(1, 0.2)),
                )
            ),
            "Extension": extension,
            "feather_density": p.feather_density if p is not None else U(25, 40),
        }
    )

    wing_coord = (
        p.wing_coord_x if p is not None else N(0.68, 0.02),
        150 / 180 * (p.wing_coord_y if p is not None else N(1, 0.1)),
        0.8,
    )
    if wing_fac.params["Extension"] > 0.5:
        wing_rot = (90, 0, 90)
    else:
        wing_rot = (90, 40, 90)
    for side in [-1, 1]:
        wing = genome.part(wing_fac)
        genome.attach(
            wing, body, coord=wing_coord, joint=Joint(rest=wing_rot), side=side
        )

    head_fac = parts.head.FlyingBirdHead()
    head = genome.part(head_fac)

    beak = genome.part(parts.beak.FlyingBirdBeak())
    genome.attach(beak, head, coord=(0.85, 0, 0.5), joint=Joint(rest=(0, 0, 0)))

    eye_fac = parts.eye.MammalEye(
        {"Radius": p.eye_radius if p is not None else N(0.02, 0.005)}
    )
    t = p.eye_t if p is not None else U(0.7, 0.85)
    splay = p.eye_splay if p is not None else U(80, 110) / 180
    r = 0.85
    for side in [-1, 1]:
        eye = genome.part(eye_fac)
        genome.attach(
            eye,
            head,
            coord=(t, splay, r),
            joint=Joint(rest=(0, 0, 0)),
            rotation_basis="normal",
            side=side,
        )

    genome.attach(
        head,
        body,
        coord=(
            p.head_coord_x if p is not None else U(0.84, 0.85),
            0,
            p.head_coord_z if p is not None else U(1.05, 1.15),
        ),
        joint=Joint(rest=(0, p.head_joint if p is not None else N(18, 5), 0)),
    )

    return genome.CreatureGenome(
        parts=body,
        postprocess_params=dict(
            animation=dict(),
            hair=_flying_bird_hair_params(p),
        ),
    )


def apply_bird_materials(bird, obj):
    bird.body_material.apply(
        joining.get_parts(obj)
        + joining.get_parts(obj, False, "BodyExtra")
        + joining.get_parts(obj, False, "Feather")
    )

    # TODO move these into the individual part generators
    bird.tongue_material.apply(joining.get_parts(obj, False, "Tongue"))
    bird.bone_material.apply(
        joining.get_parts(obj, False, "Teeth") + joining.get_parts(obj, False, "Claws")
    )
    bird.eyeball_material.apply(
        joining.get_parts(obj, False, "Eyeball"), shader_kwargs={"coord": "X"}
    )
    bird.beak_material.apply(joining.get_parts(obj, False, "Beak"))


@gin.configurable
class BirdFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = BirdParameters

    def __init__(
        self, factory_seed=None, coarse=False, bvh=None, animation_mode=None, **kwargs
    ):
        super().__init__(factory_seed, coarse)
        self.bvh = bvh
        self.animation_mode = animation_mode
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> BirdParameters:
        body_material_fac = weighted_sample(material_assignments.bird)
        return BirdParameters(
            seed=seed,
            body_material=body_material_fac(),
            tongue_material=creature_materials.tongue.Tongue(),
            bone_material=creature_materials.bone.Bone(),
            eyeball_material=creature_materials.eyeball.Eyeball(),
            beak_material=creature_materials.beak.Beak(),
        )

    def _sample_spawn_parameters(
        self, params: BirdParameters, seed: int, i: int
    ) -> BirdParameters:
        return params.model_copy(update=_sample_bird_spawn_parameters())

    def apply_parameters(
        self, params: BirdParameters, *, spawn_scope: bool = True
    ) -> None:
        self.body_material = params.body_material
        self.tongue_material = params.tongue_material
        self.bone_material = params.bone_material
        self.eyeball_material = params.eyeball_material
        self.beak_material = params.beak_material
        self._use_fixed_spawn_draws = spawn_scope
        if spawn_scope:
            self._bird_params = params

    def create_asset(self, i, placeholder, hair=True, **kwargs):
        dynamic = self.animation_mode is not None
        bird_params = self._bird_params if self._use_fixed_spawn_draws else None
        genome = duck_genome(mode=self.animation_mode, p=bird_params)
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

        joined_extras = butil.join_objects(extras)
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
                wing_mag = (
                    self._bird_params.idle_wing_mag
                    if self._use_fixed_spawn_draws
                    else U(0, 0.3)
                )
                idle.idle_body_noise_drivers(ik_targets, wing_mag=wing_mag)
            elif self.animation_mode == "swim":
                spine = [b for b in arma.pose.bones if "Body" in b.name]
                tail = [b for b in arma.pose.bones if "Tail" in b.name]
                animate_wiggle_bones(
                    arma=arma, bones=tail, mag_deg=U(0, 30), freq=U(0.5, 2)
                )
            else:
                raise ValueError(f"Unrecognized mode {self.animation_mode=}")
        return root


def _flying_bird_legacy_init(
    inst: Any,
    seed: int,
    coarse: bool,
    bvh: Any = None,
    animation_mode: str | None = None,
    altitude: Any = ("uniform", 15, 30),
) -> None:
    inst.animation_mode = animation_mode
    inst.altitude = altitude
    inst.bvh = bvh
    inst.policy = animation_policy.AnimPolicyRandomForwardWalk(
        forward_vec=(1, 0, 0),
        speed=U(7, 15),
        step_range=(5, 40),
        yaw_dist=("normal", 0, 15),
    )
    body_material_fac = weighted_sample(material_assignments.bird)
    inst.body_material = body_material_fac()
    inst.tongue_material = creature_materials.tongue.Tongue()
    inst.bone_material = creature_materials.bone.Bone()
    inst.eyeball_material = creature_materials.eyeball.Eyeball()
    inst.beak_material = creature_materials.beak.Beak()


class FlyingBirdParameters(LegacyBridgeParameters):
    pass


@gin.configurable
class FlyingBirdFactory(ParameterizedAssetFactory, AssetFactory):
    parameters_model: ClassVar[type[AssetParameters]] = FlyingBirdParameters
    max_expected_radius = 1
    max_distance = 40

    def __init__(
        self,
        factory_seed=None,
        coarse=False,
        bvh=None,
        animation_mode=None,
        altitude=("uniform", 15, 30),
    ):
        self._flying_bird_bvh = bvh
        self._flying_bird_animation_mode = animation_mode
        self._flying_bird_altitude = altitude
        super().__init__(factory_seed, coarse)
        self.init_legacy_parameters()

    def _sample_init_parameters(self, seed: int) -> FlyingBirdParameters:
        return legacy_init_to_parameters(
            FlyingBirdParameters,
            FlyingBirdFactory,
            seed,
            self.coarse,
            self._flying_bird_bvh,
            self._flying_bird_animation_mode,
            self._flying_bird_altitude,
            init_fn=_flying_bird_legacy_init,
        )

    def _sample_spawn_parameters(
        self, params: FlyingBirdParameters, seed: int, i: int
    ) -> FlyingBirdParameters:
        return params.model_copy(update=_sample_flying_bird_spawn_parameters())

    def apply_parameters(
        self, params: FlyingBirdParameters, *, spawn_scope: bool = True
    ) -> None:
        apply_bridge_parameters(self, params, spawn_scope=spawn_scope)
        if spawn_scope:
            self._flying_bird_params = params

    def create_placeholder(self, i, loc, rot):
        p = butil.spawn_cube(size=3)
        p.name = f"{self}.create_placeholder({i})"
        p.location = loc
        p.rotation_euler = rot

        if self.bvh is None:
            return p

        altitude = rg(self.altitude)
        p.location.z += altitude
        curve = animation_policy.policy_create_bezier_path(
            p,
            self.bvh,
            self.policy,
            retry_rotation=True,
            fatal=True,
        )
        curve.name = f"animhelper:{self}.create_placeholder({i}).path"

        # animate the placeholder to the APPROX location of the snake, so the camera can follow itcurve.location = (0, 0, 0)
        run_cycle.follow_path(
            p,
            curve,
            use_curve_follow=True,
            offset=0,
            duration=bpy.context.scene.frame_end - bpy.context.scene.frame_start,
        )
        p.rotation_euler.z += np.pi / 2
        curve.data.twist_mode = "Z_UP"
        curve.data.driver_add("eval_time").driver.expression = "frame"

        return p

    def create_asset(self, i, placeholder, hair=True, animate=False, **kwargs):
        bird_params = (
            self._flying_bird_params if self._use_fixed_spawn_draws else None
        )
        genome = flying_bird_genome(self.animation_mode, p=bird_params)
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

        joined_extras = butil.join_objects(extras)
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
