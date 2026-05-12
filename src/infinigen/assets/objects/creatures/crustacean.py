# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


from collections import defaultdict

import bpy
import gin
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.creatures.parts.crustacean.antenna import (
    LobsterAntennaFactory,
    SpinyLobsterAntennaFactory,
)
from infinigen.assets.objects.creatures.parts.crustacean.body import (
    CrabBodyFactory,
    LobsterBodyFactory,
)
from infinigen.assets.objects.creatures.parts.crustacean.claw import (
    CrabClawFactory,
    LobsterClawFactory,
)
from infinigen.assets.objects.creatures.parts.crustacean.eye import CrustaceanEyeFactory
from infinigen.assets.objects.creatures.parts.crustacean.fin import CrustaceanFinFactory
from infinigen.assets.objects.creatures.parts.crustacean.leg import (
    CrabLegFactory,
    LobsterLegFactory,
)
from infinigen.assets.objects.creatures.parts.crustacean.tail import (
    CrustaceanTailFactory,
)
from infinigen.assets.objects.creatures.util.creature import genome_to_creature
from infinigen.assets.objects.creatures.util.genome import (
    CreatureGenome,
    Joint,
    attach,
    part,
)
from infinigen.assets.objects.creatures.util.joining import join_and_rig_parts
from infinigen.assets.utils.decorate import read_material_index, write_material_index
from infinigen.assets.utils.misc import assign_material
from infinigen.core.nodes.node_info import Nodes
from infinigen.core.nodes.node_utils import build_color_ramp
from infinigen.core.nodes.node_wrangler import NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import read_attr_data, shaderfunc_to_material
from infinigen.core.util import blender as butil
from infinigen.core.util.color import hsv2rgba
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform

n_legs = 4
n_limbs = 5
n_side_fin = 2


def crustacean_genome(sp):
    body_fac = sp["body_fn"]()
    obj = part(body_fac)
    # Add legs
    leg_x_length = sp["leg_x_length"](body_fac.params)
    leg_x_lengths = np.sort(uniform(0.6, 1, 4))[::-1] * leg_x_length
    leg_angle = sp["leg_angle"]
    x_legs = sp["x_legs"]
    leg_joints_x, leg_joints_y, leg_joints_z = sp["leg_joint"]

    shared_leg_params = ["bottom_flat", "bottom_cutoff"]
    leg_fn = sp["leg_fn"]
    leg_params = {k: v for k, v in leg_fn().params.items() if k in shared_leg_params}
    leg_fac = [
        leg_fn({**leg_params, "x_length": leg_x_lengths[i]}) for i in range(n_legs)
    ]
    for i in range(n_legs):
        for side in [1, -1]:
            attach(
                part(leg_fac[i]),
                obj,
                (x_legs[i + 1], leg_angle, 0.99),
                Joint((leg_joints_x[i], leg_joints_y[i], leg_joints_z[i])),
                side=side,
            )
    # Add claws
    claw_angle = sp["claw_angle"]
    claw_fn = sp["claw_fn"]
    claw_fac = claw_fn({"x_length": sp["claw_x_length"](body_fac.params)})

    for side in [1, -1]:
        attach(
            part(claw_fac),
            obj,
            (x_legs[0] + sp["x_claw_offset"], claw_angle, 0.99),
            Joint(sp["claw_joint"]),
            side=side,
        )
    # Add tails
    tail_fac = sp["tail_fn"]
    if tail_fac is not None:
        shared_params = [
            "bottom_shift",
            "bottom_cutoff",
            "top_shift",
            "top_cutoff",
            "y_length",
            "z_length",
        ]
        tail_fac = tail_fac(
            {
                **{k: v for k, v in body_fac.params.items() if k in shared_params},
                "x_length": sp["tail_x_length"](body_fac.params),
            }
        )
        tail = part(tail_fac)
        attach(tail, obj, (0, 0, 0), Joint((0, 0, 180)))
        fin_fn = sp["fin_fn"]
        if fin_fn is not None:
            fin_fn = sp["fin_fn"]
            x_fins = sp["x_fins"]
            fin_joints_x, fin_joints_y, fin_joints_z = sp["fin_joints"]
            fin_x_length = sp["fin_x_length"](body_fac.params)
            fin_x_lengths = np.sort(uniform(0.6, 1, 4))[::-1] * fin_x_length
            fin_fac = [
                fin_fn({"x_length": fin_x_lengths[i]}) for i in range(n_side_fin + 1)
            ]

            for i in range(n_side_fin):
                for side in [1, -1]:
                    attach(
                        part(fin_fac[i]),
                        tail,
                        (x_fins[i], 0.5, 0.99),
                        Joint((fin_joints_x[i], fin_joints_y[i], fin_joints_z[i])),
                        side=side,
                    )
            attach(part(fin_fac[-1]), tail, (0.99, 0.5, 0.9), Joint((0, 0, 0)))

    # Add eyes
    x_eye = sp["x_eye"]
    eye_angle = sp["eye_angle"]
    eye_joint_x, eye_joint_y, eye_joint_z = sp["eye_joint"]
    eye_fac = CrustaceanEyeFactory()
    for side in [1, -1]:
        attach(
            part(eye_fac),
            obj,
            (x_eye, eye_angle, 0.99),
            Joint((eye_joint_x, eye_joint_y, eye_joint_z)),
            side=side,
        )
    # Add antenna
    antenna_fn = sp["antenna_fn"]
    if antenna_fn is not None:
        x_antenna = sp["x_antenna"]
        antenna_angle = sp["antenna_angle"]
        antenna_fac = antenna_fn({"x_length": sp["antenna_x_length"](body_fac.params)})
        for side in [1, -1]:
            attach(
                part(antenna_fac),
                obj,
                (x_antenna, antenna_angle, 0.99),
                Joint(sp["antenna_joint"]),
                side=side,
            )

    anim_params = {k: v for k, v in sp.items() if "curl" in k or "rot" in k}
    anim_params["freq"] = sp["freq"]
    postprocess_params = dict(material={"base_hue": sp["base_hue"]}, anim=anim_params)
    return CreatureGenome(obj, postprocess_params)


def build_base_hue():
    if uniform(0, 1) < 0.6:
        return uniform(0, 0.05)
    else:
        return uniform(0.4, 0.45)


def shader_crustacean(nw: NodeWrangler, params):
    value_shift = log_uniform(2, 10)
    base_hue = params["base_hue"]
    bright_color = hsv2rgba(
        base_hue, uniform(0.8, 1.0), log_uniform(0.02, 0.05) * value_shift
    )
    dark_color = hsv2rgba(
        (base_hue + uniform(-0.05, 0.05)) % 1,
        uniform(0.8, 1.0),
        log_uniform(0.01, 0.02) * value_shift,
    )
    light_color = hsv2rgba(base_hue, uniform(0.0, 0.4), log_uniform(0.2, 1.0))
    specular = uniform(0.6, 0.8)
    specular_tint = *([uniform(0, 1)] * 3), 1
    clearcoat = uniform(0.2, 0.8)
    roughness = uniform(0.1, 0.3)
    metallic = uniform(0.6, 0.8)
    x, y, z = nw.separate(nw.new_node(Nodes.NewGeometry).outputs["Position"])
    color = build_color_ramp(
        nw,
        nw.new_node(
            Nodes.MapRange,
            [
                nw.new_node(
                    Nodes.MusgraveTexture,
                    [nw.combine(x, nw.math("ABSOLUTE", y), z)],
                    input_kwargs={"Scale": log_uniform(5, 8)},
                ),
                -1,
                1,
                0,
                1,
            ],
        ),
        [0.0, 0.3, 0.7, 1.0],
        [bright_color, bright_color, dark_color, dark_color],
    )
    ratio = nw.new_node(Nodes.Attribute, attrs={"attribute_name": "ratio"}).outputs[
        "Fac"
    ]
    color = nw.new_node(Nodes.MixRGB, [ratio, light_color, color])
    bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            "Base Color": color,
            "Metallic": metallic,
            "Roughness": roughness,
            "Specular IOR Level": specular,
            "Specular Tint": specular_tint,
            "Coat Weight": clearcoat,
        },
    )
    return bsdf


def shader_eye(nw: NodeWrangler):
    return nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={"Base Color": (0.1, 0.1, 0.1, 1), "Specular IOR Level": 0},
    )


def crustacean_postprocessing(body_parts, extras, params):
    tag_list = ["body", "claw", "leg"]
    materials = [
        shaderfunc_to_material(shader_crustacean, params["material"])
        for _, t in enumerate(tag_list)
    ]
    tag_list.append("eye")
    materials.append(shaderfunc_to_material(shader_eye))
    assign_material(body_parts + extras, materials)

    for part_obj in body_parts:
        material_indices = read_material_index(part_obj)
        for i, tag_name in enumerate(tag_list):
            if f"tag_{tag_name}" in part_obj.data.attributes.keys():
                part_obj.data.attributes.active = part_obj.data.attributes[
                    f"tag_{tag_name}"
                ]
                with butil.SelectObjects(part_obj):
                    bpy.ops.geometry.attribute_convert(domain="FACE")
                has_tag = read_attr_data(part_obj, f"tag_{tag_name}", "FACE")
                material_indices[np.nonzero(has_tag)[0]] = i
        write_material_index(part_obj, material_indices)
    for extra in extras:
        material_indices = read_material_index(extra)
        material_indices.fill(tag_list.index("claw"))
        write_material_index(extra, material_indices)


def animate_crustacean_move(arma, params):
    groups = defaultdict(list)
    for bone in arma.pose.bones.values():
        groups[(bone.bone["factory_class"], bone.bone["index"])].append(bone)
    for (factory_name, part_id), bones in groups.items():
        eval(factory_name).animate_bones(arma, bones, params)


@gin.configurable
class CrustaceanFactory(AssetFactory):
    max_expected_radius = 1
    max_distance = 40

    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.species_params = {
                "lobster": self.lobster_params,
                "crab": self.crab_params,
                "spiny_lobster": self.spiny_lobster_params,
            }
            self.species = np.random.choice(list(self.species_params.keys()))

    def create_asset(self, i, animate=True, rigging=True, cloth=False, **kwargs):
        genome = crustacean_genome(self.species_params[self.species]())
        root, parts = genome_to_creature(
            genome, name=f"crustacean({self.factory_seed}, {i})"
        )
        for p in parts:
            if p.obj.name.split("=")[-1] == "CrustaceanEyeFactor":
                assign_material(p.obj, shaderfunc_to_material(shader_eye))
        joined, extras, arma, ik_targets = join_and_rig_parts(
            root,
            parts,
            genome,
            postprocess_func=crustacean_postprocessing,
            rigging=rigging,
            min_remesh_size=0.005,
            face_size=kwargs["face_size"],
            roll="GLOBAL_POS_Z",
        )
        if animate and arma is not None:
            animate_crustacean_move(arma, genome.postprocess_params["anim"])
        else:
            butil.join_objects([joined] + extras)
        return root

    def crab_params(self):
        base_leg_curl = uniform(-np.pi * 0.15, np.pi * 0.15)
        return {
            "body_fn": CrabBodyFactory,
            "leg_fn": CrabLegFactory,
            "claw_fn": CrabClawFactory,
            "tail_fn": None,
            "antenna_fn": None,
            "fin_fn": None,
            "leg_x_length": lambda p: p["y_length"] * log_uniform(2.0, 3.0),
            "claw_x_length": lambda p: p["y_length"] * log_uniform(1.5, 1.8),
            "tail_x_length": lambda p: 0,
            "antenna_x_length": lambda p: 0,
            "fin_x_length": lambda p: 0,
            "x_legs": (
                np.linspace(uniform(0.08, 0.1), uniform(0.55, 0.6), n_limbs)
                + np.arange(n_limbs) * 0.02
            )[::-1],
            "leg_angle": uniform(0.42, 0.44),
            "leg_joint": (
                np.sort(uniform(-5, 5, n_legs))[:: 1 if uniform(0, 1) > 0.5 else -1],
                np.sort(uniform(0, 10, n_legs)),
                np.sort(uniform(65, 105, n_legs) + uniform(-8, 8))
                + np.arange(n_legs) * 2,
            ),
            "x_claw_offset": uniform(0.08, 0.1),
            "claw_angle": uniform(0.44, 0.46),
            "claw_joint": (uniform(-50, -40), uniform(-20, 20), uniform(10, 20)),
            "x_eye": uniform(0.92, 0.96),
            "eye_angle": uniform(0.8, 0.85),
            "eye_joint": (0, uniform(-60, -0), uniform(10, 70)),
            "x_antenna": 0,
            "antenna_angle": 0,
            "antenna_joint": (0, 0, 0),
            "x_fins": 0,
            "fin_joints": ([0] * n_side_fin, [0] * n_side_fin, [0] * n_side_fin),
            "leg_rot": (uniform(np.pi * 0.8, np.pi * 1.1), 0, 0),
            "leg_curl": (
                (-np.pi * 1.1, -np.pi * 0.7),
                0,
                (base_leg_curl - np.pi * 0.02, base_leg_curl + np.pi * 0.02),
            ),
            "claw_curl": ((-np.pi * 0.2, np.pi * 0.1), 0, (-np.pi * 0.1, np.pi * 0.1)),
            "claw_lower_curl": ((-np.pi * 0.1, np.pi * 0.1), 0, 0),
            "tail_curl": (0, 0, 0),
            "antenna_curl": (0, 0, 0),
            "base_hue": build_base_hue(),
            "freq": 1 / log_uniform(100, 200),
        }

    def lobster_params(self):
        base_leg_curl = uniform(-np.pi * 0.4, np.pi * 0.4)
        return {
            "body_fn": LobsterBodyFactory,
            "leg_fn": LobsterLegFactory,
            "claw_fn": LobsterClawFactory,
            "tail_fn": CrustaceanTailFactory,
            "antenna_fn": LobsterAntennaFactory,
            "fin_fn": CrustaceanFinFactory,
            "leg_x_length": lambda p: p["x_length"] * log_uniform(0.6, 0.8),
            "claw_x_length": lambda p: p["x_length"] * log_uniform(1.2, 1.5),
            "tail_x_length": lambda p: p["x_length"] * log_uniform(1.2, 1.8),
            "antenna_x_length": lambda p: p["x_length"] * log_uniform(1.6, 3.0),
            "fin_x_length": lambda p: p["y_length"] * log_uniform(1.2, 2.5),
            "x_legs": (
                np.linspace(0.05, uniform(0.2, 0.25), n_limbs)
                + np.arange(n_limbs) * 0.02
            )[::-1],
            "leg_angle": uniform(0.3, 0.35),
            "leg_joint": (
                uniform(-5, 5, n_legs),
                uniform(0, 10, n_legs),
                np.sort(uniform(95, 110, n_legs) + uniform(-8, 8)),
            ),
            "x_claw_offset": uniform(0.08, 0.1),
            "claw_angle": uniform(0.4, 0.5),
            "claw_joint": (uniform(-80, -70), uniform(-10, 10), uniform(10, 20)),
            "x_eye": uniform(0.8, 0.88),
            "eye_angle": uniform(0.8, 0.85),
            "eye_joint": (0, uniform(-60, -0), uniform(10, 70)),
            "x_antenna": uniform(0.76, 0.8),
            "antenna_angle": uniform(0.6, 0.7),
            "antenna_joint": (uniform(70, 110), uniform(-40, -30), uniform(20, 40)),
            "x_fins": np.sort(uniform(0.85, 0.95, n_side_fin)),
            "fin_joints": (
                np.sort(uniform(0, 30, n_side_fin))[
                    :: 1 if uniform(0, 1) < 0.5 else -1
                ],
                [0] * n_side_fin,
                np.sort(uniform(10, 30, n_side_fin)),
            ),
            "leg_rot": (uniform(np.pi * 0.8, np.pi * 1.1), 0, 0),
            "leg_curl": (
                (-np.pi * 1.1, -np.pi * 0.7),
                0,
                (base_leg_curl - np.pi * 0.02, base_leg_curl + np.pi * 0.02),
            ),
            "claw_curl": ((-np.pi * 0.1, np.pi * 0.2), 0, 0),
            "claw_lower_curl": ((-np.pi * 0.1, np.pi * 0.1), 0, 0),
            "tail_curl": ((-np.pi * 0.6, 0), 0, 0),
            "antenna_curl": ((np.pi * 0.1, np.pi * 0.3), 0, (0, np.pi * 0.8)),
            "base_hue": build_base_hue(),
            "freq": 1 / log_uniform(400, 500),
        }

    def spiny_lobster_params(self):
        lobster_params = self.lobster_params()
        leg_joint_x, leg_joint_y, leg_joint_z = lobster_params["leg_joint"]
        leg_joint_z_min = np.min(leg_joint_z) + uniform(-10, -5)
        return {
            **lobster_params,
            "antenna_fn": SpinyLobsterAntennaFactory,
            "claw_fn": LobsterLegFactory,
            "claw_x_length": lobster_params["leg_x_length"],
            "claw_angle": lobster_params["leg_angle"],
            "claw_joint": (uniform(10, 40), uniform(0, 10), leg_joint_z_min),
            "x_antenna": uniform(0.7, 0.75),
            "antenna_angle": uniform(0.4, 0.5),
        }


@gin.configurable
class CrabFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = "crab"


@gin.configurable
class LobsterFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = "lobster"


@gin.configurable
class SpinyLobsterFactory(CrustaceanFactory):
    def __init__(self, factory_seed, coarse=False, **_):
        super().__init__(factory_seed, coarse)
        self.species = "spiny_lobster"
