# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json
import math
import random  # validate-ignore: test_determinism
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import NamedTuple

import procfunc as pf

__all__ = [
    "HumanResult",
    "body_pose_rand",
    "face_pose_rand",
    "hand_pose_rand",
    "human",
    "human_rand",
    "pose",
    "pose_rand",
    "set_bone_rotations",
]


_FACE_HANDLE_BONES = (
    "special01",
    "special03",
    "special05.L",
    "special05.R",
    "special06.L",
    "special06.R",
)
_FACE_CHAIN_BONES = (
    "special04",
    "oris02",
    "oris06.L",
    "oris06.R",
    "levator02.L",
    "levator03.L",
    "levator04.L",
    "levator02.R",
    "levator03.R",
    "levator04.R",
    "oris04.L",
    "oris04.R",
    "oris06",
    "temporalis01.L",
    "temporalis01.R",
    "oculi02.L",
    "oculi02.R",
    "temporalis02.L",
    "temporalis02.R",
    "risorius02.L",
    "risorius02.R",
)
_FACE_MUSCLE_BONES = (
    "oris01",
    "oris07.L",
    "oris07.R",
    "levator05.L",
    "levator05.R",
    "oris03.L",
    "oris03.R",
    "oris05",
    "levator06.L",
    "levator06.R",
    "oculi01.L",
    "oculi01.R",
    "risorius03.L",
    "risorius03.R",
)
_FINGER_CHAINS = tuple(
    tuple(f"finger{finger}-{joint}.{side}" for joint in range(1, 4))
    for side in ("L", "R")
    for finger in range(1, 6)
)
_METACARPALS = tuple(
    f"metacarpal{finger}.{side}" for side in ("L", "R") for finger in range(1, 5)
)
_TOE1_CHAINS = tuple(
    tuple(f"toe1-{joint}.{side}" for joint in range(1, 3)) for side in ("L", "R")
)
_TOE_CHAINS = tuple(
    tuple(f"toe{toe}-{joint}.{side}" for joint in range(1, 4))
    for side in ("L", "R")
    for toe in range(2, 6)
)


class HumanResult(NamedTuple):
    mesh: pf.MeshObject
    armature: pf.ArmatureObject


def _mpfb_services() -> dict:
    try:
        import mpfb  # keep-local
    except ImportError as e:
        raise ImportError("human() requires the mpfb package") from e
    return mpfb.initialize()["SERVICES"]


@cache
def _detail_catalog() -> dict[str, list[dict]]:
    services = _mpfb_services()
    targets_directory = Path(services["LocationService"].get_mpfb_data("targets"))
    catalog = json.loads((targets_directory / "target.json").read_text())
    sections = (
        "head",
        "forehead",
        "eyebrows",
        "eyes",
        "ears",
        "nose",
        "cheek",
        "mouth",
        "chin",
        "neck",
        "torso",
        "breast",
        "stomach",
        "hip",
        "pelvis",
        "buttocks",
        "arms",
        "hands",
        "legs",
        "feet",
    )
    return {name: catalog[name]["categories"] for name in sections}


def _human_body(
    macro: dict, detail_stack: list[dict], skin_material: pf.Material | None
) -> HumanResult:
    services = _mpfb_services()
    body = services["HumanService"].create_human(
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )
    if detail_stack:
        services["TargetService"].bulk_load_targets(body, detail_stack)
    rig = services["HumanService"].add_builtin_rig(body, "default", import_weights=True)
    rig.location.z -= services["ObjectService"].get_lowest_point(body)

    mesh = pf.MeshObject(body)
    mesh.item().name = human.__name__
    if skin_material is not None:
        pf.ops.object.set_material(mesh, skin_material)
    pf.ops.modifier.subdivide_surface(mesh, levels=2, _skip_apply=True)
    return HumanResult(mesh=mesh, armature=pf.ArmatureObject(rig))


def _mpfb_age(age: float) -> float:
    return 0.34375 + 0.65625 * age


def human(
    gender: float = 0.5,
    age: float = 0.25,
    muscle: float = 0.5,
    weight: float = 0.5,
    proportions: float = 0.5,
    height: float = 0.5,
    skin_material: pf.Material | None = None,
) -> HumanResult:
    macro = _mpfb_services()["TargetService"].get_default_macro_info_dict()
    macro["gender"] = gender
    macro["age"] = _mpfb_age(age)
    macro["muscle"] = muscle
    macro["weight"] = weight
    macro["proportions"] = proportions
    macro["height"] = height
    return _human_body(macro, [], skin_material)


def human_rand(rng: pf.RNG, skin_material: pf.Material | None = None) -> HumanResult:
    randomizer = _mpfb_services()["RandomizationService"]
    mpfb_rng = random.Random(int(rng.integers(2**32)))
    spec = randomizer.get_default_phenotype_spec()
    spec["phenotype"]["discrete_age"] = False
    age = spec["phenotype"]["attributes"]["age"]
    age["neutral"] = _mpfb_age(0.5)
    age["deviation"] = _mpfb_age(1.0) - _mpfb_age(0.5)
    macro = randomizer.randomize_macro_info_dict(spec, mpfb_rng)
    catalog = _detail_catalog()
    detail_spec = randomizer.get_default_detail_spec(list(catalog))
    stack = randomizer.pick_random_details(detail_spec, catalog, mpfb_rng)
    return _human_body(macro, stack, skin_material)


@pf.tracer.primitive(mutates=["armature"])
def set_bone_rotations(
    armature: pf.ArmatureObject,
    bone_names: tuple[str, ...],
    rotations: tuple[tuple[float, float, float], ...],
) -> pf.ArmatureObject:
    bones = armature.item().pose.bones
    for name, rotation in zip(bone_names, rotations, strict=True):
        bone = bones[name]
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = rotation
    return armature


def pose(
    armature: pf.ArmatureObject,
    rotations: Mapping[str, tuple[float, float, float]],
) -> pf.ArmatureObject:
    return set_bone_rotations(
        armature,
        tuple(rotations),
        tuple(rotations.values()),
    )


def _angle_rand(rng: pf.RNG, mean: float, std: float, low: float, high: float) -> float:
    degrees = pf.random.clip_gaussian(rng, mean, std, low, high)
    return degrees * math.pi / 180


def _rotation_rand(
    rng: pf.RNG,
    x: tuple[float, float, float, float],
    y: tuple[float, float, float, float],
    z: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    return (
        _angle_rand(rng, x[0], x[1], x[2], x[3]),
        _angle_rand(rng, y[0], y[1], y[2], y[3]),
        _angle_rand(rng, z[0], z[1], z[2], z[3]),
    )


def _body_rotations_rand(
    rng: pf.RNG,
) -> dict[str, tuple[float, float, float]]:
    bounds = {
        "spine01": ((0, 4, -10, 10), (0, 3, -8, 8), (0, 3, -8, 8)),
        "spine02": ((0, 4, -10, 10), (0, 3, -8, 8), (0, 3, -8, 8)),
        "spine03": ((0, 5, -15, 15), (0, 4, -10, 10), (0, 4, -10, 10)),
        "spine04": ((0, 3, -8, 8), (0, 3, -8, 8), (0, 3, -8, 8)),
        "spine05": ((0, 3, -8, 8), (0, 3, -8, 8), (0, 3, -8, 8)),
        "neck01": ((0, 4, -10, 10), (0, 4, -10, 10), (0, 4, -10, 10)),
        "neck02": ((0, 4, -10, 10), (0, 4, -10, 10), (0, 4, -10, 10)),
        "neck03": ((0, 4, -10, 10), (0, 4, -10, 10), (0, 4, -10, 10)),
        "head": ((0, 8, -20, 25), (0, 12, -35, 35), (0, 8, -20, 20)),
        "clavicle.L": ((0, 5, -12, 12), (0, 5, -12, 12), (0, 5, -12, 12)),
        "clavicle.R": ((0, 5, -12, 12), (0, 5, -12, 12), (0, 5, -12, 12)),
        "shoulder01.L": ((0, 6, -15, 15), (0, 6, -15, 15), (0, 6, -15, 15)),
        "shoulder01.R": ((0, 6, -15, 15), (0, 6, -15, 15), (0, 6, -15, 15)),
        "upperarm01.L": ((20, 30, -45, 90), (0, 18, -45, 45), (30, 30, -20, 100)),
        "upperarm01.R": ((20, 30, -45, 90), (0, 18, -45, 45), (-30, 30, -100, 20)),
        "lowerarm01.L": ((35, 30, 0, 120), (0, 15, -45, 45), (0, 1, -2, 2)),
        "lowerarm01.R": ((35, 30, 0, 120), (0, 15, -45, 45), (0, 1, -2, 2)),
        "wrist.L": ((0, 10, -25, 25), (0, 8, -20, 20), (0, 10, -25, 25)),
        "wrist.R": ((0, 10, -25, 25), (0, 8, -20, 20), (0, 10, -25, 25)),
        "upperleg01.L": ((-15, 25, -75, 20), (0, 12, -30, 30), (-5, 10, -30, 20)),
        "upperleg01.R": ((-15, 25, -75, 20), (0, 12, -30, 30), (5, 10, -20, 30)),
        "lowerleg01.L": ((25, 25, 0, 100), (0, 2, -5, 5), (0, 1, -2, 2)),
        "lowerleg01.R": ((25, 25, 0, 100), (0, 2, -5, 5), (0, 1, -2, 2)),
        "foot.L": ((0, 8, -20, 20), (0, 4, -10, 10), (0, 4, -10, 10)),
        "foot.R": ((0, 8, -20, 20), (0, 4, -10, 10), (0, 4, -10, 10)),
    }
    rotations = {}
    for name, axes in bounds.items():
        rotations[name] = _rotation_rand(rng, axes[0], axes[1], axes[2])

    for chain in _TOE1_CHAINS:
        curl = _angle_rand(rng, 2, 3, 0, 10)
        rotations[chain[0]] = (0.6 * curl, 0.0, 0.0)
        rotations[chain[1]] = (0.4 * curl, 0.0, 0.0)

    for chain in _TOE_CHAINS:
        curl = _angle_rand(rng, 2, 3, 0, 10)
        rotations[chain[0]] = (0.45 * curl, 0.0, 0.0)
        rotations[chain[1]] = (0.35 * curl, 0.0, 0.0)
        rotations[chain[2]] = (0.20 * curl, 0.0, 0.0)

    return rotations


def body_pose_rand(rng: pf.RNG, armature: pf.ArmatureObject) -> pf.ArmatureObject:
    return pose(armature, _body_rotations_rand(rng))


def _hand_rotations_rand(
    rng: pf.RNG,
) -> dict[str, tuple[float, float, float]]:
    rotations = {}
    for name in _METACARPALS:
        rotations[name] = _rotation_rand(
            rng,
            (0, 3, -8, 8),
            (0, 2, -5, 5),
            (0, 2, -5, 5),
        )

    for chain in _FINGER_CHAINS:
        curl = _angle_rand(rng, 25, 20, 0, 75)
        spread = _angle_rand(rng, 0, 3, -8, 8)
        rotations[chain[0]] = (0.45 * curl, 0.0, spread)
        rotations[chain[1]] = (0.75 * curl, 0.0, 0.0)
        rotations[chain[2]] = (0.55 * curl, 0.0, 0.0)

    return rotations


def hand_pose_rand(rng: pf.RNG, armature: pf.ArmatureObject) -> pf.ArmatureObject:
    return pose(armature, _hand_rotations_rand(rng))


def _face_rotations_rand(
    rng: pf.RNG,
) -> dict[str, tuple[float, float, float]]:
    small = (0, 0.7, -2.5, 2.5)
    medium = (0, 3, -9, 9)
    large = (0, 6, -20, 20)
    rotations = {
        "jaw": _rotation_rand(
            rng,
            (6, 6, 0, 20),
            (0, 3, -6, 6),
            (0, 3, -6, 6),
        )
    }

    for name in _FACE_HANDLE_BONES:
        rotations[name] = _rotation_rand(rng, small, small, small)

    for name in _FACE_CHAIN_BONES:
        rotations[name] = _rotation_rand(rng, medium, medium, medium)

    for name in _FACE_MUSCLE_BONES:
        rotations[name] = _rotation_rand(rng, large, large, large)

    for side in ("L", "R"):
        blink = _angle_rand(rng, 4, 5, 0, 18)
        rotations[f"orbicularis03.{side}"] = (-blink, 0.0, 0.0)
        rotations[f"orbicularis04.{side}"] = (blink, 0.0, 0.0)

    gaze_horizontal = _angle_rand(rng, 0, 8, -20, 20)
    gaze_vertical = _angle_rand(rng, 0, 8, -20, 20)
    rotations["eye.L"] = (
        -0.554 * gaze_horizontal - 0.832 * gaze_vertical,
        -0.039 * gaze_horizontal + 0.043 * gaze_vertical,
        0.831 * gaze_horizontal - 0.553 * gaze_vertical,
    )
    rotations["eye.R"] = (
        0.554 * gaze_horizontal - 0.832 * gaze_vertical,
        -0.039 * gaze_horizontal - 0.043 * gaze_vertical,
        0.831 * gaze_horizontal + 0.553 * gaze_vertical,
    )

    tongue_curl = _angle_rand(rng, 0, 6, -16, 16)
    tongue_spec = (0, 3, -8, 8)
    rotations["tongue00"] = _rotation_rand(rng, tongue_spec, tongue_spec, tongue_spec)
    rotations["tongue01"] = (tongue_curl, 0.0, 0.0)
    rotations["tongue02"] = (0.8 * tongue_curl, 0.0, 0.0)
    rotations["tongue03"] = (0.6 * tongue_curl, 0.0, 0.0)
    rotations["tongue04"] = (0.4 * tongue_curl, 0.0, 0.0)
    for name in (
        "tongue05.L",
        "tongue05.R",
        "tongue06.L",
        "tongue06.R",
        "tongue07.L",
        "tongue07.R",
    ):
        rotations[name] = _rotation_rand(rng, tongue_spec, tongue_spec, tongue_spec)

    return rotations


def face_pose_rand(rng: pf.RNG, armature: pf.ArmatureObject) -> pf.ArmatureObject:
    return pose(armature, _face_rotations_rand(rng))


def pose_rand(rng: pf.RNG, armature: pf.ArmatureObject) -> pf.ArmatureObject:
    rng_body, rng_hands, rng_face = rng.spawn(3)
    rotations = _body_rotations_rand(rng_body)
    rotations.update(_hand_rotations_rand(rng_hands))
    rotations.update(_face_rotations_rand(rng_face))
    return pose(armature, rotations)
