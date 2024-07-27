# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Meenal Parakh


import functools

import numpy as np
from numpy.random import uniform

from infinigen.assets.color_fits import real_color_distribution
from infinigen.assets.materials import (
    beverage_fridge_shaders,
    ceiling_light_shaders,
    ceramic,
    dirt,
    dishwasher_shaders,
    fabrics,
    glass,
    glass_volume,
    lamp_shaders,
    metal,
    microwave_shaders,
    mirror,
    oven_shaders,
    plaster,
    plastic,
    rug,
    sofa_fabric,
    table_marble,
    text,
    vase_shaders,
    velvet,
    water,
    wood,
    woods,
)
from infinigen.assets.materials.art import Art, ArtFabric, ArtRug
from infinigen.assets.materials.plastics import plastic_rough
from infinigen.assets.materials.plastics.plastic_rough import shader_rough_plastic
from infinigen.assets.materials.wear_tear import (
    procedural_edge_wear,
    procedural_scratch,
)

DEFAULT_EDGE_WEAR_PROB = 0.5
DEFAULT_SCRATCH_PROB = 0.5


class TextureAssignments:
    def __init__(self, materials, probabilities):
        self.materials = materials
        self.probabilities = probabilities

    def assign_material(self):
        p = np.array(self.probabilities)
        p = p / p.sum()
        return np.random.choice(self.materials, p=p)


class MaterialOptions:
    def __init__(self, materials_list):
        self.materials, self.probabilities = zip(*materials_list)
        self.probabilities = np.array(self.probabilities)
        self.probabilities = self.probabilities / self.probabilities.sum()

    def assign_material(self):
        return np.random.choice(self.materials, p=self.probabilities)


def get_all_metal_shaders():
    metal_shaders_list = [
        metal.brushed_metal.shader_brushed_metal,
        metal.galvanized_metal.shader_galvanized_metal,
        metal.grained_and_polished_metal.shader_grained_metal,
        metal.hammered_metal.shader_hammered_metal,
    ]
    color = metal.sample_metal_color()
    new_shaders = [
        functools.partial(shader, base_color=color) for shader in metal_shaders_list
    ]
    for idx, ns in enumerate(new_shaders):
        # fix taken from: https://github.com/elastic/apm-agent-python/issues/293
        ns.__name__ = metal_shaders_list[idx].__name__

    return new_shaders


def plastic_furniture():
    new_shader = functools.partial(
        shader_rough_plastic, base_color=real_color_distribution("sofa_leather")
    )
    new_shader.__name__ = shader_rough_plastic.__name__
    return new_shader


def get_all_fabric_shaders():
    return [
        fabrics.shader_coarse_knit_fabric,
        fabrics.shader_fine_knit_fabric,
        fabrics.shader_fabric,
        fabrics.shader_leather,
        fabrics.shader_sofa_fabric,
    ]


def beverage_fridge_materials():
    metal_shaders = get_all_metal_shaders()
    return {
        "surface": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "front": TextureAssignments([beverage_fridge_shaders.shader_glass_001], [1.0]),
        "handle": TextureAssignments(
            [beverage_fridge_shaders.shader_white_metal_001], [1.0]
        ),
        "back": TextureAssignments(
            [beverage_fridge_shaders.shader_black_medal_001], [1.0]
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def dishwasher_materials():
    metal_shaders = get_all_metal_shaders()
    return {
        "surface": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "front": TextureAssignments([dishwasher_shaders.shader_glass_002], [1.0]),
        "white_metal": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "top": TextureAssignments([dishwasher_shaders.shader_black_medal_002], [1.0]),
        "name_material": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def microwave_materials():
    metal_shaders = get_all_metal_shaders()
    return {
        "surface": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "back": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "black_glass": TextureAssignments(
            [microwave_shaders.shader_black_glass], [1.0]
        ),
        "glass": TextureAssignments([microwave_shaders.shader_glass], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def oven_materials():
    metal_shaders = get_all_metal_shaders()
    return {
        "surface": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "back": TextureAssignments([oven_shaders.shader_black_medal], [1.0]),
        "white_metal": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "black_glass": TextureAssignments(
            [oven_shaders.shader_super_black_glass], [1.0]
        ),
        "glass": TextureAssignments([oven_shaders.shader_glass], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def tv_materials():
    return {
        "surface": TextureAssignments([metal, plastic_rough], [1.0, 0.2]),
        "screen_surface": TextureAssignments([text.Text], [1.0]),
        "support": TextureAssignments([metal, plastic_rough], [1.0, 0.2]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def bathtub_materials():
    return {
        "surface": TextureAssignments([ceramic], [1]),
        "leg": TextureAssignments([metal], [1.0]),
        "hole": TextureAssignments([metal], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def bathroom_sink_materials():
    return {
        "surface": TextureAssignments([ceramic, metal], [0.9, 0.1]),
        # rest inherited from bathtub_materials
    }


def toilet_materials():
    return {
        "surface": TextureAssignments([ceramic, metal], [0.9, 0.1]),
        "hardware_surface": TextureAssignments([metal], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def hardware_materials():
    return {
        "surface": TextureAssignments([metal], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def blanket_materials():
    return {
        "surface": TextureAssignments([ArtFabric, fabrics.fabric_random], [1.0, 1.0]),
    }


def pants_materials():
    return {
        "surface": TextureAssignments([ArtFabric, fabrics.fabric_random], [1.0, 1.0]),
    }


def towel_materials():
    return {
        "surface": TextureAssignments([ArtRug, rug], [0.2, 0.8]),
    }


def acquarium_materials():
    return {
        "glass_surface": TextureAssignments([glass], [1.0]),
        "belt_surface": TextureAssignments([metal.galvanized_metal], [1.0]),
        "water_surface": TextureAssignments([water], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [0, DEFAULT_EDGE_WEAR_PROB],
    }


def ceiling_light_materials():
    return {
        "black_material": TextureAssignments(
            [ceiling_light_shaders.shader_black], [1.0]
        ),
        "white_material": TextureAssignments(
            [ceiling_light_shaders.shader_lamp_bulb_nonemissive], [1.0]
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def lamp_materials():
    return {
        "black_material": TextureAssignments([lamp_shaders.shader_black], [1.0]),
        "metal": TextureAssignments([lamp_shaders.shader_metal], [1.0]),
        "lampshade": TextureAssignments([lamp_shaders.shader_lampshade], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [0, 0],
    }


def table_cocktail_materials():
    # top materials are: choice(['marble', 'tiled_wood', 'plastic', 'glass']),
    # choice(['brushed_metal', 'grained_metal', 'galvanized_metal', 'wood', 'glass']),
    metal_shaders = get_all_metal_shaders()
    return {
        "top": TextureAssignments(
            [
                table_marble.shader_marble,
                woods.tiled_wood.shader_wood_tiled,
                shader_rough_plastic,
                glass_volume.shader_glass_volume,
            ],
            [1.0, 1.0, 1.0, 1.0],
        ),
        "leg": TextureAssignments(
            [*metal_shaders, wood.shader_wood, glass_volume.shader_glass_volume],
            [1.0] * len(metal_shaders) + [1.0, 1.0],
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def table_dining_materials():
    metal_shaders = get_all_metal_shaders()
    fabric_shaders = get_all_fabric_shaders()
    probs = [1.0 / len(metal_shaders)] * len(metal_shaders)

    return {
        "top": MaterialOptions(
            [
                (table_marble.shader_marble, 2.0),
                (wood.shader_wood, 1.0),
                (dishwasher_shaders.shader_glass_002, 1.0),
                (oven_shaders.shader_super_black_glass, 1.0),
                (woods.tiled_wood.shader_wood_tiled, 2.0),
                (glass_volume.shader_glass_volume, 1.0),
                *(zip(metal_shaders, probs)),
            ]
        ),
        "leg": MaterialOptions(
            [
                (wood.shader_wood, 1.0),
                (glass_volume.shader_glass_volume, 1.0),
                (plastic_furniture(), 1.0),
                *(zip(metal_shaders, probs)),
            ]
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def bar_chair_materials(leg_style=None):
    metal_shaders = get_all_metal_shaders()
    if leg_style == "wheeled":
        probs = [0.01 / len(metal_shaders)] * len(metal_shaders)
    else:
        probs = [1.0 / len(metal_shaders)] * len(metal_shaders)
    return {
        "seat": TextureAssignments([fabrics.shader_leather], [1.0]),
        "leg": TextureAssignments([wood.shader_wood, *metal_shaders], [1.0] + probs),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, 0.0],
    }


def chair_materials():
    return {
        "limb": TextureAssignments([metal, wood, fabrics.fabric_random], [2.0, 2.0, 2]),
        "surface": TextureAssignments(
            [plastic_rough, wood, fabrics.fabric_random], [0.3, 0.5, 0.7]
        ),
        "panel": TextureAssignments(
            [plastic_rough, wood, fabrics.fabric_random], [0.3, 0.5, 0.7]
        ),
        "arm": TextureAssignments(
            [plastic, wood, fabrics.fabric_random], [0.3, 0.5, 0.7]
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def office_chair_materials(leg_style=None):
    metal_shaders = get_all_metal_shaders()
    if leg_style == "wheeled":
        probs = [0.01 / len(metal_shaders)] * len(metal_shaders)
    else:
        probs = [1.0 / len(metal_shaders)] * len(metal_shaders)
    return {
        "top": TextureAssignments(
            [
                fabrics.shader_leather,
                wood.shader_wood,
                shader_rough_plastic,
                glass_volume.shader_glass_volume,
            ],
            [1.0, 1.0, 1.0, 1.0],
        ),
        "leg": TextureAssignments([wood.shader_wood, *metal_shaders], [1.0] + probs),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def bedframe_materials():
    return {
        "surface": TextureAssignments(
            [wood, plaster],
            [
                2.0,
                1.0,
            ],
        ),
        "limb_surface": TextureAssignments([wood, plaster], [2.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def mattress_materials():
    return {
        "surface": TextureAssignments([sofa_fabric], [1.0]),
    }


def pillow_materials():
    return {
        "surface": TextureAssignments([ArtFabric, sofa_fabric], [1.0, 1.0]),
    }


def sofa_materials():
    return {
        "sofa_fabric": MaterialOptions(
            [
                (velvet.shader_velvet, 0.5),
                (sofa_fabric.shader_sofa_fabric, 0.3),
                (fabrics.shader_leather, 0.2),
            ]
        ),
    }


def book_materials():
    return {
        "surface": TextureAssignments([plaster], [1.0]),
        "cover_surface": TextureAssignments([text.Text], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [0, 0],
    }


def vase_materials():
    return {
        "surface": TextureAssignments(
            [vase_shaders.shader_ceramic, glass_volume.shader_glass_volume], [1.0, 1.0]
        ),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def pan_materials():
    return {
        "surface": TextureAssignments([metal], [1.0]),
        "inside": TextureAssignments([metal], [1.0]),
        # no guard as it overrides over tableware_materials
    }


def cup_materials():
    return {
        "surface": TextureAssignments([glass, plastic], [1.0, 1.0]),
        "wrap_surface": TextureAssignments([text.Text], [1.0]),
    }


def bottle_materials():
    return {
        "surface": TextureAssignments([glass, plastic], [1.0, 1.0]),
        "wrap_surface": TextureAssignments([text.Text], [1.0]),
        "cap_surface": TextureAssignments([metal, plastic], [1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, 0.0],
    }


def tableware_materials(fragile=False, transparent=False):
    if fragile:
        surface_materials = TextureAssignments([ceramic, glass, plastic], [1.0, 1, 1])
    elif transparent:
        surface_materials = TextureAssignments([ceramic, glass], [1.0, 1])
    else:
        surface_materials = TextureAssignments(
            [ceramic, glass, plastic, metal, wood], [1, 1, 1.0, 1, 1]
        )

    return {
        "surface": surface_materials,
        "guard": TextureAssignments([wood, plastic], [1.0, 1.0]),
        "inside": TextureAssignments([ceramic, metal], [1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def can_materials():
    return {
        "surface": TextureAssignments([metal], [1.0]),
        "wrap_surface": TextureAssignments([text.Text], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def jar_materials():
    return {
        "surface": TextureAssignments([glass], [1.0]),
        "cap_surface": TextureAssignments([metal], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def foodbag_materials():
    return {
        "surface": TextureAssignments([text.Text], [1.0]),
    }


def lid_materials():
    return {
        "surface": TextureAssignments([ceramic, metal], [0.5, 0.5]),
        "rim_surface": TextureAssignments([metal], [1.0]),
        "handle_surface": TextureAssignments([metal, ceramic], [1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, 0.0],
    }


def glasslid_materials():
    return {
        "surface": TextureAssignments([glass], [1.0]),
        "rim_surface": TextureAssignments([metal], [1.0]),
        "handle_surface": TextureAssignments([metal, ceramic], [1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, 0.0],
    }


def plant_container_materials():
    return {
        "surface": TextureAssignments([ceramic, metal], [3.0, 1.0]),
        "dirt_surface": TextureAssignments([dirt], [1.0]),
    }


def balloon_materials():
    return {
        "surface": TextureAssignments([metal], [1.0]),
    }


def range_hood_materials():
    return {
        "surface": TextureAssignments([metal], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def wall_art_materials():
    return {
        "frame": TextureAssignments([wood, metal], [1.0, 1.0]),
        "surface": TextureAssignments([Art], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def mirror_materials():
    return {
        "frame": TextureAssignments([wood, metal], [1.0, 1.0]),
        "surface": TextureAssignments([mirror], [1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def kitchen_sink_materials():
    shaders = get_all_metal_shaders()
    sink_color = metal.sample_metal_color(metal_color="natural")
    if uniform() < 0.5:
        tap_color = metal.sample_metal_color(metal_color="plain")
    else:
        tap_color = metal.sample_metal_color(metal_color="natural")
    sink_shaders = [
        lambda nw, *args: shader(nw, *args, base_color=sink_color) for shader in shaders
    ]
    tap_shaders = [
        lambda nw, *args: shader(nw, *args, base_color=tap_color) for shader in shaders
    ]
    return {
        "sink": TextureAssignments(sink_shaders, [1.0, 1.0, 1.0, 1.0]),
        "tap": TextureAssignments(tap_shaders, [1.0, 1.0, 1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def kitchen_tap_materials():
    shaders = get_all_metal_shaders()
    if uniform() < 0.5:
        tap_color = metal.sample_metal_color(metal_color="plain")
    else:
        tap_color = metal.sample_metal_color(metal_color="natural")
    tap_shaders = [
        lambda nw, *args: shader(nw, *args, base_color=tap_color) for shader in shaders
    ]
    return {
        "tap": TextureAssignments(tap_shaders, [1.0, 1.0, 1.0, 1.0]),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


def rug_materials():
    return {
        "surface": MaterialOptions(
            [
                (rug, 3.0),
                (ArtRug, 2.0),
                (fabrics.fabric_random, 5.0),
            ]
        )
    }


def window_materials():
    metal_shaders = get_all_metal_shaders()
    plastic_shaders = [plastic_rough.shader_rough_plastic]
    wood_shaders = [wood.shader_wood]
    glass_shaders = [glass.shader_glass]

    frame_shaders = metal_shaders + plastic_shaders + wood_shaders
    return {
        "frame": TextureAssignments(frame_shaders, [1.0] * len(frame_shaders)),
        "curtain": TextureAssignments(frame_shaders, [1.0] * len(frame_shaders)),
        "curtain_frame": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "glass": TextureAssignments(metal_shaders, [1.0] * len(metal_shaders)),
        "wear_tear": [procedural_scratch, procedural_edge_wear],
        "wear_tear_prob": [DEFAULT_SCRATCH_PROB, DEFAULT_EDGE_WEAR_PROB],
    }


AssetList = {
    # appliances
    "BeverageFridgeFactory": beverage_fridge_materials,  # looks like dishwasher currently
    "DishwasherFactory": dishwasher_materials,
    "MicrowaveFactory": microwave_materials,
    "OvenFactory": oven_materials,  # looks like dishwasher currently
    "TVFactory": tv_materials,
    "MonitorFactory": None,  # inherits from TVFactory
    # bathroom
    "BathtubFactory": bathtub_materials,
    "BathroomSinkFactory": bathroom_sink_materials,  # inheriting from bathtub factory, so not used
    "HardwareFactory": hardware_materials,
    "ToiletFactory": toilet_materials,
    # clothes
    "BlanketFactory": blanket_materials,  # has Art which is a class, not func,
    # also "Normal Not Found" is printed when generating
    ############## this point onwards, using this dictionary to get corresponding
    ############## material functions except for tableware base
    "PantsFactory": pants_materials,  # same comment as above
    "ShirtFactory": pants_materials,  # same comment as above
    "TowelFactory": towel_materials,
    # decor
    "AquariumTankFactory": acquarium_materials,
    # lighting
    "CausticsLampFactory": None,  # the properties are not materials, so skipping
    "CeilingLightFactory": ceiling_light_materials,
    "PointLampFactory": None,  # the properties are not materials, so skipping
    "LampFactory": lamp_materials,  # really required bunch of changes to expose the materials
    # seating: chairs
    "BarChairFactory": bar_chair_materials,
    "ChairFactory": chair_materials,  # an internal reassignment that overrides surface with the limb material
    "OfficeChairFactory": office_chair_materials,
    # seating: sofas and beds
    "BedFactory": None,  # uses the below factories, so no materials
    "BedFrameFactory": bedframe_materials,
    "MattressFactory": mattress_materials,
    "PillowFactory": pillow_materials,
    "SofaFactory": sofa_materials,
    # shelves: todo
    "SimpleDeskFactory": None,
    "SimpleBookcaseFactory": None,
    "CellShelfFactory": None,
    "TVStandFactory": None,
    "TriangleShelfFactory": None,
    "LargeShelfFactory": None,
    "SingleCabinetFactory": None,
    "KitchenCabinetFactory": None,
    "KitchenSpaceFactory": None,
    "KitchenIslandFactory": None,
    # table decorations : they have their own materials
    "BookFactory": book_materials,
    "BookColumnFactory": None,  # use BookFactory
    "BookStackFactory": None,  # use BookFactory
    "VaseFactory": vase_materials,
    # sink and tap
    "SinkFactory": kitchen_sink_materials,
    "TapFactory": kitchen_tap_materials,
    # tables
    "TableCocktailFactory": table_cocktail_materials,
    "TableDiningFactory": table_dining_materials,
    "TableTopFactory": None,  # not sure where the materials are used in it
    # Tableware
    "TablewareFactory": tableware_materials,  # only function with arguments
    # 'TablewareFactory': tableware_materials_default,  # directly uses the following functions (not through the AssetList Dictionary)
    "SpoonFactory": None,  # uses materials from tableware base
    "KnifeFactory": None,  # uses materials from tableware base
    "ChopsticksFactory": None,  # uses materials from tableware base
    "ForkFactory": None,  # uses materials from tableware base
    "SpatulaFactory": None,  # uses materials from tableware base
    "PanFactory": pan_materials,
    "PotFactory": None,  # uses the same materials as PanFactory
    "CupFactory": cup_materials,
    "WineglassFactory": None,  # uses materials from transparent tableware
    "PlateFactory": None,  # uses materials from tableware base
    "BowlFactory": None,  # uses materials from tableware base
    "FruitContainerFactory": None,  # uses materials from tableware base
    "BottleFactory": bottle_materials,
    "CanFactory": can_materials,
    "JarFactory": jar_materials,
    "FoodBagFactory": foodbag_materials,
    "FoodBoxFactory": foodbag_materials,  # same params as above
    "LidFactory": lid_materials,
    "GlassLidFactory": glasslid_materials,
    "PlantContainerFactory": plant_container_materials,
    # wall decorations
    "BalloonFactory": balloon_materials,
    "RangeHoodFactory": range_hood_materials,  # getting RangeHoodFactory not Found.
    "WallArtFactory": wall_art_materials,
    "MirrorFactory": mirror_materials,
    # window
    "WindowFactory": window_materials,
    "RugFactory": rug_materials,
}
