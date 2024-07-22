# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Meenal Parakh: initial assignment system, separate materials from implementations
# - Alexander Raistrick: refactor

from infinigen.assets.materials import (
    art,
    ceramic,
    creature,
    dev,
    fabric,
    lamp_shaders,
    metal,
    plant,
    plastic,
    terrain,
    text,
    wood,
)
from infinigen.assets.materials.wear_tear import edge_wear, scratches

wear_tear_prob = [0.5, 0.5]

wear_tear = [scratches, edge_wear]


woods = [
    (wood.Wood, 1.0),
    (wood.WhitePlywood, 0.1),
    (wood.BlackPlywood, 0.1),
    (wood.BlondePlywood, 0.1),
]

wood_tiles = [
    (wood.CompositeWoodTile, 1.0),
    (wood.CrossedWoodTile, 1.0),
    (wood.HardwoodFloor, 1.0),
    (wood.HexagonWoodTile, 1.0),
    (wood.SquareWoodTile, 1.0),
    (wood.StaggeredWoodTile, 1.0),
]

metals = [
    (metal.MetalBasic, 3.0),
    (metal.BrushedMetal, 2.0),
    (metal.GalvanizedMetal, 1.0),
    (metal.GrainedMetal, 1.0),
    (metal.HammeredMetal, 0.7),
    (metal.BrushedBlackMetal, 0.3),
]

fabrics = [
    (fabric.CoarseKnitFabric, 1.0),
    (fabric.FineKnitFabric, 1.0),
    (fabric.PlaidFabric, 1.0),
    (fabric.Leather, 0.3),
    (fabric.SofaFabric, 1.0),
]

plastics = [
    (plastic.Plastic, 1.0),
    (plastic.PlasticRough, 1.0),
    (plastic.PlasticTranslucent, 1.0),
]

shelf_board = (
    metals
    + plastics
    + [
        (ceramic.GlassVolume, 1.0),
        (wood.Wood, 5.0),
        (wood.WhitePlywood, 10),
        (wood.BlackPlywood, 10),
        (wood.BlondePlywood, 10),
    ]
)

marble = [
    (ceramic.Marble, 1.0),
    (ceramic.MarbleRegular, 0.2),
    (ceramic.MarbleVoronoi, 0.2),
]

kitchen_appliance_hard = metals + [(plastic.PlasticRough, 2.0)]
appliance_front_maybeglass = metals + [
    (metal.BlackGlass, 7.0),
    (ceramic.Glass, 1.0),
]

appliance_handle = metals + [
    (plastic.PlasticRough, 1.0),
    (wood.Wood, 3.0),
    (metal.WhiteMetal, 2.0),
]

bathroom_touchsurface = [
    (ceramic.Ceramic, 1.0),
    (metal.BrushedMetal, 0.1),
    (metal.MetalBasic, 0.1),
]

blanket = fabrics
pants = fabrics + [(art.ArtFabric, 1.0)]
towel = fabrics + [(art.ArtRug, 2.0), (fabric.Rug, 5.0)]

lampshade = fabrics + [
    (lamp_shaders.LampShade, 7.0),
]

table_top = [
    (ceramic.Marble, 1.0),
    (wood.TiledWood, 1.0),
    (plastic.PlasticRough, 1.0),
    (ceramic.GlassVolume, 1.0),
]

decorative_metal = [
    (metal.BrushedMetal, 1.0),
    (metal.GalvanizedMetal, 1.0),
    (metal.GrainedMetal, 1.0),
    (metal.HammeredMetal, 1.0),
]

furniture_leg = decorative_metal + [
    (wood.Wood, 1.0),
    (ceramic.GlassVolume, 1.0),
    (plastic.PlasticRough, 1.0),
]

furniture_hard_surface = [
    (plastic.PlasticRough, 1.0),
    (wood.Wood, 1.0),
]

table_top = [
    (ceramic.Marble, 1.0),
    (wood.TiledWood, 1.0),
    (wood.Wood, 1.0),
    (plastic.PlasticRough, 0.5),
    (ceramic.GlassVolume, 1.0),
]

tableware = [
    (ceramic.Ceramic, 1.0),
    (ceramic.Glass, 1.0),
    (plastic.Plastic, 1.0),
    (metal.MetalBasic, 1.0),
    (wood.Wood, 1.0),
]

officechair_seat = [
    (fabric.Leather, 1.0),
    (wood.Wood, 1.0),
    (plastic.PlasticRough, 1.0),
    (ceramic.GlassVolume, 1.0),
]

bedframe = decorative_metal + [
    (wood.Wood, 7.0),
    (ceramic.Plaster, 2.0),
]

plain_fabric = [
    (fabric.SofaFabric, 1.0),
]

decorative_fabric = (
    fabrics
    + [
        (art.ArtFabric, 5.0),
    ],
)

large_seat_fabric = [
    (fabric.Velvet, 0.3),
    (fabric.SofaFabric, 0.5),
    (fabric.Leather, 0.2),
]

rug_fabric = fabrics + [
    (fabric.Rug, 7.0),
    (art.ArtRug, 5.0),
    (fabric.CoarseKnitFabric, 1.0),
]

graphicdesign = [  # bottle wrappers, books, etc
    (text.Text, 1.0),
    (dev.BasicBSDF, 0.1),
]

decorative_hard = (  # vases, plantpots
    decorative_metal
    + [
        (ceramic.VaseCeramic, 4.0),
        (ceramic.ColoredGlass, 2.0),
        (ceramic.Marble, 1.0),
        (ceramic.GlassVolume, 2.0),
        (ceramic.Ceramic, 1.0),
    ]
)

cup = decorative_hard + [
    (metal.MetalBasic, 2.0),
    (plastic.Plastic, 5.0),
    (plastic.PlasticTranslucent, 5.0),
    (ceramic.Glass, 3.0),
]
jar = cup
lid = cup

frame = decorative_metal + woods

potting_soil = [
    (terrain.Mud, 1),
    (terrain.Sand, 1),
    (terrain.Soil, 3),
    (terrain.Dirt, 6),
]

ground = [
    (terrain.Mud, 2),
    (terrain.Sand, 1),
    (terrain.CobbleStone, 1),
    (terrain.CrackedGround, 1),
    (terrain.Dirt, 1),
    (terrain.Stone, 1),
    (terrain.Soil, 1),
    (terrain.ChunkyRock, 0),
]

beach = [
    (terrain.Sand, 10),
    (terrain.CrackedGround, 1),
    (terrain.Dirt, 1),
    (terrain.Stone, 1),
    (terrain.Soil, 1),
]

eroded = [
    (terrain.Sand, 1),
    (terrain.CrackedGround, 1),
    (terrain.Dirt, 1),
    (terrain.Stone, 1),
    (terrain.Soil, 1),
]

mountain = [
    (terrain.Mountain, 10),
    (terrain.Sandstone, 2),
]

rock = [
    (terrain.Mountain, 5),
    (terrain.Stone, 1),
]

bark = [
    (plant.BarkBirch, 0.1),
    (plant.BarkRandom, 0.9),
    # ('wood', 0.01),
]

bird = [
    (creature.SpotSparse, 4),
    (creature.ReptileBrownCircle, 0.5),
    (creature.ReptileTwoColor, 0.5),
    (creature.Bird, 5),
]

carnivore = [
    (creature.Tiger, 3),
    (creature.Giraffe, 0.2),
    (creature.SpotSparse, 2),
]

reptile = [
    (creature.SnakeScale, 1),
]

fish = metals + [
    (creature.FishBody, 7),
    # (scale, 1),
]

herbivore = [
    (creature.Giraffe, 1),
    (creature.SpotSparse, 3),
]

beetle = [
    (creature.Chitin, 1),
]
