# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


class Task:
    Coarse = "coarse"
    Populate = "populate"
    FineTerrain = "fine_terrain"
    Render = "render"
    GroundTruth = "ground_truth"
    MeshSave = "mesh_save"


class Materials:
    GroundCollection = "ground_collection"
    MountainCollection = "mountain_collection"
    LiquidCollection = "liquid_collection"
    Eroded = "eroded"
    Lava = "lava"
    Snow = "snow"
    Atmosphere = "atmosphere"
    Clouds = "clouds"
    Beach = "beach"
    all = [GroundCollection, MountainCollection, Eroded, LiquidCollection, Lava, Snow, Atmosphere, Clouds, Beach]

class LandTile:
    Canyon = "Canyon"
    Canyons = "Canyons"
    Cliff = "Cliff"
    Mesa = "Mesa"
    Mountain = "Mountain"
    River = "River"
    Volcano = "Volcano"
    Coast = "Coast"
    MultiMountains = "MultiMountains"

class Assets:
    Caves = "Caves"
    UpsidedownMountains = "UpsidedownMountains"
    Ocean = "Ocean"

class AssetFile:
    Heightmap = "heightmap"
    Mask = "mask"
    TileSize = "tile_size"
    Params = "params"
    Finish = "finish"

class Process:
    Snowfall = "snowfall"
    Erosion = "erosion"
    IceErosion = 'ice_erosion'
    Eruption = "eruption"

class TerrainNames:
    OpaqueTerrain = "OpaqueTerrain"
    CollectiveTransparentTerrain = "CollectiveTransparentTerrain"

class Transparency:
    IndividualTransparent = "IndividualTransparent"
    CollectiveTransparent = "CollectiveTransparent"
    Opaque = "Opaque"

class ElementNames:
    Atmosphere = "atmosphere"
    Liquid = "liquid"
    Caves = "caves"
    Clouds = "clouds"
    LandTiles = "landtiles"
    Ground = "ground"
    Mountains = "mountains"
    WarpedRocks = "warped_rocks"
    VoronoiRocks = "voronoi_rocks"
    VoronoiGrains = "voronoi_grains"
    UpsidedownMountains = "upsidedown_mountains"
    Volcanos = "volcanos"
    FloatingIce = "floating_ice"

class Tags:
    Cave = "cave"
    LiquidCovered = "liquid_covered"
    UpsidedownMountainsLowerPart = "upsidedown_mountain_lower_part"
    Terrain = "terrain"
    Landscape = "landscape"
    OutOfView = "out_of_view"

class Attributes:
    BoundarySDF = "BoundarySDF"
    ElementTag = "ElementTag"

class SelectionCriterions:
    CloseUp = "closeup"
    Altitude = "altitude"

class ElementTag:
    Liquid = 0
    Clouds = 1
    Terrain = 2
    WarpedRocks = 3
    VoronoiRocks = 4
    VoronoiGrains = 5
    Volcanos = 6
    FloatingIce = 7
    UpsidedownMountains = 8
    total_cnt = 9
    map = [
        ElementNames.Liquid, ElementNames.Clouds, Tags.Terrain, ElementNames.WarpedRocks, ElementNames.VoronoiRocks,
        ElementNames.VoronoiGrains, ElementNames.Volcanos, ElementNames.FloatingIce, ElementNames.UpsidedownMountains,
    ]

class SurfaceTypes:
    BlenderDisplacement = "BlenderDisplacement"
    Displacement = "Displacement"
    SDFPerturb = "SDFPerturb"
