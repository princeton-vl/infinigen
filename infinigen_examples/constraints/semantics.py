# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import infinigen.assets.static_assets as static_assets
from infinigen.assets.objects import (
    appliances,
    bathroom,
    clothes,
    decor,
    elements,
    lamp,
    seating,
    shelves,
    table_decorations,
    tables,
    tableware,
    wall_decorations,
    windows,
)
from infinigen.core.tags import Semantics


def home_asset_usage():
    """Defines what generators are consider to fulfill what roles in a home setting.

    The primary effect of this to determine what types of objects are returned by the square brackets [ ] operator in home_furniture_constraints

    You can define these however you like - use

    See the `Semantics` class in `infinigen.core.tags` for a list of possible semantics, or add your own.

    """

    # TODO: this whole used_as will be integrated into the constraint language. Currently there are two paralell semantics trees, one to define the tags and one to use them.

    used_as = {}

    # region small objects

    used_as[Semantics.Dishware] = {
        tableware.PlateFactory,
        tableware.BowlFactory,
        tableware.WineglassFactory,
        tableware.PanFactory,
        tableware.PotFactory,
        tableware.CupFactory,
    }
    used_as[Semantics.Cookware] = {tableware.PotFactory, tableware.PanFactory}
    used_as[Semantics.Utensils] = {
        tableware.SpoonFactory,
        tableware.KnifeFactory,
        tableware.ChopsticksFactory,
        tableware.ForkFactory,
    }

    used_as[Semantics.FoodPantryItem] = {
        tableware.CanFactory,
        tableware.FoodBagFactory,
        tableware.FoodBoxFactory,
        tableware.JarFactory,
        tableware.BottleFactory,
    }

    used_as[Semantics.TableDisplayItem] = {
        tableware.FruitContainerFactory,
        table_decorations.VaseFactory,
        tableware.BowlFactory,
        tableware.PotFactory,
    }

    used_as[Semantics.OfficeShelfItem] = {
        table_decorations.BookStackFactory,
        table_decorations.BookColumnFactory,
        elements.NatureShelfTrinketsFactory,
    }

    used_as[Semantics.KitchenCounterItem] = set().union(
        used_as[Semantics.Dishware],
        used_as[Semantics.Cookware],
        {
            table_decorations.BookColumnFactory,
            tableware.JarFactory,
        },
    )

    used_as[Semantics.BathroomItem] = {
        tableware.BottleFactory,
        tableware.BowlFactory,
        clothes.TowelFactory,
    }

    used_as[Semantics.ClothDrapeItem] = {
        # objects that can be strewn about / draped over furniture
        # clothes.BlanketFactory,
        clothes.PantsFactory,
        clothes.ShirtFactory,
    }

    used_as[Semantics.HandheldItem] = set.union(
        used_as[Semantics.Utensils],
        used_as[Semantics.FoodPantryItem],
        used_as[Semantics.TableDisplayItem],
        used_as[Semantics.OfficeShelfItem],
        used_as[Semantics.ClothDrapeItem],
        used_as[Semantics.Dishware],
    )

    # endregion

    # region furniture

    used_as[Semantics.Sink] = {
        table_decorations.SinkFactory,
        bathroom.BathroomSinkFactory,
        bathroom.StandingSinkFactory,
    }

    used_as[Semantics.Storage] = {
        shelves.SimpleBookcaseFactory,
        shelves.CellShelfFactory,
        shelves.LargeShelfFactory,
        # static_assets.StaticShelfFactory,
        shelves.KitchenCabinetFactory,
        shelves.SingleCabinetFactory,
    }

    used_as[Semantics.SideTable] = {
        shelves.SidetableDeskFactory,
        tables.SideTableFactory,
    }

    used_as[Semantics.Table] = set.union(
        used_as[Semantics.SideTable],
        {
            tables.TableDiningFactory,
            tables.TableCocktailFactory,
            shelves.SimpleDeskFactory,
            tables.CoffeeTableFactory,
            # static_assets.StaticTableFactory,
        },
    )

    used_as[Semantics.Chair] = {
        seating.BarChairFactory,
        seating.ChairFactory,
        seating.OfficeChairFactory,
    }

    used_as[Semantics.LoungeSeating] = {
        seating.SofaFactory,
        # static_assets.StaticSofaFactory,
        seating.ArmChairFactory,
    }

    used_as[Semantics.Seating] = set.union(
        used_as[Semantics.Chair],
        used_as[Semantics.LoungeSeating],
    )

    used_as[Semantics.KitchenAppliance] = {
        appliances.DishwasherFactory,
        appliances.OvenFactory,
        appliances.BeverageFridgeFactory,
        appliances.MicrowaveFactory,
    }

    used_as[Semantics.KitchenCounter] = {
        shelves.KitchenSpaceFactory,
        shelves.KitchenIslandFactory,
    }

    used_as[Semantics.Bed] = {
        seating.BedFactory,
    }

    used_as[Semantics.Furniture] = set().union(
        used_as[Semantics.Storage],
        used_as[Semantics.Table],
        used_as[Semantics.Seating],
        used_as[Semantics.KitchenCounter],
        used_as[Semantics.KitchenAppliance],
        used_as[Semantics.Bed],
        {
            bathroom.StandingSinkFactory,
            bathroom.ToiletFactory,
            bathroom.BathtubFactory,
            seating.SofaFactory,
            # static_assets.StaticSofaFactory,
            shelves.TVStandFactory,
        },
    )

    # endregion furniture

    used_as[Semantics.WallDecoration] = {
        wall_decorations.WallArtFactory,
        wall_decorations.MirrorFactory,
        wall_decorations.BalloonFactory,
    }

    used_as[Semantics.Door] = {
        elements.doors.GlassPanelDoorFactory,
        elements.doors.LiteDoorFactory,
        elements.doors.LouverDoorFactory,
        elements.doors.PanelDoorFactory,
    }

    used_as[Semantics.Window] = {windows.WindowFactory}

    used_as[Semantics.CeilingLight] = {
        lamp.CeilingLightFactory,
    }

    used_as[Semantics.Lighting] = set().union(
        used_as[Semantics.CeilingLight],
        {
            lamp.LampFactory,
            lamp.FloorLampFactory,
            lamp.DeskLampFactory,
        },
    )

    used_as[Semantics.Object] = set().union(
        used_as[Semantics.Furniture],
        used_as[Semantics.Sink],
        used_as[Semantics.Door],
        used_as[Semantics.Window],
        used_as[Semantics.WallDecoration],
        used_as[Semantics.HandheldItem],
        used_as[Semantics.Lighting],
        {
            tableware.PlantContainerFactory,
            tableware.LargePlantContainerFactory,
            decor.AquariumTankFactory,
            appliances.TVFactory,
            appliances.MonitorFactory,
            elements.RugFactory,
            bathroom.HardwareFactory,
        },
    )

    # region Extra metadata about assets
    # TODO be move outside of the semantics heirarchy and into separate AssetFactory.metadata classvar

    used_as[Semantics.RealPlaceholder] = {
        appliances.MonitorFactory,
        appliances.TVFactory,
        bathroom.BathroomSinkFactory,
        bathroom.StandingSinkFactory,
        bathroom.ToiletFactory,
        decor.AquariumTankFactory,
        elements.RackFactory,
        elements.RugFactory,
        seating.BedFrameFactory,
        seating.BedFactory,
        seating.ChairFactory,
        shelves.KitchenSpaceFactory,
        tables.TableCocktailFactory,
        table_decorations.BookColumnFactory,
        table_decorations.BookFactory,
        table_decorations.BookStackFactory,
        table_decorations.SinkFactory,
        tableware.BowlFactory,
        tableware.FoodBoxFactory,
        tableware.FruitContainerFactory,
        tableware.LargePlantContainerFactory,
        tableware.PlantContainerFactory,
        tableware.PotFactory,
        wall_decorations.BalloonFactory,
        wall_decorations.MirrorFactory,
        wall_decorations.WallArtFactory,
        shelves.SingleCabinetFactory,
        shelves.KitchenCabinetFactory,
        shelves.CellShelfFactory,
        elements.NatureShelfTrinketsFactory,
    }

    used_as[Semantics.AssetAsPlaceholder] = set()

    used_as[Semantics.AssetPlaceholderForChildren] = {
        shelves.SimpleBookcaseFactory,
        shelves.CellShelfFactory,
        shelves.SingleCabinetFactory,
        shelves.KitchenCabinetFactory,
        shelves.LargeShelfFactory,
        static_assets.StaticShelfFactory,
        table_decorations.SinkFactory,
        tables.TableCocktailFactory,
    }

    used_as[Semantics.PlaceholderBBox] = {
        seating.SofaFactory,
        appliances.OvenFactory,
    }

    used_as[Semantics.SingleGenerator] = (
        set()
        .union(
            used_as[Semantics.Dishware],
            used_as[Semantics.Utensils],
            {
                lamp.CeilingLightFactory,
                lamp.CeilingClassicLampFactory,
                seating.ChairFactory,
                seating.BarChairFactory,
                seating.OfficeChairFactory,
            },
        )
        .difference({tableware.CupFactory})
    )

    used_as[Semantics.NoRotation] = set().union(
        used_as[Semantics.WallDecoration],
        {
            bathroom.HardwareFactory,
            lamp.CeilingLightFactory,  # rotationally symetric
        },
    )

    used_as[Semantics.NoCollision] = {
        elements.RugFactory,
    }

    used_as[Semantics.NoChildren] = {
        elements.RugFactory,
        wall_decorations.MirrorFactory,
        wall_decorations.WallArtFactory,
        lamp.CeilingLightFactory,
    }

    # endregion

    return used_as
