# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from collections import OrderedDict

from numpy.random import uniform

import infinigen.assets.static_assets as static_assets
from infinigen.assets.objects import (
    appliances,
    bathroom,
    decor,
    elements,
    lamp,
    seating,
    shelves,
    table_decorations,
    tables,
    tableware,
    wall_decorations,
)
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.constraints import usage_lookup
from infinigen.core.tags import Semantics, Subpart

from .indoor_asset_semantics import home_asset_usage
from .util import constraint_util as cu


def sample_home_constraint_params():
    return dict(
        # what pct of the room floorplan should we try to fill with furniture?
        furniture_fullness_pct=uniform(0.6, 0.9),
        # how many objects in each shelving per unit of volume
        obj_interior_obj_pct=uniform(0.5, 1),  # uniform(0.6, 0.9),
        # what pct of top surface of storage furniture should be filled with objects? e.g pct of top surface of shelf
        obj_on_storage_pct=uniform(0.1, 0.9),
        # what pct of top surface of NON-STORAGE objects should be filled with objects? e.g pct of countertop/diningtable covered in stuff
        obj_on_nonstorage_pct=uniform(0.1, 0.6),
        # meters squared of wall art per approx meters squared of FLOOR area. TODO cant measure wall area currently.
        painting_area_per_room_area=uniform(20, 60) / 40,
        # rare objects wont even be added to the constraint graph in most homes
        has_tv=uniform() < 0.5,
        has_aquarium_tank=uniform() < 0.15,
        has_birthday_balloons=uniform() < 0.15,
        has_cocktail_tables=uniform() < 0.15,
        has_kitchen_barstools=uniform() < 0.15,
    )


def home_constraints():
    """Construct a constraint graph which incentivizes realistic home layouts.

    Result will contain both hard constraints (`constraints`) and soft constraints (`score_terms`).

    Notes for developers:
    - This function is typically evaluated ONCE. It is not called repeatedly during the optimization process.
        - To debug values you will need to inject print statements into impl_bindings.py or evaluate.py. Better debugging tools will come soon.
        - Similarly, most `lambda:` statements below will only be evaluated once to construct the graph - do not assume they will be re-evaluated during optimization.
    - Available constraint options are in `infinigen/core/constraints/constraint_language/__init__.py`.
        - You can easily add new constraint functions by adding them here, and defining evaluator functions for them in `impl_bindings.py`
        - Using newly added constraint types as hard constraints may be rejected by our hard constraint solver
    - It is quite easy to specify an impossible constraint program, or one that our solver cannot solve:
        - By default, failing to solve the program correctly is just printed as a warning, and we still return the scene.
        - You can cause failed optimization results to crash instead using `-p solve_objects.abort_unsatisfied=True` in the command line.
    - More documentation coming soon, and feel free to ask questions on Github Issues!

    """

    used_as = home_asset_usage()
    usage_lookup.initialize_from_dict(used_as)

    rooms = cl.scene()[{Semantics.Room, -Semantics.Object}]
    obj = cl.scene()[{Semantics.Object, -Semantics.Room}]

    cutters = cl.scene()[Semantics.Cutter]
    window = cutters[Semantics.Window]
    doors = cutters[Semantics.Door]

    constraints = OrderedDict()
    score_terms = OrderedDict()

    # region overall fullness

    furniture = obj[Semantics.Furniture].related_to(rooms, cu.on_floor)
    wallfurn = furniture.related_to(rooms, cu.against_wall)
    storage = wallfurn[Semantics.Storage][static_assets.StaticShelfFactory]

    params = sample_home_constraint_params()

    for k, v in params.items():
        print(f"{home_constraints.__name__} params - {k}: {v}")

    score_terms["furniture_fullness"] = rooms.mean(
        lambda r: (
            furniture.related_to(r)
            .volume(dims=(0, 1))
            .safediv(r.volume(dims=(0, 1)))
            .sub(params["furniture_fullness_pct"])
            .abs()
            .minimize(weight=15)
        )
    )

    score_terms["obj_in_obj_fullness"] = rooms.mean(
        lambda r: (
            furniture.related_to(r).mean(
                lambda f: (
                    obj.related_to(f, cu.on)
                    .volume()
                    .safediv(f.volume())
                    .sub(params["obj_interior_obj_pct"])
                    .abs()
                    .minimize(weight=10)
                )
            )
        )
    )

    def top_fullness_pct(f):
        return (
            obj.related_to(f, cu.ontop)
            .volume(dims=(0, 1))
            .safediv(f.volume(dims=(0, 1)))
        )

    score_terms["obj_ontop_storage_fullness"] = rooms.mean(
        lambda r: (
            storage.related_to(r).mean(
                lambda f: (
                    top_fullness_pct(f)
                    .sub(params["obj_on_storage_pct"])
                    .abs()
                    .minimize(weight=10)
                )
            )
        )
    )

    score_terms["obj_ontop_nonstorage_fullness"] = rooms.mean(
        lambda r: (
            furniture[-Semantics.Storage]
            .related_to(r)
            .mean(
                lambda f: (
                    top_fullness_pct(f)
                    .sub(params["obj_on_nonstorage_pct"])
                    .abs()
                    .minimize(weight=10)
                )
            )
        )
    )

    # endregion

    # region furniture

    score_terms["furniture_aesthetics"] = wallfurn.mean(
        lambda t: (
            t.distance(wallfurn).hinge(0.2, 0.6).maximize(weight=0.6)
            + cl.accessibility_cost(t, furniture).minimize(weight=5)
            + cl.accessibility_cost(t, rooms).minimize(weight=10)
        )
    )

    constraints["storage"] = rooms.all(
        lambda r: (storage.related_to(r).count().in_range(1, 7))
    )
    score_terms["storage"] = rooms.mean(
        lambda r: (
            cl.accessibility_cost(
                storage.related_to(r), furniture.related_to(r), dist=0.5
            ).minimize(weight=5)
            + cl.accessibility_cost(storage.related_to(r), r, dist=0.5).minimize(
                weight=5
            )
        )
    )

    # endregion furntiure

    score_terms["portal_accessibility"] = (
        # make sure the fronts of objects are accessible where applicable
        #### disabled since its generally fine to block floor-to-ceiling windows a little
        # window.mean(lambda t: (
        #    cl.accessibility_cost(t, furniture, np.array([0, -1, 0]))
        # )).minimize(weight=2) +
        doors.mean(
            lambda t: (
                cl.accessibility_cost(t, furniture, cu.front_dir, dist=4)
                + cl.accessibility_cost(t, furniture, cu.back_dir, dist=4)
            )
        ).minimize(weight=5)
    )

    # region WALL/FLOOR COVERINGS
    walldec = obj[Semantics.WallDecoration].related_to(rooms, cu.flush_wall)
    wall_art = walldec[wall_decorations.WallArtFactory]
    mirror = walldec[wall_decorations.MirrorFactory]
    rugs = obj[elements.RugFactory].related_to(rooms, cu.on_floor)

    constraints["rugs"] = rooms.all(lambda r: (rugs.related_to(r).distance(rugs) >= 1))

    score_terms["rugs"] = rooms.all(
        lambda r: (cl.center_stable_surface_dist(rugs.related_to(r)).minimize(weight=1))
    )

    def vertical_diff(o, r):
        return (o.distance(r, cu.floortags) - o.distance(r, cu.ceilingtags)).abs()

    constraints["wall_decorations"] = rooms.all(
        lambda r: (
            wall_art.related_to(r).count().in_range(0, 6)
            * mirror.related_to(r).count().in_range(0, 1)
            * walldec.related_to(r).all(lambda t: t.distance(r, cu.floortags) > 0.6)
            # walldec.all(lambda t: (
            #    (vertical_diff(t, r).abs() < 1.5) *
            #    (t.distance(cutters) > 0.1)
            # ))
        )
    )
    score_terms["wall_decorations"] = rooms.mean(
        lambda r: (
            walldec.related_to(r).mean(
                lambda w: (
                    vertical_diff(w, r).abs().minimize(weight=1)
                    + w.distance(walldec).maximize(weight=1)
                    + w.distance(window).hinge(0.25, 10).maximize(weight=1)
                    + cl.angle_alignment_cost(w, r, cu.floortags).minimize(weight=5)
                    + cl.accessibility_cost(w, furniture, dist=1).minimize(weight=5)
                    + cl.center_stable_surface_dist(w).minimize(weight=1)
                )
            )
        )
    )

    score_terms["floor_covering"] = rugs.mean(
        lambda rug: (
            rug.distance(rooms, cu.walltags).maximize(weight=3)
            + cl.angle_alignment_cost(rug, rooms, cu.walltags).minimize(weight=3)
        )
    )
    # endregion

    # region PLANTS
    small_plants = obj[tableware.PlantContainerFactory].related_to(storage, cu.ontop)
    big_plants = (
        obj[tableware.LargePlantContainerFactory]
        .related_to(rooms, cu.on_floor)
        .related_to(rooms, cu.against_wall)
    )
    constraints["plants"] = rooms.all(
        lambda r: (
            big_plants.related_to(r).count().in_range(0, 1)
            * small_plants.related_to(storage.related_to(r)).count().in_range(0, 5)
        )
    )
    score_terms["plants"] = rooms.mean(
        lambda r: (
            big_plants.related_to(r)
            .mean(lambda p: p.distance(doors))
            .maximize(weight=5)
            + (  # small plants should be near window for sunlight
                small_plants.related_to(storage.related_to(r)).mean(
                    lambda p: p.distance(window.related_to(r))
                )
            ).minimize(weight=1)
        )
    )
    # endregion

    # region SIDETABLE
    sidetable = furniture[Semantics.SideTable].related_to(furniture, cu.side_by_side)

    score_terms["sidetable"] = rooms.mean(
        lambda r: (
            sidetable.related_to(r).mean(
                lambda t: (t.distance(r, cu.walltags).minimize(weight=1))
            )
        )
    )
    # endregion

    # region DESKS
    desks = wallfurn[shelves.SimpleDeskFactory]
    deskchair = furniture[seating.OfficeChairFactory].related_to(
        desks, cu.front_against
    )
    monitors = obj[appliances.MonitorFactory]
    constraints["desk"] = rooms.all(
        lambda r: (
            desks.related_to(r).all(
                lambda t: (
                    deskchair.related_to(r).related_to(t).count().in_range(0, 1)
                    * monitors.related_to(t, cu.ontop).count().equals(1)
                    * (obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 0)
                    * (deskchair.related_to(r).related_to(t).count() == 1)
                )
            )
        )
    )

    score_terms["desk"] = rooms.mean(
        lambda r: desks.mean(
            lambda d: (
                obj.related_to(d).count().maximize(weight=3)
                + d.distance(doors.related_to(r)).maximize(weight=0.1)
                + cl.accessibility_cost(d, furniture.related_to(r)).minimize(weight=3)
                + cl.accessibility_cost(d, r).minimize(weight=3)
                + monitors.related_to(d).mean(
                    lambda m: (
                        cl.accessibility_cost(m, r, dist=2).minimize(weight=3)
                        + cl.accessibility_cost(
                            m, obj.related_to(r), dist=0.5
                        ).minimize(weight=3)
                        + m.distance(r, cu.walltags).hinge(0.1, 1e7).minimize(weight=1)
                    )
                )
                + deskchair.distance(rooms, cu.walltags).maximize(weight=1)
            )
        )
    )

    # endregion

    # region ALL LIGHTING RULES

    lights = obj[Semantics.Lighting]
    floor_lamps = (
        lights[lamp.FloorLampFactory]
        .related_to(rooms, cu.on_floor)
        .related_to(rooms, cu.against_wall)
    )
    # constraints['lighting'] = rooms.all(lambda r: (
    #    # dont put redundant lights close to eachother (including lamps, ceiling lights, etc)
    #    lights.related_to(r).all(lambda l: l.distance(lights.related_to(r)) >= 2)
    # ))

    # endregion

    # region CEILING LIGHTS
    ceillights = lights[lamp.CeilingLightFactory]

    constraints["ceiling_lights"] = rooms.all(
        lambda r: (ceillights.related_to(r, cu.hanging).count().in_range(1, 4))
    )
    score_terms["ceiling_lights"] = rooms.mean(
        lambda r: (
            (ceillights.count() / r.volume(dims=2)).hinge(0.08, 0.15).minimize(weight=5)
            + ceillights.mean(
                lambda t: (
                    t.distance(r, cu.walltags).pow(0.5) * 1.5
                    + t.distance(ceillights).pow(0.2) * 2
                )
            ).maximize(weight=1)
        )
    )
    # endregion

    # region LAMPS
    lamps = lights[lamp.DeskLampFactory].related_to(furniture, cu.ontop)
    constraints["lamps"] = rooms.all(
        lambda r: (
            # allow 0-2 lamps per room, placed on any sensible object
            lamps.related_to(storage.related_to(r)).count().in_range(0, 2)
            # * lamps.related_to(sidetable.related_to(r)).count().in_range(0, 2)
            # * lamps.related_to(desks.related_to(r, cu.on), cu.ontop).count().in_range(0, 1)
            * (  # pull-string lamps look extremely unnatural when too far off the ground
                lamps.related_to(storage.related_to(r)).all(
                    lambda l: l.distance(r, cu.floortags).in_range(0.5, 1.5)
                )
            )
        )
    )

    score_terms["lamps"] = lamps.mean(
        lambda l: (
            cl.center_stable_surface_dist(l.related_to(sidetable)).minimize(weight=1)
            + l.distance(lamps).maximize(weight=1)
        )
    )
    # endregion

    # region CLOSETS
    closets = rooms[Semantics.Closet].excludes(cu.room_types)
    constraints["closets"] = closets.all(
        lambda r: (
            (storage.related_to(r).count() >= 1)
            * ceillights.related_to(r, cu.hanging).count().in_range(0, 1)
            * (
                walldec.related_to(r).count() == 0
            )  # special case exclusion - no paintings etc in closets
        )
    )
    score_terms["closets"] = closets.all(
        lambda r: (
            storage.related_to(r).count().maximize(weight=2)
            * obj.related_to(storage.related_to(r)).count().maximize(weight=2)
        )
    )

    # NOTE: closets also have special-case behavior below depending on what room they are adjacent to
    # endregion

    # region BEDROOMS
    bedrooms = rooms[Semantics.Bedroom].excludes(cu.room_types)
    beds = wallfurn[Semantics.Bed][seating.BedFactory]
    constraints["bedroom"] = bedrooms.all(
        lambda r: (
            beds.related_to(r).count().in_range(1, 2)
            * (
                sidetable.related_to(r)
                .related_to(beds.related_to(r), cu.leftright_leftright)
                .count()
                .in_range(0, 2)
            )
            * rugs.related_to(r).count().in_range(0, 1)
            * desks.related_to(r).count().in_range(0, 1)
            * storage.related_to(r).count().in_range(2, 5)
            * floor_lamps.related_to(r).count().in_range(0, 1)
            * storage.related_to(r).all(
                lambda s: (
                    obj[Semantics.OfficeShelfItem].related_to(s, cu.on).count() >= 0
                )
            )
        )
    )

    score_terms["bedroom"] = bedrooms.mean(
        lambda r: (
            beds.related_to(r).count().maximize(weight=3)
            + beds.related_to(r)
            .mean(lambda t: cl.distance(r, doors))
            .maximize(weight=0.5)
            + sidetable.related_to(r)
            .mean(lambda t: t.distance(beds.related_to(r)))
            .minimize(weight=3)
        )
    )

    # endregion

    # region KITCHENS
    kitchens = rooms[Semantics.Kitchen].excludes(cu.room_types)

    countertops = furniture[Semantics.KitchenCounter]
    wallcounter = countertops[shelves.KitchenSpaceFactory].related_to(
        rooms, cu.against_wall
    )
    island = countertops[shelves.KitchenIslandFactory]
    barchairs = furniture[seating.BarChairFactory]

    constraints["kitchen_counters"] = kitchens.all(
        lambda r: (
            wallcounter.related_to(r).count().in_range(1, 2)
            * island.related_to(r).count().in_range(0, 1)
        )
    )

    if params["has_kitchen_barstools"]:
        constraints["kitchen_barchairs"] = kitchens.all(
            lambda r: (
                barchairs.related_to(island.related_to(r), cu.front_against)
                .count()
                .in_range(0, 4)
            )
        )

    score_terms["kitchen_counters"] = kitchens.mean(
        lambda r: (
            # try to fill 40-60% of kitchen floorplan with countertops (additive with typical furniture incentive)
            (
                countertops.related_to(r).volume(dims=2)
                / r.volume(dims=2).clamp_min(1)  # avoid div by 0
            )
            .hinge(0.4, 0.6)
            .minimize(weight=10)
            +
            # cluster countertops together
            countertops.related_to(r)
            .mean(lambda c: countertops.related_to(r).mean(lambda c2: c.distance(c2)))
            .minimize(weight=3)
        )
    )

    constraints["kitchen_island_placement"] = kitchens.all(
        lambda r: wallcounter.related_to(r).all(
            lambda t: (t.distance(island.related_to(r)).in_range(0.7, 3))
        )
        * island.related_to(r).all(
            lambda t: (
                t.distance(wallcounter.related_to(r)).in_range(0.7, 3)
                * (t.distance(r, cu.walltags) > 2)
            )
        )
    )

    score_terms["kitchen_island_placement"] = kitchens.mean(
        lambda r: (
            island.mean(
                lambda t: (
                    cl.angle_alignment_cost(t, wallcounter)
                    + cl.angle_alignment_cost(t, r, cu.walltags)
                )
            ).minimize(weight=1)
            + island.distance(r, cu.walltags).hinge(3, 1e7).minimize(weight=10)
            + wallcounter.mean(
                lambda t: cl.focus_score(t, island.related_to(r)).minimize(weight=5)
            )
        )
    )

    sink_flush_on_counter = cl.StableAgainst(
        cu.bottom, {Subpart.SupportSurface}, margin=0.001
    )
    cl.StableAgainst(cu.back, cu.walltags, margin=0.1)
    kitchen_sink = obj[Semantics.Sink][table_decorations.SinkFactory].related_to(
        countertops, sink_flush_on_counter
    )
    constraints["kitchen_sink"] = kitchens.all(
        lambda r: (
            # those sinks can be on either type of counter
            kitchen_sink.related_to(wallcounter.related_to(r)).count().in_range(0, 1)
            * kitchen_sink.related_to(island.related_to(r))
            .count()
            .in_range(0, 1)  # island sinks dont need to be against wall
            * countertops.related_to(r).all(
                lambda c: (
                    kitchen_sink.related_to(c).all(
                        lambda s: s.distance(c, cu.side).in_range(0.05, 0.2)
                    )
                )
            )
        )
    )

    score_terms["kitchen_sink"] = kitchens.mean(
        lambda r: (
            countertops.mean(
                lambda c: kitchen_sink.related_to(c).mean(
                    lambda s: (
                        (s.volume(dims=2) / c.volume(dims=2))
                        .hinge(0.2, 0.4)
                        .minimize(weight=10)
                    )
                )
            )
            + island.related_to(r).mean(
                lambda isl: (  # sinks on islands must be near to edge and oriented outwards
                    kitchen_sink.related_to(isl).mean(
                        lambda s: (
                            cl.angle_alignment_cost(s, isl, cu.side).minimize(weight=10)
                            + cl.distance(s, isl, cu.side)
                            .hinge(0.05, 0.07)
                            .minimize(weight=10)
                        )
                    )
                )
            )
        )
    )

    kitchen_appliances = obj[Semantics.KitchenAppliance]
    kitchen_appliances_big = kitchen_appliances.related_to(
        kitchens, cu.on_floor
    ).related_to(kitchens, cu.against_wall)
    microwaves = kitchen_appliances[appliances.MicrowaveFactory].related_to(
        wallcounter, cu.on
    )

    constraints["kitchen_appliance"] = kitchens.all(
        lambda r: (
            kitchen_appliances_big[appliances.DishwasherFactory]
            .related_to(r)
            .count()
            .in_range(0, 1)
            * kitchen_appliances_big[appliances.BeverageFridgeFactory]
            .related_to(r)
            .count()
            .in_range(0, 1)
            * (
                kitchen_appliances_big[appliances.OvenFactory].related_to(r).count()
                == 1
            )
            * (wallfurn[shelves.KitchenCabinetFactory].related_to(r).count() >= 0)
            * (microwaves.related_to(wallcounter.related_to(r)).count().in_range(0, 1))
        )
    )

    score_terms["kitchen_appliance"] = kitchens.mean(
        lambda r: (
            kitchen_appliances.mean(
                lambda t: (
                    t.distance(wallcounter.related_to(r)).minimize(weight=1)
                    + cl.accessibility_cost(t, r, dist=1).minimize(weight=10)
                    + cl.accessibility_cost(
                        t, furniture.related_to(r), dist=1
                    ).minimize(weight=10)
                    + t.distance(island.related_to(r))
                    .hinge(0.7, 1e7)
                    .minimize(weight=10)
                )
            )
        )
    )

    def obj_on_counter(r):
        return obj.related_to(countertops.related_to(r), cu.on)

    constraints["kitchen_objects"] = kitchens.all(
        lambda r: (
            (obj_on_counter(r)[Semantics.KitchenCounterItem].count() >= 0)
            * (
                obj[Semantics.FoodPantryItem]
                .related_to(storage.related_to(r), cu.on)
                .count()
                >= 0
            )
            * island.related_to(r).all(
                lambda t: (
                    obj[Semantics.TableDisplayItem]
                    .related_to(t, cu.ontop)
                    .count()
                    .in_range(0, 4)
                )
            )
        )
    )

    score_terms["kitchen_objects"] = kitchens.mean(
        lambda r: (
            (
                obj.related_to(wallcounter, cu.on)
                .mean(lambda t: t.distance(r, cu.walltags))
                .minimize(weight=3)
            )
            + cl.center_stable_surface_dist(
                obj.related_to(island.related_to(r), cu.ontop)
            ).minimize(weight=1)
        )
    )

    # disabled for now bc tertiary
    # constraints['kitchen_appliance_objects'] = kitchens.all(lambda r: (
    #    wallfurn[appliances.DishwasherFactory].related_to(r).all(lambda r: (
    #        (obj[Semantics.Cookware].related_to(r, cu.on).count() >= 0) *
    #        (obj[Semantics.Dishware].related_to(r, cu.on).count() >= 0
    #    )) *
    #    wallfurn[appliances.OvenFactory].related_to(r).all(lambda r: (
    #        (obj[Semantics.Cookware].related_to(r, cu.on).count() >= 0)
    #    ))
    # )))

    closet_kitchen = closets.related_to(kitchens, cl.RoomNeighbour())
    constraints["closet_kitchen"] = closet_kitchen.all(
        lambda r: (
            obj[Semantics.FoodPantryItem]
            .related_to(storage.related_to(r), cu.on)
            .count()
            >= 0
        )
    )
    score_terms["closet_kitchen"] = closet_kitchen.mean(
        lambda r: (
            storage.related_to(r).count().maximize(weight=2)
            + obj[Semantics.FoodPantryItem]
            .related_to(storage.related_to(r), cu.on)
            .count()
            .maximize(weight=5)
        )
    )

    # score_terms['kitchen_table'] # todo diningtable or hightop

    # endregion

    # region LIVINGROOMS

    livingrooms = rooms[Semantics.LivingRoom].excludes(cu.room_types)
    sofas = furniture[static_assets.StaticVendingMachineFactory]
    tvstands = wallfurn[shelves.TVStandFactory]
    coffeetables = furniture[tables.CoffeeTableFactory]

    sofa_back_near_wall = cl.StableAgainst(
        cu.back, cu.walltags, margin=uniform(0.1, 0.3)
    )
    cl.StableAgainst(cu.side, cu.walltags, margin=uniform(0.1, 0.3))

    def freestanding(o, r):
        return o.related_to(r).related_to(r, -sofa_back_near_wall)

    constraints["sofa"] = livingrooms.all(
        lambda r: (
            # sofas.related_to(r).count().in_range(2, 3)
            sofas.related_to(r, sofa_back_near_wall).count().in_range(2, 4)
            # * sofas.related_to(r, sofa_side_near_wall).count().in_range(0, 1)
            * freestanding(sofas, r).all(
                lambda t: (  # frustrum infront of freestanding sofa must directly contain tvstand
                    cl.accessibility_cost(t, tvstands.related_to(r), dist=3) > 0.7
                )
            )
            * sofas.all(
                lambda t: (
                    cl.accessibility_cost(t, furniture.related_to(r), dist=2).in_range(
                        0, 0.5
                    )
                    * cl.accessibility_cost(t, r, dist=1).in_range(0, 0.5)
                )
            )
            # * ( # allow a storage object behind non-wall sofas
            #    storage.related_to(r)
            #    .related_to(freestanding(sofas, r))
            #    .count().in_range(0, 1)
            # )
        )
    )

    constraints["sofa_positioning"] = rooms.all(
        lambda r: (
            sofas.all(
                lambda s: (
                    (cl.accessibility_cost(s, rooms, dist=3) < 0.5)
                    * (
                        cl.focus_score(s, tvstands.related_to(r)) > 0.5
                    )  # must face or perpendicular to TVStand
                )
            )
        )
    )

    score_terms["sofa"] = livingrooms.mean(
        lambda r: (
            sofas.volume().maximize(weight=10)
            + sofas.related_to(r).mean(
                lambda t: (
                    t.distance(sofas.related_to(r)).hinge(0, 1).minimize(weight=1)
                    + t.distance(tvstands.related_to(r)).hinge(2, 3).minimize(weight=5)
                    + cl.focus_score(t, tvstands.related_to(r)).maximize(weight=5)
                    + cl.angle_alignment_cost(
                        t, tvstands.related_to(r), cu.front
                    ).minimize(weight=1)
                    + cl.focus_score(t, coffeetables.related_to(r)).maximize(weight=2)
                    + cl.accessibility_cost(t, r, dist=3).minimize(weight=3)
                )
            )
            + freestanding(sofas, r).mean(
                lambda t: (
                    cl.angle_alignment_cost(t, tvstands.related_to(r)).minimize(
                        weight=5
                    )
                    + cl.angle_alignment_cost(t, r, cu.walltags).minimize(weight=3)
                    + cl.center_stable_surface_dist(t).minimize(weight=0.5)
                )
            )
        )
    )

    tvs = obj[appliances.TVFactory].related_to(tvstands, cu.ontop)

    if params["has_tv"]:
        constraints["tv"] = livingrooms.all(
            lambda r: (
                tvstands.related_to(r).all(
                    lambda t: (
                        (tvs.related_to(t).count() == 1)
                        * tvs.related_to(t).all(
                            lambda tv: cl.accessibility_cost(tv, r, dist=1).in_range(
                                0, 0.1
                            )
                        )
                    )
                )
            )
        )

    score_terms["tvstand"] = rooms.all(
        lambda r: (
            tvstands.mean(
                lambda stand: (
                    tvs.related_to(stand).volume().maximize(weight=1)
                    + stand.distance(window).maximize(
                        weight=1
                    )  # penalize being very close to window. avoids tv blocking window.
                    + cl.accessibility_cost(stand, furniture).minimize(weight=3)
                    + cl.center_stable_surface_dist(stand).minimize(
                        weight=5
                    )  # center tvstand against wall (also tries to do vertical & floor but those are constrained)
                    + cl.center_stable_surface_dist(tvs.related_to(stand)).minimize(
                        weight=1
                    )
                )
            )
        )
    )

    constraints["livingroom"] = livingrooms.all(
        lambda r: (
            storage.related_to(r).count().in_range(1, 5)
            * tvstands.related_to(r).count().equals(1)
            * (  # allow sidetables next to any sofa
                sidetable.related_to(r)
                .related_to(sofas.related_to(r), cu.side_by_side)
                .count()
                .in_range(0, 2)
            )
            * desks.related_to(r).count().in_range(0, 1)
            * coffeetables.related_to(r).count().in_range(0, 1)
            * coffeetables.related_to(r).all(
                lambda t: (
                    obj[Semantics.OfficeShelfItem]
                    .related_to(t, cu.on)
                    .count()
                    .in_range(0, 3)
                )
            )
            * (
                rugs.related_to(r)
                # .related_to(furniture.related_to(r), cu.side_by_side)
                .count()
                .in_range(0, 2)
            )
        )
    )

    score_terms["livingroom"] = livingrooms.mean(
        lambda r: (
            coffeetables.related_to(r).mean(
                lambda t: (
                    # ideal coffeetable-to-tv distance according to google
                    t.distance(sofas.related_to(r)).hinge(0.45, 0.6).minimize(weight=5)
                    + cl.angle_alignment_cost(
                        t, sofas.related_to(r), cu.front
                    ).minimize(weight=5)
                    + cl.focus_score(sofas.related_to(r), t).maximize(weight=5)
                )
            )
        )
    )

    constraints["livingroom_objects"] = livingrooms.all(
        lambda r: (
            storage.all(
                lambda t: (
                    obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 0
                )
            )
            * coffeetables.all(
                lambda t: (
                    obj[Semantics.TableDisplayItem]
                    .related_to(t, cu.ontop)
                    .count()
                    .in_range(0, 1)
                    * (obj[Semantics.OfficeShelfItem].related_to(t, cu.on).count() >= 0)
                )
            )
        )
    )

    # endregion

    # region DININGROOMS

    diningtables = furniture[Semantics.Table][static_assets.StaticTableFactory]
    diningchairs = furniture[Semantics.Chair][seating.ChairFactory]
    constraints["dining_chairs"] = rooms.all(
        lambda r: (
            diningtables.related_to(r).all(
                lambda t: (
                    diningchairs.related_to(r)
                    .related_to(t, cu.front_against)
                    .count()
                    .in_range(3, 6)
                )
            )
        )
    )

    score_terms["dining_chairs"] = rooms.all(
        lambda r: (
            diningchairs.related_to(r).count().maximize(weight=5)
            + diningchairs.related_to(r)
            .mean(lambda t: t.distance(diningchairs.related_to(r)))
            .maximize(weight=3)
            # cl.reflectional_asymmetry(diningchairs.related_to(r), diningtables.related_to(r)).minimize(weight=1)
            # cl.rotational_asymmetry(diningchairs.related_to(r)).minimize(weight=1)
        )
    )

    constraints["dining_table_objects"] = rooms.all(
        lambda r: (
            diningtables.related_to(r).all(
                lambda t: (
                    obj[Semantics.TableDisplayItem]
                    .related_to(t, cu.ontop)
                    .count()
                    .in_range(0, 2)
                    * (obj[Semantics.Utensils].related_to(t, cu.ontop).count() >= 0)
                    * (
                        obj[Semantics.Dishware]
                        .related_to(t, cu.ontop)
                        .count()
                        .in_range(0, 2)
                    )
                )
            )
        )
    )

    score_terms["dining_table_objects"] = rooms.mean(
        lambda r: (
            cl.center_stable_surface_dist(
                obj[Semantics.TableDisplayItem].related_to(
                    diningtables.related_to(r), cu.ontop
                )
            ).minimize(weight=1)
        )
    )

    diningrooms = rooms[Semantics.DiningRoom].excludes(cu.room_types)
    constraints["diningroom"] = diningrooms.all(
        lambda r: (
            (diningtables.related_to(r).count() == 1)
            * storage.related_to(r).all(
                lambda t: (
                    (obj[Semantics.Dishware].related_to(t, cu.on).count() >= 0)
                    * (
                        obj[Semantics.OfficeShelfItem]
                        .related_to(t, cu.on)
                        .count()
                        .in_range(0, 5)
                    )
                )
            )
        )
    )
    score_terms["diningroom"] = diningrooms.mean(
        lambda r: (
            diningtables.related_to(r).distance(r, cu.walltags).maximize(weight=10)
            + cl.angle_alignment_cost(
                diningtables.related_to(r), r, cu.walltags
            ).minimize(weight=10)
            + cl.center_stable_surface_dist(diningtables.related_to(r)).minimize(
                weight=1
            )
        )
    )
    # endregion

    # region BATHROOMS
    bathrooms = rooms[Semantics.Bathroom].excludes(cu.room_types)
    toilet = wallfurn[bathroom.ToiletFactory]
    bathtub = wallfurn[bathroom.BathtubFactory]
    sink = wallfurn[bathroom.StandingSinkFactory]
    hardware = obj[bathroom.HardwareFactory].related_to(bathrooms, cu.against_wall)
    constraints["bathroom"] = bathrooms.all(
        lambda r: (
            mirror.related_to(r).related_to(r, cu.flush_wall).count().equals(1)
            * sink.related_to(r).count().equals(1)
            * toilet.related_to(r).count().equals(1)
            * storage.related_to(r).all(
                lambda t: (
                    obj[Semantics.BathroomItem].related_to(t, cu.on).count() >= 0
                )
            )
        )
    )

    score_terms["toilet"] = rooms.all(
        lambda r: (
            toilet.distance(doors).maximize(weight=1)
            + toilet.distance(furniture).maximize(weight=1)
            + toilet.distance(sink).maximize(weight=1)
            + cl.accessibility_cost(toilet, furniture, dist=2).minimize(weight=10)
        )
    )

    constraints["bathtub"] = bathrooms.all(
        lambda r: (
            bathtub.related_to(r).count().in_range(0, 1)
            * hardware.related_to(r).count().in_range(1, 4)
        )
    )
    score_terms["bathtub"] = bathrooms.all(
        lambda r: (
            bathtub.mean(lambda t: t.distance(hardware)).minimize(weight=0.2)
            + sink.mean(lambda t: t.distance(hardware)).minimize(weight=0.2)
            + hardware.mean(
                lambda t: (
                    t.distance(rooms, cu.floortags).hinge(0.5, 1).minimize(weight=15)
                )
            )
        )
    )

    score_terms["bathroom"] = mirror.related_to(bathrooms).distance(sink).minimize(
        weight=0.2
    ) + cl.accessibility_cost(mirror, furniture, cu.down_dir).maximize(weight=3)
    # endregion

    # region MISC OBJECTS

    if params["has_aquarium_tank"]:

        def aqtank(r):
            return obj[decor.AquariumTankFactory].related_to(
                storage.related_to(r), cu.ontop
            )

        constraints["aquarium_tank"] = aqtank(rooms).count().in_range(0, 1)
        score_terms["aquarium_tank"] = rooms.all(
            lambda r: (
                aqtank(r).distance(r, cu.walltags).hinge(0.05, 0.1).minimize(weight=1)
            )
        )

    if params["has_birthday_balloons"]:
        balloons = obj[wall_decorations.BalloonFactory].related_to(
            rooms, cu.against_wall
        )
        constraints["birthday_balloons"] = (
            balloons.related_to(rooms, cu.against_wall).count().in_range(0, 3)
        )
        score_terms["birthday_balloons"] = rooms.all(
            lambda r: (
                balloons.mean(
                    lambda b: b.distance(r, cu.floortags)
                    .hinge(1.6, 2.5)
                    .minimize(weight=1)
                )
            )
        )

    if params["has_cocktail_tables"]:
        cocktail_table = (
            furniture[tables.TableCocktailFactory]
            .related_to(rooms, cu.on_floor)
            .related_to(rooms, cu.against_wall)
        )

        constraints["cocktail_tables"] = diningrooms.all(
            lambda r: (
                cocktail_table.related_to(r).count().in_range(0, 3)
                * (
                    barchairs.related_to(cocktail_table.related_to(r), cu.front_against)
                    .count()
                    .in_range(0, 4)
                )
                * (
                    obj[tableware.WineglassFactory]
                    .related_to(cocktail_table.related_to(r), cu.ontop)
                    .count()
                    .in_range(0, 4)
                )
            )
        )
        score_terms["cocktail_tables"] = diningrooms.mean(
            lambda r: (
                cocktail_table.related_to(r).mean(
                    lambda t: (
                        t.distance(r, cu.walltags).hinge(0.5, 1).minimize(weight=1)
                        + t.distance(cocktail_table.related_to(r))
                        .hinge(1, 2)
                        .minimize(weight=1)
                        + barchairs.related_to(t)
                        .mean(lambda c: c.distance(barchairs.related_to(t)))
                        .maximize(weight=1)
                    )
                )
            )
        )

    # endregion

    return cl.Problem(
        constraints=constraints,
        score_terms=score_terms,
    )


all_constraint_funcs = [home_constraints]
