# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum, EnumMeta


class ABCEnumMeta(EnumMeta, ABCMeta):
    pass


class Tag:
    def __neg__(self) -> Negated:
        return Negated(self)


class StringTag(Tag):
    def __init__(self, desc: str):
        self.desc = desc


class EnumTag(Tag, Enum, metaclass=ABCEnumMeta):
    pass


class Semantics(EnumTag):
    # Mesh types
    Room = "room"
    Object = "object"
    Cutter = "cutter"

    # Room types
    Kitchen = "kitchen"
    Bedroom = "bedroom"
    LivingRoom = "living-room"
    Closet = "closet"
    Hallway = "hallway"
    Bathroom = "bathroom"
    Garage = "garage"
    Balcony = "balcony"
    DiningRoom = "dining-room"
    Utility = "utility"
    StaircaseRoom = "staircase-room"
    Warehouse = "warehouse"
    Office = "office"
    MeetingRoom = "meeting-room"
    OpenOffice = "open-office"
    BreakRoom = "break-room"
    Restroom = "restroom"
    FactoryOffice = "factory-office"

    Root = "root"
    New = "new"
    RoomNode = "room-node"
    GroundFloor = "ground"
    SecondFloor = "second-floor"
    ThirdFloor = "third-floor"
    Exterior = "exterior"
    Staircase = "staircase"
    Visited = "visited"
    RoomContour = "room-contour"

    # Object types
    Furniture = "furniture"
    FloorMat = "FloorMat"
    WallDecoration = "wall-decoration"
    HandheldItem = "handheld-item"

    # Furniture functions
    Storage = "storage"
    Seating = "seating"
    LoungeSeating = "lounge-seating"
    Table = "table"
    Bathing = "bathing"
    SideTable = "side-table"
    Watchable = "watchable"
    Desk = "desk"
    Bed = "bed"
    Sink = "sink"
    CeilingLight = "ceiling-light"
    Lighting = "lighting"
    KitchenCounter = "kitchen-counter"
    KitchenAppliance = "kitchen-appliance"

    # Small Object Functions
    TableDisplayItem = "table-display-item"
    OfficeShelfItem = "office-shelf-item"
    KitchenCounterItem = "kitchen-counter-item"
    FoodPantryItem = "food-pantry"
    BathroomItem = "bathroom-item"
    ShelfTrinket = "shelf-trinket"
    Dishware = "dishware"
    Cookware = "cookware"
    Utensils = "utensils"
    ClothDrapeItem = "cloth-drape"

    # Object Access Type
    AccessTop = "access-top"
    AccessFront = "access-front"
    AccessAnySide = "access-any-side"
    AccessAllSides = "access-all-sides"

    # Object Access Method
    AccessStandingNear = "access-stand-near"
    AccessSit = "access-stand-near"
    AccessOpenDoor = "access-open-door"
    AccessHand = "access-with-hand"

    # Special Case Objects
    Chair = "chair"
    Window = "window"
    Open = "open"
    Entrance = "entrance"
    Door = "door"

    # Solver feature flags
    # Per-Asset Behavior Config
    RealPlaceholder = "real-placeholder"
    OversizePlaceholder = "oversize-placeholder"
    AssetAsPlaceholder = "asset-as-placeholder"
    AssetPlaceholderForChildren = "asset-placeholder-for-children"
    PlaceholderBBox = "placeholder-bbox"
    SingleGenerator = "single-generator"
    NoRotation = "no-rotation"
    NoCollision = "no-collision"
    NoChildren = "no-children"

    def __str__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    def __lt__(self, other):
        return self.name < other.name

    @classmethod
    @property
    def floors(cls):
        return [Semantics.GroundFloor, Semantics.SecondFloor, Semantics.ThirdFloor]


class Subpart(EnumTag):
    SupportSurface = "support"
    Interior = "interior"
    Visible = "visible"
    Bottom = "bottom"
    Top = "top"
    Side = "side"
    Back = "back"
    Front = "front"
    Ceiling = "ceiling"
    Wall = "wall"

    StaircaseWall = "staircase-wall"  # TODO Lingjie Remove

    def __str__(self):
        return f"{self.__class__.__name__}({self.value})"

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


@dataclass(frozen=True)
class FromGenerator(Tag):
    generator: type

    def __repr__(self):
        return f"{self.__class__.__name__}({self.generator.__name__})"


@dataclass(frozen=True)
class Negated(Tag):
    tag: Tag

    def __str__(self):
        return "-" + str(self.tag)

    def __repr__(self):
        return f"-{repr(self.tag)}"

    def __neg__(self):
        return self.tag

    def __post_init__(self):
        assert not isinstance(self.tag, Negated), "dont construct double negative tags"


@dataclass(frozen=True)
class Variable(Tag):
    name: str

    def __post_init__(self):
        assert isinstance(self.name, str)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class SpecificObject(Tag):
    name: str


def decompose_tags(tags: set[Tag]):
    positive, negative = set(), set()

    for t in tags:
        match t:
            case Negated(tag):
                negative.add(tag)
            case _:
                positive.add(t)

    return positive, negative


def contradiction(tags: set[Tag]):
    pos, neg = decompose_tags(tags)

    if pos.intersection(neg):
        return True

    if len([t for t in pos if isinstance(t, FromGenerator)]) > 1:
        return True
    if len([t for t in tags if isinstance(t, SpecificObject | Variable)]) > 1:
        return True

    return False


def implies(t1: set[Tag], t2: set[Tag]):
    p1, n1 = decompose_tags(t1)
    p2, n2 = decompose_tags(t2)

    return not contradiction(t1) and p1.issuperset(p2) and n1.issuperset(n2)


def satisfies(t1: set[Tag], t2: set[Tag]):
    p1, n1 = decompose_tags(t1)
    p2, n2 = decompose_tags(t2)

    return p1.issuperset(p2) and not n1.intersection(p2) and not n2.intersection(p1)


def difference(t1: set[Tag], t2: set[Tag]):
    """Return a set of predicates representing the difference

    If the difference is empty, will return a contradictory set of predicates.
    """

    p1, n1 = decompose_tags(t1)
    p2, n2 = decompose_tags(t2)

    pos = p1.union(n2 - n1)
    neg = n1.union(p2 - p1)

    return pos.union(Negated(n) for n in neg)


def to_tag(s: str | Tag | type, fac_context=None) -> Tag:
    if isinstance(s, Tag):
        return s

    if type(s) is type:
        if not fac_context:
            raise ValueError(f"to_tag got {s=} but {fac_context=}")
        if s not in fac_context:
            raise ValueError(f"Got {s=} of type=type but it was not in fac_context")
        return FromGenerator(s)

    assert isinstance(s, str), s

    if s.startswith("-"):
        return Negated(to_tag(s[1:]))

    if fac_context is not None:
        fac = next((f for f in fac_context.keys() if f.__name__ == s), None)
        if fac:
            return FromGenerator(fac)

    s = s.strip("\"'")

    try:
        return Semantics[s]
    except KeyError:
        pass

    try:
        return Subpart[s]
    except KeyError:
        pass

    raise ValueError(
        f"to_tag got {s=} but could not resolve it. Please see tags.Semantics and tags.Subpart for available tag strings"
    )


def to_string(tag: Tag | str):
    if isinstance(tag, str):
        return tag

    match tag:
        case Semantics() | Subpart():
            return tag.value
        case StringTag():
            return tag.desc
        case FromGenerator():
            return tag.__name__
        case Negated():
            raise ValueError(f"Negated tag {tag=} is not allowed here")
        case _:
            raise ValueError(f"to_string unhandled {tag=}")


def to_tag_set(x, fac_context=None):
    match x:
        case None:
            return set()
        case set() | list() | tuple() | frozenset():
            return {to_tag(xi, fac_context=fac_context) for xi in x}
        case x:
            return {to_tag(x, fac_context=fac_context)}
