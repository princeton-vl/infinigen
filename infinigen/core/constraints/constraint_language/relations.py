# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Relation(ABC):
    @abstractmethod
    def implies(self, other) -> bool:
        """
        self must imply all parts of other, both positive and negative.
        """
        pass

    @abstractmethod
    def satisfies(self, other: Relation) -> bool:
        """
        self must imply all positive parts of other, and not contradict any negative parts
        """
        pass

    @abstractmethod
    def intersects(self, other, strict=False) -> bool:
        pass

    @abstractmethod
    def intersection(self, other: Relation) -> Relation:
        pass

    @abstractmethod
    def difference(self, other: Relation) -> Relation:
        pass

    def __neg__(self) -> Relation:
        return NegatedRelation(self)


@dataclass(frozen=True)
class AnyRelation(Relation):
    def implies(self, other) -> bool:
        return other.__class__ is AnyRelation

    def satisfies(self, other: cl.Relation) -> bool:
        return other.__class__ is AnyRelation

    def intersects(self, _other: Relation, strict=False) -> bool:
        return True

    def intersection(self, other: Relation) -> Relation:
        return deepcopy(other)

    def difference(self, other: Relation):
        return -other


@dataclass(frozen=True)
class NegatedRelation(Relation):
    rel: Relation

    def __repr__(self):
        return f"-{self.rel}"

    def __str__(self):
        return f"{self.__class__.__name__}({self.rel})"

    def __neg__(self) -> Relation:
        return self.rel

    def implies(self, other: Relation) -> bool:
        match other:
            case AnyRelation():
                return False
            case NegatedRelation(rel):
                return self.rel.implies(rel)
            case _:
                return not self.rel.implies(other) and not self.intersects(
                    other, strict=True
                )

    def satisfies(self, other: cl.Relation) -> bool:
        match other:
            case AnyRelation():
                return False
            case NegatedRelation(rel):
                return self.rel.satisfies(rel)
            case _:
                return not self.rel.satisfies(other) and not self.intersects(
                    other, strict=True
                )

    def intersects(self, other: Relation, strict=False) -> bool:
        match other:
            case NegatedRelation(rel):
                if isinstance(self.rel, AnyRelation) or isinstance(rel, AnyRelation):
                    return False
                # TODO hacky, allows false positives for GeometryRelation. very unlikely to come up however
                return True
            case _:
                # implementationn depends on other's type, let them handle it
                return other.intersects(self, strict=strict)

    def intersection(self, other: Relation) -> Relation:
        return self.rel.difference(other)

    def difference(self, other: Relation) -> Relation:
        return self.rel.intersection(other)


class ConnectorType(Enum):
    Door = "door"
    Open = "open"
    Wall = "wall"


@dataclass(frozen=True)
class RoomNeighbour(Relation):
    connector_types: frozenset[ConnectorType] = field(default_factory=frozenset)

    def __post_init__(self):
        if self.connector_types is not None:
            object.__setattr__(self, "connector_types", frozenset(self.connector_types))

    def implies(self, other: Relation) -> bool:
        if isinstance(other, AnyRelation):
            return True

        return isinstance(other, RoomNeighbour) and self.connector_types.issuperset(
            other.connector_types
        )

    def satisfies(self, other: Relation) -> bool:
        return self.implies(other)

    def intersects(self, other: Relation, strict=False) -> bool:
        if isinstance(other, AnyRelation):
            return True

        return isinstance(other, RoomNeighbour) and not self.connector_types.isdisjoint(
            other.connector_types
        )

    def intersection(self, other: Relation) -> Relation:
        if isinstance(other, AnyRelation):
            return deepcopy(self)

        return self.__class__(
            connector_types=self.connector_types.intersection(other.connector_types)
        )

    def difference(self, other: Relation) -> Relation:
        if isinstance(other, AnyRelation):
            return -AnyRelation()

        return self.__class__(
            connector_types=self.connector_types.difference(other.connector_types)
        )


def no_frozenset_repr(self: GeometryRelation):
    def is_neg(x):
        return isinstance(x, t.Negated)

    def setrepr(s):
        return f"{{{', '.join(repr(x) for x in sorted(list(s), key=is_neg))}}}"

    return f"{self.__class__.__name__}({setrepr(self.child_tags)}, {setrepr(self.parent_tags)})"


@dataclass(frozen=True)
class GeometryRelation(Relation):
    child_tags: frozenset[t.Subpart] = field(default_factory=frozenset)
    parent_tags: frozenset[t.Subpart] = field(default_factory=frozenset)

    __repr__ = no_frozenset_repr

    def __post_init__(self):
        # allow the user to init with sets that subsequently get frozen
        # use object.__setattr__ to bypass dataclass's frozen since it is guaranteed safe here
        object.__setattr__(self, "child_tags", frozenset(self.child_tags))
        object.__setattr__(self, "parent_tags", frozenset(self.parent_tags))

    def _extra_fields(self) -> list[str]:
        """Return any fields added by subclasses. Useful for implementing implies/intersects
        which must check these fields regardless of inheritance. TODO, Hacky.
        """
        return [
            f.name for f in fields(self) if f.name not in ["child_tags", "parent_tags"]
        ]

    def _compatibility_checks(
        self, other: GeometryRelation, strict_on_fields=False
    ) -> bool:
        if not issubclass(other.__class__, self.__class__):
            return False

        if strict_on_fields:
            for k in self._extra_fields():
                if not getattr(self, k) == getattr(other, k):
                    # logger.warning(f'{self._compatibility_checks} ignoring mismatch {k=} for {other=}')
                    return False

        return True

    def implies(self, other: Relation) -> bool:
        match other:
            case AnyRelation():
                return True
            case NegatedRelation(AnyRelation()):
                return False
            case GeometryRelation(ochild, oparent):
                if not self._compatibility_checks(other):
                    # logger.debug(f"{self.implies} failed compatibility for %s", other)
                    return False
                if not t.implies(self.child_tags, ochild):
                    # logger.debug(f"{self.implies} failed child tags for %s", other)
                    return False
                if not t.implies(self.parent_tags, oparent):
                    # logger.debug(f"{self.implies} failed parent tags for %s", other)
                    return False
                return True
            case NegatedRelation(GeometryRelation(ochild, oparent)):
                if not self._compatibility_checks(other.rel):
                    logger.debug(f"{self.implies} failed compatibility for %s", other)
                    return False
                if t.implies(self.child_tags, {-t for t in ochild}) and t.implies(
                    self.parent_tags, {-t for t in oparent}
                ):
                    return True
                return False
            case _:
                raise ValueError(f"{self.implies} encountered unhandled {other=}")

    def satisfies(self, other: Relation) -> bool:
        match other:
            case AnyRelation():
                return True
            case NegatedRelation(AnyRelation()):
                return False
            case GeometryRelation(ochild, oparent):
                if not self._compatibility_checks(other):
                    logger.debug(f"{self.satisfies} failed compatibility for %s", other)
                    return False
                if not t.satisfies(self.child_tags, ochild):
                    logger.debug(f"{self.satisfies} failed child tags for %s", other)
                    return False
                if not t.satisfies(self.parent_tags, oparent):
                    logger.debug(f"{self.satisfies} failed parent tags for %s", other)
                    return False
                return True
            case NegatedRelation(GeometryRelation(ochild, oparent)):
                if not self._compatibility_checks(other.rel):
                    logger.debug(f"{self.implies} failed compatibility for %s", other)
                    return False
                if t.satisfies(self.child_tags, {-t for t in ochild}) and t.satisfies(
                    self.parent_tags, {-t for t in oparent}
                ):
                    return True
                return False
            case _:
                raise ValueError(f"{self.satisfies} encountered unhandled {other=}")

    def intersects(self, other: Relation, strict=False) -> bool:
        def tags_compatible(a, b):
            if strict:
                return t.implies(a, b) or t.implies(b, a)
            else:
                return not t.contradiction(a.union(b))

        logger.debug(f"{self.intersects} other=%s", other)

        match other:
            case AnyRelation():
                return True
            case NegatedRelation(AnyRelation()):
                return False
            case GeometryRelation(ochild, oparent):
                if not self._compatibility_checks(other):
                    logger.debug(
                        "%s failed compatiblity_checks for self=%s, other=%s",
                        self.intersects.__name__,
                        self.child_tags,
                        other,
                    )
                    return False
                if not tags_compatible(self.child_tags, ochild):
                    logger.debug(
                        "%s failed child tags for self=%s, other=%s",
                        self.intersects.__name__,
                        self.child_tags,
                        other,
                    )
                    return False
                if not tags_compatible(self.parent_tags, oparent):
                    logger.debug(
                        "%s failed parent tags for self=%s, other=%s",
                        self.intersects.__name__,
                        self.child_tags,
                        other,
                    )
                    return False
                return True
            case NegatedRelation(GeometryRelation()):
                # is self compatible with NOT other.rel?
                # true unless other.rel->self
                return not other.rel.implies(self)
            case _:
                logger.warning(
                    f"{self.intersects} encountered unhandled %s, returning False",
                    other,
                )
                return False

    def intersection(self: Relation, other: Relation) -> Relation:
        """TODO: There are potentially many intersections of relations with negations."""

        match other:
            case AnyRelation():
                return deepcopy(self)
            case NegatedRelation(rel):
                return self.difference(rel)
            case GeometryRelation(ochild, oparent):
                if not self._compatibility_checks(other):
                    logger.warning(
                        f"{self.intersection} failed compatibility for {other=}"
                    )
                    return -AnyRelation()
                return self.__class__(
                    child_tags=self.child_tags.union(ochild),
                    parent_tags=self.parent_tags.union(oparent),
                    **{k: getattr(self, k) for k in self._extra_fields()},
                )
            case _:
                logger.warning(
                    f"Encountered unhandled {other=} for {self.intersection}"
                )
                return -AnyRelation()

    def difference(self: Relation, other: Relation) -> Relation:
        match other:
            case AnyRelation():
                return -AnyRelation()
            case NegatedRelation(rel):
                return self.intersection(rel)
            case GeometryRelation(ochild, oparent):
                if not self.intersects(other):
                    return deepcopy(self)
                if t.implies(self.child_tags, ochild) and t.implies(
                    self.parent_tags, oparent
                ):
                    return -AnyRelation()

                return self.__class__(
                    child_tags=t.difference(self.child_tags, ochild),
                    parent_tags=t.difference(self.parent_tags, oparent),
                    **{k: getattr(self, k) for k in self._extra_fields()},
                )
            case _:
                logger.warning(
                    f"Encountered unhandled {other=} for {self.intersection}"
                )
                return -AnyRelation()


@dataclass(frozen=True)
class Touching(GeometryRelation):
    __repr__ = no_frozenset_repr


@dataclass(frozen=True)
class SupportedBy(Touching):
    __repr__ = no_frozenset_repr


@dataclass(frozen=True)
class CoPlanar(GeometryRelation):
    margin: float = 0

    # rev_normal: if True, align the normals so they face the SAME direction, rather than two planes facing eachother.
    # typical use is for sink embedded in countertop
    rev_normal: bool = False

    __repr__ = no_frozenset_repr


@dataclass(frozen=True)
class StableAgainst(GeometryRelation):
    margin: float = 0

    # check_ if False, only check x/z stability, z is allowed to overhand.
    # typical use is chair-against-table relation
    check_z: bool = True

    rev_normal: bool = False

    __repr__ = no_frozenset_repr


class IdentityCompareRelation(Relation):
    def implies(self, other: Relation) -> bool:
        return isinstance(other, AnyRelation) or isinstance(other, self.__class__)

    def satisfies(self, other: Relation) -> bool:
        return self.implies(other)

    def intersects(self, other: Relation, strict=False) -> bool:
        return isinstance(other, AnyRelation) or isinstance(other, self.__class__)

    def intersection(self, other: Relation) -> Relation:
        return deepcopy(self)

    def difference(self, other: Relation) -> Relation:
        return -AnyRelation()


@dataclass(frozen=True)
class CutFrom(IdentityCompareRelation):
    pass


@dataclass(frozen=True)
class SharedEdge(IdentityCompareRelation):
    pass


@dataclass(frozen=True)
class Traverse(IdentityCompareRelation):
    pass
