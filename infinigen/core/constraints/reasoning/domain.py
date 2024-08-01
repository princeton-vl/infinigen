# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field

from infinigen.core import tags as t
from infinigen.core.constraints import constraint_language as cl
from infinigen.core.util.logging import lazydebug

logger = logging.getLogger(__name__)


def reldom_implies(a: tuple[cl.Relation, Domain], b: tuple[cl.Relation, Domain]):
    """If relation a is satisfied, is relation guaranteed to be satisfied?"""

    assert isinstance(a[1], Domain)
    assert isinstance(b[1], Domain)

    return a[0].implies(b[0]) and a[1].implies(b[1])


def reldom_compatible(
    a: tuple[cl.Relation, Domain],
    b: tuple[cl.Relation, Domain],
):
    """If relation a is satisfied, can relation b be satisfied?"""

    assert isinstance(a[1], Domain)
    assert isinstance(b[1], Domain)

    a_neg = isinstance(a[0], cl.NegatedRelation)
    b_neg = isinstance(b[0], cl.NegatedRelation)
    match (a_neg, b_neg):
        case True, False:
            if b[0].implies(a[0].rel) and b[1].intersects(a[1]):
                lazydebug(
                    logger,
                    lambda: f"reldom_compatible found contradicting negated {a[0]} {b[0]}",
                )
                return False
        case False, True:
            if a[0].implies(b[0].rel) and a[1].intersects(b[1]):
                lazydebug(
                    logger,
                    lambda: f"reldom_compatible found contradicting negated {a[0]} {b[0]}",
                )
                return False

    return True


def reldom_satisfies(
    a: tuple[cl.Relation, Domain],
    b: tuple[cl.Relation, Domain],
):
    return a[0].intersects(b[0], strict=True) and a[1].satisfies(b[1])


def reldom_intersects(
    a: tuple[cl.Relation, Domain], b: tuple[cl.Relation, Domain], **kwargs
):
    return a[0].intersects(b[0]) and a[1].intersects(b[1], **kwargs)


def reldom_intersection(
    a: tuple[cl.Relation, Domain],
    b: tuple[cl.Relation, Domain],
):
    return (a[0].intersection(b[0]), a[1].intersection(b[1]))


def domain_finalized(dom: Domain, check_anyrel=False, check_variable=True):
    if check_variable and any(isinstance(x, t.Variable) for x in dom.tags):
        return False

    for rel, cdom in dom.relations:
        if check_anyrel and isinstance(rel, cl.AnyRelation):
            return False
        if not domain_finalized(cdom):
            return False

    return True


@dataclass
class Domain:
    """
    Describes a class of object in the scene

    Objects are in the domain if:
    - Some part of the object is tagged with each of 'tags'
    - It is related to an object matching each Domain in relations

    WARNING: Recurive datastructure, here be dragons

    Note: Default-constructed Domain contains Everything
    """

    tags: set[t.Semantics] = field(default_factory=set)
    relations: list[tuple[cl.Relation, Domain]] = field(default_factory=list)

    def repr(self, abbrv=False, onelevel=False, oneline=False):
        def is_neg(x):
            return isinstance(x, t.Negated)

        def setrepr(s):
            inner = ", ".join(
                repr(x)
                for x in sorted(list(s), key=is_neg)
                if not (abbrv and isinstance(x, t.Negated))
            )
            return "{" + inner + "}"

        next_abbrv = abbrv or onelevel

        def repr_reldom(r, d):
            if abbrv:
                rel = (
                    f"-{r.rel.__class__.__name__}"
                    if isinstance(r, cl.NegatedRelation)
                    else f"{r.__class__.__name__}"
                )
                return f"({rel}(...), Domain({setrepr(d.tags)}, [...]))"
            else:
                return f"({repr(r)}, {d.repr(abbrv=next_abbrv)})"

        relations = [
            repr_reldom(r, d)
            for r, d in sorted(
                self.relations, key=lambda x: isinstance(x[0], cl.NegatedRelation)
            )
        ]

        if not oneline and sum(len(x) for x in relations) > 20:
            relations = [r.replace("\n", "\n\t") for r in relations]
            relations = "\n\t" + ",\n\t".join(relations) + "\n"
        else:
            relations = ", ".join(relations)
        return f"{self.__class__.__name__}({setrepr(self.tags)}, [{relations}])"

    __repr__ = repr

    def __post_init__(self):
        assert isinstance(self.tags, set)
        assert isinstance(self.relations, list)

    def implies(self, other: Domain):
        return t.implies(self.tags, other.tags) and all(
            any(reldom_implies(rel, orel) for rel in self.relations)
            for orel in other.relations
        )

    def add_relation(
        self, new_rel: cl.Relation, new_dom: Domain, optimize_check_implies=True
    ):
        """
        new_rel, new_dom: the relation and domain to be added
        optimize_check_implies: bool
            If enabled, dont add relations (aka predicates) that are already
            provably true based on existing predicates. This should not be necessary for correctness:
            object addition should check if relation constraints are satisfied/implied before adding more.
            But it may (?) help speed, and the unit tests assume it is enabled for the most part
        """

        assert new_dom is not self

        lazydebug(
            logger,
            lambda: f"add_relation {new_rel} {new_dom} to existing {len(self.relations)}",
        )

        if not optimize_check_implies:
            self.relations.append((new_rel, new_dom))
            return

        covered = False

        for i, (er, ed) in enumerate(self.relations):
            if isinstance(new_rel, cl.NegatedRelation):
                continue
            elif isinstance(er, cl.NegatedRelation):
                continue
            elif reldom_implies((er, ed), (new_rel, new_dom)):
                covered = True
            elif reldom_satisfies((er, ed), (new_rel, new_dom)) or reldom_satisfies(
                (new_rel, new_dom), (er, ed)
            ):
                lazydebug(
                    logger,
                    lambda: f"Tightening existing relation {(er, ed)} with {(new_rel, new_dom)}",
                )
                self.relations[i] = reldom_intersection((new_rel, new_dom), (er, ed))
                covered = True
            elif new_dom.intersects(ed, require_satisfies_right=True):
                lazydebug(logger, lambda: f"Tightening domain {ed} with {new_dom}")
                self.relations[i] = (er, ed.intersection(new_dom))
            else:
                lazydebug(
                    logger,
                    lambda: f"{(er, ed)} is not relevant for {(new_rel, new_dom)}",
                )

        if not covered:
            lazydebug(
                logger,
                lambda: f"optimize_check_implies found nothing, adding relation {new_rel} {new_dom}",
            )
            self.relations.append((new_rel, new_dom))

        if self.is_recursive():
            raise ValueError(
                f"Encountered recursive domain after add_relation {new_rel=} {new_dom=} onto {self.tags=} {len(self.relations)=}"
            )

    def with_relation(self, rel: cl.Relation, dom: Domain):
        new = copy.deepcopy(self)
        new.add_relation(rel, dom)
        return new

    def with_tags(self, tags: set[t.Semantics]):
        if not isinstance(tags, set):
            tags = {tags}
        new = copy.deepcopy(self)
        new.tags.update(tags)
        return new

    def satisfies(self, other: Domain):
        """

        Assumes that 'self' is fully specified: any predicates that arent listed are false.

        Different from 'implies' in that if `other` contains negative predicates, `self` need not imply these,
        it just needs to not contradict them.

        Different from 'intersects' in that
        """

        lazydebug(logger, lambda: f"{Domain.satisfies.__name__} for {self} {other}")

        if not t.satisfies(self.tags, other.tags):
            lazydebug(
                logger, lambda: f"failed tag implication {self.tags} -> {other.tags}"
            )
            return False

        def bothsat(reldom1, reldom2):
            return reldom1[0].satisfies(reldom2[0]) and reldom1[1].satisfies(reldom2[1])

        for orel in other.relations:
            match orel:
                case (cl.NegatedRelation(n), d):
                    contradictor = next(
                        (srel for srel in self.relations if bothsat(srel, (n, d))), None
                    )

                    if contradictor is not None:
                        lazydebug(
                            logger,
                            lambda: f"satisfies found {contradictor} in self, which contradicts {orel} because it satisfies {(n, d)}",
                        )
                        return False
                case _:
                    if not any(bothsat(srel, orel) for srel in self.relations):
                        lazydebug(
                            logger,
                            lambda: f"found unsatisfied {orel} for {self.relations}",
                        )
                        return False

        return True

    def intersects(
        self, other: Domain, require_satisfies_left=False, require_satisfies_right=False
    ):
        """Return True if self and other could have a non-empty intersection.

        Parameters
        ----------
        self: Domain - the domain to check
        other: Domain - the domain to check against

        require_satisfies_left: bool -
            If True, assume that `self` is exhaustively specified (ie, any predicates not listed are false),
            and therefore `other` must imply `self` for the intersection to be non-empty.
        require_satisfies_right: bool -
            If True, assume that `other` is exhaustively specified (ie, any predicates not listed are false),
            and therefore `self` must imply `other` for the intersection to be non-empty.
        """

        lazydebug(logger, lambda: f"Domain.intersects for \n\t{self} \n\t{other}")

        if t.contradiction(self.tags.union(other.tags)):
            lazydebug(logger, lambda: f"tag contradiction {self.tags}, {other.tags}")
            return False

        # no relations can contradict eachother
        for ard, brd in itertools.product(self.relations, other.relations):
            if ard is brd:
                continue
            if not reldom_compatible(ard, brd):
                lazydebug(logger, lambda: f"found incompatible {ard} {brd}")
                return False

        # any relations actually known to be present must intersect
        a_pos = [
            rd for rd in self.relations if not isinstance(rd[0], cl.NegatedRelation)
        ]
        b_pos = [
            rd for rd in other.relations if not isinstance(rd[0], cl.NegatedRelation)
        ]
        if require_satisfies_left:
            if not t.satisfies(other.tags, self.tags):
                return False
            for ard in a_pos:
                if not any(reldom_intersects(ard, brd) for brd in b_pos):
                    lazydebug(
                        logger,
                        lambda: f"require_satisfies_left found no intersecting {ard} {b_pos}",
                    )
                    return False
        if require_satisfies_right:
            if not t.satisfies(self.tags, other.tags):
                return False
            for brd in b_pos:
                if not any(reldom_intersects(ard, brd) for ard in a_pos):
                    lazydebug(
                        logger,
                        lambda: f"require_satisfies_right found no intersecting {brd} {a_pos}",
                    )
                    return False

        lazydebug(
            logger, lambda: f"Domain.intersects for {self} {other} returning True"
        )

        return True

    def intersection(self, other: Domain):
        """
        Return a domain representing the intersection of self and other.
        Result is at least as strict as self and other.

        contains(self, x) and contains(other, x) -> contains(intersection, x)

        TODO:
        - does order relations are checked for intersection matter?
            - almost certainly yes, intersection is not transitive.
            - so what order is best? fewest remaining relations? does it matter?
        """

        newtags = self.tags.union(other.tags)
        if t.contradiction(newtags):
            raise ValueError(
                f"Contradictory {newtags=} for {self.intersection} {other=}"
            )

        newdom = Domain(newtags)
        for orel, odom in *self.relations, *other.relations:
            newdom.add_relation(orel, copy.deepcopy(odom))

        return newdom

    def is_recursive(self, seen=None):
        """Check if this domain somehow references itself via its own relations.
        Domains should ideally never reach this state; this function is used to check that they dont.
        """

        if seen is None:
            seen = set()

        if id(self) in seen:
            return True

        seen.add(id(self))

        return any(d.is_recursive(seen=seen) for _, d in self.relations)

    def positive_part(self):
        return Domain(
            tags={ti for ti in self.tags if not isinstance(ti, t.Negated)},
            relations=[
                (r, d.positive_part())
                for r, d in self.relations
                if not isinstance(r, cl.NegatedRelation)
            ],
        )

    def traverse(self):
        yield self
        for rel, dom in self.relations:
            yield from dom.traverse()

    def all_vartags(self) -> set[t.Variable]:
        return {x for d in self.traverse() for x in d.tags if isinstance(x, t.Variable)}

    def get_objs_named(self):
        objnames = {x.name for x in self.tags if isinstance(x, t.SpecificObject)}
        for rel, dom in self.relations:
            if isinstance(rel, cl.NegatedRelation):
                continue
            objnames = objnames.union(dom.get_objs_named())
        return objnames
