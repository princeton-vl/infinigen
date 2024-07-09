# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix constants

from collections import defaultdict

import numpy as np
import shapely
from matplotlib import pyplot as plt
from numpy.random import uniform
from shapely import LineString, union

import infinigen.core.constraints.example_solver.room.constants as constants
from infinigen.assets.utils.shapes import shared
from infinigen.core.constraints.example_solver.room.utils import (
    canonicalize,
    compute_neighbours,
    cut_polygon_by_line,
    is_valid_polygon,
    unit_cast,
    update_exterior_edges,
    update_staircase_occupancies,
)
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class SegmentMaker:
    def __init__(self, factory_seed, contour, n, merge_alpha=-1):
        with FixedSeed(factory_seed):
            self.contour = contour
            self.n = n
            self.n_boxes = int(self.n * uniform(1.4, 1.6))

            self.box_ratio = 0.3
            self.min_segment_area = log_uniform(1.5, 2)
            self.min_segment_size = log_uniform(0.5, 1.0)

            self.divide_box_fn = lambda x: x.area**0.5

            self.n_box_trials = 200
            self.merge_fn = lambda x: x**merge_alpha

    def build_segments(self, staircase=None):
        while True:
            try:
                segments, shared_edges = self.filter_segments()
                break
            except Exception:
                pass
        exterior_edges = update_exterior_edges(segments, shared_edges)
        neighbours_all = {
            k: set(compute_neighbours(se, constants.SEGMENT_MARGIN))
            for k, se in shared_edges.items()
        }
        exterior_neighbours = set(
            compute_neighbours(exterior_edges, constants.SEGMENT_MARGIN)
        )
        staircase_occupancies = update_staircase_occupancies(segments, staircase)
        return {
            "segments": segments,
            "shared_edges": shared_edges,
            "exterior_edges": exterior_edges,
            "neighbours_all": neighbours_all,
            "exterior_neighbours": exterior_neighbours,
            "staircase_occupancies": staircase_occupancies,
            "staircase": staircase,
        }

    def divide_segments(self):
        segments = {0: self.contour}
        for _ in range(self.n_boxes):
            keys, values = zip(*segments.items())
            prob = np.array([self.divide_box_fn(v) for v in values])
            for _ in range(self.n_box_trials):
                k = np.random.choice(list(keys), p=prob / prob.sum())
                x, y, xx, yy = segments[k].bounds
                w, h = xx - x, yy - y
                r = uniform(0.25, 0.75)
                line = None
                if w >= h:
                    w_ = unit_cast(r * w)
                    bound = max(self.box_ratio * h, constants.SEGMENT_MARGIN)
                    if w_ >= bound and w - w_ >= bound:
                        line = LineString([(x + w_, -100), (x + w_, 100)])
                else:
                    h_ = unit_cast(r * h)
                    bound = max(self.box_ratio * w, constants.SEGMENT_MARGIN)
                    if h_ >= bound and h - h_ >= bound:
                        line = LineString([(-100, y + h_), (100, y + h_)])
                if line is not None:
                    i = max(segments.keys())
                    s, t = cut_polygon_by_line(segments[k], line)
                    s_ = canonicalize(s)
                    t_ = canonicalize(t)
                    if (
                        np.abs(s.area - s_.area) < 1e-3
                        and np.abs(t.area - t_.area) < 1e-3
                    ):
                        segments[k], segments[i + 1] = s_, t_
                        break
        return {k: v for k, v in segments.items()}

    def merge_segment(self, segments, shared_edges, attached, i, j):
        assert i != j
        s = canonicalize(union(segments[i], segments[j]))
        if not is_valid_polygon(s):
            return
        segments[j] = s
        segments.pop(i)
        shared_edges.pop(i)
        attached.pop(i)
        for k, ses in shared_edges.items():
            if i in ses:
                ses.pop(i)
        for k, ats in attached.items():
            if i in ats:
                ats.remove(i)
        for k, s in segments.items():
            for l, t in segments.items():
                if k != l and (k == j or l == j):
                    se = shared(s, t)
                    shared_edges[k][l] = se
                    if se.length >= constants.SEGMENT_MARGIN:
                        attached[k].add(l)
                        attached[l].add(k)
        return shared_edges

    def filter_segments(self):
        segments = self.divide_segments()
        shared_edges = defaultdict(dict)
        attached = defaultdict(set)
        for k, s in segments.items():
            for l, t in segments.items():
                if k < l:
                    se = shared(s, t)
                    shared_edges[k][l] = shared_edges[l][k] = se
                    if se.length >= constants.SEGMENT_MARGIN:
                        attached[k].add(l)
                        attached[l].add(k)

        while len(segments) > self.n:
            prob = np.array([1 / (len(attached[c]) + 1) for c in shared_edges.keys()])
            k = np.random.choice(list(shared_edges.keys()), p=prob / prob.sum())
            candidates = list(
                k for k, se in shared_edges[k].items() if se.length >= 1e-6
            )
            prob = np.array(
                [
                    len(attached[c].difference(attached[k])) ** 2 + 0.5
                    for c in candidates
                ]
            )
            n = np.random.choice(candidates, p=prob / prob.sum())
            self.merge_segment(segments, shared_edges, attached, k, n)
        return segments, shared_edges

    def plot(self, segments):
        plt.clf()
        for k, s in segments.items():
            shapely.plotting.plot_polygon(s, color=uniform(0, 1, 3))
        plt.tight_layout()
        plt.show()
