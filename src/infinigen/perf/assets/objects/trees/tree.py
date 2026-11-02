# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alejandro Newell: original Infinigen tree skeleton generation (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/trees/tree.py)
# - David Yan, Jia Deng: optimized implementation

import heapq
import warnings

import numpy as np

from infinigen.assets.objects.trees import tree as _default
from infinigen.assets.objects.trees.utils import mesh

try:
    # built only with INFINIGEN_INSTALL_PERF; without it the default
    # implementation runs
    from infinigen.perf.assets.objects.trees import _tree_ops
except ImportError:
    _tree_ops = None


class TreeVertices(_default.TreeVertices):
    def __init__(self, vtxs=None, parent=None, level=None):
        """Define vertices and edges to outline tree geometry."""
        if vtxs is None:
            vtxs = np.array([[0, 0, 0]])
        elif isinstance(vtxs, list):
            vtxs = np.array(vtxs)
        parent = [-1] * len(vtxs) if parent is None else parent
        level = [0] * len(vtxs) if level is None else level

        self._vtxs = np.asarray(vtxs)
        self._size = len(self._vtxs)
        self.parent = parent
        self.level = level

    @property
    def vtxs(self):
        return self._vtxs[: self._size]

    def get_idxs(self):
        return list(range(self._size))

    def get_edges(self):
        parents = np.asarray(self.parent)
        valid = parents != -1
        return np.stack([np.arange(self._size)[valid], parents[valid]], axis=1)

    def _reserve_for(self, values):
        required = self._size + len(values)
        if values.dtype == self._vtxs.dtype:
            result_dtype = self._vtxs.dtype
        else:
            result_dtype = np.result_type(self._vtxs.dtype, values.dtype)
        if required <= len(self._vtxs) and result_dtype == self._vtxs.dtype:
            return

        capacity = max(required, max(4, 2 * len(self._vtxs)))
        storage = np.empty((capacity, *self._vtxs.shape[1:]), dtype=result_dtype)
        storage[: self._size] = self._vtxs[: self._size]
        self._vtxs = storage

    def append(self, v, p, l=None):
        values = np.asarray(v)
        self._reserve_for(values)
        end = self._size + len(values)
        self._vtxs[self._size : end] = values
        self._size = end
        self.parent += p

        if l is None:
            l = [0] * len(values)
        elif isinstance(l, int):
            l = [l] * len(values)
        self.level += l

    def __len__(self):
        return self._size


class FineTreeVertices(_default.FineTreeVertices, TreeVertices):
    pass


def parse_tree_attributes(vtx):
    """Linear-time tree attributes when parents precede children; else the default."""
    original_n = len(vtx.vtxs)
    original_parents = np.asarray(vtx.parent, dtype=int)
    indices = np.arange(original_n)
    if (
        len(original_parents) != original_n
        or original_n == 0
        or original_parents[0] != -1
        or np.any(original_parents[1:] < 0)
        or np.any(original_parents[1:] >= indices[1:])
    ):
        return _default.parse_tree_attributes.default(vtx)

    children = [[] for _ in range(original_n)]
    depth = [0] * original_n
    for child in range(1, original_n):
        parent = int(original_parents[child])
        children[parent].append(child)
        depth[child] = depth[parent] + 1

    rev_depth = [0] * original_n
    child_idx = [0] * original_n
    for idx in range(original_n - 1, -1, -1):
        if not children[idx]:
            child_idx[idx] = -1
            continue
        best_child = children[idx][0]
        best_rev_depth = rev_depth[best_child]
        for child in children[idx]:
            if rev_depth[child] > best_rev_depth:
                best_child = child
                best_rev_depth = rev_depth[child]
        rev_depth[idx] = best_rev_depth + 1
        child_idx[idx] = best_child

    parents = original_parents.tolist()
    parents[0] = 0
    levels = list(vtx.level)
    edge_ref = [
        ([] if idx == 0 else [parents[idx]]) + list(children[idx])
        for idx in range(original_n)
    ]

    for idx in range(original_n):
        current_children = [v for v in edge_ref[idx] if v != parents[idx]]
        if len(current_children) < 2:
            continue
        deepest_child = max(current_children, key=lambda child: rev_depth[child])
        for child_to_move in current_children:
            if child_to_move == deepest_child:
                continue
            new_parent_idx = len(parents)
            parents.append(parents[idx])
            depth.append(0)
            rev_depth.append(rev_depth[child_to_move] + 1)
            child_idx.append(child_to_move)
            levels.append(levels[idx])

            edge_ref.append([child_to_move])
            edge_ref[child_to_move].remove(idx)
            edge_ref[idx].remove(child_to_move)

            new_parent_pos = vtx.vtxs[idx].reshape(1, 3)
            vtx.append(new_parent_pos, [-1], [levels[idx]])
            vtx.parent[child_to_move] = new_parent_idx

    parents = np.asarray(parents, dtype=int)
    depth = np.asarray(depth, dtype=int)
    rev_depth = np.asarray(rev_depth, dtype=int)
    child_idx = np.asarray(child_idx, dtype=int)
    n = len(parents)

    stem_id = -np.ones(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    longest_first = [(-int(rev_depth[idx]), idx) for idx in range(n)]
    heapq.heapify(longest_first)
    curr_stem_id = 1
    while longest_first:
        _, tmp_idx = heapq.heappop(longest_first)
        if not remaining[tmp_idx]:
            continue
        while tmp_idx != -1:
            remaining[tmp_idx] = False
            if len(edge_ref[tmp_idx]) <= 2:
                stem_id[tmp_idx] = curr_stem_id
            tmp_idx = child_idx[tmp_idx]
        curr_stem_id += 1

    vtx_pos = vtx.vtxs
    parent_loc = vtx_pos[parents].copy()
    self_loc = vtx_pos.copy()
    parent_loc[0] = np.array([0, 0, -1], dtype=float)

    return {
        "parent_idx": parents,
        "depth": depth,
        "rev_depth": rev_depth,
        "stem_id": stem_id,
        "parent_skeleton_loc": parent_loc,
        "skeleton_loc": self_loc,
    }


def _filter_update_attractors(atts, vtxs, dist_thr, curr_min, curr_match, idx_offset=0):
    """`remove_matched_atts` without the cumulative attractor-to-vertex deltas, which
    the native space colonization recomputes from the positions."""
    if (
        atts.dtype != np.float64
        or vtxs.dtype != np.float64
        or not atts.flags.c_contiguous
        or not vtxs.flags.c_contiguous
        or _tree_ops is None
    ):
        atts, _, curr_min, curr_match = _default.remove_matched_atts(
            atts, vtxs, dist_thr, curr_min, curr_match, idx_offset
        )
        return atts, curr_min, curr_match

    return _tree_ops.filter_update_attractors(
        atts, vtxs, dist_thr, curr_min, curr_match, idx_offset
    )


def _grow_vertices_exact(
    mean_directions,
    matched_vtxs,
    vertices,
    random_values,
    D,
    pull_dir,
    dir_rand,
    mag_rand,
):
    """One growth step of the matched vertices, norms per vector as in the default."""

    count = len(mean_directions)
    norms = np.fromiter(
        (np.sqrt(np.dot(direction, direction)) for direction in mean_directions),
        dtype=np.float64,
        count=count,
    )
    new_directions = mean_directions / norms[:, np.newaxis]
    if pull_dir is not None:
        new_directions += pull_dir
        norms = np.fromiter(
            (np.sqrt(np.dot(direction, direction)) for direction in new_directions),
            dtype=np.float64,
            count=count,
        )
        new_directions = new_directions / norms[:, np.newaxis]

    new_directions += random_values[:, :3] * dir_rand
    distances = D * np.exp(random_values[:, 3] * mag_rand)
    return vertices[matched_vtxs] + distances[:, np.newaxis] * new_directions


def space_colonization(
    tree,
    atts,
    D=0.1,
    d=10.0,
    s=0.1,
    pull_dir=None,
    dir_rand=0.1,
    mag_rand=0.15,
    n_steps=200,
    level=0,
):
    if _tree_ops is None:
        return _default.space_colonization.default(
            tree, atts, D, d, s, pull_dir, dir_rand, mag_rand, n_steps, level
        )
    if callable(atts):
        atts = atts(tree.vtxs)

    curr_min = np.zeros(len(atts)) + d
    curr_match = -np.ones(len(atts)).astype(int)
    atts, curr_min, curr_match = _filter_update_attractors(
        atts, tree.vtxs, s, curr_min, curr_match
    )

    if np.all(curr_match == -1):
        warnings.warn(
            "Space colonization attractor matching failed, all curr_match == -1"
        )
        return

    for _ in range(n_steps):
        matched_vtxs, mean_directions = _tree_ops.grouped_means(
            atts, tree.vtxs, curr_match
        )
        random_values = np.random.randn(len(matched_vtxs), 4)
        new_vtxs = _grow_vertices_exact(
            mean_directions,
            matched_vtxs,
            tree.vtxs,
            random_values,
            D,
            pull_dir,
            dir_rand,
            mag_rand,
        )

        idx_offset = len(tree)
        tree.append(new_vtxs, matched_vtxs.tolist(), level)

        atts, curr_min, curr_match = _filter_update_attractors(
            atts, new_vtxs, s, curr_min, curr_match, idx_offset
        )

        if atts.shape[0] == 0:
            break


def tree_skeleton(
    skeleton_params: dict,
    trunk_spacecol: dict,
    roots_spacecol: dict,
    init_pos,
    scale,
):
    vtx = TreeVertices(np.array(init_pos).reshape(-1, 3))
    _default.recursive_path(vtx, vtx.get_idxs(), level=0, **skeleton_params)

    if trunk_spacecol is not None:
        space_colonization(vtx, **trunk_spacecol, level=max(vtx.level) + 1)

    if roots_spacecol is not None:
        space_colonization(vtx, **roots_spacecol, level=-1)

    attributes = parse_tree_attributes(vtx)
    obj = mesh.init_mesh("Tree", vtx.vtxs, vtx.get_edges())
    attributes["level"] = np.array(vtx.level)

    for att_name, att_val in attributes.items():
        if att_val.ndim == 2:
            obj.data.attributes.new(name=att_name, type="FLOAT_VECTOR", domain="POINT")
            obj.data.attributes[att_name].data.foreach_set(
                "vector", att_val.reshape(-1) * scale
            )  # vector value should be scaled together with the obj
        else:
            obj.data.attributes.new(name=att_name, type="INT", domain="POINT")
            obj.data.attributes[att_name].data.foreach_set("value", att_val)

    obj.scale *= scale
    # transform_apply without the operator's depsgraph updates
    obj.data.transform(obj.matrix_basis)
    obj.matrix_basis.identity()

    return obj


def skin_tree_unmerged(nw, params):
    """`tree.skin_tree` without welding where nearby stems meet, for native meshing."""
    from infinigen.core.nodes.node_wrangler import Nodes
    from infinigen.perf.assets.objects.trees.utils import geometrynodes as gn

    base_geo = nw.new_node(Nodes.GroupInput).outputs["Geometry"]
    skin = nw.new_node(
        gn.set_tree_radius_unmerged().name,
        input_kwargs={
            "Geometry": base_geo,
            "Reverse depth": nw.expose_input("Reverse depth", attribute="rev_depth"),
            **params,
        },
    )
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": skin})


def rand_path(
    n_pts,
    sz=1,
    std=0.3,
    momentum=0.5,
    init_vec=[0, 0, 1],
    init_pt=[0, 0, 0],
    pull_dir=None,
    pull_init=1,
    pull_factor=0,
    sz_decay=1,
    decay_mom=True,
):
    """Default `rand_path` with the random steps drawn in one call (the same stream
    as one draw per step) and 3-vector norms taken directly."""
    init_vec = np.array(init_vec, dtype=float)
    if pull_dir is not None:
        pull_dir = np.array(pull_dir, dtype=float)
        init_vec += pull_init * pull_dir
    init_vec = init_vec / np.sqrt(np.dot(init_vec, init_vec))

    path = np.zeros((n_pts, 3))
    path[0] = init_pt
    random_deltas = np.random.randn(max(0, n_pts - 1), 3)
    for i in range(1, n_pts):
        if i == 1:
            prev_delta = init_vec * sz
        else:
            prev_delta = path[i - 1] - path[i - 2]

        prev_sz = np.sqrt(np.dot(prev_delta, prev_delta))
        new_delta = prev_delta + random_deltas[i - 1] * std
        if pull_dir is not None:
            new_delta += pull_factor * pull_dir
        new_delta = (new_delta / np.sqrt(np.dot(new_delta, new_delta))) * prev_sz

        if decay_mom:
            tmp_momentum = 1 - (1 - momentum) * (i + 1) / n_pts
        else:
            tmp_momentum = momentum
        delta = prev_delta * tmp_momentum + new_delta * (1 - tmp_momentum)
        delta = (delta / np.sqrt(np.dot(delta, delta))) * sz * (sz_decay**i)
        path[i] = path[i - 1] + delta

    return path


def recursive_path(
    tree,
    parent_idxs,
    level,
    path_kargs=None,
    spawn_kargs=None,
    n=1,
    symmetry=False,
    children=None,
):
    """Default `recursive_path`, numbering new vertices with a range."""
    if path_kargs is None:
        return

    if symmetry:
        n = 2 * n

    for branch_idx in range(n):
        curr_idx = branch_idx // 2 if symmetry else branch_idx
        curr_path = path_kargs(curr_idx)
        curr_spawn = spawn_kargs(curr_idx)
        if symmetry:
            curr_spawn["ang_sign"] = 2 * (branch_idx % 2) - 1

        parent_idx, init_pt, init_vec = _default.get_spawn_pt(
            tree.vtxs[parent_idxs], **curr_spawn
        )
        parent_idx = parent_idxs[parent_idx]

        path = _default.rand_path(**curr_path, init_pt=init_pt, init_vec=init_vec)
        new_vtxs = path[1:]
        start_idx = len(tree)
        new_idxs = list(range(start_idx, start_idx + len(new_vtxs)))
        node_idxs = [parent_idx] + new_idxs
        tree.append(new_vtxs, node_idxs[:-1], level)

        if children is not None:
            for c in children:
                _default.recursive_path(tree, node_idxs, level + 1, **c)
