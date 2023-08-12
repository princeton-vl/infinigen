# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


import pdb
from dataclasses import dataclass
import warnings

import bpy
import numpy as np
from scipy.interpolate import interp1d

from .utils import helper, mesh
from .utils import geometrynodes as gn
from infinigen.assets.leaves import leaf

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core.util import blender as butil
from ..utils.object import data2mesh, mesh2obj

C = bpy.context
D = bpy.data


class TreeVertices():
    def __init__(self, vtxs=None, parent=None, level=None):
        """Define vertices and edges to outline tree geometry."""
        if vtxs is None:
            vtxs = np.array([[0, 0, 0]])
        elif isinstance(vtxs, list):
            vtxs = np.array(vtxs)
        parent = [-1] * len(vtxs) if parent is None else parent
        level = [0] * len(vtxs) if level is None else level

        self.vtxs = vtxs
        self.parent = parent
        self.level = level

    def get_idxs(self):
        return list(np.arange(len(self.vtxs)))

    def get_edges(self):
        edges = np.stack([np.arange(len(self.vtxs)), np.array(self.parent)], 1)
        return edges[edges[:, 1] != -1]

    def append(self, v, p, l=None):
        self.vtxs = np.append(self.vtxs, v, axis=0)
        self.parent += p

        if l is None:
            l = [0] * len(v)
        elif isinstance(l, int):
            l = [l] * len(v)
        self.level += l

    def __len__(self):
        return len(self.vtxs)


def dfs(idx, edges, parents, depth, rev_depth, n_leaves, child_idx):
    children = [v for v in edges[idx] if v != parents[idx]]
    if len(children) == 0:
        # At leaf, move backwards to update rev_depth
        curr_idx = idx
        child_idx[curr_idx] = -1
        curr_depth = 0
        while curr_idx != 0:
            prev_idx = curr_idx
            curr_idx = parents[curr_idx]
            curr_depth += 1
            n_leaves[curr_idx] += 1
            if rev_depth[curr_idx] < curr_depth:
                child_idx[curr_idx] = prev_idx
                rev_depth[curr_idx] = curr_depth

    else:
        for c in children:
            parents[c] = idx
            depth[c] = depth[idx] + 1
            dfs(c, edges, parents, depth, rev_depth, n_leaves, child_idx)


def parse_tree_attributes(vtx):
    # Strong assumption, root is vertex 0
    n = len(vtx.vtxs)
    parents = np.zeros(n, dtype=int)
    depth = np.zeros(n, dtype=int)
    rev_depth = np.zeros(n, dtype=int)
    n_leaves = np.zeros(n, dtype=int)
    child_idx = np.zeros(n, dtype=int)

    vtx_pos = vtx.vtxs
    levels = vtx.level

    edge_ref = {i: [] for i in range(n)}
    for e in vtx.get_edges():
        v0, v1 = e
        edge_ref[v0] += [v1]
        edge_ref[v1] += [v0]

    # Traverse tree from root node, recording depth + parent idx
    # Upon hitting leaf, walk back up parents saving max reverse depth
    dfs(0, edge_ref, parents, depth, rev_depth, n_leaves, child_idx)

    # if there is already a longer path connected to this parent, we create a dummy
    # copy of the parent node, and connect the current child to the dummy parent.
    # This makes sure each point will have no more than one child. 
    new_p_id = n  # p for parent. start from the last of the array

    for idx in range(n):
        children = np.array([v for v in edge_ref[idx] if v != parents[idx]])
        if len(children) >= 2:
            child_depths = rev_depth[children]
            deepest_child_idx = children[child_depths.argmax()]  # we keep this untouched

            children_idxs_to_deal = np.setdiff1d(children, np.array([deepest_child_idx]))
            for child_idx_to_deal in children_idxs_to_deal:
                new_p_pos = vtx_pos[idx]  # len-3
                new_p_parent = parents[idx]
                new_p_depth = 0
                new_p_rev_depth = rev_depth[child_idx_to_deal] + 1
                new_p_n_leaves = 1
                new_p_child_idx = child_idx_to_deal
                new_p_level = levels[idx]

                # apply modifications
                parents = np.append(parents, new_p_parent)
                depth = np.append(depth, new_p_depth)
                rev_depth = np.append(rev_depth, new_p_rev_depth)
                n_leaves = np.append(n_leaves, new_p_n_leaves)
                child_idx = np.append(child_idx, new_p_child_idx)
                vtx_pos = np.append(vtx_pos, new_p_pos.reshape(1, 3), axis=0)

                # new connection
                # note we don't connect the new node with its parent
                edge_ref[new_p_id] = [child_idx_to_deal, ]

                # remove old connections
                edge_ref[child_idx_to_deal].remove(idx)
                edge_ref[idx].remove(child_idx_to_deal)

                # # modify the vertex class
                vtx.append(new_p_pos.reshape(1, 3), [-1], [new_p_level])
                vtx.parent[child_idx_to_deal] = new_p_id

                new_p_id += 1

    n = len(parents)

    # Assign stem ids in order from longest to shortest paths
    stem_id = -np.ones(n, dtype=int)
    curr_idxs = np.arange(n)
    curr_stem_id = 1

    while len(curr_idxs) > 0:
        curr_depths = rev_depth[curr_idxs]
        tmp_idx = curr_idxs[curr_depths.argmax()]
        to_remove = []
        while tmp_idx != -1:
            to_remove += [tmp_idx]
            if len(edge_ref[tmp_idx]) <= 2:
                stem_id[tmp_idx] = curr_stem_id
            tmp_idx = child_idx[tmp_idx]

        curr_idxs = np.setdiff1d(curr_idxs, to_remove)
        curr_stem_id += 1

    parent_loc = np.zeros((n, 3), dtype=float)
    self_loc = np.zeros((n, 3), dtype=float)

    for vertex_idx, parent_idx in enumerate(parents):
        parent_loc[vertex_idx] = vtx_pos[parent_idx]
        self_loc[vertex_idx] = vtx_pos[vertex_idx]

    parent_loc[0] = np.array([0, 0, -1],
                             dtype=float)  # create a fake parent location for the root, to avoid zero-length
    # vector

    return {
        'parent_idx': parents,
        'depth': depth,
        'rev_depth': rev_depth,
        'stem_id': stem_id,
        'parent_skeleton_loc': parent_loc,
        'skeleton_loc': self_loc}


def rand_path(n_pts, sz=1, std=.3, momentum=.5, init_vec=[0, 0, 1], init_pt=[0, 0, 0], pull_dir=None,
              pull_init=1, pull_factor=0, sz_decay=1, decay_mom=True):
    init_vec = np.array(init_vec, dtype=float)
    if pull_dir is not None:
        pull_dir = np.array(pull_dir, dtype=float)
        init_vec += pull_init * pull_dir
    init_vec = init_vec / np.linalg.norm(init_vec)

    path = np.zeros((n_pts, 3))
    path[0] = init_pt
    for i in range(1, n_pts):
        if i == 1:
            prev_delta = init_vec * sz
        else:
            prev_delta = path[i - 1] - path[i - 2]

        prev_sz = np.linalg.norm(prev_delta)
        new_delta = prev_delta + np.random.randn(3) * std
        if pull_dir is not None:
            new_delta += pull_factor * pull_dir
        new_delta = (new_delta / np.linalg.norm(new_delta)) * prev_sz

        if decay_mom:
            tmp_momentum = 1 - (1 - momentum) * (i + 1) / n_pts
        else:
            tmp_momentum = momentum
        delta = prev_delta * tmp_momentum + new_delta * (1 - tmp_momentum)
        delta = (delta / np.linalg.norm(delta)) * sz * (sz_decay ** i)
        path[i] = path[i - 1] + delta

    return path


def get_spawn_pt(path, rng=[.5, 1], ang_min=np.pi / 6, ang_max=.9 * np.pi / 2, rnd_idx=None, ang_sign=None,
                 axis2=None, init_vec=None, z_bias=0):
    n = len(path)
    if n == 1:
        return 0, path[0], init_vec

    if rnd_idx is None:
        rnd_idx = np.random.randint(n * rng[0], n * rng[1])

    if init_vec is None:
        curr_vec = path[rnd_idx] - path[rnd_idx - 1]
        axis1 = np.array([curr_vec[1], -curr_vec[0], 0])
        if axis2 is None:
            axis2 = helper.rodrigues_rot(curr_vec, axis1, np.pi / 2)
        if callable(axis2):  # evaluate it. could be a random generator
            axis2 = axis2()
        rnd_ang = np.random.rand() * (ang_max - ang_min) + ang_min
        if ang_sign is None:
            ang_sign = np.sign(np.random.randn())
        rnd_ang *= ang_sign
        init_vec = helper.rodrigues_rot(curr_vec, axis2, rnd_ang)

    return rnd_idx, path[rnd_idx], init_vec


def recursive_path(tree, parent_idxs, level, path_kargs=None, spawn_kargs=None, n=1, symmetry=False,
                   children=None):
    if path_kargs is None:
        return

    if symmetry:
        n = 2 * n

    for branch_idx in range(n):
        curr_idx = branch_idx // 2 if symmetry else branch_idx
        curr_path = path_kargs(curr_idx)
        curr_spawn = spawn_kargs(curr_idx)
        if symmetry:
            curr_spawn['ang_sign'] = 2 * (branch_idx % 2) - 1

        parent_idx, init_pt, init_vec = get_spawn_pt(tree.vtxs[parent_idxs], **curr_spawn)
        parent_idx = parent_idxs[parent_idx]

        path = rand_path(**curr_path, init_pt=init_pt, init_vec=init_vec)
        new_vtxs = path[1:]
        new_idxs = list(np.arange(len(new_vtxs)) + len(tree))
        node_idxs = [parent_idx] + new_idxs
        tree.append(new_vtxs, node_idxs[:-1], level)

        if children is not None:
            for c in children:
                recursive_path(tree, node_idxs, level + 1, **c)


def remove_matched_atts(atts, vtxs, dist_thr, curr_min, curr_match, idx_offset=0, prev_deltas=None):
    dists, deltas = helper.compute_dists(atts, vtxs)
    if prev_deltas is not None:
        deltas = np.append(prev_deltas, deltas, axis=1)

    min_dist = dists.min(1)
    closest = dists.argmin(1)
    to_keep = min_dist > dist_thr

    atts = atts[to_keep]
    min_dist = min_dist[to_keep]
    closest = closest[to_keep]
    deltas = deltas[to_keep]
    curr_min = curr_min[to_keep]
    curr_match = curr_match[to_keep]

    to_update = min_dist < curr_min
    curr_min[to_update] = min_dist[to_update]
    curr_match[to_update] = closest[to_update] + idx_offset

    return atts, deltas, curr_min, curr_match


def space_colonization(tree, atts, D=.1, d=10.0, s=.1, pull_dir=None, dir_rand=.1, mag_rand=.15, n_steps=200,
                       level=0):
    # D: length of each growing step
    # d: init value for distance between attractors and points. safe to set to a very large value (e.g., 10)
    # s: if distance between an attractor and any point is less than s, we remove the attractor. should be
    # larger than D (e.g., 1.5*D)
    # pull_dir: useful if you want a bias in the growing direction (e.g., trunks growing upward)
    # dir_rand/mag_rand randomness in growing direction/length
    # n_steps: if set large enough, the tree will grow until all attractors are reached.

    if callable(atts):
        atts = atts(tree.vtxs)

    curr_min = np.zeros(len(atts)) + d
    curr_match = -np.ones(len(atts)).astype(int)
    atts, deltas, curr_min, curr_match = remove_matched_atts(atts, tree.vtxs, s, curr_min, curr_match)

    if np.all(curr_match == -1):
        warnings.warn('Space colonization attractor matching failed, all curr_match == -1')
        return

    for i in range(n_steps):
        new_vtxs = []
        new_parents = []
        matched_vtxs = np.unique(curr_match)

        for n_idx in matched_vtxs:
            if n_idx != -1:
                new_dir = deltas[curr_match == n_idx, n_idx].mean(0)
                new_dir = new_dir / np.linalg.norm(new_dir)
                if pull_dir is not None:
                    new_dir += pull_dir
                    new_dir = new_dir / np.linalg.norm(new_dir)
                new_dir += np.random.randn(3) * dir_rand
                tmp_D = D * np.exp(np.random.randn() * mag_rand)

                n0 = tree.vtxs[n_idx]
                n1 = n0 + tmp_D * new_dir
                new_vtxs += [n1]
                new_parents += [n_idx]

        idx_offset = len(tree)
        new_vtxs = np.stack(new_vtxs, 0)
        tree.append(new_vtxs, new_parents, level)

        atts, deltas, curr_min, curr_match = remove_matched_atts(atts, new_vtxs, s, curr_min, curr_match,
                                                                 idx_offset, deltas)

        if atts.shape[0] == 0:
            break


@dataclass
class TreeParams:
    skeleton: dict
    trunk_spacecol: dict
    roots_spacecol: dict
    child_placement: dict
    skinning: dict


def tree_skeleton(skeleton_params: dict, trunk_spacecol: dict, roots_spacecol: dict, init_pos, scale):
    vtx = TreeVertices(np.array(init_pos).reshape(-1, 3))
    recursive_path(vtx, vtx.get_idxs(), level=0, **skeleton_params)

    if trunk_spacecol is not None:
        space_colonization(vtx, **trunk_spacecol, level=max(vtx.level) + 1)

    if roots_spacecol is not None:
        space_colonization(vtx, **roots_spacecol, level=-1)

    attributes = parse_tree_attributes(vtx)
    obj = mesh.init_mesh('Tree', vtx.vtxs, vtx.get_edges())
    attributes['level'] = np.array(vtx.level)

    for att_name, att_val in attributes.items():
        if att_val.ndim == 2:
            obj.data.attributes.new(name=att_name, type='FLOAT_VECTOR', domain='POINT')
            obj.data.attributes[att_name].data.foreach_set('vector', att_val.reshape(
                -1) * scale)  # vector value should be scaled together with the obj
        else:
            obj.data.attributes.new(name=att_name, type='INT', domain='POINT')
            obj.data.attributes[att_name].data.foreach_set('value', att_val)

    obj.scale *= scale
    with butil.SelectObjects(obj):
        bpy.ops.object.transform_apply(scale=True)

    return obj


def skin_tree(nw, params, source_obj=None):
    base_geo = nw.new_node(Nodes.GroupInput).outputs['Geometry']
    skin = nw.new_node(gn.set_tree_radius().name, input_kwargs={
        'Geometry': base_geo,
        'Reverse depth': nw.expose_input('Reverse depth', attribute='rev_depth'), **params})

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': skin})


def add_tree_children(nw, child_col, params, merge_dist=None, realize=False):
    base_geo = nw.new_node(Nodes.GroupInput).outputs['Geometry']

    rev_depth = nw.expose_input('Reverse Depth', attribute='rev_depth')

    depth_range = params.pop('depth_range', None)
    if depth_range is not None:
        min, max = depth_range
        lt = nw.new_node(Nodes.Math, [rev_depth, max + 0.01], attrs={'operation': 'LESS_THAN'})
        gt = nw.new_node(Nodes.Math, [rev_depth, min - 0.01], attrs={'operation': 'GREATER_THAN'})
        selection = nw.new_node(Nodes.BooleanMath, [lt, gt], attrs={'operation': 'AND'})
    else:
        selection = None

    children = nw.new_node(gn.coll_distribute(merge_dist=merge_dist).name, input_kwargs={
        'Geometry': base_geo,
        'Collection': child_col,
        'Selection': selection, **params})

    if realize:
        children = nw.new_node(Nodes.RealizeInstances, [children])

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': children})


class FineTreeVertices(TreeVertices):
    def __init__(self, vtxs=None, parent=None, level=None, radius_fn=None, resolution=1):
        super(FineTreeVertices, self).__init__(vtxs, parent, level)
        self.resolution = resolution
        if radius_fn is None:
            radius_fn = (lambda base_radius, size, resolution: [1] * size)
        self.radius_fn = radius_fn
        self.detailed_locations = [[0, 0, 0]]
        self.radius = [1]
        self.detailed_parents = [-1]

    def append(self, v, p, l=None):
        super(FineTreeVertices, self).append(v, p, l)
        f = interp1d(np.arange(len(v) + 1), np.concatenate([self.vtxs[p[0]:p[0] + 1], v]), axis=0,
                     kind='quadratic')
        self.detailed_locations.extend(f(np.linspace(0, len(v), len(v) * self.resolution + 1))[1:])
        base_radius = self.radius[p[0] * self.resolution]
        self.radius.extend(self.radius_fn(base_radius, len(v), self.resolution))
        self.detailed_parents.append(p[0] * self.resolution)
        self.detailed_parents.extend(
            np.arange(0, len(v) * self.resolution - 1) + len(self.detailed_parents) - 1)

    @property
    def edges(self):
        edges = np.stack([np.arange(len(self.detailed_locations)), np.array(self.detailed_parents)], 1)
        return edges[edges[:, 1] != -1]

    def fix_first(self):
        self.radius[0] = self.radius[1]


def build_radius_tree(radius_fn, branch_config, base_radius=.002, resolution=1, fix_first=False):
    vtx = FineTreeVertices(np.zeros((1, 3)), radius_fn=radius_fn, resolution=resolution)
    recursive_path(vtx, vtx.get_idxs(), level=0, **branch_config)
    if fix_first:
        vtx.radius[0] = vtx.radius[1]
    obj = mesh2obj(data2mesh(vtx.detailed_locations, vtx.edges, [], 'tree'))
    vg_a = obj.vertex_groups.new(name='radius')
    for i, r in enumerate(vtx.radius):
        vg_a.add([i], base_radius * r, 'REPLACE')
    return obj
