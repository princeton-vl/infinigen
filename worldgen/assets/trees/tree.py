import pdb
from dataclasses import dataclass
import warnings
import bpy
import numpy as np
from .utils import helper, mesh
from assets.leaves import leaf
from nodes.node_wrangler import Nodes
from util import blender as butil


        """Define vertices and edges to outline tree geometry."""
            vtxs = np.array(vtxs)
        parent = [-1] * len(vtxs) if parent is None else parent
        level = [0] * len(vtxs) if level is None else level
        self.vtxs = vtxs
        self.parent = parent
        self.level = level
    def get_idxs(self):
        return list(np.arange(len(self.vtxs)))
    def get_edges(self):
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

    # Strong assumption, root is vertex 0
    parents = np.zeros(n, dtype=int)
    depth = np.zeros(n, dtype=int)
    rev_depth = np.zeros(n, dtype=int)
    n_leaves = np.zeros(n, dtype=int)
    child_idx = np.zeros(n, dtype=int)

    edge_ref = {i: [] for i in range(n)}
        edge_ref[v0] += [v1]
        edge_ref[v1] += [v0]

    # Traverse tree from root node, recording depth + parent idx
    # Upon hitting leaf, walk back up parents saving max reverse depth
    dfs(0, edge_ref, parents, depth, rev_depth, n_leaves, child_idx)

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

    return {
        'parent_idx': parents,
        'depth': depth,
        'parent_skeleton_loc': parent_loc,


    init_vec = np.array(init_vec, dtype=float)
        pull_dir = np.array(pull_dir, dtype=float)
        init_vec += pull_init * pull_dir
    init_vec = init_vec / np.linalg.norm(init_vec)

    path = np.zeros((n_pts, 3))
    path[0] = init_pt
    for i in range(1, n_pts):
        if i == 1:
            prev_delta = init_vec * sz
        else:

        prev_sz = np.linalg.norm(prev_delta)
        new_delta = prev_delta + np.random.randn(3) * std
        if pull_dir is not None:
            new_delta += pull_factor * pull_dir
        new_delta = (new_delta / np.linalg.norm(new_delta)) * prev_sz
        if decay_mom:
        else:
            tmp_momentum = momentum
        delta = prev_delta * tmp_momentum + new_delta * (1 - tmp_momentum)
        delta = (delta / np.linalg.norm(delta)) * sz * (sz_decay ** i)
    return path
    n = len(path)
    if n == 1:
        return 0, path[0], init_vec
    if rnd_idx is None:
        rnd_idx = np.random.randint(n * rng[0], n * rng[1])
    if init_vec is None:
        curr_vec = path[rnd_idx] - path[rnd_idx - 1]
        axis1 = np.array([curr_vec[1], -curr_vec[0], 0])
        if axis2 is None:
        if callable(axis2):  # evaluate it. could be a random generator
            axis2 = axis2()
        rnd_ang = np.random.rand() * (ang_max - ang_min) + ang_min
        if ang_sign is None:
            ang_sign = np.sign(np.random.randn())
        rnd_ang *= ang_sign
        init_vec = helper.rodrigues_rot(curr_vec, axis2, rnd_ang)
    return rnd_idx, path[rnd_idx], init_vec
    if path_kargs is None:
        return
        n = 2 * n

    for branch_idx in range(n):
        curr_idx = branch_idx // 2 if symmetry else branch_idx
        curr_path = path_kargs(curr_idx)
        curr_spawn = spawn_kargs(curr_idx)
        if symmetry:
            curr_spawn['ang_sign'] = 2 * (branch_idx % 2) - 1
        parent_idx = parent_idxs[parent_idx]
        path = rand_path(**curr_path, init_pt=init_pt, init_vec=init_vec)
        new_vtxs = path[1:]
        new_idxs = list(np.arange(len(new_vtxs)) + len(tree))
        node_idxs = [parent_idx] + new_idxs
        tree.append(new_vtxs, node_idxs[:-1], level)
        if children is not None:
            for c in children:
                recursive_path(tree, node_idxs, level + 1, **c)
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
    # D: length of each growing step
    # d: init value for distance between attractors and points. safe to set to a very large value (e.g., 10)
    # pull_dir: useful if you want a bias in the growing direction (e.g., trunks growing upward)
    # dir_rand/mag_rand randomness in growing direction/length
    # n_steps: if set large enough, the tree will grow until all attractors are reached.

    if callable(atts):
        atts = atts(tree.vtxs)

    curr_min = np.zeros(len(atts)) + d
    curr_match = -np.ones(len(atts)).astype(int)

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


        if atts.shape[0] == 0:
            break
@dataclass
class TreeParams:
    skeleton: dict
    roots_spacecol: dict
    child_placement: dict
    skinning: dict


    vtx = TreeVertices(np.array(init_pos).reshape(-1, 3))
    recursive_path(vtx, vtx.get_idxs(), level=0, **skeleton_params)

    if trunk_spacecol is not None:
        space_colonization(vtx, **trunk_spacecol, level=max(vtx.level) + 1)

    if roots_spacecol is not None:
        space_colonization(vtx, **roots_spacecol, level=-1)

    obj = mesh.init_mesh('Tree', vtx.vtxs, vtx.get_edges())

    for att_name, att_val in attributes.items():

    return obj


    base_geo = nw.new_node(Nodes.GroupInput).outputs['Geometry']
    skin = nw.new_node(gn.set_tree_radius().name, input_kwargs={
        'Geometry': base_geo,

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
        'Collection': child_col,
        'Selection': selection, **params})

    if realize:
        children = nw.new_node(Nodes.RealizeInstances, [children])

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': children})
