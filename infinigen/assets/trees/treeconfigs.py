# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alejandro Newell


from logging import root
import numpy as np

import bpy
from .utils import mesh, helper
from .tree import TreeParams

subsubtwig_config = {'n': 2, 'symmetry': True,
                     'path_kargs': lambda idx: {'n_pts': 3, 'std': 1, 'momentum': 1, 'sz': .4},
                     'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .2, 'rnd_idx': 2*idx+2,
                                                 'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]}}
subtwig_config = {'n': 3, 'symmetry': True,
                  'path_kargs': lambda idx: {'n_pts': 6, 'std': 1, 'momentum': 1, 'sz': .6 - .1 * idx},
                  'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .1, 'rnd_idx': 2*idx+1,
                                              'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]},
                  'children': [subsubtwig_config]}
twig_config = {'n': 1, 'decay': .8, 'valid_leaves': [-2, -1],
               'path_kargs': lambda idx: {'n_pts': 7, 'sz': .5, 'std': .5, 'momentum': .7},
               'spawn_kargs': lambda idx: {'init_vec': [0, 1, 0]},
               'children': [subtwig_config]}


def random_pine_rot():
    theta = np.random.uniform(2*np.pi)
    return [np.sin(theta), 0.0, np.cos(theta)]


subsubtwig_config = {'n': 20, 'symmetry': False,
                     'path_kargs': lambda idx: {'n_pts': 2, 'std': 1, 'momentum': 1, 'sz': .2},
                     'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .2,
                                                 'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': random_pine_rot}}
subtwig_config = {'n': 7, 'symmetry': False,
                  'path_kargs': lambda idx: {'n_pts': 10, 'std': .3, 'momentum': 1, 'sz': .2 - .01 * idx},
                  'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .1,
                                              'ang_min': np.pi/8, 'ang_max': np.pi/8 + np.pi/16, 'axis2': random_pine_rot},
                  'children': [subsubtwig_config]}
pinetwig_config = {'n': 1,
                   'path_kargs': lambda idx: {'n_pts': 7, 'sz': .5, 'std': .2, 'momentum': .7},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 1, 0]},
                   'children': [subtwig_config]}


subsubsubtwig_config = {'n': 1, 'symmetry': True,
                        'path_kargs': lambda idx: {'n_pts': 2, 'std': 1, 'momentum': 1, 'sz': .4},
                        'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .2, 'rnd_idx': idx+1,
                                                    'ang_min': np.pi/8, 'ang_max': np.pi/8 + np.pi/32, 'axis2': [0, 0, 1]}}
subsubtwig_config = {'n': 3, 'symmetry': False,
                     'path_kargs': lambda idx: {'n_pts': 3, 'std': 1, 'momentum': 1, 'sz': .6 - .1 * idx},
                     'spawn_kargs': lambda idx: {'rng': [0.1, 1.0], 'z_bias': .1,
                                                 'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]},
                     'children': [subsubsubtwig_config]}
subtwig_config = {'n': 8, 'symmetry': False,
                  'path_kargs': lambda idx: {'n_pts': 7, 'std': 1, 'momentum': 1, 'sz': .6 - .1 * idx},
                  'spawn_kargs': lambda idx: {'rng': [0.2, 1.0], 'z_bias': .1,
                                              'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]},
                  'children': [subsubtwig_config]}
bambootwig_config = {'n': 1, 'decay': .8, 'valid_leaves': [-2, -1],
                     'path_kargs': lambda idx: {'n_pts': 15, 'sz': 1.0, 'std': .05, 'momentum': .7, 'pull_dir': [0, 0, -0.3], 'pull_factor': 0.5, 'pull_init': 0.0},
                     'spawn_kargs': lambda idx: {'init_vec': [0, 1, 0]},
                     'children': [subtwig_config]}


subtwig_config = {'n': 37, 'symmetry': True,
                  'path_kargs': lambda idx: {'n_pts': 2, 'std': 1, 'momentum': 1, 'sz': .4},
                  'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .2, 'rnd_idx': idx+2,
                                              'ang_min': 0.3*np.pi, 'ang_max': 0.3*np.pi + np.pi/16, 'axis2': [0, 0, 1]}}
palmtwig_config = {'n': 1, 'decay': .8, 'valid_leaves': [-2, -1],
                   'path_kargs': lambda idx: {'n_pts': 40, 'sz': .5, 'std': .05, 'momentum': .7, 'pull_dir': [0, 0, -0.3], 'pull_factor': 0.5, 'pull_init': 0.0},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 1, 0]},
                   'children': [subtwig_config]}


subtwig_config = {'n': 3, 'symmetry': True,
                  'path_kargs': lambda idx: {'n_pts': 3, 'std': 1, 'momentum': 1, 'sz': .6 - .1 * idx},
                  'spawn_kargs': lambda idx: {'rng': [.2, .9], 'z_bias': .1, 'rnd_idx': 2*idx+1,
                                              'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]},
                  'children': []}
shrubtwig_config = {'n': 1,
                    'path_kargs': lambda idx: {'n_pts': 6, 'sz': .5, 'std': .5, 'momentum': .7},
                    'spawn_kargs': lambda idx: {'init_vec': [0, 1, 0]},
                    'children': [subtwig_config]}


def generate_twig_config():
    n_twig_pts = np.random.randint(10) + 5
    twig_len = np.random.uniform(3, 4)
    twig_sz = twig_len / n_twig_pts
    avail_idxs = np.arange(n_twig_pts)
    start_idx = 1 + int(n_twig_pts * np.random.uniform(0, .3))
    sample_density = np.random.choice(
        np.arange(np.ceil(np.sqrt(n_twig_pts)), dtype=int) + 1)
    avail_sub_idxs = avail_idxs[start_idx::sample_density]

    init_z = np.random.uniform(0, .3)
    z_rnd_factor = np.random.uniform(0.01, .05)

    skip_subtwig = np.random.rand() < .3
    subsub_sz = np.random.uniform(.02, .1)
    subtwig_momentum = np.random.uniform(0, 1)
    subtwig_std = np.random.rand() ** 2
    sz_decay = np.random.uniform(.9, 1)
    pull_factor = np.random.uniform(0, .3)

    if not skip_subtwig:
        n_sub_pts = np.random.randint(10) + 5
        sub_sz = np.random.uniform(1, twig_len-.5) / n_sub_pts
        idx_decay = (sub_sz * (np.random.rand() * .8 + .1)) / n_sub_pts
        avail_idxs = np.arange(n_sub_pts)
        start_idx = int(n_sub_pts * np.random.rand() * .5) + 1
        sample_density = np.random.choice([1, 2, 3])
        avail_idxs = avail_idxs[start_idx::sample_density]

        ang_offset = np.random.rand() * np.pi / 3
        ang_range = np.random.rand() * ang_offset

        subsubtwig_config = {'n': len(avail_idxs), 'symmetry': True,
                             'path_kargs': lambda idx: {'n_pts': 3, 'std': 1, 'momentum': 1, 'sz': subsub_sz,
                                                        'pull_dir': [0, 0, init_z + np.random.randn() * z_rnd_factor],
                                                        'pull_factor': pull_factor},
                             'spawn_kargs': lambda idx: {'rnd_idx': avail_idxs[idx],
                                                         'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]}}
        subtwig_config = {'n': len(avail_sub_idxs), 'symmetry': True,
                          'path_kargs': lambda idx: {'n_pts': n_sub_pts,
                                                     'std': subtwig_std, 'momentum': subtwig_momentum,
                                                     'sz': sub_sz - idx_decay * idx, 'sz_decay': sz_decay,
                                                     'pull_dir': [0, 0, init_z + np.random.randn() * z_rnd_factor],
                                                     'pull_factor': pull_factor},
                          'spawn_kargs': lambda idx: {'rng': [.2, .9], 'rnd_idx': avail_sub_idxs[idx],
                                                      'ang_min': ang_offset, 'ang_max': ang_offset + ang_range, 'axis2': [0, 0, 1]},
                          'children': [subsubtwig_config]
                          }

    else:
        subtwig_config = {'n': len(avail_sub_idxs), 'symmetry': True,
                          'path_kargs': lambda idx: {'n_pts': 3, 'std': 1, 'momentum': 1, 'sz': subsub_sz,
                                                     'pull_dir': [0, 0, init_z + np.random.randn() * z_rnd_factor],
                                                     'pull_factor': pull_factor},
                          'spawn_kargs': lambda idx: {'rnd_idx': avail_sub_idxs[idx],
                                                      'ang_min': np.pi/4, 'ang_max': np.pi/4 + np.pi/16, 'axis2': [0, 0, 1]}}

    twig_config = {'n': 1,
                   'path_kargs': lambda idx: {'n_pts': n_twig_pts, 'sz': twig_sz, 'std': .5, 'momentum': .5,
                                              'pull_dir': [0, 0, init_z + np.random.randn() * z_rnd_factor],
                                              'pull_factor': pull_factor},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 1, -init_z]},
                   'children': [subtwig_config]}

    return twig_config


def basic_tree(init_pos=np.array([[0, 0, 0]])):
    def init_att_fn(nodes):
        pt_offset = init_pos[0] + np.array([0, 0, 11])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[7, 7, 7], pt_offset=pt_offset)
        return branch_pts

    def root_att_fn(nodes):
        # Pass this into root_kargs to initialize a root system
        pt_offset = init_pos[0] + np.array([0, 0, -3.5])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[5, 5, 4], pt_offset=pt_offset)
        return branch_pts

    branch_config = {'n': 5, 'spawn_kargs': lambda idx: {'rng': [.5, .8]},
                     'path_kargs': lambda idx: {'n_pts': 5, 'sz': .4, 'std': 1.4, 'momentum': .4},
                     'children': []}
    tree_config = {'n': 4,
                   'path_kargs': lambda idx: ({'n_pts': 15, 'sz': .8, 'std': 1, 'momentum': .7}
                                              if idx > 0 else
                                              {'n_pts': 15, 'sz': 1, 'std': .1, 'momentum': .7}),
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': [branch_config]}

    twig_kargs = {'config': shrubtwig_config,
                  'radii_kargs': {'Max radius': .1},
                  'leaf_kargs': {'Density': 1, 'Min scale': .4, 'Max scale': .6, 'Multi inst': 2}
                  }
    tree_kargs = {'config': tree_config, 'init_pos': init_pos,
                  'radii_kargs': {'Min radius': .04, 'Exponent': 2},
                  'leaf_kargs': {'Density': 1, 'Min scale': .35, 'Max scale': .45},
                  'space_kargs': {'atts': init_att_fn, 'D': .3, 's': .4, 'd': 10,
                                  'pull_dir': [0, 0, .5], 'n_steps': 20},
                  'root_kargs': None #{'atts': None, 'D': .2, 's': .3, 'd': 2,
                                 #'dir_rand': .3, 'mag_rand': .2,
                                 #'pull_dir': None, 'n_steps': 30},
                  }

    return tree_kargs, twig_kargs


def palm_tree(init_pos=np.array([[0, 0, 0]])):
    def tmp_att_fn(nodes):
        # pt_offset = init_pos[0] + np.array([0,0,20])
        pt_offset = nodes[-1]
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[1, 1, 1], pt_offset=pt_offset)
        return branch_pts

    # select a random horizontal angle
    pull_angle = np.random.uniform(0.0, 2*np.pi)

    tree_config = {'n': 1,
                   'path_kargs': lambda idx: {'n_pts': 20, 'sz': .8, 'std': 0.1, 'momentum': 0.95, 'pull_dir': [np.cos(pull_angle), np.sin(pull_angle), 0.0], 'pull_factor': np.random.uniform(0., 1.5), 'pull_init': 0.0},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': []}

    leaf_kargs = {'leaf_width': .1, 'alpha': 0.3, 'use_wave': False}
    twig_kargs = {'config': palmtwig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'max_density': 20, 'scale': 2.0, 'rot_x': (-0.5, -0.4), 'rot_z': (-0.1, 0.1)}
                  }
    tree_kargs = {'config': tree_config, 'D_': .3, 's': .4, 'd': 10, 'init_pos': init_pos,
                  'pull_dir': [0, 0, .5], 'n_updates': 20, 'init_att_fn': tmp_att_fn,
                  'radii_kargs': {'max_radius': 0.7, 'merge_size': .3, 'min_radius': 0.1, 'growth_amt': 1.01},
                  'leaf_kargs': {'max_density': 20, 'scale': .3, 'rot_x': (-1.0, 1.0), 'rot_z': (-0.1, 0.1)}
                  }

    return tree_kargs, twig_kargs, leaf_kargs


def baobab_tree(init_pos=np.array([[0, 0, 0]])):
    def tmp_att_fn(nodes):
        # pt_offset = init_pos[0] + np.array([0,0,20])
        pt_offset = nodes[-1]
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=50,
                                             scaling=[7, 7, 1], pt_offset=pt_offset)
        return branch_pts

    # select a random horizontal angle
    pull_angle = np.random.uniform(0.0, 2*np.pi)

    tree_config = {'n': 1,
                   'path_kargs': lambda idx: {'n_pts': 20, 'sz': .8, 'std': 0.1, 'momentum': 0.95},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': []}

    leaf_kargs = {'leaf_width': .5, 'alpha': 0.3, 'use_wave': False}
    twig_kargs = {'config': shrubtwig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'max_density': 20, 'scale': 0.5, 'rot_x': (-0.5, -0.4), 'rot_z': (-0.1, 0.1)}
                  }
    tree_kargs = {'config': tree_config, 'D_': .5, 's': .6, 'd': 10, 'init_pos': init_pos,
                  'pull_dir': [0, 0, .5], 'n_updates': 20, 'init_att_fn': tmp_att_fn,
                  'radii_kargs': {'max_radius': 2.0, 'merge_size': .3, 'min_radius': 0.1, 'growth_amt': 1.10},
                  'leaf_kargs': {'max_density': 30, 'scale': 0.7, 'rot_x': (0, 1.0), 'rot_z': (-1.0, 1.0)}
                  }

    return tree_kargs, twig_kargs, leaf_kargs


def bamboo_tree(init_pos=np.array([[0, 0, 0]])):
    height = np.random.randint(25, 35)

    def tmp_att_fn(nodes):
        # pt_offset = init_pos[0] + np.array([0,0,20])
        pt_offset = nodes[-1]
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=50,
                                             scaling=[0.5, 0.5, 4])
        # rotate the points
        rot_axis = (nodes[-1] - nodes[-2]) / \
            np.linalg.norm((nodes[-1] - nodes[-2]))
        rot_axis = (rot_axis + np.array([0, 0, 1])) / 2.

        branch_pts = np.array([helper.rodrigues_rot(
            pts, rot_axis, np.pi) for pts in branch_pts])

        branch_pts += pt_offset

        return branch_pts

    # select a random horizontal angle
    pull_angle = np.random.uniform(0.0, 2*np.pi)

    tree_config = {
        'n': 1,
        'path_kargs': lambda idx: {
            'n_pts': height, 'sz': .8, 'std': 0.1, 'momentum': 0.95,
            'pull_dir': [np.cos(pull_angle), np.sin(pull_angle), 0.0],
            'pull_factor': np.random.uniform(0.1, 0.6), 'pull_init': 0.0},
        'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
        'children': []
    }

    leaf_kargs = {'leaf_width': .1, 'alpha': 0.3, 'use_wave': False}
    twig_kargs = {'config': bambootwig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'max_density': 20, 'scale': 1.5, 'rot_x': (-0.5, -0.4), 'rot_z': (-0.1, 0.1)}
                  }
    tree_kargs = {'config': tree_config,
                  'D_': .3, 's': .4, 'd': 10, 'init_pos': init_pos,
                  'pull_dir': [0, 0, .5], 'n_updates': 20, 'init_att_fn': tmp_att_fn,
                  'radii_kargs': {'max_radius': 0.3, 'merge_size': .1, 'min_radius': 0.2, 'growth_amt': 1.01},
                  'leaf_kargs': {'max_density': 20, 'scale': .3, 'rot_x': (-1.0, 1.0), 'rot_z': (-0.1, 0.1)}
                  }

    return tree_kargs, twig_kargs, leaf_kargs


def shrub(init_pos=np.array([[0, 0, 0]]), shrub_shape=0):
    scale = 0.2



    def att_fn_ball(nodes):
        pt_offset = init_pos[0] + np.array([0, 0, 7*scale])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_uv_sphere_add, n=2000,
                                             scaling=[7*scale, 7*scale, 7*scale], pt_offset=pt_offset)
        return branch_pts

    def att_fn_cone(nodes):
        pt_offset = init_pos[0] + np.array([0, 0, 9*scale])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cone_add, n=2000,
                                             scaling=[5*scale, 5*scale, 10*scale], pt_offset=pt_offset)
        return branch_pts

    def att_fn_cube(nodes):
        pt_offset = init_pos[0] + np.array([0, 0, 9*scale])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=2000,
                                             scaling=[4*scale, 4*scale, 7*scale], pt_offset=pt_offset)
        return branch_pts

    if shrub_shape == 0:
        tmp_att_fn = att_fn_ball
    elif shrub_shape == 1:
        tmp_att_fn = att_fn_cone
    elif shrub_shape == 2:
        tmp_att_fn = att_fn_cube
    else:
        raise NotImplementedError

    leaf_kargs = {'leaf_width': np.random.rand() * .5 + .1,
                  'alpha': np.random.rand() * .3}
    branch_config = {'n': 5, 'spawn_kargs': lambda idx: {'rng': [.5, .8]},
                     'path_kargs': lambda idx: {'n_pts': 5, 'sz': .4, 'std': 1.4, 'momentum': .4},
                     'children': []}
    tree_config = {'n': 1,
                   'path_kargs': lambda idx: ({'n_pts': 3, 'sz': .8, 'std': 1, 'momentum': .7}
                                              if idx > 0 else
                                              {'n_pts': 3, 'sz': 1, 'std': .1, 'momentum': .7}),
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': [branch_config]}

    twig_kargs = TreeParams(
        skeleton=shrubtwig_config,
        trunk_spacecol=None,
        roots_spacecol=None,
        child_placement={'Density': 1, 'Min scale': .4, 'Max scale': .6, 'Multi inst': 2},
        skinning={'Max radius': .1}
    )

    tree_kargs = TreeParams(
        skeleton=tree_config,
        trunk_spacecol={'atts': tmp_att_fn, 'D': .3, 's': .4, 'd': 10},
        roots_spacecol=None,
        child_placement={'depth_range': (0, 2.7), 'Density': 0.7, 'Min scale': 1.2*scale, 'Max scale': 1.4*scale, 'Multi inst': 3, 'Pitch offset': 1., 'Pitch variance': 2., 'Yaw variance': 2.},
        skinning={'Min radius': 0.005, 'Max radius': 0.025, 'Exponent': 2}
    )

    return tree_kargs, twig_kargs, leaf_kargs

    # branch_config = {'n': 5, 'spawn_kargs': lambda idx: {'rng': [.5,.8]},
    #   'path_kargs': lambda idx: {'n_pts': 5, 'sz': .4, 'std': 1.4, 'momentum': .4},
    #   'children': []}
    # twig_config = {'n': 4,
    #   'path_kargs': lambda idx: ({'n_pts': 15, 'sz': .8, 'std': 1, 'momentum': .7}
    #                              if idx > 0 else
    #                              {'n_pts': 15, 'sz': 1, 'std': .1, 'momentum': .7}),
    #   'spawn_kargs': lambda idx: {'init_vec': [0,0,1]},
    #   'children': [branch_config]}

    # twig_kargs = {'config': shrubtwig_config,
    #               'radii_kargs': {'Max radius': .1},
    #               'leaf_kargs': {'Density': 1, 'Min scale': .4, 'Max scale': .6, 'Multi inst': 2}
    #               }
    # tree_kargs = {'config': twig_config, 'init_pos': init_pos,
    #               'radii_kargs': {'Min radius': .04, 'Exponent': 2},
    #               'leaf_kargs': {'Density': 1, 'Min scale': .35, 'Max scale': .45},
    #               'space_kargs': {'atts': init_att_fn, 'D': .3, 's': .4, 'd': 10,
    #                               'pull_dir': [0,0,.5], 'n_steps': 20},
    #               'root_kargs': {'atts': None, 'D': .2, 's': .3, 'd': 2,
    #                              'dir_rand': .3, 'mag_rand': .2,
    #                              'pull_dir': None, 'n_steps': 30},
    #               }


def basic_stem(init_pos=np.array([[0, 0, 0]])):
    branch_config = {'n': 3, 'spawn_kargs': lambda idx: {'rng': [.1 * (idx + 1), .1 * (idx + 2)]},
                     'path_kargs': lambda idx: {'n_pts': 20 - 2 * idx, 'sz': .5, 'std': 1.5,
                                                'momentum': .7, 'decay_mom': False,
                                                'pull_dir': [0, 0, 1], 'pull_factor': 1.5 + idx * .2},
                     'children': []}
    tree_config = {'n': 1,
                   'path_kargs': lambda idx: ({'n_pts': 30, 'sz': .5, 'std': 2,
                                               'momentum': .8, 'decay_mom': False,
                                               'pull_dir': [0, 0, 1], 'pull_factor': 2 + idx * .5}),
                   'spawn_kargs': lambda idx: {'init_vec': [np.random.randn(), np.random.randn(), 1]},
                   'children': [branch_config]}

    tree_kargs = {'config': tree_config, 'init_pos': init_pos,
                  'radii_kargs': {'Min radius': .02, 'Max radius': .1, 'Exponent': 2},
                  'leaf_kargs': {'Density': 0, 'Min scale': .35, 'Max scale': .45},
                  'space_kargs': {}, 'root_kargs': {},
                  }

    return tree_kargs, None, {}


def space_tree_wrap(cds, n_init=5):
    def tmp_att_fn(nodes):
        return cds

    tree_config = {'n': 1,
                   'path_kargs': lambda idx: {'n_pts': 1, 'sz': .8, 'std': 1, 'momentum': .7},
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]}}

    twig_kargs = {'config': twig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'max_density': 5, 'scale': .5}}
    tree_kargs = {'config': tree_config, 'D_': .15, 's': .2, 'd': 0.5, 'dir_rand': .3, 'mag_rand': .2,
                  'pull_dir': [0, 0, 0], 'n_updates': 40, 'init_att_fn': tmp_att_fn,
                  'radii_kargs': {'max_radius': .04, 'merge_size': 0.1, 'min_radius': .01, 'growth_amt': 1.02},
                  'leaf_kargs': {}}

    rand_pts = np.random.choice(np.arange(len(cds)), n_init, replace=False)
    tree_kargs['init_pos'] = cds[rand_pts]

    return tree_kargs, twig_kargs


def space_tree(obj, init_pos=np.array([[0, 0, 0]])):
    def init_att_fn(nodes):
        return mesh.sample_vtxs(obj, n=1000, emit_from="VOLUME", seed=np.random.randint(100))

    twig_kargs = {'config': shrubtwig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'Density': 1, 'Min scale': .4, 'Max scale': .6}}
    tree_kargs = {'config': {'n': 0}, 'init_pos': init_pos,
                  'leaf_kargs': {'Density': 0},
                  'radii_kargs': {'Min radius': .01, 'Scaling': .05, 'Exponent': 2},
                  'space_kargs': {'atts': init_att_fn, 'D': .1, 's': .2, 'd': 10,
                                  'dir_rand': .2, 'mag_rand': .2,
                                  'pull_dir': [0, .5, 0], 'n_steps': 100},
                  }

    return tree_kargs, twig_kargs


def pine_tree(init_pos=np.array([[0, 0, 0]])):
    
    def tmp_att_fn(nodes):
        tmp_v = nodes[nodes[:, 2] > 3]
        atts = [tmp_v.copy() + np.random.randn(*tmp_v.shape)
                * .5 for _ in range(5)]
        return np.concatenate(atts, 0)[::5]

    def root_att_fn(nodes):
        # Pass this into root_kargs to initialize a root system
        pt_offset = init_pos[0] + np.array([0, 0, -3.5])
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[5, 5, 4], pt_offset=pt_offset)
        return branch_pts

    per_layer = 4
    tree_ht = np.random.randint(20, 30)
    max_sz = .8
    start_ht = int(tree_ht * np.random.uniform(0.1, 0.3))
    n = tree_ht - start_ht

    branch_config = {'n': n * per_layer,
                     'path_kargs': lambda idx: {'n_pts': np.random.randint(np.floor(((n - idx // per_layer) / n) * 6),
                                                                           np.ceil(((n - idx // per_layer) / n) * 8)) + 3,
                                                'std': .3, 'momentum': .9, 'sz': max_sz - (max_sz / tree_ht) * (idx // per_layer)},
                     'spawn_kargs': lambda idx: {'rng': [.5, 1], 'z_bias': .2, 'rnd_idx': (idx // per_layer)+start_ht,
                                                 'ang_min': np.pi/2, 'ang_max': np.pi/2 + np.pi/16,
                                                 'axis2': [np.random.randn(), np.random.randn(), .5]},
                     'children': []
                     }
    pinetree_config = {'n': 1,
                       'path_kargs': lambda idx: {'n_pts': tree_ht + 1, 'sz': .8, 'std': 0.1, 'momentum': .7},
                       'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                       'children': [branch_config]}

    leaf_kargs = {'leaf_width': .05, 'alpha': 0, 'use_wave': False}
    twig_kargs = TreeParams(
        skeleton=pinetwig_config,
        trunk_spacecol=None, roots_spacecol=None,
        skinning={'Min radius': .005, 'Max radius': 0.03, 'Exponent': 1.3, 'Scaling': 0.1, 'Profile res': 3},
        child_placement={'depth_range': (0, 5.0), 'Density': 1.0, 'Min scale': .7, 'Max scale': .9},
    )

    tree_kargs = TreeParams(
        skeleton=pinetree_config,
        skinning={'Min radius': 0.02, 'Exponent': 1.5, 'Max radius': 0.2},
        trunk_spacecol={'atts': tmp_att_fn, 'D': .3, 's': .4, 'd': 10,
                                  'pull_dir': [0, 0, .5], 'n_steps': 20},
        roots_spacecol=None,#{'atts': None, 'D': .2, 's': .3, 'd': 2,
                            #     'dir_rand': .3, 'mag_rand': .2,
                            #     'pull_dir': None, 'n_steps': 30},
        child_placement={'depth_range': (0, 2.7), 'Density': 1.0, 'Min scale': .7, 'Max scale': .9}                         
    )

    return tree_kargs, twig_kargs, leaf_kargs

def coral():
    def tmp_att_fn(nodes):
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[7, 7, 7], pt_offset=[0, 0, 11])
        return branch_pts

    branch_config = {'n': 5, 'spawn_kargs': lambda idx: {'rng': [.5, .8]},
                     'path_kargs': lambda idx: {'n_pts': 5, 'sz': .4, 'std': 1.4, 'momentum': .4},
                     'children': []}
    tree_config = {'n': 4,
                   'path_kargs': lambda idx: ({'n_pts': 15, 'sz': .8, 'std': 1, 'momentum': .7}
                                              if idx > 0 else
                                              {'n_pts': 15, 'sz': 1, 'std': .1, 'momentum': .7}),
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': [branch_config]}

    twig_kargs = {'config': twig_config,
                  'radii_kargs': {'max_radius': .1, 'merge_size': .2},
                  'leaf_kargs': {'max_density': 20, 'scale': .4}}
    tree_kargs = {'config': tree_config, 'D_': .3, 's': .4, 'd': 10,
                  'pull_dir': [0, 0, .5], 'n_updates': 20, 'init_att_fn': tmp_att_fn,
                  'radii_kargs': {'max_radius': .7, 'merge_size': .3, 'min_radius': .03, 'growth_amt': 1.01},
                  'leaf_kargs': {'max_density': 5, 'scale': .3}}

    return tree_kargs, twig_kargs


def parse_genome(tree_genome):
    genome_keys = ['size', 'trunk_warp', 'n_trunks',
                   'branch_start', 'branch_angle', 'multi_branch',
                   'branch density', 'branch_len',
                   'branch_warp', 'pull_dir_vt',
                   'pull_dir_hz', 'outgrowth', 'branch_thickness',
                   'twig_density', 'twig_scale']
    return {k: tree_genome[k_idx] for k_idx, k in enumerate(genome_keys)}


def calc_height(x, min_ht=5, max_ht=30, bias=-.05, uniform=.5):
    def map_fn(val): return np.tan((val-.5+bias)*np.pi*(1.1-uniform))
    rng = map_fn(0), map_fn(1)
    y = map_fn(x)
    y = (y - rng[0]) / (rng[1] - rng[0])
    y = y * (max_ht - min_ht) + min_ht
    return y


def generate_tree_config(tree_genome=None, season='autumn'):
    """
    Main latent params that we might want to control:
    - overall size/"age"
    - trunk straightness
    - additional "trunks"
    - starting height of branches
    - outgoing branch angle (parallel to ground vs angled up vs angled proporitionally to height)
    - branch density
    - branch length (fn of height)
    - branch straightness
    - pull direction (up/down/to the side)
    - outgrowth (space filling) / "density"
    - branch thickness (ideally this behaves reasonably based on everything else)
    """
    if tree_genome is None:
        tree_genome = np.random.rand(32)

    cfg = parse_genome(tree_genome)
    sz = calc_height(cfg['size'], min_ht=12)
    n_tree_pts = int(sz)
    n_trunks = int(10 ** (cfg['n_trunks']**1.6))
    ex = np.exp((6 - (5 if n_trunks > 1 else 0)) * (cfg['trunk_warp']-.1))
    trunk_std = ((1 - (ex / (1 + ex)))*4) ** 2
    trunk_mtm = max(.2, min(.95, (1 / (trunk_std + 1)) +
                    np.random.randn() * .2))
    radial_out = False  # False # np.random.rand() < .3
    avail_idxs = np.arange(n_tree_pts)
    start_idx = 1 + int(n_tree_pts * np.random.uniform(.1, .7))
    sample_density = np.random.choice(
        np.arange(np.ceil(np.sqrt(n_tree_pts)), dtype=int) + 1)
    avail_idxs = avail_idxs[start_idx::sample_density]
    multi_branch = int(5 ** (cfg['multi_branch']**1.6))
    avail_idxs = np.repeat(avail_idxs, multi_branch).flatten()

    n = len(avail_idxs)

    start_ht = sz * (start_idx / sz)
    box_ht = (sz - start_ht) * .6

    def tmp_att_fn(nodes):
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[sz/2, sz/2, box_ht], pt_offset=[0, 0, start_ht + sz * .4])
        return branch_pts

    max_sz = 1

    if radial_out:
        start_ht = int(sz * .1)
        per_layer = np.random.randint(3, 6)
        branch_config = {'n': n * per_layer,
                         'path_kargs': lambda idx: {'n_pts': np.random.randint(np.floor(((n - idx // per_layer) / n) * 6),
                                                                               np.ceil(((n - idx // per_layer) / n) * 8)) + 3,
                                                    'std': .3, 'momentum': .9, 'sz': max_sz - (max_sz / sz) * (idx // per_layer),
                                                    'pull_dir': [0, 0, np.random.rand()],
                                                    'pull_factor': np.random.rand()},
                         'spawn_kargs': lambda idx: {'rnd_idx': avail_idxs[idx // per_layer],
                                                     'ang_min': np.pi/2, 'ang_max': np.pi/2 + np.pi/16,
                                                     'axis2': [np.random.randn(), np.random.randn(), .5]}}

    else:
        branch_config = {'n': n,
                         'path_kargs': lambda idx: {'n_pts': int(n_tree_pts*np.random.uniform(.4, .6)),
                                                    'sz': 1, 'std': 1.4, 'momentum': .4,
                                                    'pull_dir': [0, 0, np.random.rand()],
                                                    'pull_factor': np.random.rand()},
                         'spawn_kargs': lambda idx: {'rnd_idx': avail_idxs[idx]}}

    tree_config = {'n': n_trunks,
                   'path_kargs': lambda idx: ({'n_pts': n_tree_pts, 'sz': 1,
                                               'std': trunk_std, 'momentum': trunk_mtm,
                                               'pull_dir': [0, 0, 0]}),
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': [branch_config]
                   }

    tmp_D = .3 + .2 * (sz / 30)  # .3 * sz / 8
    tmp_s = tmp_D * 1.3
    if n < 5:
        n_updates = np.random.choice([2, 3, int(1 + sz // 2)])
    else:
        n_updates = np.random.choice([2, 2, 2, 3, 4, 5])

    max_radius = 0.2
    merge_size = 2.5 - cfg['branch_thickness']

    if season == "winter":
        twig_density = 0.0 if cfg['twig_density'] < 0.5 else 0.5 * cfg['twig_density']
        twig_inst = 1 + 0 * np.random.randint(3, 5)
    else:
        twig_density = 0.5 + 0.5 * cfg['twig_density']
        twig_inst = np.random.randint(1, 3)

    return TreeParams(
        skeleton=tree_config,
        skinning={'Max radius': max_radius, 'Min radius': .02, 'Exponent': merge_size},
        trunk_spacecol={'atts': tmp_att_fn, 'D': tmp_D, 's': tmp_s, 'd': 10,
                            'pull_dir': [0, 0, np.random.randn() * .3], 'n_steps': n_updates},
        roots_spacecol=None, #{'atts': None, 'D': .05, 's': .1, 'd': 2, 'dir_rand': .05, 'mag_rand': .05, 'pull_dir': None, 'n_steps': 30},
        child_placement={'depth_range': (0, 5.0), 'Density': twig_density, 'Multi inst': twig_inst,
                            'Pitch variance': 1.0, 'Yaw variance': 10.0, 'Min scale': 1.1, 'Max scale': 1.3}                    
    )


def random_tree(tree_genome=None, season='autumn'):
    leaf_kargs = {'leaf_width': np.random.rand() * .5 + .1,
                  'alpha': np.random.rand() * .3}

    if season == "winter":
        leaf_density = np.random.uniform(.0, 0.1)
        leaf_inst = 1
    elif season == "spring": # flowers should be less dense
        leaf_density = np.random.uniform(.3, 0.7)
        leaf_inst = 2
    else:
        leaf_density = np.random.uniform(.4, 1.0)
        leaf_inst = 3


    twig_kargs = TreeParams(
        skeleton=generate_twig_config(),
        skinning={'Max radius': 0.01, 'Min radius': 0.005},
        trunk_spacecol=None,
        roots_spacecol=None,
        child_placement={'Density': leaf_density, 'Multi inst': leaf_inst,
                                 'Min scale': .3, 'Max scale': .4}
    )
    tree_kargs = generate_tree_config(tree_genome, season=season)
    return tree_kargs, twig_kargs, leaf_kargs


def generate_coral_config(tree_genome=None):
    """
    Main latent params that we might want to control:
    - overall size/"age"
    - trunk straightness
    - additional "trunks"
    - starting height of branches
    - outgoing branch angle (parallel to ground vs angled up vs angled proporitionally to height)
    - branch density
    - branch length (fn of height)
    - branch straightness
    - pull direction (up/down/to the side)
    - outgrowth (space filling) / "density"
    - branch thickness (ideally this behaves reasonably based on everything else)
    """
    if tree_genome is None:
        tree_genome = np.random.rand(32)

    cfg = parse_genome(tree_genome)
    sz = calc_height(cfg['size'])
    n_tree_pts = int(sz)
    n_trunks = np.random.randint(5, 20)  # int(10 ** (cfg['n_trunks']**1.6))
    ex = np.exp((6 - (5 if n_trunks > 1 else 0)) * (cfg['trunk_warp']-.1))
    trunk_std = ((1 - (ex / (1 + ex)))*4) ** 2
    trunk_mtm = max(.2, min(.95, (1 / (trunk_std + 1)) +
                    np.random.randn() * .2))
    radial_out = False  # np.random.rand() < .3
    avail_idxs = np.arange(n_tree_pts)
    start_idx = 1 + int(n_tree_pts * np.random.uniform(0, .7))
    sample_density = np.random.choice(
        np.arange(np.ceil(np.sqrt(n_tree_pts)), dtype=int) + 1)
    avail_idxs = avail_idxs[start_idx::sample_density]
    multi_branch = int(5 ** (cfg['multi_branch']**1.6))
    avail_idxs = np.repeat(avail_idxs, multi_branch).flatten()

    n = 0  # len(avail_idxs)

    start_ht = sz * (start_idx / sz) + 1
    box_ht = (sz - start_ht) * .6

    def tmp_att_fn(nodes):
        branch_pts = mesh.get_pts_from_shape(bpy.ops.mesh.primitive_cube_add, n=500,
                                             scaling=[sz/2, sz/2, box_ht], pt_offset=[0, 0, start_ht + sz * .4])
        return branch_pts

    max_sz = 1

    branch_config = {'n': n,
                     'path_kargs': lambda idx: {'n_pts': int(n_tree_pts*np.random.uniform(.4, .6)),
                                                'sz': 1, 'std': .4, 'momentum': .8,
                                                'pull_dir': [0, 0, np.random.rand()],
                                                'pull_factor': np.random.rand()},
                     'spawn_kargs': lambda idx: {'rnd_idx': avail_idxs[idx]}}

    tree_config = {'n': n_trunks,
                   'path_kargs': lambda idx: ({'n_pts': n_tree_pts, 'sz': 1,
                                               'std': trunk_std, 'momentum': trunk_mtm,
                                               'pull_dir': [0, 0, 1]}),
                   'spawn_kargs': lambda idx: {'init_vec': [0, 0, 1]},
                   'children': [branch_config]
                   }

    tmp_D = .3 + .2 * (sz / 30)  # .3 * sz / 8
    tmp_s = tmp_D * 1.3
    if n < 5:
        n_updates = np.random.choice([2, 3, int(1 + sz // 2)])
    else:
        n_updates = np.random.choice([2, 2, 2, 3, 4, 5])
    # print(sz, n_updates)
    n_updates = 3
    max_radius = .3  # 00
    merge_size = np.random.uniform(.2, .7)
    growth_amt = 1.01

    return {'config': tree_config, 'D_': tmp_D, 's': tmp_s, 'd': 10,
            'pull_dir': [0, 0, np.random.randn() * .3],
            # np.random.randint(15) + 3,
            'init_att_fn': tmp_att_fn, 'n_updates': n_updates,
            'radii_kargs': {'max_radius': max_radius, 'merge_size': merge_size, 'min_radius': .2, 'growth_amt': growth_amt},
            'leaf_kargs': {'max_density': 0 if np.random.rand() < .1 else np.random.uniform(5, 20),
                           'scale': np.random.uniform(.5, 1)},
            }


def random_coral(genome=None):
    leaf_kargs = {}
    twig_kargs = {}
    tree_kargs = generate_coral_config(genome)
    return tree_kargs, twig_kargs, leaf_kargs
