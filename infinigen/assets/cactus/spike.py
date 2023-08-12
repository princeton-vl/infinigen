# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei


import colorsys
import numpy as np
from numpy.random import uniform

from infinigen.core.util import blender as butil
from infinigen.assets.utils.misc import sample_direction
from infinigen.assets.utils.decorate import assign_material
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.core.placement.factory import AssetFactory, make_asset_collection
from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes
from infinigen.core import surface
from infinigen.assets.trees.tree import build_radius_tree
import infinigen.core.util.blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.assets.utils.tag import tag_object, tag_nodegroup

def build_spikes(base_radius=.002, **kwargs):
    n_branch = 4
    n_major = 9
    branch_config = {
        'n': n_branch,
        'path_kargs': lambda idx: {'n_pts': n_major, 'std': .5, 'momentum': .85, 'sz': uniform(.005, .01)},
        'spawn_kargs': lambda idx: {'init_vec': sample_direction(.8)}
    }

    radius_fn = lambda base_radius, size, resolution: base_radius * .5 ** (
            np.arange(size * resolution) / (size * resolution))
    obj = build_radius_tree(radius_fn, branch_config, base_radius)
    surface.add_geomod(obj, geo_radius, apply=True, input_args=['radius', None, .001])
    tag_object(obj, 'spike')
    return obj


def make_default_selections(spike_distance, cap_percentage, density):
    def selection(nw: NodeWrangler, selected, geometry):
        z = nw.separate(nw.new_node(Nodes.InputPosition))[-1]
        z_stat = nw.new_node(Nodes.AttributeStatistic, [geometry, None, z]).outputs
        percentage = nw.scalar_divide(nw.scalar_sub(z_stat['Max'], z), z_stat['Range'])
        is_cap = nw.bernoulli(nw.build_float_curve(percentage, [(0, 1), (cap_percentage, .5), (1, 0)]))
        cap = nw.new_node(Nodes.SeparateGeometry, [geometry, is_cap])
        cap = nw.new_node(Nodes.MergeByDistance, [cap, None, spike_distance / 2])

        points = nw.new_node(Nodes.DistributePointsOnFaces,
                             input_kwargs={'Mesh': geometry, 'Selection': selected, 'Density': density
                             }).outputs['Points']
        points = nw.new_node(Nodes.MergeByDistance, [points, None, spike_distance])

        all_points = nw.new_node(Nodes.JoinGeometry, [[cap, points]])
        return all_points

    return selection


def geo_spikes(nw: NodeWrangler, spikes, points_fn=None, realize=True):
    geometry, selection = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None),
        ('NodeSocketFloat', 'Selection', None)]).outputs[:2]
    capture = nw.new_node(Nodes.CaptureAttribute,
                          input_kwargs={'Geometry': geometry, 'Value': nw.new_node(Nodes.InputNormal)})

    selected = nw.compare('GREATER_THAN', selection, .8)
    spikes = nw.new_node(Nodes.CollectionInfo, [spikes, True, True])

    rotation = nw.new_node(Nodes.AlignEulerToVector, input_kwargs={'Vector': (capture, 'Attribute')},
                           attrs={'axis': 'Z'})
    rotation = nw.new_node(Nodes.RotateEuler,
                           input_kwargs={'Rotation': rotation, 'Angle': nw.uniform(0, 2 * np.pi)},
                           attrs={'type': 'AXIS_ANGLE', 'space': 'LOCAL'})
    rotation = nw.new_node(Nodes.AlignEulerToVector, [rotation, nw.uniform(.2, .5)], attrs={'axis': 'Z'})
    rotation = nw.add(rotation, nw.uniform([-.05] * 3, [.05] * 3))

    points = surface.eval_argument(nw, points_fn, selected=selected, geometry=capture.outputs['Geometry'])
    spikes = nw.new_node(Nodes.InstanceOnPoints, input_kwargs={
        'Points': points,
        'Instance': spikes,
        'Pick Instance': True,
        'Rotation': rotation,
        'Scale': nw.uniform([.5] * 3, [1.] * 3)
    })
    if realize:
        realize_instances = nw.new_node(Nodes.RealizeInstances, [spikes])
    else:
        realize_instances = spikes
    
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': realize_instances})


def shader_spikes(nw: NodeWrangler):
    roughness = .8
    specular = .25
    mix_ratio = .9
    color = *colorsys.hsv_to_rgb(uniform(.2, .4), uniform(.1, .3), .8), 1
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF, input_kwargs={
        'Base Color': color,
        'Roughness': roughness,
        'Specular': specular,
        'Subsurface': .1
    })
    transparent_bsdf = nw.new_node(Nodes.TranslucentBSDF, [color])
    mix_rgb = nw.new_node(Nodes.MixShader, [mix_ratio, principled_bsdf, transparent_bsdf])
    return mix_rgb


def apply(obj, points_fn, base_radius=.002, realize=True):
    spikes = deep_clone_obj(obj)
    instances = make_asset_collection(build_spikes, 5, 'spikes', verbose=False, base_radius=base_radius)
    mat = surface.shaderfunc_to_material(shader_spikes)
    for o in instances.objects:
        assign_material(o, mat) 
    surface.add_geomod(spikes, geo_spikes, apply=realize, input_args=[instances, points_fn, realize],
                       input_attributes=[None, 'selection'])
    butil.delete_collection(instances)
    return spikes
