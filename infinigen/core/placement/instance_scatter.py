# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Lahav Lipson


from math import prod
import logging

import bpy
from mathutils import Vector

import numpy as np

from infinigen.assets.utils.misc import CountInstance
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.placement.camera import nodegroup_active_cam_info

logger = logging.getLogger(__name__)

def _less(a, b, nw):
    return nw.new_node(Nodes.Compare, [a, b], attrs={"data_type": "FLOAT", "operation": "LESS_THAN"})


def _greater(a, b, nw):
    return nw.new_node(Nodes.Compare, [a, b], attrs={"data_type": "FLOAT", "operation": "GREATER_THAN"})


def _band(a, b, nw):
    return nw.new_node(Nodes.BooleanMath, [a, b], attrs={"operation": "AND"})


def _in_bucket(val, lower, upper, nw):
    lt = _less(val, upper, nw)
    gt = _greater(val, lower, nw)
    return _band(gt, lt, nw)


def _valnode(label, val, nw):
    node = nw.new_node(Nodes.Value, label=label)
    node.outputs[0].default_value = val
    return node


def _vecnode(val, nw):
    return nw.new_node(Nodes.Vector, attrs={"vector": Vector(val)})


def camera_cull_points(nw, fov=25, camera=None, near_dist_margin=5):
    instance_position = nw.new_node(Nodes.InputPosition)
    if camera is None:
        camera = bpy.context.scene.camera
    camera_info = nw.new_node(nodegroup_active_cam_info().name)

    distance = nw.new_node(Nodes.VectorMath, [instance_position, camera_info], attrs={"operation": "DISTANCE"})
    pt_to_cam = nw.new_node(Nodes.VectorMath, [instance_position, camera_info], attrs={"operation": "SUBTRACT"})
    pt_to_cam_normalized = nw.new_node(Nodes.VectorMath, [pt_to_cam], attrs={"operation": "NORMALIZE"})
    cam_dir = nw.new_node(Nodes.VectorRotate, attrs={"rotation_type": 'EULER_XYZ'},
                          input_kwargs={"Vector": _vecnode((0., 0., -1.), nw),
                                        "Rotation": (camera_info, "Rotation")})
    dot_prod = nw.new_node(Nodes.VectorMath, [pt_to_cam_normalized, cam_dir], {"operation": "DOT_PRODUCT"})
    angle_rad = nw.new_node(Nodes.Math, [dot_prod], {"operation": "ARCCOSINE"})
    angle_deg = nw.new_node(Nodes.Math, [angle_rad], {"operation": "DEGREES"})
    visible = nw.new_node(Nodes.BooleanMath, [_less(angle_deg, fov, nw), _less(distance, near_dist_margin, nw)], 
                          attrs={'operation': 'OR'})

    return visible, distance

def bucketed_instance(nw, points, collection, distance, buckets, selection, scaling, rotation, instance_index=None):
    instance_index = {'Instance Index': surface.eval_argument(nw, instance_index, n=len(
        collection.objects))} if instance_index is not None else {}
    collection_info = nw.new_node(Nodes.CollectionInfo, [collection, True, True])

    instance_groups = []
    prev_upper_val = 0
    for idx, (cutoff, merge_dist) in enumerate(buckets):
        if idx != 0:
            prev_upper_val = nw.expose_input(f"Cutoff_{idx}", val=buckets[idx - 1][0])
        upper_val = nw.expose_input(f"Cutoff_{idx + 1}", val=cutoff)

        distance_thresh = _in_bucket(distance, prev_upper_val, upper_val, nw)
        lower_res_collection = nw.new_node(Nodes.MergeByDistance, [collection_info], input_kwargs={
            "Distance": nw.expose_input(f"Merge_By_Dist_{idx + 1}", merge_dist)})
        separate_points = nw.new_node(Nodes.SeparateGeometry, [points, distance_thresh],
                                      attrs={"domain": "POINT"})
        instance_on_points = nw.new_node(Nodes.InstanceOnPoints, [separate_points],
                                         input_kwargs={"Instance": collection_info, "Pick Instance": True,
                                                       **instance_index, "Scale": scaling, "Selection": selection,
                                                       "Rotation": rotation})
        instance_groups.append(instance_on_points)

    return nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': instance_groups})

def geo_instance_scatter(
    nw: NodeWrangler, base_obj, collection, density, 
    fov=None, dist_max=None, no_culling_dist=5, min_spacing=0,
    scaling=Vector((1, 1, 1)), normal=None, normal_fac=1, rotation_offset=None,
    selection=True, taper_scale=False, taper_density=False,
    ground_offset=0, instance_index=None,
    transform_space='RELATIVE', reset_children=True
):
    base_geo = nw.new_node(Nodes.ObjectInfo, [base_obj], attrs={'transform_space':transform_space}).outputs['Geometry']

    overall_density = nw.expose_input("Overall Density", val=density)
    selection_val = surface.eval_argument(nw, selection)
    if isinstance(selection_val, tuple):
        selection_val, density_scalar = selection_val
    else:
        density_scalar = None
    scaling = surface.eval_argument(nw, scaling)

    if density_scalar is not None:
        if taper_density:
            overall_density = nw.new_node(Nodes.Math, [density_scalar, overall_density], attrs={'operation': 'MULTIPLY'})
        if taper_scale:
            scaling = nw.new_node(Nodes.VectorMath, input_kwargs={0: scaling, 'Scale': density_scalar}, attrs={'operation': 'SCALE'})

    points = nw.new_node(Nodes.DistributePointsOnFaces, 
        [base_geo], input_kwargs={"Density": overall_density, "Selection": selection_val})
    distribute_points = points
        
    if min_spacing > 0:
        points = nw.new_node(Nodes.MergeByDistance,
            input_kwargs={'Geometry': points, 'Distance': surface.eval_argument(nw, min_spacing)})

    point_fields = {}

    normal = (distribute_points, "Normal") if normal is None else surface.eval_argument(nw, normal)
    rotation_val = nw.new_node(Nodes.AlignEulerToVector, attrs={"axis": "Z"},
                                input_kwargs={"Factor": surface.eval_argument(nw, normal_fac), "Vector": normal})
    rotation_val = nw.new_node(Nodes.RotateEuler, [rotation_val], {"type": "AXIS_ANGLE", "space": "LOCAL"},
                            input_kwargs={"Axis": Vector((0., 0., 1.)), "Angle": nw.uniform(0, 1e4)})
    if rotation_offset is not None:
        rotation_val = nw.new_node(Nodes.RotateEuler, attrs=dict(space='OBJECT'), input_kwargs={
            'Rotation':surface.eval_argument(nw, rotation_offset), 'Rotate By':rotation_val})
    point_fields['rotation'] = (rotation_val, 'FLOAT_VECTOR')
    
    if instance_index is not None:
        inst = surface.eval_argument(nw, instance_index, n=len(collection.objects))
        point_fields['instance_index'] = (inst, 'INT')
    if scaling is not None:
        point_fields['scaling'] = (surface.eval_argument(nw, scaling), 'FLOAT_VECTOR')
    if ground_offset != 0:
        point_fields['ground_offset'] = (surface.eval_argument(nw, ground_offset), 'FLOAT')

    if dist_max is not None or fov is not None:
        
        for k, (soc, dtype) in point_fields.items():
            points = nw.new_node(Nodes.CaptureAttribute, input_kwargs={'Geometry': points, 'Value': soc}, attrs={'data_type': dtype})
            point_fields[k] = points

        # camera-based culling
        visible, distance = camera_cull_points(nw, fov=nw.expose_input("FOV", val=fov), near_dist_margin=no_culling_dist)
        points = nw.new_node(Nodes.SeparateGeometry, 
            [points, visible], attrs={"domain": "POINT"})
        if dist_max is not None:
            in_range = _less(distance, dist_max)
            points = nw.new_node(Nodes.SeparateGeometry, 
                [points, in_range], attrs={"domain": "POINT"})
    else:
        for k, v in point_fields.items():
            point_fields[k] = v[0]
    
    collection_info = nw.new_node(Nodes.CollectionInfo, [collection, True, reset_children])

    instances = nw.new_node(Nodes.InstanceOnPoints, [points], input_kwargs={
        "Instance": collection_info, "Pick Instance": True,
        'Instance Index': point_fields.get('instance_index'), 
        "Rotation": point_fields.get('rotation'),
        "Scale": point_fields.get('scaling')})
    
    if ground_offset != 0:
        instances = nw.new_node(Nodes.TranslateInstances, [instances],
            input_kwargs={ "Translation": nw.combine(0, 0, point_fields['ground_offset']), "Local Space": True})

    instances = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': instances, 'Shade Smooth': False})

    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': instances})

def scatter_instances(
    collection, 
    density=None, vol_density=None, max_density=5000,
    scale=None, scale_rand=0, scale_rand_axi=0,
    **kwargs
):
    
    if np.sum([density is None, vol_density is None]) != 1:
        raise ValueError(f'Scatter instances got {density=} and {vol_density=} expected only one of the three')

    name = 'scatter:' + collection.name.split(':')[-1]

    avg_scale = scale * (1 - scale_rand/2) * (1 - scale_rand_axi/2) 

    if vol_density is not None:
        assert scale is not None, 'Cannot compute expected collection vol when using legacy scaling= func'
        assert density is None # ensured by check above

        avg_vol = np.mean([prod(list(o.dimensions)) for o in collection.objects])
        density = vol_density / (avg_vol * avg_scale ** 2) # TODO cube power?

        if density > max_density:
            logger.warning(f'scatter_instances with {collection.name=} {vol_density=} {avg_scale=:.4f} {avg_vol=:.4f} attempted {density=:.4f}, clamping to {max_density=}')
            density = max_density
        
    if scale is not None:
        assert 'scaling' not in kwargs
        def scaling(nw: NodeWrangler):
            axis_scaling = nw.new_node(Nodes.RandomValue,
                input_kwargs={0: 3*(1-scale_rand_axi,), 1:3*(1,)},
                attrs={"data_type": 'FLOAT_VECTOR'})
            overall = nw.uniform(1 - scale_rand, 1)
            return nw.multiply(axis_scaling, overall, 3*(scale,))
        kwargs['scaling'] = scaling

    scatter_obj = butil.spawn_vert(name)
    kwargs.update(dict(collection=collection, density=density))
    with CountInstance(name):
        surface.add_geomod(scatter_obj, geo_instance_scatter, apply=False, input_kwargs=kwargs)
    butil.put_in_collection(scatter_obj, butil.get_collection('scatters'))
    return scatter_obj