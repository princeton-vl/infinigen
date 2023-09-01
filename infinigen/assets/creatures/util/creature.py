# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
from dataclasses import dataclass, field
import typing
import warnings
import logging

import bpy
import mathutils
from mathutils.bvhtree import BVHTree

import numpy as np

from infinigen.assets.creatures.util.geometry import lofting, skin_ops
from infinigen.assets.creatures.util.creature_util import interp_dict, euler
from infinigen.assets.creatures.util import genome
from infinigen.assets.creatures.util import tree, join_smoothing

from infinigen.assets.creatures.util import genome

from infinigen.core import surface
from infinigen.core.placement import detail
from infinigen.core.util import blender as butil, logging as logging_util
from infinigen.core.util.math import homogenize, lerp_sample, lerp
from infinigen.core.nodes.node_wrangler import Nodes

logger = logging.getLogger(__name__)


def infer_skeleton_from_mesh(obj):
    vs = np.array([v.co for v in obj.data.vertices]).reshape(-1, 3)
    try:
        v_xmin = vs[vs[:, 0].argmin()]
        v_xmax = vs[vs[:, 0].argmax()]
        return np.array([v_xmin, v_xmax])
    except ValueError:
        warnings.warn(f'infer_skeleton_from_mesh({obj=}) failed, returning null skeleton')
        return np.array([[0, 0, 0], [0.1, 0, 0]])


@dataclass
class Part:
    skeleton: np.array
    obj: bpy.types.Object
    attach_basemesh: bpy.types.Object = None

    joints: dict = field(default_factory=dict)
    iks: dict = field(default_factory=dict)

    settings: dict = field(default_factory=dict)
    _bvh: BVHTree = None
    side: int = 1

    def __post_init__(self):
        if self.joints is None:
            self.joints = {}
        if self.iks is None:
            self.iks = {}

    def bvh(self):
        if self._bvh is None:
            logger.debug(f'Computing part bvh for {self.obj.name}')
            target = self.attach_basemesh or self.obj
            assert target.type == 'MESH'
            depsgraph = bpy.context.evaluated_depsgraph_get()
            self._bvh = BVHTree.FromObject(target, depsgraph)
        return self._bvh

    def __repr__(self):
        return f'{self.__class__.__name__}(obj.name={repr(self.obj.name)}, skeleton.shape=' \
               f'{self.skeleton.shape if self.skeleton is not None else None})'

    def skeleton_global(self):
        return homogenize(self.skeleton) @ np.array(self.obj.matrix_world)[:-1].T


ALL_TAGS = ['body', 'neck', 'head', 'jaw', 'head_detail', 'limb', 'foot', 'rigid']


class PartFactory:

    def __init__(self, params=None, sample=True):
        if sample:
            self.params = self.sample_params()
            if params is not None:
                self.params.update(params)
        else:
            assert params is not None
            self.params = params

    def sample_params(self):
        raise NotImplementedError

    def make_part(self, params) -> Part:
        raise NotImplementedError

    def __call__(self, rand=0) -> Part:
        params = self.params

        if rand > 0:
            other_sample = self.sample_params()
            params = interp_dict(params, other_sample, rand)

        logger.debug(f'Computing {self}.make_part()')
        part = self.make_part(params)

        if part is None:
            raise ValueError(f'{self}.make_part() returned None, did you forget a return?')

        return part

    @staticmethod
    def animate_bones(arma, bones, params):
        return


def quat_align_vecs(a, b):
    assert a is not None
    assert b is not None

    if not isinstance(a, mathutils.Vector):
        a = mathutils.Vector(a)
    if not isinstance(b, mathutils.Vector):
        b = mathutils.Vector(b)

    return mathutils.Quaternion(a.cross(b), a.angle(b))


def raycast_surface(part: Part, idx_pct, dir_rot: mathutils.Quaternion, r=1, debug=False):
    # figure out axis of rotation
    idx = np.array([idx_pct]) * (len(part.skeleton) - 1)
    tangents = lofting.skeleton_to_tangents(part.skeleton)
    forward = lerp_sample(tangents, idx)

    # raycast to find surface of the part
    origin = mathutils.Vector(lerp_sample(part.skeleton, idx).reshape(-1))
    basis = part.obj.rotation_euler.to_quaternion() @ quat_align_vecs((1, 0, 0), forward.reshape(-1))
    direction = basis @ dir_rot @ mathutils.Vector([1, 0, 0])

    location, normal, index, dist = part.bvh().ray_cast(origin, direction)

    if location is None:
        logger.warning(f'Raycast did not intersect {part} with {dist=} {dir_rot=} {idx_pct=}')
        location = origin
        dist = 0
        normal = (1, 0, 0)
    elif debug:
        o = butil.spawn_empty('origin')
        o.location = origin

        d = butil.spawn_empty('dir')
        d.location = (origin + 0.05 * direction)

        e = butil.spawn_empty('hit')
        e.location = location

        for v in [o, d, e]:
            v.parent = part.obj

    location = part.obj.matrix_world @ mathutils.Vector(lerp(origin, location, r))

    return location, normal, forward.reshape(3)


def write_local_attributes(part, idx, tags):
    # local attributes must come before posing / attachment of parts
    assert part.obj.location == mathutils.Vector((0, 0, 0)), part.obj.location

    n = len(part.obj.data.vertices)

    # local position
    surface.write_attribute(part.obj, lambda nw: nw.new_node(Nodes.InputPosition), name='local_pos', apply=True)

    # float repr of integer part idx, useful after join/remesh
    part_idx_attr = part.obj.data.attributes.new('part_idx', 'FLOAT', 'POINT')
    part_idx_attr.data.foreach_set('value', np.full(n, idx))

    for t in tags:
        attr = part.obj.data.attributes.new(f'tag_{t}', 'FLOAT', 'POINT')
        attr.data.foreach_set('value', np.ones(n))


def write_global_attributes(part):
    skeleton = part.skeleton_global()
    verts = np.array([part.obj.matrix_world @ v.co for v in part.obj.data.vertices])

    dists = np.linalg.norm(skeleton.reshape(1, -1, 3) - verts.reshape(-1, 1, 3), axis=-1)
    closest_idx = dists.argmin(axis=1)

    rads = dists[np.arange(dists.shape[0]), closest_idx]
    rad_attr = part.obj.data.attributes.new('skeleton_rad', 'FLOAT', 'POINT')
    rad_attr.data.foreach_set('value', rads)

    # location of nearest skeleton point
    skeleton_loc_attr = part.obj.data.attributes.new('skeleton_loc', 'FLOAT_VECTOR', 'POINT')
    skeleton_loc_attr.data.foreach_set('vector', skeleton[closest_idx].reshape(-1))

    # location of the parent of the nearest skeleton point
    parent_loc = skeleton[np.clip(closest_idx - 1, 0, len(skeleton))]
    if skeleton.shape[0] > 1:
        parent_loc[closest_idx == 0] = skeleton[0] - (skeleton[1] - skeleton[0])
    parent_skeleton_loc_attr = part.obj.data.attributes.new('parent_skeleton_loc', 'FLOAT_VECTOR', 'POINT')
    parent_skeleton_loc_attr.data.foreach_set('vector', parent_loc.reshape(-1))


def sanitize_for_boolean(o):
    '''
    Attempt to clean up `o` to make boolean operations more likely to succeed
    '''

    with butil.SelectObjects(o), logging_util.Suppress():
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all()
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')


def apply_attach_transform(part, target, att):
    u, v, rad = att.coord

    loc, normal, tangent = raycast_surface(target, idx_pct=u, dir_rot=euler(180 * v, 0, 0) @ euler(0, 90, 0),
                                           r=rad, debug=False)

    if att.rotation_basis == 'global':
        basis_rot = mathutils.Quaternion()
    elif att.rotation_basis == 'normal':
        basis_rot = quat_align_vecs((1, 0, 0), normal)
    elif att.rotation_basis == 'tangent':
        basis_rot = quat_align_vecs((1, 0, 0), tangent)
    else:
        raise ValueError(f'Unrecognized {att.rotation_basis=}')
    rot = basis_rot @ euler(*att.joint.rest)
    att.joint.rest = np.rad2deg(
        np.array(rot.to_euler()))  # write back so subsequent steps can use updated global pose

    part.obj.parent = target.obj
    part.obj.location = loc
    part.obj.rotation_euler = rot.to_euler()
    bpy.context.view_layer.update()

    assert att.side in [-1, 1]
    for obj in [part.obj, part.attach_basemesh]:
        if obj is None:
            continue
        part.side = target.side * att.side
        obj.matrix_world = mathutils.Matrix.Scale(att.side, 4, (
            0, 1, 0)) @ obj.matrix_world  # # butil.apply_transform(obj, loc=False, rot=False, scale=True)


def attach(part: Part, target: Part, att: genome.Attachment):
    if target.obj.type != 'MESH':
        raise ValueError(
            f'attach() recieved {target.obj=} with {target.obj.type=} which is not valid for raycast '
            f'attachment, please convert to type=MESH')

    apply_attach_transform(part, target, att)

    # Create a joining part if necessary
    # if att.smoothing_width > 0:
    #    bevel_obj = join_smoothing.create_bevel_connection(part.obj, target.obj, part.bvh(), target.bvh(),
    #    att.smoothing_width)
    #    bevel_obj.parent = part.obj

    # Cut any cutters from the parent
    cutter_extras = [o for o in butil.iter_object_tree(part.obj) if 'Cutter' in o.name]
    for o in cutter_extras:
        sanitize_for_boolean(o)
        butil.modify_mesh(target.obj, 'BOOLEAN', object=o, operation='DIFFERENCE', apply=True, solver='FAST')
        butil.delete(o)


def genome_to_creature(genome: genome.CreatureGenome, name: str):
    parts = tree.map(genome.parts, lambda g: g.part_factory())

    for i, (part, cnode) in enumerate(zip(parts, genome.parts)):
        factory_class = cnode.part_factory.__class__.__name__
        part.obj.name = f'{name}.parts({i}, factory={factory_class})'
        part.obj['factory_class'] = factory_class
        part.obj['index'] = i
        for extra in part.obj.children:
            extra.name = f'{name}.parts({i}).extra({extra.name}, {i})'
            extra.parent = part.obj
            extra['factory_class'] = factory_class
            extra['index'] = i

    # write attribute values that must come before posing/arrangement
    logger.debug(f'Writing local attributes')
    for i, (part, genode) in enumerate(tree.tzip(parts, genome.parts)):
        tags = genode.part_factory.tags
        write_local_attributes(part, i, tags)

    for genome, (parent, part) in zip(tree.iter_items(genome.parts, postorder=True),
                                      tree.iter_parent_child(parts, postorder=True)):
        if parent is None:
            continue  # root object doesnt need attaching
        logger.debug(f'Attaching {part} to {parent}')
        attach(part, parent, genome.att)

    # write any attributes that must come after posign/arrangement
    logger.debug(f'Writing global attributes')
    for part in parts:
        write_global_attributes(part)

    root = butil.spawn_empty(name)
    for p in parts:
        if p.attach_basemesh is not None:
            butil.delete(p.attach_basemesh)
            p.attach_basemesh = None
            p._bvh = None
        with butil.SelectObjects(p.obj):
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
            p.obj.parent = root

    return root, parts
