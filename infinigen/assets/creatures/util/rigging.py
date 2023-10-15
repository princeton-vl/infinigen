# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import re
import math
from numbers import Number
from functools import partial
import logging

import bpy
import mathutils
import numpy as np
from tqdm import tqdm

from infinigen.core.util import blender as butil, math as mutil
from infinigen.assets.creatures.util import tree
from infinigen.assets.creatures.util.creature import Part, infer_skeleton_from_mesh
from infinigen.assets.creatures.util.genome import Joint, IKParams

logger = logging.getLogger(__name__)

IK_TARGET_PREFIX = 'ik_target'


def bone(editbones, head, tail, parent):
    bone = editbones.new('bone')  # name overriden later
    bone.head = head
    bone.tail = tail
    bone.parent = parent
    return bone


def get_bone_idxs(part_node: tree.Tree):

    part, att = part_node.item
    child_ts = [c.item[1].coord[0] for c in part_node.children]

    for t in child_ts:
        assert t >= 0 and t <= 1

    bounds = [0.0, 1.0]

    tr = part.settings.get('trim_bounds_child_margin', 0.15)
    if tr > 0 and len(child_ts):
        if min(child_ts) < tr:
            bounds[0] = min(child_ts)
        if max(child_ts) > 1 - tr:
            bounds[1] = max(child_ts)

    idxs = set(bounds)
    if part.joints is not None:
        idxs = idxs.union(part.joints.keys())
    return sorted(list(idxs))

def create_part_bones(part_node: tree.Tree, editbones, parent):
    
    bones = {}
    part, att = part_node.item
    skeleton = part.skeleton_global()
    idxs = get_bone_idxs(part_node)

    if part.settings.get('rig_reverse_skeleton', False):
        idxs = list(reversed(idxs))

    for idx1, idx2 in zip(idxs[:-1], idxs[1:]):
        head = mutil.lerp_sample(skeleton, idx1 * (len(skeleton) - 1)).reshape(-1)
        tail = mutil.lerp_sample(skeleton, idx2 * (len(skeleton) - 1)).reshape(-1)
        parent = bone(editbones, head, tail, parent)
        bones[idx1] = parent

    if part.settings.get('rig_extras', False):
        for i, extra in enumerate(part.obj.children):
            if extra.type != 'MESH':
                continue
            skeleton = mutil.homogenize(infer_skeleton_from_mesh(extra)) @ np.array(extra.matrix_world)[:-1].T
            head = mutil.lerp_sample(skeleton, 0 * (len(skeleton) - 1)).reshape(-1)
            tail = mutil.lerp_sample(skeleton, 1 * (len(skeleton) - 1)).reshape(-1)

            extra_id = re.fullmatch('.*\.extra\((.*),.*', extra.name).group(1)
            bones[extra_id] = bone(editbones, head, tail, parent)

    return bones

def create_bones(parts_atts, arma):

    def make_parent_connector_bone(part, att, parent_bones, parent_bone_t):
        
        u, v, r = att.coord

        parent_bone = parent_bones[parent_bone_t]
        bonekeys = [k for k in parent_bones.keys() if not isinstance(k, str)]

        try:
            next_t = next(i for i in sorted(bonekeys) if i >= u)
        except StopIteration:
            next_t = parent_bone_t

        if next_t == parent_bone_t:
            head = parent_bone.head
        else:
            pct = (u - parent_bone_t) / (next_t - parent_bone_t)
            assert 0 <= pct and pct <= 1, (pct, next_t, parent_bone_t)
            head = mutil.lerp(parent_bone.head, parent_bone.tail, pct)

        tail = part.skeleton_global()[0]
        parent_bone = bone(editbones, head, tail, parent_bone)

        return parent_bone

    def make_bones(node: tree.Tree, parent_bones: dict, editbones):
        part, att = node.item

        bones = {}
        parent_bone = None
            
        if parent_bones is not None:
            bonekeys = [k for k in parent_bones.keys() if not isinstance(k, str)]
            parent_bone_t = max((i for i in bonekeys if i <= att.coord[0]), default=min(bonekeys))
            parent_bone = parent_bones[parent_bone_t]
            
            if att.coord[-1] > part.settings.get('connector_collapse_margin_radpct', 0.5):
                bones[-1] = parent_bone = make_parent_connector_bone(part, att, parent_bones, parent_bone_t)

        part_bones = create_part_bones(node, editbones, parent=parent_bone)
        bones.update(part_bones)

        return bones

    def finalize_bonedict_to_leave_editmode(bones):
        # the edit bones wont continue to exist once we leave edit mode, store their names instead
        for j, b in bones.items():
            partname = part.obj.name.split('.')[-1]

            if isinstance(j, (int, float)):
                b.name = f'{partname}.side({part.side}).bone({j:.2f})'
            elif isinstance(j, str):
                b.name = f'{partname}.side({part.side}).extra_bone({j})'
            else:
                raise ValueError(f'Unrecognized {j=}')
            b['side'] = part.side
            b['factory_class'] = part.obj['factory_class']
            b['index'] = part.obj['index']
            b['length'] = j
            bones[j] = b.name

    with butil.ViewportMode(arma, mode='EDIT'):
        editbones = arma.data.edit_bones
        part_bones = tree.map_parent_child(parts_atts, partial(make_bones, editbones=editbones))
        for (part, _), bones in tree.tzip(parts_atts, part_bones):
            finalize_bonedict_to_leave_editmode(bones)

    return part_bones


def compute_chain_length(parts_atts: tree.Tree, bones, part, ik: IKParams):

    if ik.chain_parts is None:
        assert ik.chain_length is not None
        return ik.chain_length

    nodes, parents = tree.to_node_parent(tree.tzip(parts_atts, bones))
    curr_idx = next(i for i, ((p, _), _) in enumerate(nodes) if p is part)
    chain_length = 0
    for i in range(math.ceil(ik.chain_parts)):
        p = 1 if i < int(ik.chain_parts) else (ik.chain_parts - int(ik.chain_parts))
        n_skeleton_bones = len([b for b in nodes[curr_idx][1].values() if 'extra' not in b])
        chain_length += math.ceil(p * n_skeleton_bones)
        if curr_idx not in parents:
            break
        curr_idx = parents[curr_idx]

    if ik.chain_length is not None:
        chain_length += ik.chain_length

    logger.debug(f'Com')

    return chain_length


def create_ik_targets(arma, parts_atts: tree.Tree, bones):

    def make_target(part_node, part_bones, ik: IKParams):

        part, att = part_node.item

        joint_ts = get_bone_idxs(part_node)
        bone_idx = t if t != joint_ts[-1] else joint_ts[
            -2]  # the last idx doesnt have its own bone, it is just the endpoint
        base_keys =[k for k in part_bones.keys() if isinstance(k,Number)]
        bone_idx = max((i for i in base_keys if i <= bone_idx), default=min(base_keys))
        name = part_bones[bone_idx]

        pbone = arma.pose.bones[name]

        if ik.mode == 'iksolve':
            con = pbone.constraints.new('IK')
            con.chain_count = compute_chain_length(parts_atts, bones, part, ik)
        elif ik.mode == 'pin':
            con = pbone.constraints.new('COPY_LOCATION')
        else:
            raise ValueError(f'Unrecognized {ik.mode=}')

        con.target = butil.spawn_empty(f'{IK_TARGET_PREFIX}({ik.name})', disp_type='CUBE', s=ik.target_size)
        con.target.location = pbone.tail if t != 0 else pbone.head

        if ik.rotation_weight > 0:
            if ik.mode == 'iksolve':
                con.use_rotation = True
                con.orient_weight = ik.rotation_weight
                con.target.rotation_euler = (pbone.matrix).to_euler()
            else:
                rot_con = pbone.constraints.new('COPY_ROTATION')
                rot_con.target = con.target
                rot_con.influence = ik.rotation_weight

        return con.target

    targets = []
    with butil.ViewportMode(arma, mode='POSE'):
        # TODO: risky zip, silent fail on non-matching topology
        data_iter = zip(
            tree.iter_nodes(parts_atts), 
            tree.iter_nodes(bones)
        )
        for part_node, bones_node in data_iter: 
            part, att = part_node.item
            assert part.iks is not None, part
            for t, ik in part.iks.items():
                targets.append(make_target(part_node, bones_node.item, ik))

    col = butil.get_collection('ik_targets')
    for t in targets:
        butil.put_in_collection(t, col)

    return targets


def apply_joint_constraint(joint: Joint, pose_bone, eps=1e-2):
    pb = pose_bone

    if joint.bounds is not None:
        bounds = np.deg2rad(joint.bounds)

        if not bounds.shape == (2, 3):
            raise ValueError(f'Encountered invalid {joint.bounds=}, {joint.bounds.shape=}')

        ranges = bounds[1] - bounds[0]

        for i, ax in enumerate('xyz'):
            if ranges[i] > eps:
                setattr(pb, f'use_ik_limit_{ax}', True)
                setattr(pb, f'ik_min_{ax}', bounds[0, i])
                setattr(pb, f'ik_max_{ax}', bounds[1, i])
            else:
                setattr(pb, f'lock_ik_{ax}', True)
    else:
        for ax in 'xyz':
            setattr(pb, f'use_ik_limit_{ax}', False)
            setattr(pb, f'lock_ik_{ax}', False)

    if joint.stretch is not None:
        pb.ik_stretch = joint.stretch

    if joint.stiffness is not None:
        s = joint.stiffness
        if not (hasattr(s, '__len__') and len(s) == 3):
            s = (s,) * 3
        pb.ik_stiffness_x, pb.ik_stiffness_y, pb.ik_stiffness_z = s


def constrain_bones(arma, parts_atts, bones, shoulder_auto_stiffness=0.85):
    
    def constrain_bone(part, att, skeleton_idx, bname):
        pb = arma.pose.bones[bname]

        if skeleton_idx == 0:
            # the orientation of bone 0 really controls the attachment angle
            # of the whole part, and should be constrained by att.joint
            if att is not None:
                apply_joint_constraint(att.joint, pb)
        elif part.joints is not None and skeleton_idx in part.joints:
            joint = part.joints[skeleton_idx]
            apply_joint_constraint(joint, pb)

        if skeleton_idx < 0 and shoulder_auto_stiffness > 0:
            # shoulder bones have index < 1, and were added automatically
            # make them stiff to minimally affect final outcome
            pb.ik_stiffness_x, pb.ik_stiffness_y, pb.ik_stiffness_z = (shoulder_auto_stiffness,) * 3
            pb.lock_ik_x = True
            pb.lock_ik_y = True
            pb.lock_ik_z = True
    
    with butil.ViewportMode(arma, mode='POSE'):
        for (part, att), part_bones in tree.tzip(parts_atts, bones):
            for skeleton_idx, bname in part_bones.items():
                if not isinstance(skeleton_idx, int):
                    continue
                constrain_bone(part, att, skeleton_idx, bname)

def pose_bones(arma, parts_atts, bones):
    with butil.ViewportMode(arma, mode='POSE'):
        for (part, att), part_bones in tree.tzip(parts_atts, bones):
            if part.joints is None:
                continue
            for skeleton_idx, bname in part_bones.items():
                if skeleton_idx == 0:
                    j = att.joint if att is not None else None
                else:
                    j = part.joints.get(skeleton_idx)
                if j is None or j.pose is None:
                    continue
                off = np.deg2rad(j.pose - j.rest)  # TODO, handle att.rotation_basis
                arma.pose.bones[bname].rotation_euler = tuple(off)


def parent_to_bones(objs, arma):
    for obj in objs:
        save_pos = obj.location
        with butil.SelectObjects([obj, arma], active=arma):
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        obj.location = save_pos


def parent_bones_by_part(creature, arma, part_bones):
    assert creature.parts[0] is creature.root
    for i, part in enumerate(creature.parts[1:]):
        with butil.SelectObjects([part.obj, arma]), butil.ViewportMode(arma, mode='POSE'):
            for bone in arma.pose.bones:
                select = (bone.name in part_bones[i].values())
                arma.data.bones[bone.name].select = select
                bone.bone.select = select
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    with butil.ViewportMode(arma, mode='POSE'):
        for bone in arma.pose.bones:
            bone.bone.select = False


def creature_rig(root, genome, parts, constraints=True, roll='GLOBAL_POS_Y'):
    data = bpy.data.armatures.new(name=f'{root.name}.armature_data')
    arma = bpy.data.objects.new(f'{root.name}_armature', data)
    bpy.context.scene.collection.objects.link(arma)

    parts_atts = tree.tzip(parts, tree.map(genome.parts, lambda n: n.att))
    bones = create_bones(parts_atts, arma)

    # force recalculate roll to eliminate bad guesses made by blender
    with butil.ViewportMode(arma, mode='EDIT'):
        bpy.ops.armature.select_all(action='SELECT')
        bpy.ops.armature.calculate_roll(type=roll)

    targets = create_ik_targets(arma, parts_atts, bones)
    for t in targets:
        t.parent = root

    if constraints:
        constrain_bones(arma, parts_atts, bones)

    pose_bones(arma, parts_atts, bones)

    return arma, targets


def create_ragdoll(root, arma, min_col_length=0.1, col_joint_margin=0.2, col_radius=0.07):
    def include_bone(b):
        if '-1' in b.name:
            return False
        if (b.head - b.tail).length < min_col_length:
            return False
        return True

    def create_bone_collider(pbone):
        col_head = mutil.lerp(pbone.head, pbone.tail, col_joint_margin)
        col_tail = mutil.lerp(pbone.head, pbone.tail, 1 - col_joint_margin)

        col = butil.spawn_line(pbone.name + '.col', np.array([col_head, col_tail]))
        with butil.SelectObjects(col), butil.CursorLocation(col_head):
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

        skin_mod = butil.modify_mesh(col, 'SKIN', apply=False)
        for svert in col.data.skin_vertices[0].data:
            svert.radius = (col_radius, col_radius)
        butil.apply_modifiers(col, mod=skin_mod)

        con = pbone.constraints.new('CHILD_OF')
        con.target = col
        with butil.SelectObjects(col):
            bpy.ops.rigidbody.object_add()

        col.rigid_body.mass = 5

        return col

    def configure_rigidbody_joint(child_bone, child_obj, parent_obj):
        o = butil.spawn_empty(child_bone.name + '.phys_joint')
        o.location = child_bone.head

        with butil.SelectObjects(o):
            bpy.ops.rigidbody.constraint_add()
            cons = bpy.context.object.rigid_body_constraint

        cons.type = 'GENERIC_SPRING'
        cons.object1 = child_obj
        cons.object2 = parent_obj

        # no linear sliding
        for ax in 'xyz':
            setattr(cons, f'use_limit_lin_{ax}', True)
            setattr(cons, f'limit_lin_{ax}_lower', 0)
            setattr(cons, f'limit_lin_{ax}_upper', 0)

        # copy over any angle constraints
        for ax in 'xyz':
            do_limit = getattr(child_bone, f'use_ik_limit_{ax}')
            setattr(cons, f'use_limit_ang_{ax}', do_limit)
            for ck, bk in (('lower', 'min'), ('upper', 'max')):
                lim = getattr(child_bone, f'ik_{bk}_{ax}')
                setattr(cons, f'limit_ang_{ax}_{ck}', lim / 3)

        for ax in 'xyz':
            setattr(cons, f'use_spring_ang_{ax}', True)

        return o

    def ancestors(pbone):
        while pbone.parent is not None:
            pbone = pbone.parent
            yield pbone

    with butil.ViewportMode(arma, mode='POSE'):
        # remove any ik constraints
        for b in arma.pose.bones:
            for c in b.constraints:
                b.constraints.remove(c)

        col_bones = [b for b in arma.pose.bones if include_bone(b)]

        # create colliders for all bones
        col_objs = {b.name: create_bone_collider(b) for b in col_bones}
        for o in col_objs.values():
            o.parent = root
            o.hide_render = True

        # create hinge constraints
        for b in col_bones:
            try:
                hinge_target = next(b for b in ancestors(b) if b in col_bones)
            except StopIteration:
                continue
            joint_obj = configure_rigidbody_joint(b, col_objs[b.name], col_objs[hinge_target.name])
            joint_obj.parent = root

        col_bone_names = [b.name for b in
            col_bones]  # store names so we can reference outside of pose mode in the next step

    # animation will be applied wrong if children inherit physics transformations from parents - unparent all
    # bones
    with butil.ViewportMode(arma, mode='EDIT'):
        for b in arma.data.edit_bones:
            b.select = b.name in col_bone_names
        bpy.ops.armature.parent_clear(type='CLEAR')
