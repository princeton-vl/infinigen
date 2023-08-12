# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


from pathlib import Path
import pdb

import numpy as np

from infinigen.core.util import blender as butil
from infinigen.core.nodes.node_transpiler.transpiler import transpile, indent
from infinigen.assets.creatures.util.geometry import lofting

def prefix():
    return (
        "import numpy as np\n"
        "from infinigen.assets.creatures.util.creature import CreatureGenome, PartGenome, Attachment, Joint\n"
    )

def repr_np_array(v):
    v = np.round(v, 3)
    return f'np.{repr(v)}'

def basename(obj):
    return obj.name.split('.')[0]

def parse_nurbs_data(obj, i=0):

    '''
    Given a blender object, read it's handles out as a (n,m,3) vertex array

    TODO: Read out knotvector. Function should yield all data necessary to define that NURBS
    '''
    
    assert obj.type == 'SURFACE'
    
    spline = obj.data.splines[i]
    m, n = spline.point_count_u, spline.point_count_v
    
    points = np.array([p.co for p in spline.points])
    points = points.reshape(n, m, -1)

    return points

def parse_part(nurbs_part, mesh_part, profiles_folder):
    
    name = basename(nurbs_part)

    part_genome_kwargs = {}
    handles = parse_nurbs_data(nurbs_part)
    skeleton, ts, rads, profiles_norm = lofting.factorize_nurbs_handles(handles[..., :-1])

    part_genome_kwargs['skeleton'] = repr_np_array(skeleton)
    part_genome_kwargs['rads'] = repr_np_array(rads.reshape(-1))

    path = Path(profiles_folder)/f'profile_{name}.npy'
    np.save(path, profiles_norm)
    print(f'Saving {path}')
    part_genome_kwargs['profile'] = f'np.load({repr(str(path))})'

    body = f"return {repr_function_call('PartGenome', part_genome_kwargs)}"
    code = f'def {name}():\n' + indent(body)
    return name, code

def find_approx_uvr_coord(child, parent_mesh, parent_nurbs):
    assert parent_mesh.type == 'MESH'

    loc = np.array(child.matrix_world.translation)
    verts = np.array([parent_mesh.matrix_world @ v.co for v in parent_mesh.data.vertices])

    dists = np.linalg.norm(verts - loc, axis=-1)
    i = dists.argmin()

    d = parent_nurbs.data
    verts_u = d.splines[0].point_count_u * d.resolution_u
    verts_v = d.splines[0].point_count_v * d.resolution_v
    assert verts_u * verts_v == len(parent_mesh.data.vertices)

    u = (i % verts_v) / verts_u
    v = (i // verts_v) / verts_u

    handles = parse_nurbs_data(parent_nurbs)
    skeleton, *_ = lofting.factorize_nurbs_handles(handles[..., :-1])
    skeleton_point = lofting.lerp_sample(skeleton, u * (len(skeleton) - 1))
    r = np.linalg.norm(verts[i] - skeleton_point) / np.linalg.norm(loc - skeleton_point)

    return np.array([u, v, r])

def parse_attachment(part, parent_mesh, parent_nurbs):

    uvr = find_approx_uvr_coord(part, parent_mesh, parent_nurbs)

    kwargs  = {
        'target': repr(basename(parent_mesh)),
        'coord': tuple(np.round(uvr, 2)),
        'joint': f'Joint(rest={tuple(np.round(part.rotation_euler, 2))})',
    }

    return repr_function_call('Attachment', kwargs, spacing=' ')

def repr_function_call(funcname, kwargs, spacing='\n', multiline=True):
    kwargs_str = f',{spacing}'.join([f'{k}={v}' for k, v in kwargs.items()])
    paren_sep = '\n' if multiline else ''
    return f'{funcname}({paren_sep}{indent(kwargs_str)}{paren_sep})'

def parse_creature(nurbs_root, mesh_root, profiles_folder):
    
    assert nurbs_root.type == 'SURFACE'
    assert mesh_root.type == 'MESH'

    code = prefix() + '\n'

    nurbs_parts = list(butil.iter_object_tree(nurbs_root))
    mesh_parts = list(butil.iter_object_tree(mesh_root))
    assert len(nurbs_parts) == len(mesh_parts)

    names = []
    atts = {}
    for nurbs_part, mesh_part in zip(nurbs_parts, mesh_parts):
        
        assert basename(nurbs_part) == basename(mesh_part)
        print(f'Processing {basename(nurbs_part)}')

        name, new_code = parse_part(nurbs_part, mesh_part, profiles_folder)
        names.append(name)
        code += new_code + '\n\n'

        if mesh_part.parent is not None:
            atts[name] = parse_attachment(mesh_part, mesh_part.parent, nurbs_part.parent)

    joiningome_args = {
        'parts': repr_function_call('dict', {name: f'{name}()' for name in names}),
        'attachments': repr_function_call('dict', atts)
    }

    body = f"return {repr_function_call('CreatureGenome', joiningome_args)}"
    code += f"def creature():\n {indent(body)}"

    return code
