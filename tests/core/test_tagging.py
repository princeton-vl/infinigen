# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

from infinigen.core.util import blender as butil

import numpy as np
import bpy

def test_tagging_basic():

    tagging.tag_system.clear()
    butil.clear_scene()
    
    cube = butil.spawn_cube()

    assert len([n for n in cube.data.attributes.keys() if n.startswith(tagging.PREFIX)]) == 0
    assert list(tagging.tag_system.tag_dict.keys()) == [tag_name]

    tagint_attr = cube.data.attributes.get(tagging.COMBINED_ATTR_NAME)
    assert tagint_attr is not None
    assert tagint_attr.domain == 'FACE'

    tagint_vals = surface.read_attr_data(cube, tagging.COMBINED_ATTR_NAME, domain='FACE')

    cubey_tag_int = tagging.tag_system.tag_dict.get(tag_name)
    assert cubey_tag_int == 1

    n_poly = len(cube.data.polygons)
    assert len(tagint_vals) == n_poly
    assert np.all(tagint_vals == cubey_tag_int)

    mask = np.arange(n_poly) >= (n_poly // 2)

    assert list(tagging.tag_system.tag_dict.keys()) == [tag_name, combined_half_name]


    with butil.ViewportMode(cube, mode='EDIT'):
        butil.select(cube)
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')


def get_canonical_tag_cube():

    cube = butil.spawn_cube()

    with butil.ViewportMode(cube, mode='EDIT'):
        butil.select(cube)
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')

    tagging.tag_canonical_surfaces(cube)
    return cube

def test_tag_canonical():
    
    tagging.tag_system.clear()
    butil.clear_scene()
    cube = get_canonical_tag_cube() 
    
    for tag in tagging.CANONICAL_TAGS:
        mask = tagging.tagged_face_mask(cube, tag)
        assert mask.sum() == 2 # expect 2 triangles forming every side of the cube
        
        idx1, idx2 = np.where(mask)[0]
        norm1 = cube.data.polygons[idx1].normal
        norm2 = cube.data.polygons[idx2].normal
        assert norm1 == norm2

def test_tag_canonical_negated():

    tagging.tag_system.clear()
    butil.clear_scene()
    cube = get_canonical_tag_cube() 

    assert len(cube.data.polygons) == 12


    assert all_but_top.sum() == 10 # 4*2 side triangles, 2 bottom triangles

    assert side.sum() == 8 # 4 sides, 2 triangles 