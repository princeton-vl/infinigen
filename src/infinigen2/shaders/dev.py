# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import bpy
import procfunc as pf
from procfunc import types as t

__all__ = [
    "developer_grid",
    "flat_bsdf",
]


@pf.tracer.generator
def developer_grid(
    vector: pf.nodes.ProcNode[t.Vector],
) -> pf.Material:
    # TODO add primitive texture funcs and replace with those
    texture = bpy.data.textures.new(name="UV_Grid_Texture", type="IMAGE")
    image = bpy.data.images.new(name="UV_Grid_Image", width=2048, height=2048)
    image.generated_type = "UV_GRID"
    texture.image = image
    texture = pf.Texture(texture)

    vec_mapped = pf.nodes.shader.mapping(vector=vector, scale=t.Vector((1, 1, 1)))

    image_node = pf.nodes.texture.image(
        vector=vec_mapped,
        image=image,
        extension="REPEAT",
    )

    shader = pf.nodes.shader.principled_bsdf(base_color=image_node.color)
    return pf.Material(surface=shader)


@pf.tracer.generator
def flat_bsdf(
    vector: pf.nodes.ProcNode[pf.Vector] | None = None,
) -> pf.Material:
    if vector is None:
        vector = pf.nodes.shader.coord().generated

    shader = pf.nodes.shader.principled_bsdf(base_color=t.Vector((1, 0, 1)))
    return pf.Material(surface=shader)
