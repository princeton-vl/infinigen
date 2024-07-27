# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import bpy

from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler


def shader_invisible(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    light_path = nw.new_node(Nodes.LightPath)

    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF, input_kwargs={"Roughness": 0.7697}
    )

    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)

    mix_shader = nw.new_node(
        Nodes.MixShader,
        input_kwargs={
            "Fac": light_path.outputs["Is Camera Ray"],
            1: principled_bsdf,
            2: transparent_bsdf,
        },
    )

    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={"Surface": mix_shader},
        attrs={"is_active_output": True},
    )


def apply(obj, selection=None, **kwargs):
    if not isinstance(obj, list):
        obj = [obj]

    for o in obj:
        for i in range(len(o.material_slots)):
            bpy.ops.object.material_slot_remove({"object": o})
    surface.add_material(obj, shader_invisible, selection=selection)
