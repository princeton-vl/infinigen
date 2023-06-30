# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma
# Date Signed: June 5 2023

import os
import sys

sys.path.append(f"{os.path.split(os.path.abspath(__file__))[0]}/../..")
import argparse

import bpy
import numpy as np
from nodes.node_wrangler import Nodes, NodeWrangler
from terrain.utils import Mesh, read
from util.blender import clear_scene
from util.organization import AssetFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--overlay', type=int, default=False)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    folder = os.path.dirname(args.input)
    tile_size = float(np.loadtxt(f"{folder}/{AssetFile.TileSize}.txt"))
    image = read(args.input)
    mesh = Mesh(heightmap=image, L=tile_size)
    if args.overlay:
        image = read(args.input.replace(AssetFile.Heightmap, AssetFile.Mask))
        mesh.vertex_attributes["attribute"] = image.reshape(-1).astype(np.float32)
    clear_scene()
    obj = mesh.export_blender("preview")
    if args.overlay:
        material = bpy.data.materials.new(name="preview_material")
        material.use_nodes = True
        nw = NodeWrangler(material.node_tree)
        new_attribute_node = nw.new_node(Nodes.Attribute, [], {"attribute_name": "attribute"})
        material.node_tree.links.new(new_attribute_node.outputs['Color'], material.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
        obj.active_material = material
