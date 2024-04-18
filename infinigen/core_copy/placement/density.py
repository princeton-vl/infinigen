# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import pdb
import logging

import bpy
import mathutils
import numpy as np

from infinigen.core.nodes.node_wrangler import NodeWrangler, Nodes
from infinigen.core.nodes import node_utils as nu
from infinigen.core.surface import eval_argument

logger = logging.getLogger(__name__)

tag_dict = None

def set_tag_dict(tag_dict_):
    global tag_dict
    tag_dict = tag_dict_

def placement_mask(scale=0.05, select_thresh=0.55, normal_thresh=0.5, normal_thresh_high=2.,
                               normal_dir=(0, 0, 1), tag=None, return_scalar=False, altitude_range=None):
    def selection(nw):
        mask = nw.new_node(Nodes.Value)
        mask.outputs["Value"].default_value = 1

        if select_thresh is not None:
            mininum_val = nw.new_node(Nodes.Value)
            mininum_val.outputs[0].default_value = np.random.normal(select_thresh, 0.025)
            noise_node = nu.noise(nw, scale)
            noise_mask = nw.new_node(Nodes.Math, input_args=[noise_node, mininum_val],
                                     attrs={'operation': 'GREATER_THAN'})
            mask = nw.scalar_multiply(mask, noise_mask)

        if normal_thresh is not None:
            facing_mask = nu.facing_mask(nw, normal_dir, thresh=normal_thresh)
            mask = nw.scalar_multiply(mask, facing_mask)
            if normal_thresh_high is not None:
                facing_mask = nu.facing_mask(nw, - mathutils.Vector(normal_dir), thresh=-normal_thresh_high)
                mask = nw.scalar_multiply(mask, facing_mask)

        if tag is not None:
            keys = list(tag_dict.keys())
            tag_parts = tag.split(',')
            logger.debug(f'Parsing {tag=} into {len(tag_parts)=}, matching against {len(tag_dict)=}')
            for part in tag_parts:
                if part.startswith("-"):
                    keys = [k for k in keys if part[1:] not in k.split('.')]
                else:
                    keys = [k for k in keys if part in k.split('.')]
            conditions = []
            for k in keys:
                conditions.append(nw.new_node(Nodes.Compare, attrs={'operation': "EQUAL", "data_type": "FLOAT"}, input_args=[eval_argument(nw, "MaskTag"), tag_dict[k]]))
            if len(conditions) > 0:
                mask = nw.scalar_multiply(
                    mask,
                    nw.scalar_add(*conditions)
                )
        if altitude_range is not None:
            z = (nw.new_node(Nodes.SeparateXYZ, [nw.new_node(Nodes.InputPosition)]), 2)
            start, end = altitude_range
            mask = nw.scalar_multiply(
                mask,
                nw.new_node(Nodes.Compare, attrs={'operation': "GREATER_THAN", "data_type": "FLOAT"}, input_args=[z, start]),
                nw.new_node(Nodes.Compare, attrs={'operation': "LESS_THAN", "data_type": "FLOAT"}, input_args=[z, end]),
            )
        if (select_thresh is not None) and return_scalar:
            map_range = nw.new_node(Nodes.MapRange, input_kwargs={'Value': noise_node, 1: mininum_val, 2: 0.75},
                                    attrs={'interpolation_type': 'SMOOTHSTEP'})
            return mask, map_range

        return mask

    return selection
