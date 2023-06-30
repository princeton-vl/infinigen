import pdb
import logging

import bpy
import mathutils

from nodes import node_utils as nu

logger = logging.getLogger(__name__)


def placement_mask(scale=0.05, select_thresh=0.55, normal_thresh=0.5, normal_thresh_high=2.,
    def selection(nw):
        mask = nw.new_node(Nodes.Value)
        mask.outputs["Value"].default_value = 1

        if select_thresh is not None:
            mask = nw.scalar_multiply(mask, noise_mask)

        if normal_thresh is not None:
            facing_mask = nu.facing_mask(nw, normal_dir, thresh=normal_thresh)
            mask = nw.scalar_multiply(mask, facing_mask)
            tag_parts = tag.split(',')
            logger.debug(f'Parsing {tag=} into {len(tag_parts)=}, matching against {len(tag_dict)=}')
            for part in tag_parts:
            if len(conditions) > 0:
                mask = nw.scalar_multiply(
                    mask,
                    nw.scalar_add(*conditions)
                )
        return mask

