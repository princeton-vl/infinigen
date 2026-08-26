# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging

import bpy
import procfunc as pf

from infinigen2 import context
from infinigen2.exporters.render_error_check.util import (
    context_materials,
    flattened_node_count,
)

logger = logging.getLogger(__name__)

SHADER_NODE_COUNT_FAIL = 1000


class ShaderTooComplexError(RuntimeError):
    pass


def count_material_nodes(material: bpy.types.Material) -> int:
    if not material.use_nodes or material.node_tree is None:
        return 0
    return flattened_node_count(material.node_tree, {})


def assert_shader_complexity_ok(objects: list[pf.MeshObject] | None = None):
    over = {
        m.name: c
        for m in context_materials(objects)
        if (c := count_material_nodes(m)) >= SHADER_NODE_COUNT_FAIL
    }
    if over:
        error = ShaderTooComplexError(
            f"materials exceed {SHADER_NODE_COUNT_FAIL} flattened nodes: {over}"
        )
        context.raise_or_warn(
            context.globals.error_mode_shader_complexity, error, logger
        )
