# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import bpy

bpy.ops.preferences.addon_enable(module='flip_fluids_addon')
bpy.ops.flip_fluid_operators.complete_installation()
