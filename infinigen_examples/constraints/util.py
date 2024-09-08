# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors: Alexander Raistrick

import numpy as np

from infinigen.core import tags as t
from infinigen.core.constraints import (
    constraint_language as cl,
)

room_types = {
    t.Semantics.Kitchen,
    t.Semantics.Bedroom,
    t.Semantics.LivingRoom,
    t.Semantics.Closet,
    t.Semantics.Hallway,
    t.Semantics.Bathroom,
    t.Semantics.Garage,
    t.Semantics.Balcony,
    t.Semantics.DiningRoom,
    t.Semantics.Utility,
    t.Semantics.StaircaseRoom,
}

all_sides = {t.Subpart.Bottom, t.Subpart.Top, t.Subpart.Front, t.Subpart.Back}
walltags = {
    t.Subpart.Wall,
    t.Subpart.Visible,
    -t.Subpart.SupportSurface,
    -t.Subpart.Ceiling,
}
floortags = {
    t.Subpart.SupportSurface,
    t.Subpart.Visible,
    -t.Subpart.Wall,
    -t.Subpart.Ceiling,
}
ceilingtags = {
    t.Subpart.Visible,
    t.Subpart.Ceiling,
    -t.Subpart.Wall,
    -t.Subpart.SupportSurface,
}

front_dir = np.array([0, 1, 0])
back_dir = np.array([0, -1, 0])
down_dir = np.array([0, 0, -1])

bottom = {t.Subpart.Bottom, -t.Subpart.Top, -t.Subpart.Front, -t.Subpart.Back}
back = {t.Subpart.Back, -t.Subpart.Top, -t.Subpart.Front}
top = {t.Subpart.Top, -t.Subpart.Back, -t.Subpart.Bottom, -t.Subpart.Front}
side = {-t.Subpart.Top, -t.Subpart.Bottom, -t.Subpart.Back, -t.Subpart.SupportSurface}
front = {t.Subpart.Front, -t.Subpart.Top, -t.Subpart.Bottom, -t.Subpart.Back}
leftright = {
    -t.Subpart.Top,
    -t.Subpart.Bottom,
    -t.Subpart.Back,
    -t.Subpart.Front,
    -t.Subpart.SupportSurface,
}

on_floor = cl.StableAgainst(bottom, floortags, margin=0.01)
flush_wall = cl.StableAgainst(back, walltags, margin=0.02)
against_wall = cl.StableAgainst(back, walltags, margin=0.07)
spaced_wall = cl.StableAgainst(back, walltags, margin=0.8)
hanging = cl.StableAgainst(top, ceilingtags, margin=0.05)
side_against_wall = cl.StableAgainst(side, walltags, margin=0.05)

front_coplanar_front = cl.CoPlanar(front, front, margin=0.05, rev_normal=True)
back_coplanar_back = cl.CoPlanar(back, back, margin=0.05, rev_normal=True)

ontop = cl.StableAgainst(bottom, top)
on = cl.StableAgainst(bottom, {t.Subpart.SupportSurface})

front_against = cl.StableAgainst(front, side, margin=0.05, check_z=False)
front_to_front = cl.StableAgainst(front, front, margin=0.05, check_z=False)
leftright_leftright = cl.StableAgainst(leftright, leftright, margin=0.05)
side_by_side = cl.StableAgainst(side, side)
back_to_back = cl.StableAgainst(back, back)

variable_room = t.Variable("room")
variable_obj = t.Variable("obj")
