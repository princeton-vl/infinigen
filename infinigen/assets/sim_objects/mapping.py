# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

from infinigen.assets.objects.appliances.toaster import ToasterFactory
from infinigen.assets.objects.shelves.cabinet import CabinetFactory
from infinigen.assets.sim_objects.dishwasher import DishwasherFactory
from infinigen.assets.sim_objects.door import SimDoorFactory
from infinigen.assets.sim_objects.doublefridge import DoublefridgeFactory
from infinigen.assets.sim_objects.drawer import DrawerFactory
from infinigen.assets.sim_objects.lamp import LampFactory
from infinigen.assets.sim_objects.multidoublefridge import MultiDoublefridgeFactory
from infinigen.assets.sim_objects.multifridge import MultifridgeFactory
from infinigen.assets.sim_objects.singlefridge import SinglefridgeFactory

# add newly transpiled assets here

OBJECT_CLASS_MAP = {
    "door": SimDoorFactory,
    "toaster": ToasterFactory,
    "dishwasher": DishwasherFactory,
    "multifridge": MultifridgeFactory,
    "singlefridge": SinglefridgeFactory,
    "doublefridge": DoublefridgeFactory,
    "multidoublefridge": MultiDoublefridgeFactory,
    "lamp": LampFactory,
    "cabinet": CabinetFactory,
    "drawer": DrawerFactory,
    # add newly transpiled assets here
}
