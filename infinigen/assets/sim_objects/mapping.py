# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

from infinigen.assets.sim_objects.box import BoxFactory
from infinigen.assets.sim_objects.cabinet import CabinetFactory
from infinigen.assets.sim_objects.dishwasher import DishwasherFactory
from infinigen.assets.sim_objects.door import SimDoorFactory
from infinigen.assets.sim_objects.door_handle import DoorHandleFactory
from infinigen.assets.sim_objects.drawer import DrawerFactory
from infinigen.assets.sim_objects.faucet import FaucetFactory
from infinigen.assets.sim_objects.lamp import LampFactory
from infinigen.assets.sim_objects.microwave import MicrowaveFactory
from infinigen.assets.sim_objects.oven import OvenFactory
from infinigen.assets.sim_objects.pepper_grinder import PepperGrinderFactory
from infinigen.assets.sim_objects.plier import PlierFactory
from infinigen.assets.sim_objects.refrigerator import RefrigeratorFactory
from infinigen.assets.sim_objects.soap_dispenser import SoapDispenserFactory
from infinigen.assets.sim_objects.stovetop import StovetopFactory
from infinigen.assets.sim_objects.toaster import ToasterFactory
from infinigen.assets.sim_objects.trash import TrashFactory
from infinigen.assets.sim_objects.window import WindowFactory

# add newly transpiled assets here

OBJECT_CLASS_MAP = {
    "door": SimDoorFactory,
    "toaster": ToasterFactory,
    "dishwasher": DishwasherFactory,
    "lamp": LampFactory,
    "cabinet": CabinetFactory,
    "drawer": DrawerFactory,
    "refrigerator": RefrigeratorFactory,
    "oven": OvenFactory,
    "microwave": MicrowaveFactory,
    "soap_dispenser": SoapDispenserFactory,
    "faucet": FaucetFactory,
    "plier": PlierFactory,
    "window": WindowFactory,
    "box": BoxFactory,
    "pepper_grinder": PepperGrinderFactory,
    "trash": TrashFactory,
    "door_handle": DoorHandleFactory,
    "stovetop": StovetopFactory,
    # add newly transpiled assets here
}


def print_sim_objects():
    for i, obj_class in enumerate(OBJECT_CLASS_MAP.keys()):
        print(f"{i} - {obj_class}")


if __name__ == "__main__":
    print("Valid simulation assets:")
    print_sim_objects()
