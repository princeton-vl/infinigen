from infinigen.assets.objects.appliances.toaster import ToasterFactory
from infinigen.assets.objects.elements.doors import DoorFactory
from infinigen.assets.sim_objects.dishwasher import DishwasherFactory
from infinigen.assets.sim_objects.doublefridge import DoublefridgeFactory
from infinigen.assets.sim_objects.lamp import LampFactory
from infinigen.assets.sim_objects.multidoublefridge import MultiDoublefridgeFactory
from infinigen.assets.sim_objects.multifridge import MultifridgeFactory
from infinigen.assets.sim_objects.singlefridge import SinglefridgeFactory

# add newly transpiled assets here

OBJECT_CLASS_MAP = {
    "door": DoorFactory,
    "toaster": ToasterFactory,
    "dishwasher": DishwasherFactory,
    "multifridge": MultifridgeFactory,
    "singlefridge": SinglefridgeFactory,
    "doublefridge": DoublefridgeFactory,
    "multidoublefridge": MultiDoublefridgeFactory,
    "lamp": LampFactory,
    # add newly transpiled assets here
}
