from infinigen.assets.objects.appliances.toaster import ToasterFactory
from infinigen.assets.objects.elements.doors import DoorFactory
from infinigen.assets.sim_objects.dishwasher import DishwasherFactory
from infinigen.assets.sim_objects.doublefridge import DoublefridgeFactory
from infinigen.assets.sim_objects.lamp import LampFactory
from infinigen.assets.sim_objects.multidoublefridge import MultiDoublefridgeFactory

# add newly transpiled assets here

OBJECT_CLASS_MAP = {
    "door": DoorFactory,
    "toaster": ToasterFactory,
    "dishwasher": DishwasherFactory,
    "lamp": LampFactory,
    # add newly transpiled assets here
}
