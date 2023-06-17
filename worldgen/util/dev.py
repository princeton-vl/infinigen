# Copyright (c) Princeton University.
# This source code is licensed under the GPL license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick
# Date Signed: May 30, 2023

import os, sys
from importlib import reload
from types import ModuleType

import gin

def rreload(module, filter_kw, mdict=None):
    """Recursively reload modules."""
    if mdict is None:
        mdict = {}
    if module not in mdict:
        mdict[module] = [] 
        
    with gin.unlock_config():
        reload(module)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if type(attribute) is ModuleType:
                if attribute not in mdict[module]:
                    try:
                        if attribute.__name__ in sys.builtin_module_names:
                            continue
                        if attribute.__file__ is None:
                            continue
                    except AttributeError:
                        continue
                    if filter_kw not in attribute.__file__:
                        continue
                    mdict[module].append(attribute)
                    rreload(attribute, filter_kw, mdict)

        reload(module)