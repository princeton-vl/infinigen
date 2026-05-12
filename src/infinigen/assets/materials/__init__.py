# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes
from infinigen.infinigen_gpl.surfaces import snow


class Snow:
    shader = snow.shader_snow
    type = SurfaceTypes.SDFPerturb
    mod_name = "geo_snowtexture"
    name = "snow"

    def apply(self, obj, selection=None, **kwargs):
        surface.add_geomod(
            obj,
            snow.geo_snowtexture,
            selection=selection,
        )
        surface.add_material(obj, snow.shader_snow, selection=selection)


snow.Snow = Snow
