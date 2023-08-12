# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


import bpy

from infinigen.core.util import blender as butil

class Curve:

    def __init__(
        self, points, 
        profile=None, taper=None, 
        closed=False, sharp=None, 
        scale=None
    ):
        self.points = points
        self.profile = profile
        self.taper = taper
        self.closed = closed
        self.sharp = sharp
        self.scale = scale

    def to_curve_obj(self, name='curve', 
        resu=4, curvetype='NURBS', extrude=0, fill_caps = True,
        to_mesh=False, cleanup=True
    ):

        curveData = bpy.data.curves.new(f'{name}_curve', type='CURVE')
        curveData.dimensions = '3D'
        curveData.resolution_u = resu
        curveData.use_fill_caps = fill_caps
        curveData.twist_mode = 'MINIMUM'
        curveData.extrude = extrude

        polyline = curveData.splines.new(curvetype)

        def get_pos(p):
            if len(p) == 3:
                x, y, z = p
            elif len(p) == 2:
                x, y = p
                z = 0
            else:
                raise ValueError(f'Unrecognized point dim {len(p)} in Curve.to_curve_obj')
            return x, y, z, 1

        for i, p in enumerate(self.points):
            if i != 0:
                polyline.points.add(1)
            polyline.points[-1].co = get_pos(p)

            end = ((i == 0) or (i == len(self.points) - 1)) and not self.closed
            sharp = self.sharp is not None and self.sharp[i]
            if end or sharp:
                polyline.points.add(1)
                polyline.points[-1].co = get_pos(p)

        if self.profile is not None:
            curveData.bevel_mode = 'OBJECT'
            curveData.bevel_object = self.profile

        if self.taper is not None:
            curveData.taper_object = self.taper

        obj = bpy.data.objects.new(name, curveData)
        bpy.context.scene.collection.objects.link(obj)

        if self.closed:
            with butil.ViewportMode(obj, mode='EDIT'):
                bpy.ops.curve.select_all()
                bpy.ops.curve.cyclic_toggle()

        if self.scale is not None:
            obj.scale = self.scale

        if to_mesh:

            bevel = curveData.bevel_object
            taper = curveData.taper_object

            newobj = butil.to_mesh(obj)

            if cleanup:
                butil.select_none()
                for o in [obj, bevel, taper]:
                    if o is not None:
                        o.select_set(True)
                bpy.ops.object.delete(use_global=False, confirm=False)
                
                self.profile = None
                self.taper = None

            obj = newobj

        return obj
