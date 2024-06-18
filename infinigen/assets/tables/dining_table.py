# Authors: Yiming Zuo

import bpy
import bpy
import mathutils
from numpy.random import uniform, normal, randint, choice
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core.util.color import color_category
from infinigen.core import surface

from infinigen.core.util.math import FixedSeed
from infinigen.core.placement.factory import AssetFactory

from infinigen.assets.tables.table_utils import nodegroup_create_anchors, nodegroup_create_legs_and_strechers
from infinigen.assets.tables.table_top import nodegroup_generate_table_top

from infinigen.assets.tables.legs.single_stand import nodegroup_generate_single_stand
from infinigen.assets.tables.legs.straight import nodegroup_generate_leg_straight
from infinigen.assets.tables.legs.square import nodegroup_generate_leg_square

from infinigen.assets.tables.strechers import nodegroup_strecher


@node_utils.to_nodegroup('geometry_create_legs', singleton=False, type='GeometryNodeTree')
def geometry_create_legs(nw: NodeWrangler, **kwargs):

    if kwargs['Leg Style'] == "single_stand":

            'Table Height': kwargs['Top Height'],
            'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'],
            'Align Leg X rot': True

    elif kwargs['Leg Style'] == "straight":

        strecher = nw.new_node(nodegroup_strecher().name,

            'Table Height': kwargs['Top Height'],
            'Strecher Instance': strecher,
            'Strecher Index Increment': kwargs['Strecher Increament'],
            'Strecher Relative Position': kwargs['Strecher Relative Pos'],
            'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'],
            'Align Leg X rot': True

    elif kwargs['Leg Style'] == "square":
            'Table Height': kwargs['Top Height'],
            'Leg Bottom Relative Scale': kwargs['Leg Placement Bottom Relative Scale'],
            'Align Leg X rot': True

    else:
        raise NotImplementedError


def geometry_assemble_table(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler


    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [tabletop_instance, legs]})


class TableDiningFactory(AssetFactory):
        super(TableDiningFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
    @staticmethod
    def sample_parameters(dimensions):
        # all in meters
        x, y, z = dimensions

        NGon = 4

        leg_style = choice(['straight', 'single_stand', 'square'], p=[0.5, 0.1, 0.4])
        # leg_style = choice(['straight'])

        if leg_style == "single_stand":
            leg_number = 2


            top_scale = uniform(0.6, 0.7)
            bottom_scale = 1.0

        elif leg_style == "square":
            leg_number = 2
            leg_diameter = uniform(0.07, 0.10)

            leg_curve_ctrl_pts = None
            top_scale = 0.8
            bottom_scale = 1.0

        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)

            leg_number = 4

            leg_curve_ctrl_pts = [(0.0, 1.0), (0.4, uniform(0.85, 0.95)), (1.0, uniform(0.4, 0.6))]

            top_scale = 0.8
            bottom_scale = uniform(1.0, 1.2)

        else:
            raise NotImplementedError


        parameters = {
            'Top Profile N-gon': NGon,
            'Top Profile Width': 1.414 * x,
            'Top Profile Aspect Ratio': y / x,
            'Top Profile Fillet Ratio': uniform(0.0, 0.02),
            'Top Thickness': top_thickness,
            'Top Vertical Fillet Ratio': uniform(0.1, 0.3),
            'Height': z,
            'Top Height': z - top_thickness,
            'Leg Number': leg_number,
            'Leg Style': leg_style,
            'Leg NGon': 4,
            'Leg Placement Top Relative Scale': top_scale,
            'Leg Placement Bottom Relative Scale': bottom_scale,
            'Leg Height': 1.0,
            'Leg Diameter': leg_diameter,
            'Leg Curve Control Points': leg_curve_ctrl_pts,
            'Strecher Relative Pos': uniform(0.2, 0.6),
            'Strecher Increament': choice([0, 1, 2])
        }

        return parameters

    def create_asset(self, **params):
        obj = bpy.context.active_object

        # surface.add_geomod(obj, geometry_assemble_table, apply=False, input_kwargs=self.params)
        surface.add_geomod(obj, geometry_assemble_table, apply=True, input_kwargs=self.params)
        tagging.tag_system.relabel_obj(obj)

