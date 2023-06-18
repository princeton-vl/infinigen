# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this
# source tree.

# Authors: Lingjie Mei
# Date Signed: April 13 2023 


from nodes.node_info import Nodes
from nodes.node_wrangler import NodeWrangler
from surfaces import surface


def geo_shortest_path(nw: NodeWrangler, end_index, weight, trim_threshold=.1, offset=0., merge_threshold=.005,
                      subdiv=1):
    weight = surface.eval_argument(nw, weight)
    end_index = surface.eval_argument(nw, end_index)
    geometry = nw.new_node(Nodes.GroupInput, expose_input=[('NodeSocketGeometry', 'Geometry', None)])

    if subdiv > 0:
        geometry = nw.new_node(Nodes.SubdivisionSurface, input_kwargs={'Mesh': geometry, 'Level': subdiv})

    geometry = nw.new_node(Nodes.StoreNamedAttribute,
                           [geometry, 'custom_normal', nw.new_node(Nodes.InputNormal)],
                           attrs={'data_type': 'FLOAT_VECTOR'})
    curve = nw.new_node(Nodes.EdgePathToCurve,
                        [geometry, None, nw.new_node(Nodes.ShortestEdgePath, [end_index, weight]).outputs[0]])
    curve = nw.new_node(Nodes.SplineType, [curve], attrs={'spline_type': 'NURBS'})
    curve = nw.new_node(Nodes.TrimCurve, [curve, trim_threshold])
    curve = nw.new_node(Nodes.ResampleCurve, [curve], input_kwargs={'Length': .001}, attrs={'mode': 'LENGTH'})
    curve = nw.new_node(Nodes.StoreNamedAttribute,
                        [curve, 'spline_parameter', None, nw.new_node(Nodes.SplineParameter)])
    geometry = nw.new_node(Nodes.MergeByDistance, [nw.curve2mesh(curve), None, merge_threshold])

    distance = nw.vector_math('DISTANCE', *nw.new_node(Nodes.InputEdgeVertices).outputs[2:])
    curve = nw.new_node(Nodes.EdgePathToCurve, [geometry, None, nw.new_node(Nodes.ShortestEdgePath, [
        nw.compare('EQUAL', nw.new_node(Nodes.Index), 0), distance]).outputs[0]])

    curve = nw.new_node(Nodes.StoreNamedAttribute, [curve, 'tangent', nw.new_node(Nodes.CurveTangent)],
                        attrs={'data_type': 'FLOAT_VECTOR'})
    geometry = nw.new_node(Nodes.MergeByDistance, [nw.curve2mesh(curve)])

    geometry = nw.new_node(Nodes.SetPosition, [geometry, None, None,
        nw.scale(nw.new_node(Nodes.InputNormal), nw.scalar_multiply(nw.musgrave(), offset))])
    nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': geometry})
    return geometry
