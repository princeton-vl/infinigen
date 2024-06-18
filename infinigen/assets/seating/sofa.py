
import random

    group_input = nw.new_node(Nodes.GroupInput,
            ('NodeSocketVector', 'Dimensions', (0.0000, 0.9000, 2.5000)),
            ('NodeSocketFloat', 'Baseboard Height', 0.1300),
            ('NodeSocketFloat', 'Backrest Width', 0.1100),
            ('NodeSocketFloat', 'Backrest Angle', -0.2000),
            ('NodeSocketFloatFactor', 'arm_width', 0.7000),
            ('NodeSocketFloatFactor', 'Arm_height', 0.7318),
            ('NodeSocketFloatAngle', 'arms_angle', 0.8727),
            ('NodeSocketBool', 'Footrest', False),
            ('NodeSocketInt', 'Count', 4),
            ('NodeSocketFloat', 'Scaling footrest', 1.5000),
            ('NodeSocketInt', 'Reflection', 0),
            ('NodeSocketBool', 'leg_type', False),
            ('NodeSocketFloat', 'leg_dimensions', 0.5000),
            ('NodeSocketFloat', 'leg_z', 1.0000),
        input_kwargs={0: group_input.outputs["Dimensions"], 1: (0.0000, 0.5000, 0.0000)},
    reroute = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Arm Dimensions"]})
        input_kwargs={'Location': multiply.outputs["Vector"], 'CenteringLoc': (0.0000, 1.0000, 0.0000), 'Dimensions': reroute, 'Vertices Z': 10},
    reroute_1 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': arm_cube})
    separate_xyz = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position})
    separate_xyz_1 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': reroute})
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Z"], 1: -0.1000, 2: separate_xyz_1.outputs["Z"], 3: -0.1000, 4: 0.2000})
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["arm_width"], 'Value': map_range.outputs["Result"]})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0092, 0.7688), (0.1011, 0.5937), (0.1494, 0.4062), (0.3954, 0.0781), (1.0000, 0.2187)])
    separate_xyz_2 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply.outputs["Vector"]})
        input_kwargs={0: separate_xyz.outputs["Y"], 1: separate_xyz_2.outputs["Y"]},
    position_1 = nw.new_node(Nodes.InputPosition)
    separate_xyz_14 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': position_1})
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz_14.outputs["X"], 1: -1.0000, 2: 0.6000, 3: 2.1000, 4: -1.1000})
    float_curve_1 = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Factor': group_input.outputs["Arm_height"], 'Value': map_range_1.outputs["Result"]})
    node_utils.assign_curve(float_curve_1.mapping.curves[0], [(0.1341, 0.2094), (0.7386, 1.0000), (0.9682, 0.0781), (1.0000, 0.0000)])
    separate_xyz_15 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': (-2.9000, 3.3000, 0.0000)})
        input_kwargs={0: separate_xyz_14.outputs["Z"], 1: separate_xyz_15.outputs["Z"]},
    multiply_2 = nw.new_node(Nodes.Math, input_kwargs={0: float_curve_1, 1: subtract_1}, attrs={'operation': 'MULTIPLY'})
    
    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': multiply_1, 'Z': multiply_2})
    
    vector_rotate = nw.new_node(Nodes.VectorRotate,
        input_kwargs={'Vector': combine_xyz, 'Axis': (1.0000, 0.0000, 0.0000), 'Angle': group_input.outputs["arms_angle"]})
    
    set_position = nw.new_node(Nodes.SetPosition, input_kwargs={'Geometry': reroute_1, 'Offset': vector_rotate})
    
    multiply_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Dimensions"], 1: (0.0000, 0.5000, 0.0000)},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz_3 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Arm Dimensions"]})
    
    subtract_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_3.outputs["Z"], 1: separate_xyz_3.outputs["Y"]},
        attrs={'operation': 'SUBTRACT'})
    
    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_3.outputs["X"], 'Y': separate_xyz_3.outputs["Y"], 'Z': subtract_2})
    
    reroute_2 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz_1})
        input_kwargs={'Location': multiply_3.outputs["Vector"], 'CenteringLoc': (0.0000, 1.0000, 0.0000), 'Dimensions': reroute_2},
    separate_xyz_4 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': reroute_2})
    multiply_4 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_4.outputs["X"], 1: 1.0001}, attrs={'operation': 'MULTIPLY'})
    reroute_3 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': multiply_4})
        input_kwargs={'Side Segments': 4, 'Radius': separate_xyz_4.outputs["Y"], 'Depth': reroute_3},
    divide = nw.new_node(Nodes.Math, input_kwargs={0: reroute_3, 1: 2.0000}, attrs={'operation': 'DIVIDE'})
    separate_xyz_5 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply_3.outputs["Vector"]})
    combine_xyz_2 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': divide, 'Y': separate_xyz_5.outputs["Y"], 'Z': separate_xyz_4.outputs["Z"]})
    separate_xyz_6 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Back Dimensions"]})
    
    separate_xyz_7 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Arm Dimensions"]})
    
    separate_xyz_8 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Dimensions"]})
    
        input_kwargs={0: separate_xyz_7.outputs["Y"], 1: -2.0000, 2: separate_xyz_8.outputs["Y"]},
    
    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_6.outputs["X"], 'Y': multiply_add, 'Z': separate_xyz_6.outputs["Z"]})
    
        input_kwargs={'CenteringLoc': (0.0000, 0.5000, -1.0000), 'Dimensions': combine_xyz_3, 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
    join_geometry_3 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [join_geometry_2, back_board]})
    
    multiply_5 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz_3, 1: (1.0000, 0.0000, 0.0000)},
    
        input_kwargs={0: group_input.outputs["Arm Dimensions"], 1: (0.0000, -2.0000, 0.0000), 2: group_input.outputs["Dimensions"]},
    
        input_kwargs={0: group_input.outputs["Back Dimensions"], 1: (-1.0000, 0.0000, 0.0000), 2: multiply_add_1.outputs["Vector"]},
    
    separate_xyz_9 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply_add_2.outputs["Vector"]})
    
    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_9.outputs["X"], 'Y': separate_xyz_9.outputs["Y"], 'Z': group_input.outputs["Baseboard Height"]})
    
        input_kwargs={'Location': multiply_5.outputs["Vector"], 'CenteringLoc': (0.0000, 0.5000, -1.0000), 'Dimensions': combine_xyz_4, 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
    reroute_13 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Count"]})
    
    equal = nw.new_node(Nodes.Compare, input_kwargs={2: reroute_13, 3: 4}, attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    reroute_5 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': separate_xyz_9.outputs["Y"]})
    
    separate_xyz_10 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Seat Dimensions"]})
    
    divide_1 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: separate_xyz_10.outputs["Y"]}, attrs={'operation': 'DIVIDE'})
    
    ceil = nw.new_node(Nodes.Math, input_kwargs={0: divide_1}, attrs={'operation': 'CEIL'})
    
    combine_xyz_14 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.0000, 'Y': ceil, 'Z': 1.0000})
    
    divide_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: combine_xyz_4, 1: combine_xyz_14}, attrs={'operation': 'DIVIDE'})
    
    reroute_12 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': divide_2.outputs["Vector"]})
    
    base_board_1 = nw.new_node(nodegroup_corner_cube().name,
        input_kwargs={'Location': multiply_5.outputs["Vector"], 'CenteringLoc': (0.0000, 0.5000, -1.0000), 'Dimensions': reroute_12, 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
        label='BaseBoard')
    
    equal_1 = nw.new_node(Nodes.Compare,
        input_kwargs={0: 4.0000, 2: reroute_13, 3: 4},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    switch_8 = nw.new_node(Nodes.Switch,
        input_kwargs={0: equal_1, 8: divide_2.outputs["Vector"], 9: combine_xyz_4},
        attrs={'input_type': 'VECTOR'})
    
    separate_xyz_16 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': switch_8.outputs[3]})
    
    multiply_6 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_16.outputs["Y"], 1: 0.7000}, attrs={'operation': 'MULTIPLY'})
    
    grid_1 = nw.new_node(Nodes.MeshGrid, input_kwargs={'Size Y': multiply_6, 'Vertices X': 1, 'Vertices Y': 2})
    
    combine_xyz_18 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': 0.1000, 'Y': separate_xyz_16.outputs["Y"], 'Z': separate_xyz_16.outputs["Z"]})
    
    subtract_3 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: switch_8.outputs[3], 1: combine_xyz_18},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_7 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Back Dimensions"], 1: (1.0000, 0.0000, 0.0000)},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.VectorMath, input_kwargs={0: subtract_3.outputs["Vector"], 1: multiply_7.outputs["Vector"]})
    
    transform_geometry_10 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': grid_1.outputs["Mesh"], 'Translation': add.outputs["Vector"], 'Scale': (1.0000, 1.0000, 0.9000)})
    
    cone = nw.new_node('GeometryNodeMeshCone',
        input_kwargs={'Vertices': group_input.outputs["leg_faces"], 'Side Segments': 4, 'Radius Top': 0.0100, 'Radius Bottom': 0.0250, 'Depth': 0.0700})
    
    reroute_9 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["leg_dimensions"]})
    
    combine_xyz_17 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': reroute_9, 'Y': reroute_9, 'Z': group_input.outputs["leg_z"]})
    
    transform_geometry_9 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': cone.outputs["Mesh"], 'Translation': (0.0000, 0.0000, 0.0100), 'Rotation': (0.0000, 3.1416, 0.0000), 'Scale': combine_xyz_17})
    
    foot_cube = nw.new_node(nodegroup_corner_cube().name,
        input_kwargs={'CenteringLoc': (0.5000, 0.5000, 0.9000), 'Dimensions': group_input.outputs["Foot Dimensions"]},
        label='FootCube')
    
    transform_geometry_12 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': foot_cube, 'Scale': (0.5000, 0.8000, 0.8000)})
    
    switch_6 = nw.new_node(Nodes.Switch,
        input_kwargs={1: group_input.outputs["leg_type"], 14: transform_geometry_9, 15: transform_geometry_12})
    
    transform_geometry_8 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': switch_6.outputs[6]})
    
    instance_on_points_1 = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': transform_geometry_10, 'Instance': transform_geometry_8, 'Scale': (1.0000, 1.0000, 1.2000)})
    
    realize_instances_1 = nw.new_node(Nodes.RealizeInstances, input_kwargs={'Geometry': instance_on_points_1})
    
    join_geometry_10 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [base_board_1, realize_instances_1]})
    
    subtract_4 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz_14, 1: (1.0000, 1.0000, 1.0000)},
        attrs={'operation': 'SUBTRACT'})
    
    multiply_8 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: subtract_4.outputs["Vector"], 1: (0.0000, 0.5000, 0.0000)},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_9 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: divide_2.outputs["Vector"], 1: multiply_8.outputs["Vector"]},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_16 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.0000, 'Y': group_input.outputs["Reflection"], 'Z': 1.0000})
    
    multiply_10 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_9.outputs["Vector"], 1: combine_xyz_16},
        attrs={'operation': 'MULTIPLY'})
    
    combine_xyz_12 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Scaling footrest"], 'Y': 1.0000, 'Z': 1.0000})
    
    transform_geometry_5 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': join_geometry_10, 'Translation': multiply_10.outputs["Vector"], 'Scale': combine_xyz_12})
    
    switch_2 = nw.new_node(Nodes.Switch, input_kwargs={1: group_input.outputs["Footrest"], 15: transform_geometry_5})
    
    combine_xyz_19 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Scaling footrest"], 'Y': 1.3000, 'Z': 1.0000})
    
    transform_geometry_11 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': realize_instances_1, 'Scale': combine_xyz_19})
    
    base_board_2 = nw.new_node(nodegroup_corner_cube().name,
        input_kwargs={'Location': multiply_5.outputs["Vector"], 'CenteringLoc': (0.0000, 0.5000, -1.0000), 'Dimensions': combine_xyz_4, 'Vertices X': 3, 'Vertices Y': 3, 'Vertices Z': 3},
        label='BaseBoard')
    
    combine_xyz_13 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Scaling footrest"], 'Y': 1.0000, 'Z': 1.0000})
    
    transform_geometry_6 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': base_board_2, 'Scale': combine_xyz_13})
    
    join_geometry_11 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [transform_geometry_11, transform_geometry_6]})
    
    switch_4 = nw.new_node(Nodes.Switch, input_kwargs={1: group_input.outputs["Footrest"], 15: join_geometry_11})
    
    switch_5 = nw.new_node(Nodes.Switch, input_kwargs={1: equal, 14: switch_2.outputs[6], 15: switch_4.outputs[6]})
    
    join_geometry_4 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [join_geometry_3, base_board, switch_5.outputs[6]]})
    
    grid = nw.new_node(Nodes.MeshGrid, input_kwargs={'Vertices X': 2, 'Vertices Y': 2})
    multiply_11 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Dimensions"], 1: (0.5000, 0.0000, 0.0000)},
    multiply_12 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Dimensions"], 1: (1.0000, 1.0000, 0.0000)},
    multiply_13 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Foot Dimensions"], 1: (2.5000, 2.5000, 0.0000)},
        attrs={'operation': 'MULTIPLY'})
    
    subtract_5 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_12.outputs["Vector"], 1: multiply_13.outputs["Vector"]},
    
    transform_geometry_2 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': grid.outputs["Mesh"], 'Translation': multiply_11.outputs["Vector"], 'Scale': subtract_5.outputs["Vector"]})
    
    instance_on_points = nw.new_node(Nodes.InstanceOnPoints,
        input_kwargs={'Points': transform_geometry_2, 'Instance': transform_geometry_8})
    
    join_geometry_5 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [join_geometry_4, realize_instances]})
    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Count"]})
    
    equal_2 = nw.new_node(Nodes.Compare,
        input_kwargs={1: 4.0000, 2: reroute_10, 3: 4},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    reroute_4 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz_4})
    multiply_14 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: reroute_4, 1: (0.0000, -0.5000, 1.0000)},
    multiply_15 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: reroute_4, 1: (0.0000, 0.5000, 1.0000)},
        attrs={'operation': 'MULTIPLY'})
    equal_3 = nw.new_node(Nodes.Compare,
        input_kwargs={1: 4.0000, 2: reroute_10, 3: 4},
        attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    reroute_11 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': group_input.outputs["Reflection"]})
    switch_7 = nw.new_node(Nodes.Switch, input_kwargs={0: equal_3, 4: reroute_11, 5: 1}, attrs={'input_type': 'INT'})
    combine_xyz_15 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.0000, 'Y': switch_7.outputs[1], 'Z': 1.1000})
    
    multiply_16 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_15.outputs["Vector"], 1: combine_xyz_15},
        attrs={'operation': 'MULTIPLY'})
    divide_3 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_5, 1: ceil}, attrs={'operation': 'DIVIDE'})
    combine_xyz_5 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': separate_xyz_10.outputs["X"], 'Y': divide_3, 'Z': separate_xyz_10.outputs["Z"]})
    
    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': combine_xyz_5})
    
    multiply_17 = nw.new_node(Nodes.VectorMath, input_kwargs={0: reroute_6, 1: combine_xyz_15}, attrs={'operation': 'MULTIPLY'})
    
    multiply_18 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: combine_xyz_5, 1: (1.0000, 1.0300, 1.0000)},
        input_kwargs={'CenteringLoc': (0.0000, 0.5000, 0.0000), 'Dimensions': multiply_18.outputs["Vector"], 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
    
    index = nw.new_node(Nodes.Index)
    
    equal_4 = nw.new_node(Nodes.Compare, input_kwargs={2: index, 3: 1}, attrs={'operation': 'EQUAL', 'data_type': 'INT'})
    
    store_named_attribute_1 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': seat_cushion, 'Selection': equal_4, 'Name': 'TAG_support', 6: True},
        attrs={'data_type': 'BOOLEAN', 'domain': 'FACE'})
    
    value = nw.new_node(Nodes.Value)
    value.outputs[0].default_value = 1.0000
    
    store_named_attribute_2 = nw.new_node(Nodes.StoreNamedAttribute,
        input_kwargs={'Geometry': store_named_attribute_1, 'Selection': value, 'Name': 'TAG_cushion', 6: True},
        attrs={'data_type': 'BOOLEAN', 'domain': 'FACE'})
    
    combine_xyz_6 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Seat Margin"], 'Y': group_input.outputs["Seat Margin"], 'Z': 1.0000})
    
    transform_geometry_3 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': store_named_attribute_2, 'Scale': combine_xyz_6})
    
    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Scaling footrest"], 'Y': 1.0000, 'Z': 1.1000})
    
    transform_geometry_7 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': transform_geometry_3, 'Scale': combine_xyz_11})
    
    nodegroup_array_fill_line_002 = nw.new_node(nodegroup_array_fill_line().name,
        input_kwargs={'Line Start': multiply_14.outputs["Vector"], 'Line End': multiply_16.outputs["Vector"], 'Instance Dimensions': multiply_17.outputs["Vector"], 'Count': reroute_10, 'Instance': transform_geometry_7})
    
    separate_xyz_17 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': multiply_16.outputs["Vector"]})
    
    combine_xyz_21 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Z': separate_xyz_17.outputs["Z"]})
    
    reroute_14 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': ceil})
    
    combine_xyz_20 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': 1.0000, 'Y': reroute_14, 'Z': 1.0000})
    
    transform_geometry_13 = nw.new_node(Nodes.Transform, input_kwargs={'Geometry': transform_geometry_7, 'Scale': combine_xyz_20})
    
    nodegroup_array_fill_line_002_1 = nw.new_node(nodegroup_array_fill_line().name,
        input_kwargs={'Line End': combine_xyz_21, 'Count': 1, 'Instance': transform_geometry_13})
    switch_9 = nw.new_node(Nodes.Switch,
        input_kwargs={1: equal_2, 14: nodegroup_array_fill_line_002, 15: nodegroup_array_fill_line_002_1})
    
    switch_3 = nw.new_node(Nodes.Switch, input_kwargs={1: group_input.outputs["Footrest"], 15: switch_9.outputs[6]})
    
    nodegroup_array_fill_line_002_2 = nw.new_node(nodegroup_array_fill_line().name,
        input_kwargs={'Line Start': multiply_14.outputs["Vector"], 'Line End': multiply_15.outputs["Vector"], 'Instance Dimensions': reroute_6, 'Count': reroute_14, 'Instance': transform_geometry_3})
    
    join_geometry_9 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [switch_3.outputs[6], nodegroup_array_fill_line_002_2]})
    
    subdivide_mesh = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': join_geometry_9, 'Level': 2})
    separate_xyz_11 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Seat Dimensions"]})
    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': group_input.outputs["Backrest Width"], 'Z': separate_xyz_11.outputs["Z"]})
    add_1 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_14.outputs["Vector"], 1: combine_xyz_7})
    add_2 = nw.new_node(Nodes.VectorMath, input_kwargs={0: multiply_15.outputs["Vector"], 1: combine_xyz_7})
    separate_xyz_12 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Dimensions"]})
    subtract_6 = nw.new_node(Nodes.Math,
        input_kwargs={0: separate_xyz_12.outputs["Z"], 1: separate_xyz_11.outputs["Z"]},
        attrs={'operation': 'SUBTRACT'})
    
    subtract_7 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract_6, 1: group_input.outputs["Baseboard Height"]},
    
    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ,
        input_kwargs={'X': subtract_7, 'Y': divide_3, 'Z': group_input.outputs["Backrest Width"]})
        input_kwargs={'CenteringLoc': (0.1000, 0.5000, 1.0000), 'Dimensions': combine_xyz_8, 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
    store_named_attribute_3 = nw.new_node(Nodes.StoreNamedAttribute,
    multiply_19 = nw.new_node(Nodes.Math,
        input_kwargs={0: group_input.outputs["Backrest Width"], 1: -1.0000},
    separate_xyz_13 = nw.new_node(Nodes.SeparateXYZ, input_kwargs={'Vector': group_input.outputs["Back Dimensions"]})
    add_3 = nw.new_node(Nodes.Math, input_kwargs={0: separate_xyz_13.outputs["X"], 1: 0.1000})
    add_4 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_19, 1: add_3})
    combine_xyz_9 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'X': add_4})
    add_5 = nw.new_node(Nodes.Math, input_kwargs={0: group_input.outputs["Backrest Angle"], 1: -1.5708})
    combine_xyz_10 = nw.new_node(Nodes.CombineXYZ, input_kwargs={'Y': add_5})
    transform_geometry_4 = nw.new_node(Nodes.Transform,
        input_kwargs={'Geometry': store_named_attribute_3, 'Translation': combine_xyz_9, 'Rotation': combine_xyz_10, 'Scale': combine_xyz_6})
    nodegroup_array_fill_line_003 = nw.new_node(nodegroup_array_fill_line().name,
        input_kwargs={'Line Start': add_1.outputs["Vector"], 'Line End': add_2.outputs["Vector"], 'Instance Dimensions': reroute_6, 'Count': ceil, 'Instance': transform_geometry_4})
    join_geometry_6 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [subdivide_mesh, nodegroup_array_fill_line_003]})
    join_geometry_7 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [join_geometry_5, realize_instances, join_geometry_6]})
    subdivide_mesh_1 = nw.new_node(Nodes.SubdivideMesh, input_kwargs={'Mesh': join_geometry_5, 'Level': 2})
    join_geometry_8 = nw.new_node(Nodes.JoinGeometry, input_kwargs={'Geometry': [subdivide_mesh_1, realize_instances, join_geometry_6]})
    switch_1 = nw.new_node(Nodes.Switch, input_kwargs={1: True, 14: join_geometry_7, 15: subdivision_surface_2})
        input_kwargs={'CenteringLoc': (0.0000, 0.5000, -1.0000), 'Dimensions': group_input.outputs["Dimensions"], 'Vertices X': 2, 'Vertices Y': 2, 'Vertices Z': 2},
    reroute_7 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': bounding_box})
    reroute_8 = nw.new_node(Nodes.Reroute, input_kwargs={'Input': reroute_7})
        input_kwargs={'Geometry': switch_1.outputs[6], 'BoundingBox': reroute_8},


            uniform(0.06, 0.15),
            uniform(0.7, 1),
        'Baseboard Height': uniform(0.05, 0.09),
        'arm_width': uniform(0.6, 0.9),
        'Arm_height': uniform(0.7,1.0),
        'arms_angle': uniform(0.0, 1.08),
        'Count': 1 if uniform()>0.2 else 4,
        'Reflection':1 if uniform()>0.5 else -1,
        'leg_type': True if uniform()>0.5 else False,
        'leg_dimensions': uniform(0.4,0.9),
        'leg_z':uniform(1.1, 2.5),
        'leg_faces':uniform(4,25)

    def __init__(self, factory_seed):
        from infinigen.assets.clothes import blanket
        super().__init__(factory_seed)
        with FixedSeed(factory_seed):
            self.params = sofa_parameter_distribution()

    def create_placeholder(self, **_):
        obj = butil.spawn_vert()
        butil.modify_mesh(
            obj,
            'NODES',
            node_group=nodegroup_sofa_geometry(),
            apply=True
        )
        tagging.tag_system.relabel_obj(obj)
        return obj

    def create_asset(self, i, placeholder, face_size, **_):
        hipoly = butil.copy(placeholder, keep_materials=True)
        butil.modify_mesh(hipoly, 'SUBSURF', levels=1, apply=True)

