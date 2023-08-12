# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma



import re
from collections import OrderedDict

import numpy as np
from infinigen.terrain.utils import SocketType, Vars, KernelDataType, usable_name, Nodes, NODE_ATTRS_AVAILABLE, SOCKETTYPE_KERNEL, \
    sanitize, special_sanitize, special_sanitize_float_curve, concat_string, value_string, var_list, NODE_FUNCTIONS, \
    collecting_vars, get_imp_var_name, special_sanitize_constant

functional_nodes = [
    Nodes.SetPosition, Nodes.InputPosition, Nodes.InputNormal,
    Nodes.GroupOutput, Nodes.GroupInput,
]

def my_getattr(x, a):
    if "." not in a:
        return getattr(x, a)
    else:
        return getattr(my_getattr(x, ".".join(a.split(".")[:-1])), a.split(".")[-1])

class Kernelizer:

    def get_inputs(self, node_tree):
        inputs = OrderedDict()
        for node_input in node_tree.inputs:
            if node_input.type != SocketType.Geometry:
                assert(node_input.type != SocketType.Image)
                inputs[node_input.identifier] = SOCKETTYPE_KERNEL[node_input.type]
        return inputs

    def get_output(self, node_tree):
        outputs = OrderedDict()
        for node in node_tree.nodes:
            if node.bl_idname == Nodes.SetPosition:
                outputs[Vars.Offset] = KernelDataType.float3
        for node_output in node_tree.outputs:
            if node_output.type != SocketType.Geometry:
                outputs[node_output.identifier] = SOCKETTYPE_KERNEL[node_output.type]
        return outputs

    def regularize(self, node_tree):
        use_position = False
        use_normal = False
        nodes = []
        links = []
        n_set_position = 0
        for node in node_tree.nodes:
            if node.bl_idname not in functional_nodes:
                nodes.append(node)
            elif node.bl_idname == Nodes.SetPosition:
                n_set_position += 1
            if node.bl_idname in [Nodes.InputPosition, Nodes.SetPosition]:
                use_position = 1
            elif node.bl_idname == Nodes.InputNormal:
                use_normal = 1
        # only accept a single set position node, please add multiple ones together
        assert(n_set_position <= 1)

        for link in node_tree.links:
            from_node = link.from_node
            to_node = link.to_node
            from_socket = link.from_socket
            to_socket = link.to_socket
            from_socket_dtype = link.from_socket.type
            to_socket_dtype = link.to_socket.type
            need_link0 = False
            if from_node.bl_idname == Nodes.GroupInput:
                if from_socket_dtype != SocketType.Geometry:
                    need_link0 = True
                    from_node = None
                    from_socket = from_socket.identifier
            elif from_node.bl_idname == Nodes.InputPosition:
                need_link0 = True
                from_node = None
                from_socket = Vars.Position
            elif from_node.bl_idname == Nodes.InputNormal:
                need_link0 = True
                from_node = None
                from_socket = Vars.Normal
            else:
                if from_socket_dtype != SocketType.Geometry:
                    need_link0 = True
            need_link1 = False
            if to_node.bl_idname == Nodes.GroupOutput:
                if to_socket_dtype != SocketType.Geometry:
                    need_link1 = True
                    to_node = None
                    to_socket = to_socket.identifier
            elif to_node.bl_idname == Nodes.SetPosition:
                if to_socket_dtype != SocketType.Geometry:
                    need_link1 = True
                    to_node = None
                    to_socket = Vars.Offset
            else:
                if to_socket_dtype != SocketType.Geometry:
                    need_link1 = True
            if need_link0 and need_link1:
                links.append((from_node, from_socket, to_node, to_socket))

        while any(node.bl_idname == Nodes.CaptureAttribute for node in nodes):
            for i, node in enumerate(nodes):
                if node.bl_idname == Nodes.CaptureAttribute:
                    capture_node = i
                    break
            tail_links = []
            for i, link in enumerate(links):
                from_node, from_socket, to_node, to_socket = link
                if from_node is not None and from_node.name == node.name:
                    tail_links.append(i)
            for i, link in enumerate(links):
                from_node, from_socket, to_node, to_socket = link
                if to_node is not None and to_node.name == node.name:
                    head_link = i
                    break
            new_links = []
            for i, link in enumerate(links):
                if i != head_link and i not in tail_links:
                    new_links.append(link)
                elif i != head_link:
                    from_node, from_socket, to_node, to_socket = link
                    new_links.append(
                        (links[head_link][0], links[head_link][1], to_node, to_socket))
            links = new_links
            nodes = nodes[:capture_node] + nodes[capture_node + 1:]
        
        outlets_count = {}
        for node in nodes:
            outlets_count[node.name] = 0
        for link in links:
            from_node, from_socket, to_node, to_socket = link
            if from_node is not None:
                outlets_count[from_node.name] += 1
        flag = False
        for node in outlets_count:
            if outlets_count[node] == 0:
                flag = True
                break
        while flag:
            # print(f"{node} is unused, pruning")
            new_nodes = []
            for node0 in nodes:
                if node0.name != node:
                    new_nodes.append(node0)
            nodes = new_nodes
            new_links = []
            for link in links:
                from_node, from_socket, to_node, to_socket = link
                if to_node is not None and to_node.name == node: continue
                new_links.append(link)
            links = new_links

            outlets_count = {}
            for node in nodes:
                outlets_count[node.name] = 0
            for link in links:
                from_node, from_socket, to_node, to_socket = link
                if from_node is not None:
                    outlets_count[from_node.name] += 1
            flag = False
            for node in outlets_count:
                if outlets_count[node] == 0:
                    flag = True
                    break

        return nodes, links, use_position, use_normal

    def code(self, nodes, links, kernel_inputs, kernel_impl_inputs, kernel_outputs, node_tree_name):
        nodes = {node.name: node for node in nodes}
        dependency_count = {}
        for node in nodes:
            dependency_count[node] = 0
        for link in links:
            from_node, from_socket, to_node, to_socket = link
            if from_node is not None and to_node is not None:
                dependency_count[to_node.name] += 1
        code = ""
        
        while len(dependency_count) > 0:
            for node in dependency_count:
                if dependency_count[node] == 0:
                    head_node = nodes[node]
                    break
            params = [str(sanitize(my_getattr(head_node, p), head_node, p))
                      for p in NODE_ATTRS_AVAILABLE[head_node.bl_idname]]
            inputs = []
            outputs = []
            for socket in head_node.inputs:
                var_input = False
                for link in links:
                    from_node, from_socket, to_node, to_socket = link
                    if to_node is not None and to_socket.identifier == socket.identifier and to_node.name == head_node.name:
                        var_input = True
                        if from_node is None:
                            input_raw = from_socket
                        else:
                            input_raw = usable_name(
                                from_node.name) + "__" + from_socket.identifier
                        if from_node is None:
                            if from_socket in kernel_inputs:
                                from_type = kernel_inputs[from_socket]
                            else:
                                from_type = kernel_impl_inputs[from_socket][0]
                        else:
                            from_type = SOCKETTYPE_KERNEL[from_socket.type]
                        if SOCKETTYPE_KERNEL[socket.type] != from_type:
                            input_raw = f"{SOCKETTYPE_KERNEL[socket.type]}({input_raw})"
                        inputs.append(input_raw)
                        break
                if not var_input:
                    inputs.append(value_string(socket.default_value))
            for socket in head_node.outputs:
                used = False
                for link in links:
                    from_node, from_socket, to_node, to_socket = link
                    if from_node is not None and from_node.name == head_node.name and from_socket.identifier == socket.identifier:
                        used = True
                        break
                if used:
                    varname = usable_name(head_node.name) + \
                        "__" + socket.identifier
                    code += f"{SOCKETTYPE_KERNEL[socket.type]} {varname};\n"
                    outputs.append("&" + varname)
                else:
                    outputs.append("NULL")
            if head_node.bl_idname == Nodes.ColorRamp:
                if head_node.name.endswith("_VAR"):
                    code += special_sanitize(usable_name(head_node.name), my_getattr(head_node, NODE_ATTRS_AVAILABLE[head_node.bl_idname][0]), node_tree_name)
                else:
                    code += special_sanitize_constant(usable_name(head_node.name), my_getattr(head_node, NODE_ATTRS_AVAILABLE[head_node.bl_idname][0]))
            elif head_node.bl_idname == Nodes.FloatCurve:
                code += special_sanitize_float_curve(
                    usable_name(head_node.name),
                    head_node.mapping,
                )
            if head_node.bl_idname in [Nodes.Value, Nodes.Vector, Nodes.InputColor]:
                if used:
                    # code += f'''
                    #     {varname} = {value_string(socket.default_value)};
                    # '''
                    code += f'''
                        {varname} = {get_imp_var_name(node_tree_name, head_node.name)};
                    '''
            else:
                if head_node.bl_idname == Nodes.Group:
                    func = usable_name(head_node.node_tree.name)
                    _, imp_inputs = self.node_tree_dict[head_node.node_tree.name]
                    inputs.extend(sorted(list(imp_inputs.keys())))
                else:
                    func = NODE_FUNCTIONS[head_node.bl_idname]
                code += f'''{func}(
                    {concat_string(params)}
                    {','.join(inputs)},
                    {','.join(outputs)}
                );
                '''
            for link in links:
                from_node, from_socket, to_node, to_socket = link
                if from_node is not None and to_node is not None and from_node.name == head_node.name:
                    dependency_count[to_node.name] -= 1
            del dependency_count[head_node.name]
        for link in links:
            from_node, from_socket, to_node, to_socket = link
            if from_node is None:
                from_socket_name = from_socket
            else:
                from_socket_name = usable_name(
                    from_node.name) + "__" + from_socket.identifier
            if to_node is None:
                to_type = kernel_outputs[to_socket][0]
                if SOCKETTYPE_KERNEL[from_socket.type] != to_type:
                    code += f'''
                        if ({to_socket} != NULL) *{to_socket} = {from_socket_name};
                    '''
                else:
                    code += f'''
                        if ({to_socket} != NULL) *{to_socket} = {to_type}({from_socket_name});
                    '''
        return code

    def execute_node_tree(self, node_tree, collective_style=False):
        code = ""
        nodes, links, use_position, use_normal = self.regularize(node_tree)
        inputs = self.get_inputs(node_tree)
        imp_inputs = {}
        if use_position:
            imp_inputs[Vars.Position] = KernelDataType.float3, None
        if use_normal:
            imp_inputs[Vars.Normal] = KernelDataType.float3, None

        for node in nodes:
            if node.bl_idname == Nodes.Value:
                imp_inputs[get_imp_var_name(node_tree.name, node.name)] = KernelDataType.float, np.array([node.outputs[0].default_value], dtype=np.float32)
            elif node.bl_idname == Nodes.InputColor:
                imp_inputs[get_imp_var_name(node_tree.name, node.name)] = KernelDataType.float4, np.array([node.color[i] for i in range(4)], dtype=np.float32)
            elif node.bl_idname == Nodes.Vector:
                imp_inputs[get_imp_var_name(node_tree.name, node.name)] = KernelDataType.float3, np.array([node.vector[i] for i in range(3)], dtype=np.float32)
            elif node.bl_idname == Nodes.ColorRamp and node.name.endswith("_VAR"):
                for i in range(len(node.color_ramp.elements)):
                    imp_inputs[get_imp_var_name(node_tree.name, node.name) + f"_pos{i}"] = (KernelDataType.float, np.array([node.color_ramp.elements[i].position], dtype=np.float32))
                    imp_inputs[get_imp_var_name(node_tree.name, node.name) + f"_color{i}"] = (KernelDataType.float4, np.array([node.color_ramp.elements[i].color[j] for j in range(4)], dtype=np.float32))

        for node in nodes:
            if node.bl_idname == Nodes.Group and node.node_tree.name not in self.node_tree_dict:
                subcode, sub_imp_inputs, _ = self.execute_node_tree(node.node_tree)
                imp_inputs.update(sub_imp_inputs)
                code += subcode

        outputs = self.get_output(node_tree)

        code += f'''
            DEVICE_FUNC void {usable_name(node_tree.name)}(
                {var_list(inputs, imp_inputs, outputs, collective_style=collective_style)}
            ) {{
        '''
        if collective_style:
            code += collecting_vars(imp_inputs)
        code += self.code(nodes, links, inputs, imp_inputs, outputs, node_tree.name)
        code += "}"
        self.node_tree_dict[node_tree.name] = code, imp_inputs
        return code, imp_inputs, outputs

    def __call__(self, modifier):
        node_tree = modifier.node_group
        self.node_tree_dict = {}
        code, imp_inputs, outputs = self.execute_node_tree(node_tree, collective_style=True)
        for nodeoutput in node_tree.outputs:
            id = nodeoutput.identifier
            if id != 'Output_1': # not Geometry
                code = re.sub(rf"\b{id}\b", modifier[f'{id}_attribute_name'], code)
                outputs[modifier[f'{id}_attribute_name']] = outputs.pop(id)
        return code, imp_inputs, outputs