import logging


import bpy

from nodes.node_info import Nodes, NODE_ATTRS_AVAILABLE
from nodes import node_info

logger = logging.getLogger(__name__)

        message.args = f"{message.args[0]}. The issue is probably coming from {traceback_files.pop()}",
def isnode(x):
    return isinstance(x, (bpy.types.ShaderNode, bpy.types.NodeInternal, bpy.types.GeometryNode))


    """
    Figure out if `item` somehow represents a node with an output we can use. 
    If so, return that output socket
    """

    if isinstance(item, bpy.types.NodeSocket):
        res = item
    elif isnode(item):
        # take the first active socket
        try:
            res = next(o for o in item.outputs if o.enabled)
        except StopIteration:
            raise ValueError(f'Attempted to get output socket for {item} but none are enabled!')
    elif isinstance(item, tuple) and isnode(item[0]):
        node, socket_name = item
        if isinstance(socket_name, int):
            return node.outputs[socket_name]
        try:
            res = next(o for o in node.outputs if o.enabled and o.name == socket_name)
        except StopIteration:
    else:
        return None

    if not res.enabled:

    return res


    if isinstance(input_socket_name, str):
        try:
            input_socket = next(i for i in node.inputs if i.name == input_socket_name and i.enabled)
        except:
            input_socket = node.inputs[input_socket_name]
    else:
        input_socket = node.inputs[input_socket_name]

    if not input_socket.enabled:

    return input_socket

class NodeWrangler():

        if issubclass(type(node_group), bpy.types.NodeTree):
            self.modifier = None
            self.node_group = node_group
        elif issubclass(type(node_group), bpy.types.NodesModifier):
            self.modifier = node_group
            self.node_group = self.modifier.node_group
        else:
            raise ValueError(f'Couldnt initialize NodeWrangler with {node_group=}, {type(node_group)=}')

        self.nodes = self.node_group.nodes
        self.links = self.node_group.links

        self.nodegroup_input_data = {}

        self.input_attribute_data = {}
        self.position_translation_seed = {}

        if input_args is None:
            input_args = []
        if input_kwargs is None:
            input_kwargs = {}

        node = self._make_node(node_type)
        if label is not None:
            node.label = label

        if attrs is not None:
            for key, val in attrs.items():
                try:
                    setattr(node, key, val)
                except AttributeError:
                    logger.debug(f"{w}, it is fixed automatically by using position for consistency")
        input_keyval_list = list(enumerate(input_args)) + list(input_kwargs.items())
        for input_socket_name, input_item in input_keyval_list:
            if input_item is None:
                continue
            if node_type == Nodes.GroupOutput:
                if input_socket_name not in node.inputs:
                    nodeclass = infer_output_socket(input_item).bl_idname
                    self.node_group.outputs.new(nodeclass, input_socket_name)
                    assert input_socket_name in node.inputs and node.inputs[input_socket_name].enabled
            input_socket = infer_input_socket(node, input_socket_name)

                nodeclass, name, val = inp
                self.expose_input(name, val=val, dtype=nodeclass)
        return node

        '''
        Expose an input to the nodegroups interface, making it able to be specified externally

        If this nodegroup is 
        '''

        if attribute is not None:
            if self.modifier is None and val is None:
                raise ValueError(


        if name in self.node_group.inputs:
            assert len([o for o in group_input.outputs if o.name == name]) == 1
            return group_input.outputs[name]

        # Infer from args what type of node input to make (NodeSocketFloat / NodeSocketVector / etc)
        nodeclass = self._infer_nodeclass_from_args(dtype, val)
        inp = self.node_group.inputs.new(nodeclass, name)

        def prepare_cast(to_type, val):
            # cast val only when necessary, and only when type(val) wont crash
            if to_type not in [bpy.types.bpy_prop_array, bpy.types.bpy_prop]:
                val = to_type(val)
            return val

        if val is not None:
            if not hasattr(inp, 'default_value') or inp.default_value is None:
            inp.default_value = prepare_cast(type(inp.default_value), val)

        if self.modifier is not None:
            if val is not None:
                if hasattr(curr_mod_inp_val, 'real'):
            if attribute is not None:
        assert len([o for o in group_input.outputs if o.name == name]) == 1
        return group_input.outputs[name]

    @staticmethod
    def _infer_nodeclass_from_args(dtype, val=None):
        '''

        All of these must be mapped to some node_info.NODECLASS in order to create a node.
        Optionally, we can try to infer a nodeclass from the type of a provided `val`
        '''

        if dtype is None:
            if val is not None:
                datatype = node_info.PYTYPE_TO_DATATYPE[type(val)]
            else:
                datatype = 'FLOAT_VECTOR'
            return node_info.DATATYPE_TO_NODECLASS[datatype]
        else:
            if dtype in node_info.NODECLASSES:
            else:
                if dtype in node_info.NODETYPE_TO_DATATYPE:
                    datatype = node_info.NODETYPE_TO_DATATYPE[dtype]
                elif dtype in node_info.PYTYPE_TO_DATATYPE:
                    datatype = node_info.PYTYPE_TO_DATATYPE[dtype]
                else:
                    raise ValueError(f'Could not parse {dtype=}')
                return node_info.DATATYPE_TO_NODECLASS[datatype]
    def _update_socket(self, input_socket, input_item):
        output_socket = infer_output_socket(input_item)

        if output_socket is None and hasattr(input_socket, 'default_value'):
            # we couldnt parse the inp to be any kind of node, it must be a default_value for us to assign
            try:
                input_socket.default_value = input_item
                return
            except TypeError as e:
                print(f'TypeError while assigning {input_item=} as default_value for {input_socket.name}')
                raise e
        self.links.new(output_socket, input_socket)

    def connect_input(self, input_socket, input_item):
        if isinstance(input_item, list) and any(infer_output_socket(i) is not None for i in input_item):
            if not input_socket.is_multi_input:
            for inp in input_item:
                self._update_socket(input_socket, inp)
        else:

    def _make_node(self, node_type):
        if node_type in node_info.SINGLETON_NODES:
            # for nodes in the singletons list, we should reuse an existing one if it exists
            try:
                node = next(n for n in self.nodes if n.bl_idname == node_type)
            except StopIteration:
                node = self.nodes.new(node_type)
            nodegroup_type = {
                'ShaderNodeTree': 'ShaderNodeGroup', 
                'GeometryNodeTree': 'GeometryNodeGroup',
                'CompositorNodeTree': 'CompositorNodeGroup'
            }[
            node = self.nodes.new(nodegroup_type)
            node = self.nodes.new(node_type)

        return node

    def get_position_translation_seed(self, i):
        if not i in self.position_translation_seed:
            self.position_translation_seed[i] = random_vector3()
        return self.position_translation_seed[i]

    def add2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes))

    def scalar_add2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes))

    def scalar_multiply2(self, *nodes):
        return self.new_node(Nodes.Math, list(nodes), {"operation": "MULTIPLY"})

    def sub2(self, *nodes):
        return self.new_node(Nodes.VectorMath, list(nodes), {"operation": "SUBTRACT"})

        return self.new_node(Nodes.Math, list(nodes), {"operation": "SUBTRACT"})



    def add(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.add2(*nodes)
        return self.add2(nodes[0], self.add(*nodes[1:]))

        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:

    def scalar_add(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.scalar_add2(*nodes)
        return self.scalar_add2(nodes[0], self.scalar_add(*nodes[1:]))

    def scalar_multiply(self, *nodes):
        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 2:
            return self.scalar_multiply2(*nodes)
