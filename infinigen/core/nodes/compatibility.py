import logging
from collections import OrderedDict

from .node_info import Nodes

logger = logging.getLogger(__name__)

def map_dict_keys(d, m):
    for m_from, m_to in m.items():
        if m_from not in d:
            continue
        if m_to in d:
            raise ValueError(f'{m_from} would map to {m_to} but {d} already contains that key')
        d[m_to] = d.pop(m_from)
    return d

def make_virtual_mixrgb(nw, orig_type, input_args, attrs, input_kwargs):
    attrs['data_type'] = 'RGBA'

    key_mapping =  OrderedDict({'Fac': 'Factor', 'Color1': 'A', 'Color2': 'B'})
    map_dict_keys(input_kwargs, key_mapping)

    # any previous uses of input_args are no longer valid, since the node has lots of hidden type-based sockets now
    # we will convert any input_args present into input_kwargs instead
    for k, a in zip(key_mapping.values(), input_args):
        if k in input_kwargs:
            raise ValueError(f'In {make_virtual_mixrgb}, encountered {orig_type} with conflicting {len(input_args)=} and {input_kwargs.keys()}')
        input_kwargs[k] = a
    input_args = []

    return nw.new_node(node_type=Nodes.Mix, input_args=input_args, 
                       attrs=attrs, input_kwargs=input_kwargs, compat_mode=False)

def make_virtual_transfer_attribute(nw, orig_type, input_args, attrs, input_kwargs):
    if attrs is None:
        raise ValueError(f'{attrs=} in make_virtual_transfer_attribute, cannot infer correct node type mapping')

    if attrs['mapping'] == 'NEAREST_FACE_INTERPOLATED':
        mapped_type = Nodes.SampleNearestSurface
        map_dict_keys(input_kwargs, {'Source': 'Mesh', 'Attribute': 'Value', 'Source Position': 'Sample Position'})
    elif attrs['mapping'] == 'NEAREST':
        raise ValueError("Compatibility mapping for mode='NEAREST' is not supported, please modify the code to resolve this outdated instance of TransferAttribute")
    elif attrs['mapping'] == 'INDEX':
        mapped_type = Nodes.SampleIndex
        map_dict_keys(input_kwargs, {'Source': 'Geometry', 'Attribute': 'Value'})
    else:
        assert False

    logger.warning(f'Converting request for Nodes.TransferAttribute to {mapped_type}'
                    f'to ensure compatibility with bl3.3 code, but this is unsafe. Please update to avoid {Nodes.TransferAttribute}')
   
    return nw.new_node(node_type=mapped_type, input_args=input_args, 
                       attrs=attrs, input_kwargs=input_kwargs, compat_mode=False)    

def compat_args_sample_curve(nw, orig_type, input_args, attrs, input_kwargs):
    map_dict_keys(input_kwargs, {'Curve': 'Curves'})
    return nw.new_node(node_type=orig_type, input_args=input_args, 
                       attrs=attrs, input_kwargs=input_kwargs, compat_mode=False)

COMPATIBILITY_MAPPINGS = {
    Nodes.MixRGB: make_virtual_mixrgb,
    Nodes.TransferAttribute: make_virtual_transfer_attribute,
    Nodes.SampleCurve: compat_args_sample_curve
}