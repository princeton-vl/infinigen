# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import numpy as np
from infinigen.core.nodes.node_info import Nodes as oNodes
from infinigen.core.nodes.node_info import NODE_ATTRS_AVAILABLE as o_NODE_ATTRS_AVAILABLE
from infinigen.core.surface import Registry

class Vars:
    Position = "position"
    Normal = "normal"
    Offset = "offset"
    SDF = "sdf"

class Nodes(oNodes):
    Group = "GeometryNodeGroup"

NODE_ATTRS_AVAILABLE = o_NODE_ATTRS_AVAILABLE.copy()

NODE_ATTRS_AVAILABLE.update({
    Nodes.ColorRamp: ["color_ramp.elements", "color_ramp.color_mode", "color_ramp.interpolation", "color_ramp.hue_interpolation"],    
    Nodes.MixRGB: ['use_clamp', 'blend_type'],
    Nodes.Mix: ['data_type', 'blend_type', 'clamp_result', 'clamp_factor', 'factor_mode'],
    Nodes.FloatCurve: ["mapping"],
    Nodes.Value: [],
    Nodes.Vector: [],
    Nodes.InputColor: [],
    Nodes.WaveTexture: ["wave_type", "bands_direction", "rings_direction", "wave_profile"],
    Nodes.SeparateXYZ: [],
    Nodes.Group: [],
})

class SocketType:
    Boolean = "BOOLEAN"
    Vector = "VECTOR"
    Int = "INT"
    Geometry = "GEOMETRY"
    Value = "VALUE"
    RGBA = "RGBA"
    Image = "IMAGE"

class AttributeType:
    Float = "FLOAT"
    Int = "INT"
    FloatVector = "FLOAT_VECTOR"
    FloatColor = "FLOAT_COLOR"
    Boolean = "BOOLEAN"

class FieldsType:
    Value = "value"
    Vector = "vector"
    Color = "color"
    Boolean = "boolean"

ATTRTYPE_DIMS = {
    AttributeType.Float: 1,
    AttributeType.Int: 1,
    AttributeType.FloatVector: 3,
    AttributeType.FloatColor: 4,
    AttributeType.Boolean: 1,
}

ATTRTYPE_FIELDS = {
    AttributeType.Float: FieldsType.Value,
    AttributeType.Int: FieldsType.Value,
    AttributeType.FloatVector: FieldsType.Vector,
    AttributeType.FloatColor: FieldsType.Color,
    AttributeType.Boolean: FieldsType.Boolean,
}

ATTRTYPE_NP = {
    AttributeType.Float: np.float32,
    AttributeType.Int: np.int32,
    AttributeType.FloatVector: np.float32,
    AttributeType.FloatColor: np.float32,
    AttributeType.Boolean: bool,
}

NPTYPEDIM_ATTR = {
    ("float32", 1): AttributeType.Float,
    ("float32", 3): AttributeType.FloatVector,
    ("float32", 4): AttributeType.FloatColor,
    ("int32", 1): AttributeType.Int,
}


# class ParamType:
#     Pointwise = "Pointwise"
#     Constant = "Constant"

class KernelDataType:
    float = "float"
    float2 = "float2_nonbuiltin"
    float3 = "float3_nonbuiltin"
    float4 = "float4_nonbuiltin"
    int = "int"

KERNELDATATYPE_DIMS = {
    KernelDataType.float: [],
    KernelDataType.float2: [2],
    KernelDataType.float3: [3],
    KernelDataType.float4: [4],
    KernelDataType.int: [],
}

KERNELDATATYPE_NPTYPE = {
    KernelDataType.float: np.float32,
    KernelDataType.float2: np.float32,
    KernelDataType.float3: np.float32,
    KernelDataType.float4: np.float32,
    KernelDataType.int: np.int32,
}

SOCKETTYPE_KERNEL = {
    SocketType.Value: KernelDataType.float,
    SocketType.Vector: KernelDataType.float3,
    SocketType.Int: KernelDataType.int,
    SocketType.RGBA: KernelDataType.float4,
    SocketType.Image: KernelDataType.float4,
    # BOOL todo when necessary
}


NODE_FUNCTIONS = {
    Nodes.WaveTexture: "node_shader_tex_wave",
    Nodes.NoiseTexture: "node_shader_tex_noise",
    Nodes.Math: "node_texture_math",
    Nodes.VectorMath: "node_shader_vector_math",
    Nodes.VoronoiTexture: "node_shader_tex_voronoi",
    Nodes.MapRange: "node_shader_map_range",
    Nodes.MusgraveTexture: "node_shader_tex_musgrave",
    Nodes.MixRGB: "node_shader_mix_rgb",
    Nodes.ColorRamp: "node_texture_valToRgb",
    Nodes.SeparateXYZ: "node_shader_sep_xyz",
    Nodes.CombineXYZ: "node_shader_comb_xyz",
    Nodes.FloatCurve: "node_float_curve",
    Nodes.Mix: "node_shader_mix",
}


def special_sanitize_constant(node_name, x):
    positions = ",".join([str(x[i].position) for i in range(len(x))])
    colors = ",".join(
        [f"float4_nonbuiltin({x[i].color[0]}, {x[i].color[1]}, {x[i].color[2]}, {x[i].color[3]})" for i in range(len(x))])
    return f'''
        float {node_name}_positions[{len(x)}]{{{positions}}};
        float4_nonbuiltin {node_name}_colors[{len(x)}]{{{colors}}};
    '''

def special_sanitize(node_name, x, node_tree_name):
    positions = ",".join([get_imp_var_name(node_tree_name, node_name) + f"_pos{i}" for i in range(len(x))])
    colors = ",".join(
        [get_imp_var_name(node_tree_name, node_name) + f"_color{i}" for i in range(len(x))])
    return f'''
        float {node_name}_positions[{len(x)}]{{{positions}}};
        float4_nonbuiltin {node_name}_colors[{len(x)}]{{{colors}}};
    '''

def get_imp_var_name(node_tree_name, node_name):
    return usable_name(node_name) + "_FROM_" + usable_name(node_tree_name)


def special_sanitize_float_curve(node_name, mapping, N=256):
    positions = np.linspace(0, 1, N)
    values = []
    for p in positions:
        values.append(mapping.evaluate(mapping.curves[0], p))
    values = ",".join([str(v) for v in values])
    return f'''
        float {node_name}_values[{N}]{{{values}}};
        int {node_name}_table_size = {N};
    '''

def usable_name(x):
    return x.replace(".", "_DOT_").replace(" ", "_SPACE_").replace("~", "_WAVE_").replace("(", "_LBR_").replace(")", "_RBR_").replace(",", "_COMMA_")


def sanitize(x, node, param):
    node_type = node.bl_idname
    if node_type == Nodes.NoiseTexture and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return x[:-1]
    elif node_type == Nodes.Math and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return "NODE_MATH_" + x
    elif node_type == Nodes.Math and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return int(x)
    elif node_type == Nodes.VectorMath and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return "NODE_VECTOR_MATH_" + x
    elif node_type == Nodes.VoronoiTexture and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return x[:-1]
    elif node_type == Nodes.VoronoiTexture and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "SHD_VORONOI_" + x
    elif node_type == Nodes.VoronoiTexture and param == NODE_ATTRS_AVAILABLE[node_type][2]:
        return "SHD_VORONOI_" + x
    elif node_type == Nodes.MapRange and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return x
    elif node_type == Nodes.MapRange and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "NODE_MAP_RANGE_" + x
    elif node_type == Nodes.MapRange and param == NODE_ATTRS_AVAILABLE[node_type][2]:
        return int(x)
    elif node_type == Nodes.MusgraveTexture and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return x[:-1]
    elif node_type == Nodes.MusgraveTexture and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "SHD_MUSGRAVE_" + x
    elif node_type == Nodes.MixRGB and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return int(x)
    elif node_type == Nodes.MixRGB and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "MA_RAMP_" + x
    elif node_type == Nodes.Mix and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return x + "_TYPE"
    elif node_type == Nodes.Mix and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "MA_RAMP_" + x
    elif node_type == Nodes.Mix and param == NODE_ATTRS_AVAILABLE[node_type][2]:
        return int(x)
    elif node_type == Nodes.Mix and param == NODE_ATTRS_AVAILABLE[node_type][3]:
        return int(x)
    elif node_type == Nodes.Mix and param == NODE_ATTRS_AVAILABLE[node_type][4]:
        return int(x == "UNIFORM")
    elif node_type == Nodes.ColorRamp and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return f"{len(x)}, {usable_name(node.name)}_positions, {usable_name(node.name)}_colors"
    elif node_type == Nodes.ColorRamp and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "COLBAND_BLEND_" + x
    elif node_type == Nodes.ColorRamp and param == NODE_ATTRS_AVAILABLE[node_type][2]:
        return "COLBAND_INTERP_" + x
    elif node_type == Nodes.ColorRamp and param == NODE_ATTRS_AVAILABLE[node_type][3]:
        return "COLBAND_HUE_" + x
    elif node_type == Nodes.WaveTexture and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return "SHD_WAVE_" + x
    elif node_type == Nodes.WaveTexture and param == NODE_ATTRS_AVAILABLE[node_type][1]:
        return "SHD_WAVE_BANDS_DIRECTION_" + x
    elif node_type == Nodes.WaveTexture and param == NODE_ATTRS_AVAILABLE[node_type][2]:
        return "SHD_WAVE_RINGS_DIRECTION_" + x
    elif node_type == Nodes.WaveTexture and param == NODE_ATTRS_AVAILABLE[node_type][3]:
        return "SHD_WAVE_PROFILE_" + x
    elif node_type == Nodes.FloatCurve and param == NODE_ATTRS_AVAILABLE[node_type][0]:
        return f"{usable_name(node.name)}_values, {usable_name(node.name)}_table_size"
    else:
        return x

def concat_string(strs):
    ret = ""
    for s in strs:
        ret += s + ", "
    return ret

def var_list(in_vars, imp_vars, out_vars, collective_style):
    code = []
    for var in in_vars:
        dtype = in_vars[var]
        code.append(f"{dtype} {var}")
    if collective_style:
        imp_vars_of_type = {}
        for var in sorted(imp_vars.keys()):
            if var in [Vars.Position, Vars.Normal]: continue
            dtype = imp_vars[var][0]
            if dtype in imp_vars_of_type:
                imp_vars_of_type[dtype].append(var)
            else:
                imp_vars_of_type[dtype] = [var]
        if Vars.Position in imp_vars:
            code.append(f"float3_nonbuiltin {Vars.Position}")
        if Vars.Normal in imp_vars:
            code.append(f"float3_nonbuiltin {Vars.Normal}")
        for dtype in sorted(imp_vars_of_type.keys()):
            code.append(f"POINTER_OR_REFERENCE_ARG {dtype} *{dtype}_vars")
    else:
        for var in sorted(imp_vars.keys()):
            dtype = imp_vars[var][0]
            code.append(f"{dtype} {var}")
    for var in out_vars:
        dtype = out_vars[var]
        code.append(f"POINTER_OR_REFERENCE_ARG {dtype} *{var}")
    return ",".join(code)

def collecting_vars(imp_vars):
    code = ""
    imp_vars_count = {}
    for var in sorted(imp_vars.keys()):
        if var in [Vars.Position, Vars.Normal]: continue
        dtype = imp_vars[var][0]
        if dtype in imp_vars_count:
            imp_vars_count[dtype] += 1
        else:
            imp_vars_count[dtype] = 0
        code += f"{dtype} {var} = {dtype}_vars[{imp_vars_count[dtype]}];\n"
    return code

def value_string(value):
    if isinstance(value, float):
        return str(value)
    else:
        l = len(value)
        vs = ",".join([str(value[i]) for i in range(l)])
        return f"float{l}_nonbuiltin({vs})"
