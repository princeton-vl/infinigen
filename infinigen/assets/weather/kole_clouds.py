# Derived from https://blenderartists.org/t/fluffy-mostly-procedural-clouds-for-cycles/1448689
# Original node-graph created by ThomasKole https://blenderartists.org/u/ThomasKole and licensed CC-0

import bpy
import gin
import numpy as np
from mathutils import Vector
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.nodes import node_utils
from infinigen.core import surface
from infinigen.core.util.random import random_general as rg
from infinigen.terrain.utils import drive_param

@gin.configurable
def kole_clouds_shader(
    nw: NodeWrangler,
    coverage_frame_start=("clip_gaussian", -0.1, 0.1, -0.3, 0.3), coverage_frame_end=("clip_gaussian", -0.1, 0.1, -0.3, 0.3),
    density=("uniform", .01, .04),
    translation_animation=("bool", 0.5),
    translation=0,
    anisotropy=("clip_gaussian", 0.1, 0.1, 0, 0.5),
):
    density = rg(density)
    anisotropy = rg(anisotropy)
    
    # Code generated using version 2.4.3 of the node_transpiler
    transparent_bsdf = nw.new_node(Nodes.TransparentBSDF)
    
    # PARAMETER: Coverage
    value = nw.new_node(Nodes.Value)
    coverage_frame_start = rg(coverage_frame_start)
    coverage_frame_end = rg(coverage_frame_end)
    drive_param(
        value.outputs[0], (coverage_frame_end - coverage_frame_start) / (bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1),
        offset=coverage_frame_start - (coverage_frame_end - coverage_frame_start) * bpy.context.scene.frame_start / (bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1)
    )
    
    geometry = nw.new_node('ShaderNodeNewGeometry')
    
    vector_transform = nw.new_node('ShaderNodeVectorTransform',
        input_kwargs={'Vector': geometry.outputs["Position"]})
    
    vector_transform = nw.add(vector_transform, Vector([translation, 0, 0]))
    if rg(translation_animation):
        drive_param(vector_transform.inputs[1], 0.001, offset=-(bpy.context.scene.frame_start + bpy.context.scene.frame_end) / 2 * 0.001 + translation, index=0)

    multiply = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: vector_transform, 1: (1.0, 1.0, 1.0)},
        attrs={'operation': 'MULTIPLY'})
    
    separate_xyz = nw.new_node(Nodes.SeparateXYZ,
        input_kwargs={'Vector': multiply.outputs["Vector"]})
    
    map_range = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': separate_xyz.outputs["Z"], 1: 0.1, 2: 0.3, 4: -0.2})
    
    multiply_1 = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"], 1: (1.0, 1.0, 10 ** np.random.uniform(-1, 0))},
        attrs={'operation': 'MULTIPLY'})
    
    add = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply_1.outputs["Vector"], 1: (0.0, 0.0, np.random.uniform(0, 2))})
    
    musgrave_texture = nw.new_node(Nodes.MusgraveTexture,
        input_kwargs={'Vector': add.outputs["Vector"], 'Scale': 3.0, 'Detail': 10.0, 'Dimension': 0.6, 'Lacunarity': 2.6})
    
    map_range_1 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': musgrave_texture, 1: -1.0},
        attrs={'clamp': False})
    
    add_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range.outputs["Result"], 1: map_range_1.outputs["Result"]})
    
    add_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: value, 1: add_1})
    
    map_range_2 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': add_2, 1: 0.4, 2: 0.5},
        attrs={'clamp': False})
    
    length = nw.new_node(Nodes.VectorMath,
        input_kwargs={0: multiply.outputs["Vector"]},
        attrs={'operation': 'LENGTH'})
    
    # This value should change with the solidify thickness
    value_1 = nw.new_node(Nodes.Value)
    value_1.outputs[0].default_value = 0.5
    
    add_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: 1.0, 1: value_1})
    
    map_range_3 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': length.outputs["Value"], 1: 1.0, 2: add_3})
    
    geometry_1 = nw.new_node('ShaderNodeNewGeometry')
    
    voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
        input_kwargs={'Vector': geometry_1.outputs["Position"], 'Scale': 0.01})
    
    map_range_4 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': voronoi_texture.outputs["Distance"], 3: 0.5, 4: 2.0})
    
    power = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_3.outputs["Result"], 1: map_range_4.outputs["Result"]},
        attrs={'operation': 'POWER'})
    
    float_curve = nw.new_node(Nodes.FloatCurve,
        input_kwargs={'Value': power})
    node_utils.assign_curve(float_curve.mapping.curves[0], [(0.0, 1.0), (0.0273, 0.0063), (0.2455, 0.6), (0.6682, 0.3188), (0.9955, 1.0)])
    
    map_range_5 = nw.new_node(Nodes.MapRange,
        input_kwargs={'Value': float_curve, 4: 5.0})
    
    greater_than = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_2.outputs["Result"], 1: map_range_5.outputs["Result"]},
        attrs={'operation': 'GREATER_THAN'})
    
    subtract = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"], 1: map_range_5.outputs["Result"]},
        attrs={'operation': 'SUBTRACT', 'use_clamp': True})
    
    multiply_2 = nw.new_node(Nodes.Math,
        input_kwargs={0: subtract, 1: 0.3},
        attrs={'operation': 'MULTIPLY'})
    
    multiply_3 = nw.new_node(Nodes.Math,
        input_kwargs={0: multiply_2, 1: 0.01},
        attrs={'operation': 'MULTIPLY'})
    
    add_4 = nw.new_node(Nodes.Math,
        input_kwargs={0: greater_than, 1: multiply_3})
    
    power_1 = nw.new_node(Nodes.Math,
        input_kwargs={0: map_range_1.outputs["Result"]},
        attrs={'operation': 'POWER'})

    density_mul = nw.new_node(Nodes.Math,
        input_kwargs={0: power_1, 1: density},
        attrs={'operation': 'MULTIPLY', 'use_clamp': True})
    
    volume_scatter = nw.new_node('ShaderNodeVolumeScatter',
        input_kwargs={'Color': add_4, 'Density': density_mul, 'Anisotropy': anisotropy})
    
    nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': transparent_bsdf, 'Volume': volume_scatter})

@gin.configurable("kole_clouds")
def add_kole_clouds(height=0):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    sphere = bpy.context.active_object
    sphere.name = "KoleClouds"

    surface.add_material(sphere, kole_clouds_shader, selection=None)

    # Don't change the solidify modifier
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    sphere.modifiers["Solidify"].thickness = 0.5
    sphere.modifiers["Solidify"].offset = 1
    sphere.modifiers["Solidify"].use_even_offset = True
    sphere.scale = (1200, 1200, 80)
    sphere.location = (0, 0, height)
    sphere.rotation_euler[1] = np.pi

if __name__ == "__main__":
    add_kole_clouds()