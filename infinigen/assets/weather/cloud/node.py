# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hei Law
# Acknowledgement: This file draws inspiration from https://www.youtube.com/watch?v=lPAYX8z9i8M by CGCookie

from infinigen.core.nodes.node_wrangler import Nodes
import numpy as np

def cloud_geometry_func(
    points_only=False,
    resolution=256,
):
    def cloud_geometry_node(
        nw, 
        density,
        noise_scale,
        noise_detail,
        voronoi_scale,
        mix_factor,
        rotate_angle,
        material,
        curve_func,
        **kwargs,
    ):
        scale = (1.5, 1.5, 2.0)
        
        group_input = nw.new_node(Nodes.GroupInput)
        position    = nw.new_node(Nodes.InputPosition)

        vector_rotate = nw.new_node(
            Nodes.VectorRotate,
            input_kwargs={
                'Vector': position,
                'Angle': rotate_angle,
            },
        )

        noise_texture_1 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                'Vector': vector_rotate,
                'Scale': 2.0000,
            },
        )
    
        subtract = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: noise_texture_1.outputs["Color"],
                1: (0.5000, 0.5000, 0.5000),
            },
            attrs={
                'operation': 'SUBTRACT',
            },
        )
        
        scale = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: subtract.outputs["Vector"],
                'Scale': 0.1000,
            },
            attrs={
                'operation': 'SCALE',
            },
        )
        
        multiply = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_rotate,
                1: (1.5000, 1.5000, 2.0000),
            },
            attrs={
                'operation': 'MULTIPLY',
            },
        )
        
        vector_curves = nw.new_node(
            Nodes.VectorCurve,
            input_kwargs={
                'Vector': multiply.outputs["Vector"],
            },
        )
        curve_func(vector_curves.mapping.curves)
        
        add = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: scale.outputs["Vector"],
                1: vector_curves,
            },
        )
        
        noise_texture = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                'Vector': vector_rotate,
                'Scale': noise_scale,
                'Detail': noise_detail,
            },
        )
        
        subtract_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: noise_texture.outputs["Color"],
                1: (0.5000, 0.5000, 0.5000),
            },
            attrs={
                'operation': 'SUBTRACT',
            },
        )
        
        scale_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: subtract_1.outputs["Vector"],
                'Scale': 0.1000,
            },
            attrs={
                'operation': 'SCALE',
            },
        )
        
        add_1 = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: vector_rotate,
                1: scale_1.outputs["Vector"],
            },
        )
        
        voronoi_texture = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                'Vector': add_1.outputs["Vector"],
                'Scale': voronoi_scale,
            },
        )
        
        noise_texture_2 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                'Vector': vector_rotate,
                'Detail': 5.0000,
            },
        )
        
        subtract_2 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: noise_texture_2.outputs["Fac"],
            },
            attrs={
                'operation': 'SUBTRACT',
            },
        )
        
        multiply_1 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: subtract_2,
                1: 0.1000,
            },
            attrs={
                'operation': 'MULTIPLY',
            },
        )
        
        add_2 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: voronoi_texture.outputs["Distance"],
                1: multiply_1,
            },
        )
        
        mix_1 = nw.new_node(
            Nodes.MixRGB,
            input_kwargs={
                'Fac': mix_factor,
                'Color1': add.outputs["Vector"],
                'Color2': add_2,
            },
            attrs={
                'blend_type': 'OVERLAY',
            },
        )
        
        length = nw.new_node(
            Nodes.VectorMath,
            input_kwargs={
                0: mix_1,
            },
            attrs={
                'operation': 'LENGTH',
            },
        )
        
        noise_texture_3 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                'Vector': vector_rotate,
                'Scale': 2.0000,
                'Detail': 5.0000,
            },
        )
        
        multiply_2 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: noise_texture_3.outputs["Fac"],
                1: 2.0000,
            },
            attrs={
                'operation': 'MULTIPLY',
            },
        )
        
        noise_texture_4 = nw.new_node(
            Nodes.NoiseTexture,
            input_kwargs={
                'Vector': vector_rotate,
                'Scale': 1.5000,
                'Detail': 5.0000,
            },
        )
        
        divide = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: noise_texture_4.outputs["Fac"],
                1: 100.0000,
            }, attrs={
                'operation': 'DIVIDE',
            },
        )
        
        map_range = nw.new_node(
            Nodes.MapRange,
            input_kwargs={
                'Value': length.outputs["Value"],
                3: multiply_2,
                4: divide,
            },
            attrs={
                'clamp': False,
            },
        )
        
        multiply_3 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: map_range.outputs["Result"],
                1: density,
            },
            attrs={
                'operation': 'MULTIPLY',
            },
        )
        
        greater_than = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: multiply_3,
                1: 0.01,
            },
            attrs={
                'operation': 'GREATER_THAN',
            },
        )
        
        separate_geometry = nw.new_node(
            Nodes.SeparateGeometry,
            input_kwargs={
                'Geometry': group_input.outputs["Geometry"],
                'Selection': greater_than,
            },
        )
        
        points_to_volume = nw.new_node(
            Nodes.PointsToVolume,
            input_kwargs={
                'Points': separate_geometry.outputs["Selection"],
                'Radius': 0.0150,
            },
        )
        
        volume_to_mesh = nw.new_node(
            Nodes.VolumeToMesh,
            input_kwargs={
                'Volume': points_to_volume,
            },
        )
        
        set_material = nw.new_node(
            Nodes.SetMaterial,
            input_kwargs={
                'Geometry': volume_to_mesh,
                'Material': material,
            },
        )
        return set_material
    return cloud_geometry_node


def geometry_func(
    points_only=False,
    resolution=256,
):
    cloud_func = cloud_geometry_func(
        points_only=points_only,
        resolution=resolution,
    )

    def geometry_nodes(
        nw, 
        density,
        noise_scale,
        noise_detail,
        voronoi_scale,
        mix_factor,
        rotate_angle,
        material,
        curve_func,
        **kwargs,
    ):
        cloud_mesh = cloud_func(
            nw, 
            density,
            noise_scale,
            noise_detail,
            voronoi_scale,
            mix_factor,
            rotate_angle,
            material,
            curve_func,
            **kwargs,
        )

        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={
                'Geometry': cloud_mesh,
            },
        )
    return geometry_nodes


def shader_material(
    nw,
    density,
    anisotropy,
    noise_scale,
    noise_detail,
    voronoi_scale,
    mix_factor,
    emission_strength,
    **kwargs,
):
    location = (0.0, 0.0, 0.0)
    scale    = (0.9, 0.9, 0.9)

    texture_coordinate = nw.new_node(Nodes.TextureCoord)
    
    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={
            'Vector': texture_coordinate.outputs["Object"],
        },
    )
    
    noise_texture_3 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            'Vector': mapping,
            'Scale': 2.0000,
        },
    )
    
    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: noise_texture_3.outputs["Color"],
            1: (0.5000, 0.5000, 0.5000),
        },
        attrs={
            'operation': 'SUBTRACT',
        },
    )
    
    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract.outputs["Vector"],
            'Scale': 0.1000,
        },
        attrs={
            'operation': 'SCALE',
        },
    )
    
    add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: scale.outputs["Vector"],
            1: mapping,
        },
    )
    
    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            'Vector': mapping,
            'Scale': noise_scale,
            'Detail': noise_detail,
        })
    
    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: noise_texture.outputs["Color"],
            1: (0.5000, 0.5000, 0.5000),
        },
        attrs={
            'operation': 'SUBTRACT',
        },
    )
    
    scale_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: subtract_1.outputs["Vector"],
            'Scale': 0.1000,
        },
        attrs={
            'operation': 'SCALE',
        },
    )
    
    add_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            1: scale_1.outputs["Vector"],
        },
    )
    
    voronoi_texture_1 = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={
            'Vector': add_1.outputs["Vector"],
            'Scale': voronoi_scale,
        },
    )
    
    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            'Scale': mapping,
            'Detail': 5.0000,
        },
    )
    
    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture_2.outputs["Fac"],
        },
        attrs={
            'operation': 'SUBTRACT',
        },
    )
    
    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: subtract_2,
            1: 0.1000,
        },
        attrs={
            'operation': 'MULTIPLY',
        },
    )
    
    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: voronoi_texture_1.outputs["Distance"],
            1: multiply,
        },
    )
    
    mix_1 = nw.new_node(
        Nodes.MixRGB,
        input_kwargs={
            'Fac': 0.3,
            'Color1': add.outputs["Vector"],
            'Color2': add_2,
        },
        attrs={
            'blend_type': 'OVERLAY',
        },
    )
    
    length = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: mix_1,
        },
        attrs={
            'operation': 'LENGTH',
        },
    )
    
    noise_texture_4 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            'Vector': mapping,
            'Scale': 2.0000,
            'Detail': 5.0000,
        },
    )
    
    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture_4.outputs["Fac"],
            1: 2.0000,
        },
        attrs={
            'operation': 'MULTIPLY',
        },
    )
    
    noise_texture_5 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            'Vector': mapping,
            'Scale': 1.5000,
            'Detail': 5.0000,
        },
    )
    
    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: noise_texture_5.outputs["Fac"],
            1: 100.0000,
        },
        attrs={
            'operation': 'DIVIDE',
        },
    )
    
    map_range_1 = nw.new_node(
        Nodes.MapRange,
        input_kwargs={
            'Value': length.outputs["Value"],
            3: multiply_1,
            4: divide,
        },
        attrs={
            'clamp': False,
        },
    )
    
    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: map_range_1.outputs["Result"],
            1: density,
        },
        attrs={
            'operation': 'MULTIPLY',
        },
    )
    
    principled_volume = nw.new_node(
        Nodes.PrincipledVolume,
        input_kwargs={
            'Color': (1.0000, 1.0000, 1.0000, 1.0000),
            'Density': multiply_2,
            'Anisotropy': anisotropy,
            'Absorption Color': (1.0000, 1.0000, 1.0000, 1.0000),
            'Temperature': 0.0,
            'Emission Strength': emission_strength,
        },
    )
    
    material_output = nw.new_node(
        Nodes.MaterialOutput,
        input_kwargs={
            'Volume': principled_volume,
        },
        attrs={
            'is_active_output': True,
        },
    )


def scatter_func(
    points_only=False,
    resolution=256,
):
    cloud_func = cloud_geometry_func(
        points_only=points_only,
        resolution=resolution,
    )

    def scatter_nodes(
        nw, 
        densities,
        noise_scales,
        noise_details,
        voronoi_scales,
        mix_factors,
        rotate_angles,
        materials,
        curve_funcs,
        scatter_params,
        **kwargs,
    ):
        params = zip(
            densities,
            noise_scales,
            noise_details,
            voronoi_scales,
            mix_factors,
            rotate_angles,
            materials,
            curve_funcs,
        )

        cloud_meshes = [
            cloud_func(
                nw,
                *param,
                **kwargs,
            ) for param in params
        ]

        # Selection
        voronoi_texture_2 = nw.new_node(
            Nodes.VoronoiTexture,
            input_kwargs={
                'Scale': scatter_params['voronoi_scale'],
            },
        )
        map_range = nw.new_node(
            Nodes.MapRange,
            input_kwargs={
                'Value': voronoi_texture_2.outputs["Distance"],
            },
        )
        greater_than = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: map_range.outputs["Result"],
                1: 0.6,
            },
            attrs={
                'operation': 'GREATER_THAN',
            },
        )

        # Scatter locations
        grid = nw.new_node(
            Nodes.MeshGrid,
            input_kwargs={
                'Size X': 2.0,
                'Size Y': 2.0,
                'Vertices X': scatter_params['vertices_x'],
                'Vertices Y': scatter_params['vertices_y'],
            },
        )
        distribute_points_on_faces = nw.new_node(
            Nodes.DistributePointsOnFaces,
            input_kwargs={
                'Mesh': grid,
                'Distance Min': 0.3,
                'Density Max': 64.0,
            },
            attrs={
                'distribute_method': 'POISSON',
            },
        )

        # Convert cloud geometry to instance
        geometry_to_instance = nw.new_node(
            'GeometryNodeGeometryToInstance',
            input_kwargs={
                'Geometry': cloud_meshes,
            },
        )

        random_value_2 = nw.new_node(
            Nodes.RandomValue,
            attrs={
                'data_type': 'INT',
            },
        )
        instance_on_points = nw.new_node(
            Nodes.InstanceOnPoints,
            input_kwargs={
                'Points': distribute_points_on_faces,
                'Instance': geometry_to_instance,
                'Pick Instance': True,
                'Instance Index': random_value_2.outputs[2],
            },
        )
        random_value = nw.new_node(
            Nodes.RandomValue,
            input_kwargs={
                0: (0.5, 0.5, 0.5),
                'Seed': np.random.randint(int(1e5)),
            },
            attrs={
                'data_type': 'FLOAT_VECTOR',
            },
        )
        random_value_4 = nw.new_node(
            Nodes.RandomValue,
            input_kwargs={
                2: 0.1,
                3: 0.4,
            },
        )
        multiply_4 = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: random_value.outputs["Value"],
                1: random_value_4.outputs[1],
            },
            attrs={
                'operation': 'MULTIPLY',
            },
        )
        scale_instances = nw.new_node(
            'GeometryNodeScaleInstances',
            input_kwargs={
                'Instances': instance_on_points,
                'Scale': multiply_4,
            },
        )
        random_value_1 = nw.new_node(
            Nodes.RandomValue,
            input_kwargs={
                2: -45.0,
                3: 45.0,
            },
        )
        radians = nw.new_node(
            Nodes.Math,
            input_kwargs={
                0: random_value_1.outputs[1],
            },
            attrs={
                'operation': 'RADIANS',
            },
        )
        combine_xyz = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                'Z': radians,
            },
        )
        rotate_instances = nw.new_node(
            'GeometryNodeRotateInstances',
            input_kwargs={
                'Instances': scale_instances,
                'Rotation': combine_xyz,
            },
        )
        random_value_3 = nw.new_node(Nodes.RandomValue)
        combine_xyz_2 = nw.new_node(
            Nodes.CombineXYZ,
            input_kwargs={
                'Z': random_value_3.outputs[1],
            },
        )
        translate_instances = nw.new_node(
            Nodes.TranslateInstances,
            input_kwargs={
                'Instances': rotate_instances,
                'Translation': combine_xyz_2,
            },
        )
        realize_instances = nw.new_node(
            Nodes.RealizeInstances,
            input_kwargs={
                'Geometry': translate_instances,
            },
        )
        group_output_1 = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={
                'Geometry': realize_instances,
            },
        )
    return scatter_nodes
