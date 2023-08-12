# COPYRIGHT
# adapted from Blender Real Snow add-on https://docs.blender.org/manual/en/latest/addons/object/real_snow.html
# License: GPL

from infinigen.core.nodes.node_wrangler import Nodes
from infinigen.core import surface
from infinigen.core.util.organization import SurfaceTypes

type = SurfaceTypes.SDFPerturb
mod_name = "geo_snowtexture"
name = "snow"


def shader_snow(nw, subsurface=1.0, **kwargs):
    nw.force_input_consistency()
    position = nw.new_node('ShaderNodeNewGeometry', [])
    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={'X': 0.36, 'Y': 0.46, 'Z': 0.6}
    )
    vector_math = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: combine_xyz, 1: (0.5, 0.5, 0.5)},
        attrs={'operation': 'MULTIPLY'}
    )
    mapping = nw.new_node(
        Nodes.Mapping,
        input_kwargs={'Vector': position, 'Scale': (12.0, 12.0, 12.0)}
    )
    voronoi_texture = nw.new_node(
        Nodes.VoronoiTexture,
        input_kwargs={'Vector': mapping, 'Scale': 30.0},
        attrs={'feature': 'N_SPHERE_RADIUS'}
    )
    colorramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': voronoi_texture.outputs["Radius"]})
    colorramp.color_ramp.elements[0].position = 0.525
    colorramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp.color_ramp.elements[1].position = 0.58
    colorramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    principled_bsdf = nw.new_node(
        Nodes.PrincipledBSDF,
        input_kwargs={
            'Base Color': (0.904, 0.904, 0.904, 1.0),
            'Subsurface': subsurface,
            'Subsurface Radius': vector_math.outputs["Vector"],
            'Subsurface Color': (0.904, 0.904, 0.904, 1.0),
            'Specular': 0.224,
            'Roughness': 0.1,
            'Clearcoat': colorramp.outputs["Color"],
            'Clearcoat Roughness': 0.1,
        },
        attrs={'distribution': 'MULTI_GGX'}
    )
    
    return principled_bsdf


def geo_snowtexture(nw, selection=None, **kwargs):
    nw.force_input_consistency()
    group_input = nw.new_node(Nodes.GroupInput)
    normal_dir = nw.new_node(Nodes.InputNormal)
    position0 = nw.new_node(Nodes.InputPosition)
    position = nw.multiply(position0, [12]*3)
    
    noise_texture = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'Scale': 12.0, 'Detail': 2}
    )
        
    noise_texture_1 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'Scale': 2.0, 'Detail': 4}
    )
    colorramp_1 = nw.new_node(
        Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"]}
    )
    colorramp_1.color_ramp.elements[0].position = 0.069
    colorramp_1.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_1.color_ramp.elements[1].position = 0.757
    colorramp_1.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    noise_texture_2 = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={'Vector': position, 'Scale': 1.0, 'Detail': 4}
    )
    colorramp_2 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"]}
    )
    colorramp_2.color_ramp.elements[0].position = 0.069
    colorramp_2.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_2.color_ramp.elements[1].position = 0.757
    colorramp_2.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    height = nw.scalar_add(
        nw.scalar_multiply(0.6, noise_texture),
        nw.scalar_multiply(0.4, colorramp_1),
        colorramp_2,
    )

    map_range = nw.new_node(
        Nodes.MapRange,
        input_kwargs={'Value': height, 1: 0.0, 2: 2.0, 3: -0.03, 4: 0.03}
    )
    
    modulation = nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={'Vector': position0, 'Scale': 0.5}
    )
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
        input_kwargs={'Fac': modulation}
    )
    colorramp_3.color_ramp.elements[0].position = 0.25
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.75
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    

    offset = nw.multiply(normal_dir, map_range, colorramp_3)
    
    if selection is not None:
        offset = nw.multiply(offset, surface.eval_argument(nw, selection))
    
    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={'Geometry': group_input.outputs["Geometry"], 'Offset': offset}
    )
    
    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={'Geometry': set_position})

def apply(objs, selection=None, **kwargs):
    surface.add_geomod(objs, geo_snowtexture, selection=selection)
    surface.add_material(objs, shader_snow, selection=selection, input_kwargs=kwargs)