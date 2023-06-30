import bpy
import mathutils
from numpy.random import uniform, normal, randint
from nodes import node_utils
from nodes.color import color_category
from surfaces import surface


def shader_rocks(nw, rand=True, **input_kwargs):
    
    colorramp_3 = nw.new_node(Nodes.ColorRamp,
    colorramp_3.color_ramp.elements[0].position = 0.0285
    colorramp_3.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    colorramp_3.color_ramp.elements[1].position = 0.1347
    colorramp_3.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    
    mapping = nw.new_node(Nodes.Mapping,
    
    noise_texture_1 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 15.0})
    
    rock_color1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture_1.outputs["Fac"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': (0.01, 0.024, 0.0283, 1.0)})

        sample_color(rock_color1.inputs["Color1"].default_value)
        sample_color(rock_color1.inputs["Color2"].default_value)

    noise_texture_2 = nw.new_node(Nodes.NoiseTexture,
        input_kwargs={'Vector': mapping, 'Detail': 15.0})

    rock_color2 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': noise_texture_2.outputs["Fac"], 'Color1': (0.0, 0.0, 0.0, 1.0), 'Color2': (0.0694, 0.1221, 0.0693, 1.0)})

        sample_color(rock_color2.inputs["Color1"].default_value)
        sample_color(rock_color2.inputs["Color2"].default_value)

    mix_1 = nw.new_node(Nodes.MixRGB,
        input_kwargs={'Fac': colorramp_3.outputs["Color"], 'Color1': rock_color1, 'Color2': rock_color2})
    
    principled_bsdf = nw.new_node(Nodes.PrincipledBSDF,
        input_kwargs={'Base Color': mix_1})

def geo_rocks(nw: NodeWrangler, rand=True, selection=None, random_seed=0, geometry=True, **input_kwargs):
    
            voronoi_texture_scale = nw.new_value(sample_ratio(1, sample_min, sample_max), "voronoi_texture_scale")
            voronoi_texture_w = nw.new_value(sample_range(0, 5), "voronoi_texture_w")
        else:
            voronoi_texture_scale = 1.0
            voronoi_texture_w = 0
        voronoi_texture = nw.new_node(Nodes.VoronoiTexture,
            input_kwargs={'Vector': mix, 'Scale': voronoi_texture_scale, 'W': voronoi_texture_w},
            attrs={'feature': 'DISTANCE_TO_EDGE', 'voronoi_dimensions': '4D'})

            input_kwargs={'Fac': voronoi_texture.outputs["Distance"]},
            label="colorramp_VAR",
        )

    


    surface.add_material(obj, shader_rocks, selection=selection, input_kwargs=shader_kwargs)

if __name__ == "__main__":
    mat = 'rock'
    if not os.path.isdir(os.path.join('outputs', mat)):
        os.mkdir(os.path.join('outputs', mat))
    for i in range(10):
        bpy.ops.wm.open_mainfile(filepath='test.blend')
        apply(bpy.data.objects['SolidModel'], geo_kwargs={'rand':True}, shader_kwargs={'rand': True})
        #fn = os.path.join(os.path.abspath(os.curdir), 'giraffe_geo_test.blend')
        #bpy.ops.wm.save_as_mainfile(filepath=fn)
        bpy.context.scene.render.filepath = os.path.join('outputs', mat, '%s_%d.jpg'%(mat, i))
        bpy.context.scene.render.image_settings.file_format='JPEG'
        bpy.ops.render.render(write_still=True)