# compare_scenes.py
import bpy
from pathlib import Path

def count_objects_in_scene(blend_file):
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    
    object_count = len(bpy.data.objects)
    mesh_count = len([obj for obj in bpy.data.objects if obj.type == 'MESH'])
    
    # count funitures
    furniture = len([obj for obj in bpy.data.objects if 'furniture' in obj.name.lower()])
    decorative = len([obj for obj in bpy.data.objects if any(keyword in obj.name.lower() for keyword in ['scatter', 'plant', 'book', 'decor'])])
    
    print(f"environment: {blend_file}")
    print(f"  total object count: {object_count}")
    print(f"  mesh count: {mesh_count}")
    print(f"  furniture: {furniture}")
    print(f"  decorative: {decorative}")
    print()

# compare coarse vs populated objects in scene
count_objects_in_scene("outputs/indoors/APT0_fast/scene.blend")
count_objects_in_scene("outputs/indoors/APT0_fast_populated/scene.blend")
