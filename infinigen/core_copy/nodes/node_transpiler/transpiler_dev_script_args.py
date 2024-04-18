# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick


'''
1. Create a blender file
2. Click the "Scripting" tab
3. Copy this script into a new script
4. Set the 'mode' to one of the following options:
   - mode='print' will print the result script to your terminal
   - mode='make_script' will create a new script in your blender UI, which you can open
    and run to apply the node code to an object of your choice
   - mode='write_file' will write the script to a new file called 'generated_surface_script.py.
        Make sure not to rename / move the script before committing it to git.     
5. Select an object which has some materials and/or geometry nodes on it
6. Click the play button at the top of this script to run it!
7. You should see one python function printed for each material/geonodes on your object
'''

import os
import argparse

import sys
import importlib
from pathlib import Path
import pdb

import bpy
import mathutils

from infinigen.core.nodes.node_transpiler import transpiler
from infinigen.core.nodes import node_wrangler, node_info

parser = argparse.ArgumentParser(description='Blender material node transpiler')
parser.add_argument('--background', type=str, default='')
parser.add_argument('--python', type=str, default='')
parser.add_argument('--mode', type=str, default='write_file')
parser.add_argument('--save_path', type=str, default='/Users/richardguyunqi/infinigen/my_test/') # Dir for all files, including images and script
parser.add_argument('--script_name', type=str, default='example.py') # The name for the script

parser.add_argument('--target', type=str, default='object')
parser.add_argument('--target_obj', type=str, default=bpy.context.active_object.name)
parser.add_argument('--dependencies', type=str, default='', help='Comma-separated file paths of dependencies')

# mode = 'write_file'
# target = 'object'
# target_obj = 'Cube2'
# save_path = '/Users/richardguyunqi/infinigen/my_test/'
# dependencies = [
#     # if your transpile target is using nodegroups taken from some python file,
#     # add those filepaths here so the transpiler imports from them rather than creating a duplicate definition.
# ]


find_image_func_str = '''

def find_images_from_local():
    # Get the current working directory
    cwd = os.getcwd()
    
    # Patterns for each file type
    jpg_pattern = os.path.join(cwd, '*.jpg')
    exr_pattern = os.path.join(cwd, '*.exr')
    png_pattern = os.path.join(cwd, '*.png')

    # Get lists of files for each pattern
    jpg_files = glob.glob(jpg_pattern)
    exr_files = glob.glob(exr_pattern)
    png_files = glob.glob(png_pattern)
    
    # Combine the lists
    all_files = jpg_files + exr_files + png_files
    
    return all_files

'''

load_image_func_str = '''

def load_images_from_local(image_files):
    for image_file_name in image_files:
        # Construct the full file path
        cwd = os.getcwd()
        image_file_path = os.path.join(cwd, image_file_name)

        # Check if the image is already loaded
        image = bpy.data.images.get(image_file_name)

        # If not, load the image
        if not image:
            try:
                image = bpy.data.images.load(image_file_path)
            except RuntimeError as e:
                print(f"Error loading image {image_file_path}: {e}")
'''


# Main function
def run_transpiler(args):
    # Unpack all packed files with the 'USE_LOCAL' option
    bpy.ops.file.unpack_all(method='USE_LOCAL')

    # Separate the two kinds of targets and transpile their nodes(both geometry and material)
    if args.target == 'object':
        res = transpiler.transpile_object(bpy.data.objects[args.target_obj], args.save_path, args.dependencies)
    elif target == 'world':
        res = transpiler.transpile_world(args.save_path, args.dependencies)
    else:
        raise ValueError(f'Unrecognized {target=}')

    # Output the transpiled code version of nodes in the selected way 
    if args.mode == 'print':
        print('START')
        print('\n')
        print(res)
        print('END')


    # In make_script mode, the output script is directly runnable on other objects
    # because of the added "import bpy" and the apply argument
    elif args.mode == 'make_script':
        res_debug = (
            "import bpy\n" +
            res +
            "\napply(bpy.context.active_object)"
        )
        script = bpy.data.texts.new('generated_surface_script')
        script.from_string(res_debug)

    # The output script is not directly runnable, but saved.
    elif args.mode == 'write_file':
        
        res_debug = (
            "import os\n"+ f'os.chdir("{args.save_path}")\n\n' + "import bpy\n" + 'import glob\n' + load_image_func_str + '\n' + find_image_func_str + '\n' +
            res + 
            "\nimage_files = find_images_from_local()" + "\nload_images_from_local(image_files)" + "\napply(bpy.context.active_object)"
        )

        print(f'Writing generated script to {args.save_path + args.script_name}')
        with Path(args.save_path + args.script_name).open('w') as f:
            f.write(res_debug)
    else:
        raise ValueError(f'Unrecognized {args.mode=}')


if __name__ == '__main__':
    # argv = sys.argv
    # argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # print(argv)  # --> ['example', 'args', '123']

    args = parser.parse_args()
    args.dependencies = args.dependencies.split(',') if args.dependencies else []

    #print(args)
    run_transpiler(args)