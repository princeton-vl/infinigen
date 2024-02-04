
# Asset Exporter

This documentation details how to create an OBJ, FBX, STL, PLY or OpenUSD file from a `.blend` file, such as those produced by [Hello World](HelloWorld.md) or [Generating Individual Assets](./GeneratingIndividualAssets.md). 

Blender does provide a built-in exporter, but it wont work out of the box for Infinigen since our files contain procedural materials and assets defined using shader programs. This tool's job is to "bake" all these procedural elements into more standard graphics formats (i.e, simple meshes with materials made of texture maps), before invoking the standard blender exporter. This process can be slow, since it uses a rendering engine, and lossy, since the resulting textures have a finite resolution.

To convert a folder of blender files into USD files (our recommmended format), use the command below:
```bash
python -m infinigen.tools.export --input_folder {PATH_TO_FOLDER_OF_BLENDFILES} --output_folder outputs/my_export -f usdc -r 1024
```

If you want a different output format, please use the "--help" flag or use one of the options below:
- `-f obj` will export in .obj format, 
- `-f fbx` will export in .fbx format
- `-f stl` will export in .stl format
- `-f ply` will export in .ply format. 
- `-f usdc` will export in .usdc format. 
- `-v` enables per-vertex colors (only compatible with .fbx and .ply formats). 
- `-r {INT}` controls the resolution of the baked texture maps. For instance, `-r 1024` will export 1024 x 1024 texture maps.
- `--individual` will export each object in a scene in its own individual file.

## :warning: Exporting full Infinigen scenes is only supported for USDC files. 

:bulb: Note: exporting OBJ/FBX files of **single objects** *generally works fine; this discussion only refers to large-scale scenes.

Infinigen uses of *instancing* to represent densely scattered objects. That is, rather than storing millions of unique high-detail pebbles or leaves to scatter on the floor, we use a smaller set of unique objects which are stored in memory only once, but are repeated all over the scene with many different transforms. 

To our knowledge, no file formats except '.blend' and '.usdc' support saving 3D files that contain instanced geometry. For all file formats besides these two, instanced will be *realized*: instead of storing just a few unique meshes, the meshes will be copied, pasted and transformed thousands of times (once for each unique scatter location). This creates a simple mesh that, but the cost is so high that we do not recommend attempting it for full Infinigen scenes. 

If you require OBJ/FBX/PLY files for your research, you have a few options:
- You can use individual objects, rather than full scenes. These *generally dont contain instancing so can be exported to simple mesh formats.
- You can use advice in [Configuring Infinigen](./ConfiguringInfinigen.md) to create a scene that has very small extent or low detail, such that the final realized mesh will still be small enough to fit in memory.
- You can use the advice in [Configuring Infinigen](./ConfiguringInfinigen.md) to create a scene which simply doesnt contain any instanced objects. Specifically, you should turn off trees and all scattered objects.
    - The simplest way to do this is to turn off everything except terrain, by including the config `no_assets.gin`. 

*caveat for the above: Infinigen's implementation for trees uses instances to represent leaves and branches. Trees are also generally large and high detail enough to cause issues if you realize them before exporting. Therefore, exporting whole trees as OBJs also generally isnt supported, unless you do so at very low resolution, or you turn off the tree's branches / leaves first. 

## Other Known Issues and Limitations

* Some material features used in Infinigen are not yet supported by this exporter. Specifically, this script only handles Albedo, Roughness and Metallicity maps. Any other procedural parameters of the material will be ignored, so you should not expect complex materials (e.g skin, translucent leaves, glowing lava) to be perfectly reproduced outside of Blender. Depending on file format, there is limited support for materials with non-procedural, constant values of transmission, clearcoat, and sheen.

* Exporting *animated* 3D files is generally untested and not officially supported. This includes exporting particles, articulated creatures, deforming plants, etc. These features are *in principle* supported by OpenUSD, but are untested by us and not officially supported by this export script.

* Assets with transparent materials (water, glass-like materials, etc.) may have incorrect textures for all material parameters after export.

* Large scenes and assets may take a long time to export and will crash Blender if you do not have a sufficiently large amount of memory. The export results may also be unusably large.

* When exporting in .fbx format, the embedded roughness texture maps in the file may sometimes be too bright or too dark. The .png roughness map in the folder is accurate, however.

* .fbx exports ocassionally fail due to invalid UV values on complicated geometry. Adjusting the 'island_margin' value in bpy.ops.uv.smart_project() sometimes remedies this





