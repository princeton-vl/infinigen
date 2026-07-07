

# Asset Exporter


This documentation details how to create an OBJ, FBX, STL, PLY or OpenUSD file from a `.blend` file, such as those produced by [Hello World](HelloWorld.md) or [Generating Individual Assets](./GeneratingIndividualAssets.md).


Blender does provide a built-in exporter, but it wont work out of the box for Infinigen’s blend files, since they contain procedural materials and assets. The commands below will bake these procedural elements into more standard graphics formats, such as static meshes with materials made of texture maps, before invoking the standard blender exporter. 


This process can be slow, since it uses a rendering engine, and lossy, since the resulting textures have a finite resolution. 


To convert a folder of blender files into USD files, use the command below:
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
- `--omniverse` will prepare the scene for import to IsaacSim or other NVIDIA Omniverse programs. See more in [Exporting to Simulators](./ExportingToSimulators.md).


## :warning: Exporting full Infinigen scenes is only supported for USDC files.


:bulb: Note: exporting OBJ/FBX files of **single objects** *generally works fine; this discussion only refers to large-scale nature scenes.


Infinigen uses *instancing* to represent densely scattered objects. That is, rather than storing millions of unique high-detail pebbles or leaves to scatter on the floor, we use a smaller set of unique objects which are stored in memory only once, but are repeated all over the scene with many different transforms.


To our knowledge, no file formats except '.blend' and '.usdc' support saving instanced geometry. For all file formats besides these two, instances will be *realized*: instead of storing just a few unique meshes, the meshes will be copied, pasted and transformed thousands of times (once for each unique scatter location). This creates a simple mesh that can be stored as an OBJ, but the cost is so high that we do not recommend attempting it for full Infinigen scenes.


If you require OBJ/FBX/PLY files for your research, you have a few options:
- You can use individual objects, rather than full scenes. These *generally don't contain instancing so can be exported to simple mesh formats.
- You can use advice in [Configuring Infinigen](./ConfiguringInfinigen.md) to create a scene that has very small extent or low detail, such that the final realized mesh will still be small enough to fit in memory.
- You can use the advice in [Configuring Infinigen](./ConfiguringInfinigen.md) to create a scene which simply does not contain any instanced objects. Specifically, you should turn off trees and all scattered objects.
   - The simplest way to do this is to turn off everything except terrain, by including the config `no_assets.gin`.


*Infinigen's implementation of trees uses instances to represent leaves and branches. Trees are also generally large and detailed to cause issues if you realize them before exporting. Therefore, exporting whole trees as OBJs also generally isn't supported, unless you do so at very low resolution, or you turn off the tree's branches / leaves first.


## Other Known Issues and Limitations


* Some material features used in Infinigen are not yet supported by this exporter. Specifically, this script only handles Albedo, Roughness, Normal, and Metallicity maps. Any other procedural parameters of the material will be ignored. Many file formats also have limited support for spatially varying transmission, clearcoat, and sheen. Generally, you should not expect any materials (but especially skin, translucent leaves, glowing lava) to be perfectly reproduced outside of Blender. 

* Exporting *animated* 3D files is generally untested and not officially supported. This includes exporting particles, articulated creatures, deforming plants, etc. These features are *in principle* supported by OpenUSD, but are untested by us and not officially supported by this export script.

* Assets with transparent materials (water, glass-like materials, etc.) may have incorrect textures for all material parameters after export.

* Large scenes and assets may take a long time to export and will crash Blender if you do not have enough RAM. The export results may also be unusably large.


* .fbx exports occasionally fail due to invalid UV values on complicated geometry. Adjusting the 'island_margin' value in bpy.ops.uv.smart_project() sometimes remedies this















