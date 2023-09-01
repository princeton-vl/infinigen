
# Asset Exporter

Export individaully generated assets in .blend files to various general-purpose file formats.

Create a folder of ```.blend``` files and another empty folder for the export results.

Then, run the following:
```
python -m infinigen.datagen.export -b {PATH_TO_BLEND_FILE_FOLDER} -e {PATH_TO_OUTPUT_FOLDER} -o -r 1024
```

Commandline options summary:
- ```-o``` will export in .obj format, 
- ```-f``` will export in .fbx format
- ```-s``` will export in .stl format
- ```-p``` will export in .ply format. 
- ```-v``` enables per-vertex colors (only compatible with .fbx and .ply formats). 
- ```-r {INT}``` controls the resolution of the baked texture maps. For instance, ```-r 1024``` will export 1024 x 1024 texture maps.

Only one file type can be specified for each export.

## Known Issues and Limitations

* Assets that use transparency or have fur will have incorrect textures when exporting. This is unavoidable due to texture maps being generated from baking.

* When using the vertex color export option, no roughness will be exported, only diffuse color

* Very big assets (e.g. full trees with leaves) may take a long time to export and will crash Blender if you do not have a sufficiently large amount of memory. The export results may also be unusably large.

* When exporting in .fbx format, the embedded roughness texture maps in the file may sometimes be too bright or too dark. The .png roughness map in the folder is correct, however.

* .ply bush exports will have missing leaves when uploaded to SketchFab, but are otherwise intact in other renderers such as Meshlab.

* .fbx exports ocassionally fail due to invalid UV values on complicated geometry. Adjusting the 'island_margin' value in bpy.ops.uv.smart_project() sometimes remedies this

* If using exported .obj files in PyTorch3D, make sure to use the TexturesAtlas because a mesh may have multiple associated texture maps

* Loading .obj files with a PyTorch3D inside a online Google Colabs sessions often inexplicably fails - try hosting locally

* The native PyTorch 3D renderer does not support roughness maps on .objs





