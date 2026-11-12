# Exporting Infinigen 2 Assets

Infinigen 2 can save generated assets as `.blend` files with the `--save_blend` flag. This guide shows how to convert those files into baked FBX, OBJ, or USD assets for use in external tools such as Unity, Unreal, or Blender.

The conversion reuses the Infinigen 1 material-baking exporter in `src/infinigen/tools/export.py`. Infinigen 2 does not have a native FBX or GLB exporter yet (see [#248](https://github.com/princeton-vl/infinigen/issues/248)), so this is currently the most reliable path for individual assets and small scenes.

## Requirements

- A working Infinigen 2 installation, for example from `uv sync --extra dev`.
- `gin-config`, which the Infinigen 1 exporter imports. It is not part of the default Infinigen 2 dependencies:

  ```bash
  uv pip install gin-config
  ```

  Alternatively, add it to a single command with `uv run --with gin-config ...`.

## Quick start

Generate a single asset and save its blend file. Use an absolute output path, especially on Windows:

```bash
uv run infinigen sofa_rand --seed 0 --output /absolute/path/to/outputs/sofa --save_blend
```

This writes `/absolute/path/to/outputs/sofa/scene.blend`.

Export the blend to a baked FBX:

```bash
uv run --with gin-config python scripts/export_v2_asset.py \
    --input /absolute/path/to/outputs/sofa/scene.blend \
    --output /absolute/path/to/outputs/sofa/export \
    --format fbx \
    --resolution 1024
```

The exporter writes:

```text
outputs/sofa/export/sofa/
├── sofa.fbx
└── textures/
    ├── mesh_single_vertex_DIFFUSE.png
    ├── mesh_single_vertex_NORMAL.png
    └── mesh_single_vertex_ROUGHNESS.png
```

The asset folder is named after the parent folder of `scene.blend` (`sofa` above). Pass `--name` to choose a different name.

For a scene with multiple objects, select the asset you want:

```bash
uv run --with gin-config python scripts/export_v2_asset.py \
    --input /absolute/path/to/outputs/room/scene.blend \
    --object mesh_single_vertex \
    --output /absolute/path/to/outputs/room/export \
    --format fbx
```

Replace `mesh_single_vertex` with the object name shown in Blender's Outliner. `--object` can be passed multiple times and keeps the object's children. Cameras, lights, and empties are removed by default; use `--keep-non-mesh` to keep them.

Other useful flags:

| Flag | Description |
| --- | --- |
| `--format obj`, `usdc`, `usda`, `stl`, `ply` | Output format. FBX is the default. |
| `--resolution 2048` | Resolution of the baked texture maps. |
| `--individual` | Write each mesh object to its own file. |
| `--vertex-colors` | Bake colors into vertex colors instead of textures (FBX, OBJ, and PLY only). |

Run `python scripts/export_v2_asset.py --help` for the full list.

## Importing into Unity

1. Copy the exported asset folder (the `.fbx` and its `textures` folder) into your Unity project's `Assets` folder. Keep the textures next to the FBX, or move them to a shared `Assets/Textures` folder.
2. Select the FBX in the Project window and open the **Model** tab. Set **Scale Factor** to `1` and leave the default axis settings. Infinigen exports in meters, and Blender's FBX exporter already converts Z-up to Y-up.
3. Open the **Materials** tab and choose **Extract Materials...** so you can edit the generated material. Unity will usually create a material from the FBX, but the normal and roughness maps often need to be assigned manually.
4. Assign the baked maps:
   - `*_DIFFUSE.png` to **Base Map** (sRGB enabled).
   - `*_NORMAL.png` to **Normal Map** (Texture Type: Normal map, sRGB disabled).
   - `*_ROUGHNESS.png` as a roughness or smoothness input. The Standard shader expects smoothness, so invert the roughness map or use a shader that reads roughness directly. URP and HDRP Lit shaders store smoothness in the metallic map's alpha channel.
5. If the model imports at the wrong scale, check the **Scale Factor** and **Convert Units** settings before changing the axis conversion. Changing the axes on top of Blender's conversion usually makes the orientation worse.

:bulb: The exporter bakes **DIFFUSE**, **ROUGHNESS**, and **NORMAL** maps. Metallic and transmission are not baked into textures, so glass and other transmissive materials may need manual setup in Unity.

## Limitations

- **Individual assets and small scenes only.** The exporter realizes instances before writing the mesh. Full Infinigen scenes with dense scatters can produce enormous files or run out of memory. See [ExportingToExternalFileFormats](./ExportingToExternalFileFormats.md) for the same limitation in Infinigen 1.
- **Modifiers are applied.** Subdivision surfaces are baked into the exported mesh. A single sofa can easily reach several hundred thousand triangles. Lower the subdivision level in the source asset or decimate after import if you need a game-ready mesh.
- **Static meshes only.** Animations, armatures, and skinned meshes are not supported by this path.
- **Material coverage is approximate.** The baker understands the material graphs produced by Infinigen 1 and common Principled BSDF setups. Unusual V2 node graphs may export with flat colors or missing maps.
- **No native V2 exporter.** This script is a bridge to the Infinigen 1 exporter, not a replacement for a proper V2 FBX or GLB exporter.

## Troubleshooting

**`ModuleNotFoundError: No module named 'gin'`**  
Install `gin-config` with `uv pip install gin-config`, or run the script with `uv run --with gin-config ...`.

**`No mesh objects are left to export`**  
The blend contains only cameras, lights, or empties, or the name passed to `--object` does not match an object in the file. Open the blend in Blender and check the object names in the Outliner.

**`Found multiple .blend files in ...`**  
Pass the specific file with `--input` instead of a folder.

**Textures are missing in Unity**  
Keep the `textures` folder next to the exported FBX and reimport the model. Check that the texture import settings use sRGB for the diffuse map and linear color space for the roughness and normal maps.

**The export path is wrong on Windows**  
Use absolute paths for both `--input` and `--output`. Some Infinigen 2 commands resolve relative paths against the drive root on Windows, and Blender resolves relative texture paths against the blend file rather than the current directory.

## Related documentation

- [ExportingToExternalFileFormats](./ExportingToExternalFileFormats.md) - the Infinigen 1 exporter and its options.
- [GeneratingIndividualAssets](./GeneratingIndividualAssets.md) - generating individual Infinigen 1 assets.
- [Command Line Interface](./cli.md) - Infinigen 2 generator commands.
