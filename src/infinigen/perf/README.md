# infinigen.perf

The optional optimized implementations. Nothing here runs unless its package
is selected (`optimized.packages`, `infinigen.use_optimized()`); what each
package does and how to select it is in `docs/source/OptimizedImplementations.md`.

## Layout

Every module mirrors the path of the module it optimizes:
`perf/assets/objects/trees/tree.py` holds the counterparts of definitions in
`assets/objects/trees/tree.py`. A module with no default counterpart is a helper
of the module beside it (`trees/tessellate.py`, `mushroom/meshing.py`), or,
when several packages use it, lives under `perf/assets/utils` or `perf/core/util`:

| | |
|---|---|
| `__init__.py` | `status()`, and applying the gin selection |
| `kernels.py` | `optional_import`: load a native kernel, or fall back to the default |
| `options.py` | the gin settings of the optimized implementations |
| `assets/utils/mesh_io.py` | bulk mesh reads and writes (mushroom, pinecone) |
| `assets/utils/surface_fields.py` | deferred scalar displacement (tree bark) |
| `assets/utils/heightfield/` | height-field leaf meshing |
| `assets/utils/native/` | the `_native_fields` module and the field operations it and the pinecone share |
| `core/util/isolation.py`, `blender_gc.py`, `prefetch.py` | scene isolation, orphan removal, input prefetch |

## Counterparts

A default definition with an optimized counterpart carries a gate
(`infinigen/_optimized.py`); the counterpart has the same name and signature.
It is a whole definition, so a counterpart that changes three lines of a
function repeats the rest of it -- keep such changes in the counterpart
rather than adding hooks to the default code. A counterpart identical to its
default is never needed: leave the definition out and it is inherited or
reached through the gate (`tests/infinigen/performance_v1/test_perf_layout.py`
checks this).

## Native code

Kernels live in a `native/` directory beside the code that uses them, and all
of them build under `INFINIGEN_INSTALL_PERF=True` from `setup.py`:

- Cython extensions (`perf_extensions`): the leaf, bark and height-field
  fields, the height-field triangulation, the tree cylinder mesher and
  skeleton operations, and the mushroom meshing.
- ctypes libraries (`perf_ctypes_extensions`): the pinecone kernels, loaded
  with `load_library` rather than imported.
- The optimized SoilMachine erosion (`terrain/source/cpu/soil_machine`),
  built by `scripts/install/compile_terrain.sh` into `SoilMachine_perf.so`.

GPL-2.0-or-later kernels are in the optional `infinigen_gpl` submodule,
under `perf/`. Third-party and adapted code keeps
its notice and licence beside it (`assets/utils/native/NOTICE.md`,
`assets/utils/heightfield/native/third_party/`, `terrain/mesh_to_sdf/LICENSE`).
