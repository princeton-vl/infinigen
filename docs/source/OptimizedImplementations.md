# Optimized Implementations

Infinigen ships optional alternative implementations of parts of the
generator. They are **off by default**. With them off, generation is what it
has always been; nothing in this page changes unless you ask for it.

They exist because a few generators dominate scene time. With everything
selected and the land-process pool on, the nature scenes generate 3 to 16
times faster at full quality on eight CPUs (see
[Full-quality scenes](#full-quality-scenes)). Where an optimized
implementation produces different geometry, that is stated below.

## Enabling

Everything at once:

```bash
python -m infinigen_examples.generate_nature --configs performance ...
```

Some of them:

```bash
python -m infinigen_examples.generate_nature -p "optimized.packages=['trees', 'leaves']" ...
python -m infinigen_examples.generate_nature -p "optimized.exclude=['terrain']" --configs performance ...
```

From Python, for a script or a notebook:

```python
import infinigen

infinigen.use_optimized()                     # every one
infinigen.use_optimized(["trees", "leaves"])  # only these
infinigen.use_optimized(exclude=["trees"])    # every one but trees
infinigen.use_optimized(False)                # none
```

The selection is made by a gin binding (`optimized.packages`,
`optimized.exclude`, which `--configs performance` sets) or by a
`use_optimized()` call. The gin binding is written into each stage's
`operative_gin_*.txt` alongside the scene.

The test suite takes `--optimized`:

```bash
python -m pytest tests/infinigen/performance_v1 --optimized
python -m pytest tests/infinigen/performance_v1 --optimized=trees,placement
```

The choice can be made or changed at any point: each gated definition looks
it up on its next call. Objects built before a change keep the implementation
they were built with.

To see what is selected:

```python
import infinigen
from infinigen import perf

print(infinigen.optimized_packages())  # {'terrain': True, 'trees': True, ...}
print(perf.status())                   # infinigen.perf: core, creatures, ...
```

| Name | Covers |
|---|---|
| `terrain` | erosion, snowfall, the meshers, the cave SDF, the voronoi rocks element, the terrain mesh utilities, and optionally land processes in worker processes |
| `leaves` | broadleaf, ginko, maple, v2 heightfield meshing |
| `monocot` | grasses, agave, growth, pinecone, the monocot leaf sheet |
| `small_plants` | fern node-group templating |
| `trees` | skeleton, placeholder and asset builds, attractor sampling, native trunk meshing and bark displacement |
| `corals` | brain and honeycomb reaction diffusion, cauliflower and star coral bodies built from arrays |
| `creatures` | data-level joining of bird parts |
| `placement` | camera pose search and animation, population batching, scene isolation |
| `mushroom` | the mushroom factories: cluster, cap and stem |
| `core` | the shared helpers every asset path goes through: attribute reads and writes, tag lookup, UV transfer, orphan removal, collections, random parameter draws |
| `runtime` | process-level settings: input-scene prefetch, compressed stage outputs, the OptiX kernel cache |

Any combination is valid: each is a drop-in alternative to the default one,
so turning one on never requires turning on another. Where an optimized
package could use something another one adds, it checks the selection and
otherwise does what the default implementation does. One such case exists
today: without `placement`, optimized trees that remesh let the remesh
interpolate the skinning attributes, exactly as the default implementation
does.

### Pinecone

There is no separate pinecone name. `PineconeFactory` subclasses
`MonocotGrowthFactory` and the optimization is of the shared growth path, so
it comes with `monocot`, together with grasses and agave.

## How it works

The optimized code lives under `infinigen/perf`, at the same relative path as
the code it replaces, and contains only the definitions that differ. Its
layout, and where the native kernels live, are in `infinigen/perf/README.md`. Each
default definition that has an optimized counterpart carries a gate:

```python
@gin.configurable
@optimizable("TREES", "infinigen.perf.assets.objects.trees.generate")
def make_leaf_collection(seed, leaf_params, n_leaf, leaf_types, decimate_rate=0.0, season=None):
    ...  # the default implementation, unchanged
```

With its name unselected the gate calls the default body; selected, it calls
the counterpart of the same name, whose signature must match exactly (a
mismatch is an error when the gate first resolves, not a failure deep in a
scene). A gate goes on exactly the definition that is replaced for every
caller. A gate on a method is inherited by every subclass; a class whose
representation differs is substituted as a whole (`optimizable_class`).
Gin decorates the gated definition, so config bindings apply to the optimized
implementation just as to the default one. The gate helper is
`infinigen/_optimized.py`; the selection is `infinigen/_perf_selection.py`.

A tree shipped without `infinigen/perf` runs every default body.

## Settings

These are gin configurables of the optimized implementations, registered
whenever `infinigen.perf` is imported (which the pipeline does before parsing
configs). Binding one has no effect unless its implementation is selected.

| Binding | Default | Effect |
|---|---|---|
| `isolation_enabled.enabled` | `True` | build each asset, and evaluate each animated camera or creature path, in a scratch view layer (placement); see below |
| `only_needed_land_processes.enabled` | `True` | skip land-process simulations the scene never reads (terrain) |
| `land_process_pool.enabled` | `False` | run erosion, snowfall and the cave assets in worker processes, several at once, on the CPUs the job has (terrain); see below |
| `compress_output_blend.enabled` | `True` | write stage blends in Blender's lossless zstd container (runtime) |
| `prefetch_input_scene.enabled` | `True` | warm the page cache before reading an input scene and its terrain assets (runtime, terrain) |
| `optix_kernel_cache.enabled`, `.directory` | `True`, `~/.cache/infinigen/optix` | keep the OptiX kernels GPU renders compile where later jobs find them; an `OPTIX_CACHE_PATH` already set is kept (runtime); see below |
| `direct_monocot_leaves.enabled` | `True` | build monocot leaf sheets from arrays; `False` keeps the operator-based construction, which differs in face count (monocot) |
| `native_pinecone.enabled` | `True` | build pinecones with the native kernels; `False` keeps the default pinecone, which builds about 2% fewer triangles (monocot) |
| `native_tree_meshing.enabled` | `True` | mesh each stem of a tree as its own tube and displace the bark on that mesh, instead of voxel-remeshing the welded trunk; `False` keeps the remesh (trees); see below |
| `native_tree_edge_length.ceiling_factor` | `1.7` | the longest edge native tree meshing leaves, as a multiple of the target face size; 1.0 builds about three times as many triangles (trees) |

Scene isolation matters: every operator call re-evaluates the depsgraph of
the current view layer, so the cost of building one asset grows with how much
is already in the scene. On a coarse forest scene one mushroom takes 33.8 s
built alone and 83.7 s built in the populated scene, and that difference is
what it removes. It changes the evaluation context every build runs in, so
it has its own switch. Its scratch view layer is saved with the scene and is
never rendered.

Camera animation uses the same scratch view layer. `Scene.frame_set`
evaluates every view layer of the scene, and the trajectory search sets the
frame once per frame of every candidate step. While a camera or a creature's
path is searched, the optimized placement evaluates the animated object
alone, at the same time and through the same fcurve and transform
evaluation, so the keyframes are the default's bit for bit. It also evaluates
the terrain SDF at all of a step's camera positions in one call, and stops
the pose validator's ray cast as soon as its coverage test is decided. A
policy or validator that may read other objects (following a creature, say)
keeps `frame_set`. Coarse stage, 16 threads, frames 1 to 10, against
`placement` without it:

| scene | `animate_cameras` | `flying_creatures` | keyframes and paths |
|---|---|---|---|
| coast, seed 2 | 8:01 → 0:05 | 27.6 s → 0.8 s | identical |
| cave, seed 2 | 4:47 → 0:09 | 28.8 s → 0.3 s | identical |

Compressed stage blends: a 4 GB fine-terrain scene becomes about 1.5 GB for
roughly two seconds of extra write time, and the next stage's cold read
scales with file size. The datablocks are unchanged and Blender opens either
form without being told which it is.

The OptiX kernel cache: before a GPU render, Cycles compiles its OptiX
kernels for the features the scene uses, and the driver caches the result in
`/var/tmp/OptixCache_<user>`. On a cluster that gives each job its own
`/tmp` and `/var/tmp`, as Slurm's `job_container/tmpfs` does, the cache is
deleted with the job and every render job compiles again. With `runtime`
selected the cache is kept in `$XDG_CACHE_HOME/infinigen/optix` (by default
`~/.cache/infinigen/optix`) instead. On a coral reef scene on an A40, a
128-pixel test render took 181 s with the driver's cache and 168 s with an
empty one, and 12 s in a later job that found it. The cache holds about
20 MB per GPU architecture and set of scene features. Only compiled code is
cached, so renders are unchanged: a render with the cache differed from one
without it by at most 2e-6, as much as the renders before and after a cache
hit differed from each other, the GPU's own variation between runs. On a
shared file system, jobs on several
nodes then write the same cache file; `optix_kernel_cache.directory` can
point it at storage local to each node instead.

## Optimized terrain

Terrain is not built by a default install at all — `INFINIGEN_MINIMAL_INSTALL`
defaults to `True` — so none of this applies unless you built terrain from
source, which additionally requires the `infinigen_gpl` submodule.

Erosion, snowfall, the spherical meshers, the cave SDF queries and the
terrain mesh utilities are faster, each on one CPU: the erosion simulation
does the same arithmetic with fewer lookups and allocations (221 s to 130 s
on a 2048² tile), the snowfall diffusion is numpy instead of Python loops
(46 s to 1.1 s, bitwise identical to landlab), and the cave SDF uses a tree
better suited to its query distribution.

One thing changes behaviour rather than only speed: simulations whose
outputs the scene never reads are skipped, so the on-the-fly asset folder
holds fewer files than before. Asset-library generation
(`tools/terrain/generate_terrain_assets.py`) still produces everything.

### Land processes in worker processes (off by default)

Every optimization above is faster on one CPU. This one is not: it makes a
job with several CPUs finish sooner and does nothing for a job with one, so
it is off unless `land_process_pool.enabled = True`. The simulations are
single threaded, so without it a job given four CPUs erodes its land tiles on
one of them.

With it, the erosion and snowfall of each land tile and each cave asset run
in worker processes, several at once. Heightmaps are still sampled in the
main process, in order, since they consume the random state, and each step
is planned there with its parameters and seed resolved, so a worker writes
exactly the files the step writes inline. The pool takes as many workers as
the job has CPUs, fewer if the job's memory limit would not hold them
(`worker_mem_gb`, 3 GB a worker by default: eroding a 2048² tile peaks at 1.2
to 2.9 GB), and a cave worker's KD-tree queries use its share of the CPUs.
With one CPU no pool is created at all. `land_process_pool.max_workers` sets
the count directly.

Coarse stage, everything selected, off and then on in the same job:

| scene, seed 2 | CPUs | terrain stage | whole stage | peak memory |
|---|---|---|---|---|
| coast | 4 | 26:46 → 12:05 (2.2×) | 36:51 → 22:04 | 10.4 → 11.2 GB |
| coast | 8 | 27:50 → 7:28 (3.7×) | 36:10 → 16:09 | 10.5 → 11.6 GB |
| cave | 4 | 19:24 → 13:49 (1.4×) | 30:24 → 23:46 | 9.1 → 10.4 GB |
| cave | 8 | 18:39 → 8:06 (2.3×) | 26:17 → 15:41 | 9.5 → 10.4 GB |
| mountain | 4 | 10:49 → 5:53 (1.8×) | 21:37 → 17:07 | 8.9 → 10.2 GB |
| mountain | 8 | 9:02 → 3:06 (2.9×) | 17:25 → 11:15 | 9.2 → 10.3 GB |

In every run the terrain asset files were the same bit for bit and the
camera keyframes identical. Four CPUs is what the repository's Slurm config
gives the coarse stage (`queue_coarse.cpus`).

### Compiled kernels

- **SoilMachine erosion** replaced code rather than wrapping it, so the
  optimized sources are a copy of the files they change
  (`infinigen/perf/terrain/source/cpu/soil_machine`), built by
  `compile_terrain.sh` into `SoilMachine_perf.so` alongside the default
  `SoilMachine.so`. The optimized erosion loads the first, and the default
  one if it was not built.

- **Voronoi rocks** caches, per thread, the two values each query evaluates at
  every nearby Voronoi cell centre (the mask noise and the attachment SDF):
  they depend only on the centre, and neighbouring queries share centres, so
  the results are bitwise the same. The cache is in the element's header,
  which every terrain element library compiles, so it is built into the
  default library but stays off there; the optimized terrain turns it on
  through the library's `set_attachment_cache`.

Equivalence was checked against the default implementation on the cases in
`tests/infinigen/performance_v1` and, for the C++ kernels, by comparing
checksums of full-scale outputs. The fine terrain mesh depends on the number
of CPUs the process has, in both implementations, so the two are compared at
equal CPU counts.

## Building the native kernels

Most optimized implementations use compiled extensions, built by installing
with `INFINIGEN_INSTALL_PERF`:

```bash
INFINIGEN_INSTALL_PERF=True pip install -e .
```

Without them, a selected implementation logs one line and leaves the default
in place:

```
WARNING infinigen.perf.kernels: optional native kernel ... unavailable (...);
  using the default implementation. Build with: ...
```

Some kernels are GPL-2.0-or-later. They live in the optional `infinigen_gpl`
submodule (`infinigen_gpl/perf`) and are built when it is checked out; an
implementation that needs one uses it when it is there.

## Writing an optimized implementation

Every optimized implementation is reached through a gate on the default
definition (see [How it works](#how-it-works)). Put
the counterpart at the same relative path under `infinigen/perf` and gate the
default. A factory whose optimized form is a subclass -- the mushroom
factories, for example -- is gated as a class:

```python
# infinigen/assets/objects/mushroom/generate.py
@optimizable_class("MUSHROOM", "infinigen.perf.assets.objects.mushroom.factories")
class MushroomFactory(AssetFactory):
    ...

# infinigen/perf/assets/objects/mushroom/factories.py
class MushroomFactory(_generate.MushroomFactory):
    def build_mushrooms(self, i, face_size=0.01):
        ...
```

A new name also goes in `SELECTABLE` in `infinigen/_perf_selection.py`.

- Load every kernel through `perf.kernels.optional_import`, and check all of
  them when the counterpart's module is imported, raising ImportError if one
  is missing, so that the gate keeps the default.
- Override only the methods that differ. Anything inherited stays on the
  default implementation and follows its changes.

## What changed in the default implementation

With nothing selected, the default implementation is the previous one plus
the gates, a few hooks that do nothing unless an optimized implementation
registers with them (scene isolation), the voronoi rocks cache, off unless
the optimized terrain turns it on, and these fixes:

- `ReedMonocotFactory` crashed reading its ear attribute;
- bushes grew differently in every process: the UV sphere and cone their
  attractors are sampled from came out of Blender in a per-process face
  order, and are now rebuilt in a fixed one before sampling;
- the deformed tree factories crashed on their bark material, and hollow and
  rotten trees looked their rings material up by a name Blender may suffix;
- `geo_instance_scatter` crashed whenever `dist_max` was given;
- the reptile factories (snakes, lizards, chameleons) crashed in their
  postprocessing, which still took the arguments of an older
  `join_and_rig_parts` and applied material modules instead of materials;
- `uvs_to_attribute` left vertices without loops holding uninitialized memory;
- `require_blender_addon` trusted an add-on being listed as it being
  registered, and the cave assets did not require the add-on they use;
- a node-transpiler log call raised whenever INFO logging was on;
- `delete_collection` iterated a collection's object list after freeing the
  collection, reading freed memory. It now only removes the collection: the
  loop came back empty in practice, leaving the objects in no scene, and that
  is what it still does.

### Where the geometry differs

Most optimized packages build a different mesh from the default, even where
the triangle counts agree. Only the fern, the bird and the brain and
honeycomb corals are bit-identical.
Measured with
`infinigen.tools.perf_validation` at ten seeds per asset on one CPU,
comparing each optimized build with the default one by a digest of its
geometry and, where they differ, by the distance between points sampled on
the two surfaces (as a fraction of the asset's size, beside the floor that
two samplings of one surface give):

| asset | triangles, default → optimized | build time, one CPU | same mesh | deviation, mean / p99 (floor) |
|---|---|---|---|---|
| `TreeFactory` | 133,724,996 → 100,907,144 (-24.5%) | 5,421 s → 262 s, 20.7× | 0 of 10 | 0.11% / 0.89% (0.07% / 0.18%) |
| `BushFactory` | 1,921,608 → 1,291,356 (-32.8%) | 331 s → 15 s, 21.9× | 0 of 10 | 0.12% / 0.91% (0.09% / 0.22%) |
| `LeafFactoryV2` | 71,193 → 71,193 (+0.0%) | 30 s → 3 s, 9.8× | 0 of 10 | 0.08% / 0.20% (0.07% / 0.18%) |
| `LeafFactoryBroadleaf` | 76,543 → 76,543 (+0.0%) | 107 s → 6 s, 17.2× | 0 of 10 | 0.10% / 0.38% (0.07% / 0.17%) |
| `LeafFactoryMaple` | 166,882 → 166,938 (+0.0%) | 195 s → 55 s, 3.6× | 0 of 10 | 0.11% / 0.29% (0.07% / 0.17%) |
| `LeafFactoryGinko` | 36,874 → 36,878 (+0.0%) | 24 s → 3 s, 8.5× | 0 of 10 | 0.08% / 0.17% (0.07% / 0.16%) |
| `GrassesMonocotFactory` | 2,938,964 → 2,939,367 (+0.0%) | 28 s → 15 s, 1.9× | 0 of 10 | 0.06% / 0.15% (0.05% / 0.13%) |
| `AgaveMonocotFactory` | 48,605,795 → 48,633,728 (+0.1%) | 157 s → 47 s, 3.3× | 0 of 10 | 0.23% / 0.55% (0.20% / 0.49%) |
| `PineconeFactory` | 8,644,852 → 8,827,774 (+2.1%) | 28 s → 2 s, 12.3× | 0 of 10 | 0.40% / 1.14% (0.29% / 0.72%) |
| `FernFactory` | 17,832,510 → 17,832,510 (+0.0%) | 145 s → 36 s, 4.1× | 10 of 10 | -- |
| `MushroomFactory` | 17,683,898 → 17,565,378 (-0.7%) | 199 s → 18 s, 11.2× | 0 of 10 | 0.45% / 2.73% (0.15% / 0.38%) |
| `BirdFactory` | 1,468,772 → 1,468,772 (+0.0%) | 334 s → 256 s, 1.3× | 10 of 10 | -- |
| `BrainCoralFactory` | 47,652,104 → 47,652,104 (+0.0%) | 923 s → 336 s, 2.8× | 10 of 10 | -- |
| `CauliflowerCoralFactory` | 61,402,576 → 61,406,241 (+0.0%) | 798 s → 464 s, 1.7× | 0 of 10 | 0.08% / 0.41% (0.29% / 0.69%) |
| `StarCoralFactory` | 38,788,240 → 38,802,582 (+0.0%) | 723 s → 264 s, 2.7× | 0 of 10 | 0.23% / 0.67% (0.23% / 0.62%) |

Build times are summed over the ten seeds; the deviation columns are medians
over the seeds whose meshes differ.

Leaves, grasses and agave keep their counts and move their vertices: most
seeds deviate at the sampling floor, a few by more (broadleaf seeds 1, 2, 4,
8 and 9, maple 0, 7 and 8 reach a 99th percentile of 0.6--1.3% of the leaf's
size, agave seeds 1 and 6 5.5--7.9%). The trees, the bushes, the mushroom and
the pinecone change what is built; the tool renders each pair side by side.

The brain coral runs its reaction diffusion as one compiled loop and builds
the same mesh; so does the honeycomb coral, which takes the same route (also
10 of 10, not in the table). The cauliflower coral computes the limit surface
of its subdivision on arrays instead of applying the modifier, which agrees to
float rounding, and the star coral builds its flowers on arrays. The voxel
remesh that finishes both turns those differences into a few triangles more or
fewer on most seeds and up to 0.4% on a few, with the surfaces at the sampling
floor. These times are of corals built alone. In a coral reef scene, where
every operator call also evaluates the scene around them, the `corals` stage
of the coarse step takes 1:15 with everything selected, against 39:15 with
everything but `corals` (seed 1, 16 threads).

**The optimized star corals have flowers in a scene.** The default extracts
the boundary of each subdivided face in edit mode, which depends on the mesh
select mode it finds: in vertex mode, as in the validation tool, it builds the
flowers, but a nature scene is in another mode by the time its corals are
built, and every default star coral there has none. The optimized star coral
builds them in any mode. In the coral reef scene above, its seven star corals
have 1.3 to 2.0 million triangles each by default and 3.1 to 4.5 million
optimized, so a reef looks different with `corals` selected.

Comparing reef scenes, expect the table corals to differ in any case: their
differential growth is not reproducible, even with nothing selected and on
one CPU. Two identical runs built three table corals 12 to 27% apart in
triangles.

Compare triangles, not polygons. A quad mesh and a triangulated mesh of the
same surface differ by a factor of two in polygon count while agreeing in
triangles, and these paths differ in topology.

**The optimized mushrooms have 0.8% fewer triangles.** On the seeds the matched-detail work validated, one mushroom has
3,263,624 triangles by default and 3,237,672 optimized, 0.8% fewer, while
taking 33.8 s against 3.6 s. The default mesh is entirely quads and the
optimized one is mixed, so the same surface reads as 1,631,812 against
2,458,288 polygons. With every package but `mushroom` selected, a scene has
the default mushroom counts exactly.

**The optimized pinecones are slightly denser.** The optimized monocot
replaces the pinecone with a native implementation, and across ten seeds it
builds between 0.25% fewer and 3.35% more triangles -- more on nine of the
ten -- in 0.26 s per seed instead of 4.1 s. Each side reproduces its
own counts exactly when re-run.

**The optimized trees and bushes are much faster to build, and lighter.**
Native trunk meshing keeps every stem as its own tube, where the voxel remesh
welds them into one skin, and retessellates each tube so that no edge is
longer than 1.7 times the target face size -- the density the tree benchmarks
were run at; the
bark is then displaced on that mesh, before any children are joined into it.
Across the ten seeds above, trees build in 262 s instead of 5,421 s (20.7
times faster, against 3.0 times with the remesh kept) and bushes in 15 s
instead of 331 s (21.9 times, against 3.0),
with 24.5% and 32.8% fewer triangles. The surfaces agree with the remeshed
ones to about a tenth of a percent of the tree's size on average, and to
about 1% at the 99th percentile.

`native_tree_edge_length.ceiling_factor` sets that multiple. At 1.0, the
density native meshing was validated at (sub-pixel edges
at production viewing distance, a silhouette IoU of 0.994, the forest gate),
the same ten seeds build 2.1 and 1.8 times the default's triangles, 7.9 and
8.3 times faster, with the same surface deviation.
`native_tree_meshing.enabled = False` keeps the remesh and the other tree
optimizations, which build the default triangle counts.

**The optimized leaves agree in count.** On the path by which a leaf reaches
a scene, the two routes agree to within a triangle; the vertices differ.

`LeafFactoryV2` never decimates. `make_leaf_collection` does, at
`decimate_rate=0.97`, and it is the only route by which a leaf reaches a
scene. The height-field route reads the same rate, builds at the density it
implies, and skips the destructive pass, so the two arrive at the same place
from opposite directions. Measured through `random_leaf_collection` -- the
call the `ground_leaves` scatter makes -- at five leaves per seed:

| seed | default | optimized |
| ---- | ------- | --------- |
| 0    | 6,609   | 6,610     |
| 1    | 5,909   | 5,909     |
| 2    | 5,454   | 5,455     |

The `LeafFactoryV2` template object is stored before decimation by the
default path and at the density it is used at by the optimized one, so its
own count differs by the decimation ratio (0.03); nothing instances the
template.

### A whole coarse scene

Generated both ways at seed 0, `--configs forest simple dev`:

| | default | optimized |
| --- | --- | --- |
| objects | 3,506 | 2,641 |
| triangles | 39,113,021 | 37,499,778 |
| cameras | identical | identical |

The 865 extra objects of the default scene are `Tree.NNN` meshes with zero
triangles in bush placeholder collections. Of the objects both scenes have, 48 differ in count:

| objects | default | optimized | |
| --- | --- | --- | --- |
| 20 twig assets (`GenericTreeFactory`), meshed natively | 2,501,172 | 2,066,794 | -17.4% |
| 6 `LeafFactoryV2` templates | 1,154,716 | 44,582 | stored at the density used |
| 10 mushrooms | 20,797,548 | 20,752,692 | -0.22% |
| 5 pinecones | 4,300,860 | 4,276,992 | -0.55% |
| 7 broadleaf leaves | 59,226 | 59,219 | 7 triangles |

The trees themselves are placeholders in a coarse scene until they are
populated. The cameras and every other object are unchanged.

### Full scenes

The nine scene types that generate at seed 0 with `simple dev`, each
generated coarse, populated and with its fine terrain on one CPU, then
rendered on a GPU as a separate job
(`perf_validation.run scenes --full --stage generate`, then `--stage render
--gpu`). The cave scene fails to place its cameras at this seed on both
sides, as it does on main. Triangles are those of the generated assets
(`assets:<factory>` collections), which the scene draws as instances:

| scene | generate, one CPU | generated-asset triangles |
|---|---|---|
| arctic | 6.8 → 3.9 min, 1.7× | 393,216, unchanged |
| canyon | 16.1 → 6.5 min, 2.5× | 12,102,659 → 11,681,611 (-3.5%) |
| coast | 15.3 → 6.2 min, 2.5× | 12,102,659 → 11,681,611 (-3.5%) |
| desert | 8.7 → 3.6 min, 2.4× | 2,077,845 → 1,656,793 (-20.3%) |
| forest | 125.4 → 8.3 min, 15.1× | 37,415,514 → 36,911,492 (-1.3%) |
| mountain | 15.7 → 7.6 min, 2.0× | 12,102,659 → 11,681,611 (-3.5%) |
| plain | 11.7 → 5.5 min, 2.1× | 12,206,735 → 11,785,687 (-3.4%) |
| river | 15.0 → 6.9 min, 2.2× | 12,134,204 → 11,713,156 (-3.5%) |
| snowy_mountain | 9.0 → 1.5 min, 6.0× | none |

The assets that change are the natively meshed bush twigs and tree-like
assets (33--36% and 15% fewer triangles), the mushrooms and pinecones.
Terrain and cameras are the same on both sides, and eight of the nine
renders cannot be told apart.

The forest is the one scene that populates detailed trees, and there they
are heavier: its 26 trees carry 23,466,824 triangles by
default and 26,948,050 natively (+14.8%). The five largest trees (over two
million triangles each) are 3--11% lighter, as isolated trees are; the
smaller, lower-detail ones are 1.2 to 2.0 times heavier. At a coarse
target the voxel remesh merges and drops thin branches, while native
meshing keeps every stem as a tube of at least its profile's rings. In
its render the trunks are smooth where the remeshed ones are faceted, and
the leaves match the default ones.

### Full-quality scenes

Every nature scene type at seeds 1 and 2, at full quality (`--configs
<scene>`, without `simple` or `dev`), generated for frame 1 the way
datagen chains the stages: `coarse`, `fine_terrain` and `populate` as
separate processes, each timed from start to exit. Rendering is not
included. The default side is main at `224598d`, the commit this one is
based on. The optimized side is `--configs <scene> performance -p
land_process_pool.enabled=True`. Both sides of a scene ran at the same time
on one node, each on its own 8 physical cores (16 threads), Xeon Silver
4114 or 4214:

| scene | seed 1 | seed 2 |
|---|---|---|
| arctic | 38.1 → 11.5 min, 3.3× | 19.4 → 3.5 min, 5.5× |
| canyon | 129.8 → 7.9 min, 16.5× | 121.3 → 15.7 min, 7.7× |
| cave | 90.7 → 10.8 min, 8.4× | 62.7 → 13.2 min, 4.7× |
| cliff | 59.6 → 8.9 min, 6.7× | 54.8 → 10.8 min, 5.1× |
| coast | 111.0 → 13.0 min, 8.5× | 296.3 → 21.0 min, 14.1× |
| desert | fails → 7.9 min | fails → 7.7 min |
| forest | 153.4 → 11.1 min, 13.8× | 48.7 → 7.1 min, 6.9× |
| mountain | 104.7 → 10.1 min, 10.4× | 83.8 → 13.1 min, 6.4× |
| plain | 43.9 → 7.3 min, 6.0× | 27.2 → 7.7 min, 3.5× |
| river | 65.7 → 8.2 min, 8.0× | 57.6 → 12.6 min, 4.6× |
| snowy_mountain | 28.3 → 5.2 min, 5.5× | 26.6 → 4.4 min, 6.1× |

On main both desert scenes fail in `populate` on the reptile crash fixed
above. Over the other 20, the geometric mean is 6.9× and the median 6.6×,
and by stage:

| stage | default | optimized | |
|---|---|---|---|
| `coarse` | 17.2 h | 2.0 h | 8.6× |
| `fine_terrain` | 1.7 h | 0.8 h | 2.2× |
| `populate` | 8.2 h | 0.6 h | 13.5× |
| all three | 27.1 h | 3.4 h | 8.0× |

Part of the `coarse` gain is the land-process pool putting the eight CPUs
to work on erosion and snowfall, which the default runs on one (see
[the pool's own timings](#land-processes-in-worker-processes-off-by-default)).
Most of what is left of the optimized runs is the terrain (39 of the 203
minutes), the chopped trees, which have no optimized implementation (36
minutes, 1.6× from the shared helpers), and meshing the fine terrain (20
minutes, 2.6×).

## Reproducibility

Blender's multithreaded Subdivision Surface is not bit-reproducible: on more
than one CPU, the same input comes out a few float32 ulps apart from one run
to the next, in either implementation. It can change a count: `BirdFactory`
places its tail by raycasting from a subdivided skeleton, and at seed 7 the
jitter moves the tail by 3e-8, enough for the remesh after joining to build
134,048 triangles instead of 134,052 in about a quarter of runs on four or
more CPUs. A count that looks stable on several CPUs need not be the one-CPU
count either: seed 8 built 120,212 on eight CPUs in every run recorded, and
builds 120,208 on one.

For output that is the same on every run, run on one CPU (`taskset -c 0`, or
Slurm's `-c 1`). Blender sizes its thread pool once, from the CPUs the
process may use, so it has to be set before Blender starts. The validation
tool builds its assets that way.

The default mushroom is built with Blender's spin operator, whose face order
follows a pointer-keyed map, so a mushroom seed can still come out a few
triangles apart between runs (seed 9: 964,518 in six runs of seven, 964,522
in one). The optimized mushroom spins every surface natively -- caps, gills,
stems and both kinds of web -- and is the same mesh, bit for bit, on every
run (ten seeds, three runs each).
