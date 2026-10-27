# Validating the optimized implementations

Builds each asset twice from the same seed -- once with nothing selected,
once with what you ask for -- and puts the two side by side, shaded and in
wireframe, with the triangle counts and whether the two meshes are the same.
It covers every asset an optimized implementation reaches.

```bash
# every asset, ten seeds, default against everything on
python -m infinigen.tools.perf_validation.run assets

# one package
python -m infinigen.tools.perf_validation.run --optimized=trees assets \
    --factories infinigen.assets.objects.trees.generate.TreeFactory

# ten scene types, coarse
python -m infinigen.tools.perf_validation.run scenes

# populated and with the fine terrain, which is where the optimized trees,
# bushes and other populated assets appear in a scene
python -m infinigen.tools.perf_validation.run scenes --full

# as two jobs: generate on one CPU (what is timed), then render on a GPU
python -m infinigen.tools.perf_validation.run --output out scenes --full --stage generate
python -m infinigen.tools.perf_validation.run --output out scenes --full --stage render --gpu
```

`--optimized` goes before the subcommand and defaults to everything. The
tool selects implementations as `docs/source/OptimizedImplementations.md`
describes -- a gin config for a generated scene, `use_optimized()` in a
render worker -- so it compares what a user gets. The baseline side runs
with nothing selected.

## Checking against the release

`expected/` holds the counts this release was validated at, and the seed-0
views beside them:

```bash
python -m infinigen.tools.perf_validation.run --output mine assets
python -m infinigen.tools.perf_validation.run check mine/*
```

`check` compares the triangle counts and exits non-zero if any differs;
`--tolerance 0.001` allows a fraction. The images in `expected/` are
reference views: Cycles renders vary across machines and versions.

Six assets differ in count from the default, by the amounts recorded in
`expected/`: the trees build about 25% fewer triangles and the bushes 33%
fewer (native trunk meshing), the mushroom about 0.7% fewer and the
pinecone about 2% more, and the cauliflower and star corals a few triangles
more or fewer per seed (up to 0.4%). `docs/source/OptimizedImplementations.md`
covers these and lists which assets build the same mesh.

## Mesh comparison

A triangle count measures detail, not where it is, so each asset row also
records a digest of its geometry -- world-space vertex positions and faces,
bit for bit -- and 100,000 points sampled on its surface, twice, with two
fixed seeds (`geometry.py`). The page's mesh column, and the lines `merge`
prints, say either *identical mesh* or how far the optimized surface lies
from the default one: the mean and 99th percentile nearest-sample distance,
as fractions of the asset's bounding-box diagonal, beside the same distance
between the default's own two sample sets. A deviation at that floor is
below what the samples resolve.

## One CPU

The asset worker runs on one CPU. Blender's multithreaded Subdivision
Surface varies by a few float32 ulps from run to run, and the bird turns
that into a different count: it places its tail by raycasting from a
subdivided skeleton, and at seed 7 the tail moves by 3e-8, enough for the
remesh after joining to build two vertices more or fewer in about a quarter
of runs on four or more CPUs. On one CPU Blender runs one thread and every
run builds the same mesh, so every count in `expected/` is a one-CPU count.

The default mushroom is built with Blender's spin operator, whose face order
follows a pointer-keyed map, so a seed occasionally comes out a few
triangles apart even on one CPU. Its manifest allows a relative difference
of 1e-5; the optimized mushroom, which spins every surface natively, is
checked exactly.

The confinement uses `os.sched_setaffinity` (Linux). On other systems the
worker runs on every CPU, says so, and a count can come out a few triangles
from `expected/`.

## Views

The shaded view frames the whole asset. The wireframe is a close-up of 18%
of the asset around its median vertex, seen from the same direction, so
each triangle is legible: the wire is one pixel wide at any distance, and
an asset framed whole would fill with solid wire. Topology changes that
keep the silhouette show in the wireframe.

`--modes trunk` adds a third view for trees and bushes: the wireframe of the
asset's own mesh, with its twigs and leaves hidden, around the lower trunk.
The window spans fifty of the mesh's median edges there (at most 6% of the
asset), and each side is framed on its own mesh.

Counts are triangles. A quad mesh and a triangulated mesh of the same
surface differ by a factor of two in polygon count and agree in triangles.

## Scenes

A scene row's mesh column compares the generated assets, collection by
collection (`assets:<factory>`), in triangles: a scene draws most of its
assets as instances of those collections, placed the same way on both
sides.

## Cost

Each asset row is two spawns and four renders on one CPU. Twelve factories
at ten seeds, shaded view only, one factory per job, took 3.6 CPU-hours,
more than half of it `TreeFactory`; the wireframe view adds to that.
`--seeds`, `--factories` and `--modes` narrow a run, and factories can run
as separate jobs.

The scene comparison generates each scene twice, so the ten scene types
take several hours on a compute node. Generate and render as separate jobs:
a full scene renders in about a minute on a GTX 1080 and in 17 to 68 minutes
on one CPU. The render stage records each blend's generation time, and with
`--gpu` it requires a GPU.
