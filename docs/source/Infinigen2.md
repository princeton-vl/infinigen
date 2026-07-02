# Infinigen 2.0

[Infinigen](https://infinigen.org/) is a procedural generator of 3D scenes, developed by [Princeton Vision & Learning Lab](https://pvl.cs.princeton.edu/). Infinigen is optimized for computer vision and robotics research and generates diverse high-quality 3D training data. Infinigen is based on Blender and is free and open-source (BSD 3-Clause License).

Infinigen 2.0 is top-to-bottom refactor of the system. It aims to create simple and controllable APIs for all asset generators while also improving efficiency, detail, and diversity. It is built on top of [`procfunc`](https://procfunc.readthedocs.io), a library for composing procedural generators out of small, randomizable building blocks. 

However, Infinigen 2.0 is in preview. It does not yet contain all asset types --- currently it only has new high quality indoor assets, but does not provide nature or articulated assets. 

For instructions on using Infinigen 1.0, please see the [Infinigen 1.0 Getting Started guide](Infinigen1).

## Installation

Install `uv` if you don't already have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then install Infinigen 2.0 from PyPI. It ships inside the `infinigen` package, and since 2.0 is currently an alpha you pin the pre-release:
```bash
uv pip install "infinigen==2.0.0a1"
```
or, inside a project:
```bash
uv add "infinigen==2.0.0a1"
```
Within a project, we recommend fixing a specific version of the alpha, as some interfaces may change gradually until we reach the 2.0 full release.

If you are modifying Infinigen itself, follow the [Contributing Guide](Contributing) to install from cloned source code. This is not necessary for most research projects - you can follow our [Example Projects](ExampleProjects) to create customized data using only the documented APIs.

## Examples

Every generator is a small composable building block: pick a *distribution* (a `*_rand` function) and a *target* (what to render it onto), and combine them on the command line.

### Materials

Materials render onto a demo mesh, here a torus, cube, and monkey:

<div class="example-grid">
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/landing/bricks_torus.png" alt="bricks material on a torus">
<pre><code>uv run infinigen bricks_rand material_torus_uv render_cycles</code></pre>
</figure>
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/infinigen2.shaders.composites.scratches_overlay.scratched_metal_rand/0.png" alt="scratched metal material on a cube">
<pre><code>uv run infinigen scratched_metal_rand material_cube render_cycles</code></pre>
</figure>
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/landing/fabric_patterned_monkey.png" alt="patterned fabric material on a monkey">
<pre><code>uv run infinigen fabric_patterned_rand material_monkey render_cycles</code></pre>
</figure>
</div>

### Objects

Objects render onto a neutral turntable-style demo scene:

<div class="example-grid">
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/infinigen2.objects.sofa.sofa_rand/0.png" alt="a procedurally generated sofa">
<pre><code>uv run infinigen sofa_rand object_demo render_cycles</code></pre>
</figure>
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/infinigen2.objects.lamp.lamp_rand/0.png" alt="a procedurally generated lamp">
<pre><code>uv run infinigen lamp_rand object_demo render_cycles</code></pre>
</figure>
</div>

### Scenes

Scenes are full rooms; each seed reshuffles layout, furniture, and materials:

<div class="example-grid">
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/infinigen2.scenes.room.room.livingroom_with_smallobj_rand/0.png" alt="livingroom seed 0">
<pre><code>uv run infinigen livingroom_with_smallobj_rand render_cycles --seed 0</code></pre>
</figure>
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/assets/images/infinigen2.scenes.room.room.livingroom_with_smallobj_rand/1.png" alt="livingroom seed 1">
<pre><code>uv run infinigen livingroom_with_smallobj_rand render_cycles --seed 1</code></pre>
</figure>
<figure class="example-card">
<img src="https://infinigen.cs.princeton.edu/docs/v2.0.0a1/images/infinigen2.scenes.room.room.livingroom_with_smallobj_rand/2.png" alt="livingroom seed 2">
<pre><code>uv run infinigen livingroom_with_smallobj_rand render_cycles --seed 2</code></pre>
</figure>
</div>

### Other examples

Run a render with ground truth annotations:
```bash
uv run infinigen bricks_rand material_torus_uv render_cycles render_cycles_ground_truth visualize_gt --show --debug --passes rgb surface-normal depth
```

Run many renders locally (increase `-P` to run more in parallel! warning - may crash your laptop):
```bash
seq 4 | xargs -P 1 -t -I{} uv run --no-sync python -m infinigen2.generate bricks_rand material_torus_uv render_cycles --seed {} --output outputs/bricks/{} --quiet | ./scripts/image-grid.py 4x render.png -o
```

Run different types of base geometry:
```bash
uv run infinigen bricks_rand material_sphere render_cycles
uv run infinigen bricks_rand material_monkey render_cycles
uv run infinigen bricks_rand material_torus_uv render_cycles
uv run infinigen bricks_rand material_plane_uv render_cycles
uv run infinigen bricks_rand material_banana render_cycles # to show small details
```

Trace execution to generate code or compute stats:
```bash
uv run infinigen bricks_rand material_torus_uv render_cycles --trace codegen
uv run infinigen bricks_rand --trace codestats
uv run infinigen bricks_rand --trace codegen codestats  # both on one graph
uv run infinigen bricks_rand --trace codestats --trace_level RANDOM_CONTROL  # finer granularity
```

See the [Contributing Guide](Contributing) for the full development workflow, including how to transpile Blender node graphs into `procfunc` generators.
