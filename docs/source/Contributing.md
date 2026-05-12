
### Installation

install `uv` if you dont already have it `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then install the repo:
```bash
git clone git@github.com:princeton-vl/infinigen_internal.git
cd infinigen_internal/
git checkout develop_materials
git submodule update --init --recursive
uv venv
uv pip install -e ".[dev]"
uv run pre-commit install
```

To update `procfunc` to a newer commit:
```bash
uv add "procfunc @ git+https://github.com/princeton-vl/procfunc@NEW_COMMIT_HASH"
```

**Note on formatting**: 
- auto-code-formatting is mandatory and very important
- `pre-commit` will format your files whenever you commit. if it detects any changes,  it cancels the commit, and you have to `git add` and commit them again. This is quite annoying but is intended only as a backstop. 
- to avoid `pre-commit` firing too often, ive added a `.vscode/settings.json` which should auto-format your files upon save. 

### Git

Create yourself a branch for each small task you tackle:
```bash
git checkout develop_materials
git checkout -b myinitials/taskname
```
E.g. Alex Raistrick would use `ar/bark_material` to work on a bark material.

Its great to `git commit` whenever you have a new version and can describe your changes.

Create a "Draft PR" as soon as you have any commits!
This makes it easy to share links to your code with others (e.g. to ask questions)

Use `git rebase develop_materials` if you ever want to update your branch. Do not use `git merge` or github wont let you push. You should learn what rebasing means and how to use it ([tutorial](https://www.atlassian.com/git/tutorials/rewriting-history/git-rebase)). 

##### Common issues

Never `git add` anything you didnt intentionally modify

Especially: dont `git add` or commit any badly synced submodules ! These show as `(new commits)` or `(modified content)`. Oftentimes you instead want to reset them:
```bash
git submodule update --init --recursive
```

##### Commit Messages

If you dont follow a certain style of commit messages, your branch will be "squash merged", meaning your changes show as a single commit.

Required style:
- Imperative mood e.g.:
    - "Fixes excessive saturation in bricks material"
    - "Add bricks_distribution generator"
- Single feature per commit - split your commits
- Informative but concise
    - "Fix" "temp" "ruff" "format" are strictly banned

### Procedural Generator Conventions

##### All procedural generators

When creating any procedural generator there will be two public functions, e.g:
- `bricks(vector, param1, param2 ...)` (called the `generate` function)
- `bricks_distribution(rng, vector, ...overrides)` (called the `distribution` function)

You can have other functions, but they wont be exposed to the documentation page, and should be "private" and ideally marked with an underscore like `def _my_helper_function()`. 
All functions which make assets (Object, Material, Texture, Collection/Scene) must be public and follow the generate/distribution conventions.

Both functions can only use very simple function arguments types:
- float int str
- MeshObject, Material, Texture, Collection
- Color Vector Euler Quaternion Matrix
- np.array of 1 2 3 or 4 elements (often also representing a vector/color)
- (No complicated dictionaries or lists. This includes list\[Object\] in most cases - use a Collection instead)

The `_distribution` has restricted control flow rules:
- Absolutely no `for` / `while` loops
- No `if` statements, EXCEPT for the sole case of doing `if argument is None` to fill in default arguemnts
- Almost all control flow should just use a `pf.control.choice()` with constant weights.
If you strictly need `if`/`for`/`while`, either put them inside the generate function (not `_distribution`), or we will have to add special functions to `pf.control` which are planned but WIP

Randomness can only come from `rng` input arguments, not `np.random.uniform()` or `import random`.  Only the `_distribution` function has an `rng`, not the regular function, which means the generate 

##### Material functions

Must have a `vector: ProcNode[Vector]`input. This determines the coordinate space for the material, which is plugged into all `noise` / `bricks` etc, or else they will crash.

Input variable conventions:
- `vector` determines the coordinate space (not `coord` or similar).
    - units are in *meters* - you must have correct scale for your material.
    - the vector is 3D, but you can ignore `z` and use only `x` and `y` if you wish.
- `color` for the main color of the material, if one exists. avoid using `base_color` etc. 
- `roughness` for the main roughness parameter, if one exists. avoid using `surface_roughness` etc. 

Output variable conventions:
- `surface` output for any BSDF / non-volume shaders
- `volume` for volume shader outputs
- `displacement` for displacement vector outputs. Avoid using `height` as a float

See [concrete.py](../src/infinigen_v2/generators/shaders/materials/concrete.py) for an example.

### Tools

##### Rendering

Run a quick render during development
```bash
uv run infinigen_v2 bricks_distribution material_torus_uv render_cycles
uv run infinigen_v2 sofa_distribution object_demo render_cycles
```

Run a render with GT
```bash
uv run infinigen_v2 bricks_distribution material_torus_uv render_cycles render_cycles_ground_truth visualize_gt --show --debug --passes rgb surface-normal depth
```

Run many renders locally (increase -P to run more in paralell! warning - may crash your laptop)
```bash
seq 4 | xargs -P 1 -t -I{} uv run --no-sync python -m infinigen_v2.generate bricks_distribution material_torus_uv render_cycles --seed {} --output outputs/bricks/{} --quiet | ./scripts/image-grid.py 4x render.png -o
```

View frames as a video slideshow instead:
```bash
sudo apt-get install ffmpeg vlc libx264-dev # if needed
ffmpeg -y -r 1 -pattern_type glob -i "outputs/bricks/*/image*.png" -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
vlc out.mp4
```

Trace execution to generate code or compute stats:
```bash
uv run infinigen_v2 --trace codegen bricks_distribution material_torus_uv render_cycles
uv run infinigen_v2 --trace codestats bricks_distribution
uv run infinigen_v2 --trace codegen codestats bricks_distribution  # both on one graph
uv run infinigen_v2 --trace codestats --trace_level RANDOM_CONTROL bricks_distribution  # finer granularity
```

Run different types of base geometry:
```bash
uv run infinigen_v2 bricks_distribution material_sphere render_cycles
uv run infinigen_v2 bricks_distribution material_monkey render_cycles
uv run infinigen_v2 bricks_distribution material_torus_uv render_cycles
uv run infinigen_v2 bricks_distribution material_plane_uv render_cycles
uv run infinigen_v2 bricks_distribution material_banana render_cycles # to show small details
```

Generate samples from all materials: `bash scripts/integration_v2/launch.sh outputs/new_renders <number parallel jobs>`

View Material samples: `uv run scripts/integration_v2/compare.py --scan-dir outputs`

##### Testing

Run linting
```bash
uv run ruff format
uv run ruff check --fix
```

Run unit tests
```bash
uv run pytest
```

##### Transpiling assets to python

Transpile a material from a file
```bash
uv run python -m procfunc.transpiler.main input.blend --materials materialname1 materialname2 --output newfile.py
```

Transpile and infer parameter distributions for all subcomponents:
```bash
uv run python -m procfunc.transpiler.main input.blend --materials materialname1 materialname2 --output newfile.py --transforms colors_to_hsv_definition infer_nodegroup_distributions
```

example result: file will contain some extra `_distribution` functions e.g. `brick_concrete_distribution` shown below:

```python
def brick_concrete_distribution(
    rng: pf.RNG,
    coord: t.SocketOrVal[pf.Vector],
):
    # Code generated by procfunc v0.3.0

    color_hsv = pf.random.uniform(
        rng,
        np.array([0.0, 0.09677423, 0.093]),
        np.array([0.06349206, 0.57142854, 0.294]),
    )
    color = pf.color.hsv_to_rgba(color_hsv)
    roughness = pf.random.uniform(rng, 0.7, 0.9409)
    specular_ior_level = pf.random.uniform(rng, 0.0753, 0.1941)
    color_noise_1_spread = pf.random.uniform(rng, 0.8, 0.8)
    color_noise_2_sharpness = pf.random.uniform(rng, 0.25, 0.25)

    brick_concrete_result = brick_concrete(
        rng=rng,
        coord=coord,
        color=color,
        roughness=roughness,
        specular_ior_level=specular_ior_level,
        color_noise_1_spread=color_noise_1_spread,
        color_noise_2_sharpness=color_noise_2_sharpness,
    )
    return brick_concrete_result
```

Transpile starting from a v1 python file, rather than starting from a fresh blender file:
```bash
uv run python scripts/transpile_v1_to_v2.py infinigen.assets.objects.seating.sofa.nodegroup_sofa_geometry infinigen_v2.generators.objects.sofa.sofa_distribution
```
- You may need to edit the existinv 1 file to use `apply=False` and `singleton=True` first.
- Be aware that this will overwrite the existing infinigen_v2.generators.objects.sofa file