
# Contributing

### Installation

Install `uv` if you don't already have it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then install the repo for development:
```bash
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
uv sync --extra dev
```

### Formatting

Run this once to enable the formatting hooks:
```bash
uv run pre-commit install
```

- auto-code-formatting is mandatory and very important
- `pre-commit` will format your files whenever you commit. if it detects any changes,  it cancels the commit, and you have to `git add` and commit them again. This is quite annoying but is intended only as a backstop. 
- to avoid `pre-commit` firing too often, ive added a `.vscode/settings.json` which should auto-format your files upon save. 

### Procedural Generator Conventions

##### All procedural generators

When creating any procedural generator there will be two public functions, e.g:
- `bricks(vector, param1, param2 ...)` (called the `generate` function)
- `bricks_rand(rng, vector, ...overrides)` (called the `rand` function)

You can have other functions, but they wont be exposed to the documentation page, and should be "private" and ideally marked with an underscore like `def _my_helper_function()`. 
All functions which make assets (Object, Material, Texture, Collection/Scene) must be public and follow the generate/rand conventions.

Both functions can only use very simple function arguments types:
- [float](https://docs.python.org/3/library/functions.html#float) [int](https://docs.python.org/3/library/functions.html#int) [str](https://docs.python.org/3/library/functions.html#func-str)
- [MeshObject](https://procfunc.readthedocs.io/en/latest/types.html#procfunc.types.MeshObject), [Material](https://procfunc.readthedocs.io/en/latest/types.html#procfunc.types.Material), [Texture](https://procfunc.readthedocs.io/en/latest/types.html#procfunc.types.Texture), [Collection](https://procfunc.readthedocs.io/en/latest/types.html#procfunc.types.Collection)
- [Color](https://docs.blender.org/api/current/mathutils.html#mathutils.Color) [Vector](https://docs.blender.org/api/current/mathutils.html#mathutils.Vector) [Euler](https://docs.blender.org/api/current/mathutils.html#mathutils.Euler) [Quaternion](https://docs.blender.org/api/current/mathutils.html#mathutils.Quaternion) [Matrix](https://docs.blender.org/api/current/mathutils.html#mathutils.Matrix)
- [np.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html) of 1 2 3 or 4 elements (often also representing a vector/color)
- (No complicated dictionaries or lists. This includes list\[Object\] in most cases - use a Collection instead)

The `_rand` has restricted control flow rules:
- Absolutely no `for` / `while` loops
- No `if` statements, EXCEPT for the sole case of doing `if argument is None` to fill in default arguemnts
- Almost all control flow should just use a `pf.control.choice()` with constant weights.
If you strictly need `if`/`for`/`while`, either put them inside the generate function (not `_rand`), or we will have to add special functions to `pf.control` which are planned but WIP

Randomness can only come from `rng` input arguments, not `np.random.uniform()` or `import random`.  Only the `_rand` function has an `rng`, not the regular function, which means the generate 

##### Material functions

Must have a `vector: ProcNode[Vector]`input. This determines the coordinate space for the material, which is plugged into all `noise` / `bricks` etc, or else they will crash.

Input variable conventions:
- `vector` determines the coordinate space (not `coord` or similar).
    - units are in *meters* - you must have correct scale for your material.
    - the vector is 3D, but you can ignore `z` and use only `x` and `y` if you wish.
- `roughness` for the main roughness parameter, if one exists. avoid using `surface_roughness` etc. 

Output variable conventions:
- `surface` output for any BSDF / non-volume shaders
- `volume` for volume shader outputs
- `displacement` for displacement vector outputs. Avoid using `height` as a float

See [concrete.py](../src/infinigen2/shaders/materials/concrete.py) for an example.

### Tools

##### Rendering

See the [landing page](Infinigen2) for the everyday render commands (quick render, GT render, many renders locally).

View frames as a video slideshow instead:
```bash
sudo apt-get install ffmpeg vlc libx264-dev # if needed
ffmpeg -y -r 1 -pattern_type glob -i "outputs/bricks/*/image*.png" -c:v libx264 -crf 18 -pix_fmt yuv420p out.mp4
vlc out.mp4
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

See the [procfunc transpiling example](https://github.com/princeton-vl/procfunc#transpile-a-blender-file-to-procfunc-code) for a worked end-to-end example.

Transpile a material from a file
```bash
uv run python -m procfunc.transpiler.main input.blend --materials materialname1 materialname2 --output newfile.py
```

Transpile and infer parameter distributions for all subcomponents:
```bash
uv run python -m procfunc.transpiler.main input.blend --materials materialname1 materialname2 --output newfile.py --transforms colors_to_hsv_definition infer_nodegroup_distributions
```

example result: file will contain some extra `_rand` functions e.g. `brick_concrete_rand` shown below:

```python
def brick_concrete_rand(
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
