# infinigen_v2 examples

Self-contained example projects: each renders a multi-pass video dataset with `infinigen_v2`, locally or as a SLURM array. The intended workflow is to **copy these scripts into your own project and edit them**, not to clone the whole repo.

## Get the code

Grab just the `examples/` folder from the current GitHub `develop2`:

```bash
curl -L https://github.com/princeton-vl/infinigen_internal/tarball/develop2 \
  | tar -xz --wildcards '*/examples/*' --strip-components=1
cd examples
```

All commands below assume you are inside this `examples/` folder, with `infinigen_v2` and `procfunc` installed (`uv run` resolves them).

## clay_pan_video

Render a linear camera pan of a livingroom in several passes, for many seeds, then assemble a demo video that swipes between the passes.

Local single seed:

```bash
uv run python clay_pan_video/render_clay_pan_video.py --seed 0 --output outputs/clay_pan_video
```

This launches `render_clay_pan_video.py`, which you can also customize yourself. Its current design is as follows:

- Builds a livingroom (`livingroom_with_smallobj_distribution`) and a `linear_pan_camera` for `--seed`, then renders the pan in passes.
- Each pass writes `%c/<name>-%f.png` under `--output` (`%c` = camera, `%f` = frame):
  - **clay-flat** + **ao-flat** — undisplaced mesh (`DisplacementMode.NONE`), with a camera-parented fill light.
  - **clay** + **ao-disp** — displaced mesh (`DISPLACEMENT_AND_BUMP`); fine surface detail appears in the AO pass.
  - **rgb** — full materials, plus `camera.npz`.
  - **gt** — `depth`/`surface-normal`/`object`/`optical-flow` `.npy`; skipped with `--skip_gt`.
- Writes `metadata.json` (seed, hardware, per-pass runtimes, exports).
- Flags: `--frames START END` (default `0 35`), `--resolution X Y` (default `1280 720`), `--samples` (clay/AO, default 256), `--rgb_samples` (default 2048), `--fps` (default 24), `--skip_gt`. The AO pass uses a short occlusion distance so only tight crevices and fine displacement darken; it is shown directly as the "clay" image.

SLURM array, one seed per array task:

```bash
sbatch clay_pan_video/sbatch.sh
```

Each task renders one seed (`SEED_OFFSET` + array task id), writes raw frames to fast local scratch, then packs them to permanent storage (`PACK_MODE`, default `direct_copy`; also `targz`, `cvdpack`). The submitted config is a 72-frame `1280x720` pan at `--samples 256 --rgb_samples 512 --skip_gt`. Override resources / array size at the top of `sbatch.sh`.

Output layout per scene:

```
<root>/<scene>/
  Camera/
    clay-flat-0000.png  ao-flat-0000.png  clay-0000.png  ao-disp-0000.png  rgb-0000.png  ...
    depth-0000.npy  surface-normal-0000.npy  object-0000.npy  optical-flow-0000.npy
    camera.npz
  metadata.json
```

Assemble the demo video from a multi-scene render dir:

```bash
bash clay_pan_video/make_video.sh <root> clay_pan_video.mp4
```

`<root>` is a directory of scene subdirs (the layout above). Per scene the pan is cut by frame position into 0.5s `ao-flat` / 0.5s `ao-disp` / 1.0s `rgb` (seamless — each segment uses the frames at its own point in the pan), joined with a diagonal right-to-left reveal swipe and a 2px black cut line, with white-on-black seed/runtime text burned in. All scene clips are concatenated. Timing is overridable via env: `NFLAT=12 NCLAY=12 NRGB=48 SWIPE=0.4 bash clay_pan_video/make_video.sh <root>`.

## dataset_generation

Render a stereo video of a livingroom with animated floating objects and lights, with ground truth for the left camera.

Local single seed:

```bash
uv run python dataset_generation/render_stereo_video.py --seed 0 --output outputs/stereo_video
```

`render_stereo_video.py` (flags `--seed`, `--output`) builds a `livingroom_distribution`, scatters animated `floating_objects` and up to 3 `floating_lights` (biased random walks), and a `stereo_random_walk_camera`. It renders a fixed 24-frame `1280x720` clip:

- **left** — rgb + `camera.npz`, plus `material-index`, `diffuse-color`, `environment` passes.
- **right** — rgb + `camera.npz`.
- **left gt** — `depth`/`surface-normal`/`object`/`optical-flow` `.npy`.
- **right gt** — `depth` `.npy`.
- `metadata.json` (seed, hardware, per-pass runtimes, exports).

SLURM array, one seed per array task:

```bash
sbatch dataset_generation/sbatch.sh
```

Same scratch / packing scheme as above, but `PACK_MODE` defaults to `cvdpack`.

## Output directories on SLURM

Both `sbatch.sh` scripts read an optional env var **`INFINIGEN_OUTPUT_DIRS`** — a colon-separated, ordered list of candidate output directories, fastest first (local scratch before bulk/permanent storage). The job writes raw frames to the first writable candidate with enough free space, then packs results to permanent storage. If `INFINIGEN_OUTPUT_DIRS` is unset, output defaults to `outputs/`.

To use fast scratch storage you must export it **before** submitting:

```bash
export INFINIGEN_OUTPUT_DIRS=/fast/scratch/me:/bulk/storage/me
sbatch clay_pan_video/sbatch.sh
```

On clusters where non-interactive `sbatch` submits don't inherit `uv` on `PATH`, also set **`INFINIGEN_UV_ENV`** to an env file that adds it (sourced only when `uv` is missing):

```bash
export INFINIGEN_UV_ENV=/path/to/uv/env
```
