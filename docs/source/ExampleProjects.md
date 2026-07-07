# infinigen2 examples

Each example is a single standalone script that renders a multi-pass video dataset with `infinigen2`. Each example script is extremely customizeable. You can call additional object / scene generators from the docs, change what ExportTypes are rendered, or add additional rendering steps with objects or settings changed.

## clay_pan_video

A linear camera pan of a livingroom, rendered in several passes: clay, ambient occlusion, rgb, and ground truth.

```bash
wget https://raw.githubusercontent.com/princeton-vl/infinigen/develop2/examples/render_clay_pan_video.py
uv run python render_clay_pan_video.py --seed 0 --output outputs/clay_pan_video
```

Each pass writes `%c/<name>-%f.png` under `--output` (`%c` = camera, `%f` = frame):

- **clay-flat** + **ao-flat** — undisplaced mesh (`DisplacementMode.NONE`), with a camera-parented fill light.
- **clay** + **ao-disp** — displaced mesh (`DISPLACEMENT_AND_BUMP`); fine surface detail shows up in the AO pass.
- **rgb** — full materials, plus `camera.npz`.
- **gt** — `depth`/`surface-normal`/`object`/`optical-flow` `.npy`; skipped with `--skip_gt`.

## stereo_video

A stereo video of a livingroom with animated floating objects and lights, with ground truth for the left camera. Renders a fixed 24-frame `1280x720` clip.

```bash
wget https://raw.githubusercontent.com/princeton-vl/infinigen/develop2/examples/render_floatingobj_stereo.py
uv run python render_floatingobj_stereo.py --seed 0 --output outputs/stereo_video
```

Passes written per clip:

- **left** — rgb + `camera.npz`, plus `material-index`, `diffuse-color`, `environment` passes.
- **right** — rgb + `camera.npz`.
- **left gt** — `depth`/`surface-normal`/`object`/`optical-flow` `.npy`.
- **right gt** — `depth` `.npy`.
- `metadata.json` (seed, hardware, per-pass runtimes, exports).

To render many seeds on SLURM, one per array task:

```bash
wget https://raw.githubusercontent.com/princeton-vl/infinigen/develop2/examples/stereo_video_sbatch.sh
sbatch stereo_video_sbatch.sh
```

💡 You should customize `OUTPUT_PATH=` to suit your cluster, as well as changing `--partition`, `--account`, or other SLURM configs.
