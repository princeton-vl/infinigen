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

## flying_indoor

A stereo video of a furnished room with animated floating objects and lights, with ground truth for the left camera. Renders a fixed 24-frame `1280x720` clip. This example is grouped under `examples/flying_indoor/`:

- `render.py` — the render script.
- `cvdpack.json` — the [cvdpack](https://github.com/princeton-vl/cvdpack) config that packs a rendered clip into per-camera videos.
- `sbatch.sh` — a SLURM array script that renders, packs, and offloads a dataset.

```bash
uv run python examples/flying_indoor/render.py --seed 0 --camera_idx 0 --output outputs/flying_indoor
```

Passes written per clip:

- **left** — rgb + `camera.npz`, plus `material-index`, `diffuse-color`, `environment` passes.
- **right** — rgb + `camera.npz`.
- **left gt** — `depth`/`surface-normal`/`object`/`optical-flow` `.npy`.
- **right gt** — `depth` `.npy`.
- `metadata.json` (seed, hardware, per-pass runtimes, exports).

We rendered a [downloadable flying-indoor dataset](DatasetDownloads) with this example at Infinigen 2.0.0a2, if you would rather use scenes than render them.

`--camera_idx {0,1}` (required) renders one camera of the stereo pair, so the two cameras of a seed are sharded across separate tasks; both write into the same `--output` so their packed videos land in one scene.

`--trajectory_seed` keys the camera path alone, while the room, objects, lights and their motion stay keyed to `--seed`. Hold `--seed` fixed and vary it to render several trajectories through one identical world; the stereo baseline stays scene-keyed, since the rig is a property of the scene rather than of the path. It defaults to `--seed`, so a single-trajectory render needs nothing extra.

To render a dataset on SLURM, `sbatch.sh` runs one array task per `(scene, trajectory, camera)`, with `NUM_TRAJECTORIES` paths through each scene. Each task renders raw frames to fast local `SCRATCH_DIR`, packs them with cvdpack, and offloads only the compressed videos to `FINAL_DIR`, then deletes the raw frames:

```bash
sbatch examples/flying_indoor/sbatch.sh
```

💡 Edit the hardcoded `SCRATCH_DIR` (node-local scratch) and `FINAL_DIR` (permanent storage) at the top of `sbatch.sh` to suit your cluster, size `--array` as `N_SCENES * NUM_TRAJECTORIES * 2`, and adjust `--partition`/`--account`/other SLURM configs.
