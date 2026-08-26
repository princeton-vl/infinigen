# CHANGELOG

v2.0.0a2

**Breaking changes and renames**

- Rename the `collision_set` `existing=` parameter to `cache=`
- Remove `stereo_cameras_in_bbox_rand`, `stereo_random_walk_camera` and `material_orbit_camera_rand`; compose stereo from `attach_stereo_right`, `sample_baseline` and `stereo_accept_pred`
- Split `scenes.placement_utils` into a `scenes.placement` package of `retry`, `snap`, `distribute`, `culling` and `collision`
- Rename `objects.cabinet` and `objects.slot_cabinet` to `objects.storage`, `scenes.asset_demo` to `scenes.demo_material` and `scenes.demo_object`, `primitives_rand` to `primitive_with_effect_rand`, and `plastic_opaque_rand` to `plastic_rand`

**Dependencies**

- Require `procfunc>=0.35,<0.36`
- Raise dependency floors, drop the `imageio`, `opencv-python` and `trimesh` upper caps, require `cvdpack>=0.6.0`, and split a `test` extra out of `dev`

**New assets and scenes**

- Add v2 chair generators with office and wooden-dining families, assembled from independently-sampled seat, back and base parts
- Add lever, bar-pull, knob and curved-pull handle generators, wired into doors and storage
- Add geonode doors with 1-5 panel bodies in same-material, two-tone or opaque-plus-glass finishes
- Rework storage into traceable shelves and cabinet-with-door generators, replacing the old cabinet and slot-cabinet assets
- Add a `room_rand` scene with weighted dining and sofa furniture setups plus a placement culling pass
- Register `curve_demo`, `trim_demo`, `dining_setup_rand` and `centered_sofa_setup_rand` scenes
- Add leather and patterned wall paint base materials

**Geometry, subdivision and UVs**

- Rebuild table legs, lamps, ceiling lights and handles at low base polycount with edge creasing and subsurf
- Unify table pedestal bases into one swept generator; `base_single_stand_rand` is now `pedestal_base_rand`
- Add a symmetric swept trim profile, rebuilding painting frames as mitered swept quads and adding window casing
- Add metric and swap-UV options to swept-curve UVs, and radial or metric UVs to handles
- Fix UV seams on swept curve-to-mesh profiles so textures no longer snap to zero at the seam
- Gap skirting boards under doorways and square off the profile back against the wall
- Fit window size and margin to the wall they are placed on, and crease glass panes so frame subsurf no longer shrinks the glass inward
- Diversify lamp shade shapes and fix patterned lampshade materials
- Fix lever handle cap shading under subdivision

**Materials**

- Unify carpet, granite, plastic and smooth-stone into continuous distributions, replacing the discrete style variants
- Vary wall paint colors more widely, reaching deeper and more saturated tones
- Concentrate mirror splat and spotting along a UV gradient rather than uniform speckle
- Drop displacement from `plastic_tough_packaging` and rein in `plastic_rand` displacement magnitude
- Fix invisible scratches in the deep-dirty and shallow wood presets
- Fix a rare shader stack-budget (SVM) overflow that produced black wall materials

**Rendering and ground truth**

- Raise the default render sample count from 256 to 1024
- Make render exporters respect the objects and lights lists, taking them as separate typed inputs
- Add an `object-data` export pass writing per-object index, pose, scale and bounding box
- Add stereo rigid-body point-track export, plus point-cloud and 3D-box ground-truth visualizers behind `scripts/visualize_gt.py`, replacing the removed `visualize_renders_as_videos.py`
- Write the index, depth and surface-normal EXR passes at 32-bit
- Add `--sampling_noise_threshold` and validate `--passes` values against the available export types
- Keep subdivision on instanced ceiling lamps and scattered clutter through mesh realization
- Preserve generator names on warped primitives, vases and small objects so ground-truth index tables name them correctly

**Render checks and diagnostics**

- Reorganize `render_error_check` into a package with a shared severity context and `generate.py --error_severity CHECK=MODE`
- Add render-validity checks for geometry, transforms, visibility, attributes, black frames, non-converged sampling, Cycles shader errors, displacement coordinates, object-index consistency and degenerate UVs
- Add material strict-mode checks flagging normal-map inputs, textures sampled without an explicit Vector, and floating interface nodes
- Standardize RNG draws and generator determinism so codegen reproductions match
- Categorize wandb render-watch crashes and alert when the crash rate over recent jobs exceeds a threshold

**Example projects**

- Add generic random-walk animation and camera motion primitives
- Add a `flying_indoor` example project with grouped stereo video and per-camera sharding, replacing `examples/render_floatingobj_stereo.py` and `examples/stereo_video_sbatch.sh`
- Link pregenerated flying-indoor data on HuggingFace from the example-projects docs
- Expand the floating-object pool with per-generator sampling weights and shared material overrides

**Integration renders and CI**

- Gate PR integration renders to the assets whose coverage-relevant sources changed, and shard presets and environments once instead of 12x
- Render a surface-normal pass on every integration seed, record per-asset triangle count and render time, and group viewer images into per-category sections

v2.0.0a1
- Complete rewrite on top of the procfunc procedural-generation engine
- Add 60 new procedural materials
- Add new scene arrangement system
- Add new render & ground-truth (GT) APIs

v1.16.0
- Refactored scatters into classes
- Change blender_gt surface normals convention from world coordinates to camera coordinates.
- Allow user specification of floor plans as a series of shapely primitives.

v1.15.5
- Fix mismatched USD textures due to unhandled slashes in object names

v1.15.4
- fix empty material slot crash in blendergt
- fix empty material slot crash in export glass materials #442
- fix displacement not properly disabled when requested 443
- fix mvs cameras undefined variable #437
- fix crash in overhead run_stage #439
- disable optimize_disk_usage unless requested
- attempted to fix pypi
- fix missing studio.gin #417 by @jerrylingjiemei

v1.15.2
- Fix house ocmesher camera crash
- Add house ocmesher example command, warnings and example images
- Update face_size_visualizer
- Standardize use of set_displacement_mode

v1.15.1
- Fix occlusion boundaries when polygons project to negative z in camera space

v1.15.0
- Initial code release for Infinigen-Sim! See ExportingToSimulators.md for guide on generating articulated doors, toasters, lamps, fridges, & dishwashers. Added articulation exporters to USD, MJCF, URDF.

v1.14.0
- Add option to densely subdivide room meshes using OcMesher
- Add RRT-Star camera pathfinding for video viewpoint animation

v1.13.2
- Bugfix invalid Brick kwargs and `1-2 args execution context is supported` during indoor room generation

v1.13.1
- Refactor transpiler, fix transpiling disabled input sockets

v1.13.0
- Refactor materials into classes
- Separate test lists for new-style and deprecated apply()-style materials

v1.12.3
- Fix populate_collection missing argument 'cameras'
- Fix split_in_view crash for dist_max=None

v1.12.2
- Fix excessive time/memory/crashes in nature scenes due to inactive viewpoint filter
- Fix blendergt not set to 1hr timelimit by slurm_1h.gin
- Add get_cmd.child_debug flag
- Usability improvements for integration test scripts
- Fix static asset import #391
- Fix indoor_asset_semantics.py typo #398

v1.12.1
- Fix blender_gt crash from errored object in global_flat_shading
- Replace diameter with radius in butil.spawn_capsule
- Fix ignored blender_gt sample count config
- Fix outdated bbox input for camera_pose_proposal
- Bugfix stdout passthrough mode crashing due to no logfile created
- Add normalmaps to integration test viewer, misc test fixes
- Avoid rare duplicate names in indoor solver

v1.12.0
- Publish to PyPi

v1.11.4
- Fix circular / segfaulting imports when modules imported individually
- Fix ordering-dependence in unit tests 
- Fix scenetype.gin crashes for underwater/kelpforest
- Increase integration test timelimit
- Add `analyze_crash_reasons` crash summary script
- Improve success rate of camera / creature animations via increased retry attempts

v1.11.3
- Increase max camera / object animation random walk trials
- Fix scenetype gin recognition causing crashes for underwater.gin / kelpforest.gin

v1.11.2
- Fix opengl_gt input file symlink missing

v1.11.1
- Fix failed camera search when canyon/cliff/cave loaded as nature background
- Fix scrambled GT maps in blender_gt due to incorrect OpenEXR pixel unpack ordering
- Fix save_mesh kwarg mismatch from v1.10.0
- Remove `frozendict` dependency, make `geomdl` optional if not using creatures
- Make `submitit` optional if not using SLURM
- Make blender addons optional if not using relevant assets (rocks/terrain/snowlayer)
- Make `bnurbs` CPython module optional and not installed by default

v1.11.0
- Update to Blender == 4.2.0

v1.10.1
- Fix missing validity checks for camera-rig cameras which are not the 0th index
- Fix missing seat in dining chair

v1.10.0
- Add Configuring Cameras documentation
- Add config for multiview cameras surrounding a point of interest
- Add MaterialSegmentation output pass
- Add passthrough mode to direct manage_jobs stdout directly to terminal
- Add "copyfile:destination" upload mode

v1.9.1
- Reduce excessive polycount in bottles and tableware objects
- Fix alignment of windows
- Fix wall materials not being deterministic w.r.t random seed
- Fix gin configs not correctly passed to slurm jobs in generate_individual_assets
- Fix integration test image titles 
- Fix integration test asset image alignment
- Make multistory houses disabled by default

v1.9.0
- Add CoPlanar indoor constraint, fix backwards tvs/monitors/sinks
- Fix empty scene / null objects selected during export
- Add full system visual check / integration script

v1.8.3
- Fix landlab import error message, add no_landlab.gin config

v1.8.2
- Remove nonessential opengl_gt packages
- Fix CrabFactory crash, FruitContainerFactory unparent object, wall parts
- Fix nature particles not visible in render
- Add smbpy du and df commands
- Fix fineterrain not included in export for optimize_diskusage=True
- Update mesher_backend config name & default commands

v1.8.1
- Fix bug causing hard constraints on scalar inequalities (e.g distance > x) to be ignored
- Fix bug causing livingroom sofa alignment to be incorrect
- Fix bias in camera trajectory starting direction
- Improve visual quality of home.py via constraint tweaks and new generate_indoors stages
- Fix silent output from upload stage, remove export from upload
- Reduce solving time spent on small objects

v1.8.0
- Implement tools for importing external assets into Indoors
- Use constraint language to configure room solving
- Add pillars, vertically split wall materials

v1.7.1
- Bugfix fine terrain in arctic scenes

v1.7.0
- Implement camera IMU calculation and export
- Add point tracking ground truth

v1.6.0
- Add geometric tile pattern materials
- Tune window parameters and materials
- Add floating object placement generator and example command
- Add logging to terrain asset creation & simulations 
- Add packaged font files to assets/fonts, fix too-many-open-fonts crash
- Fix fish school disappearing at last frame in video
- Fix crash from `fabrics.apply`

v1.5.1
- Fix "base.gin" crash in generate_individual_assets
- Fix individual_export in export.py
- Fix Dockerfile
- Remove dependabot
- Add scatter unit tests and fix scatter imports
- Fix black renders due to non-hidden particle emitter

v1.5.0
- ruff & auto-lint-fix the entire codebase
- move mesh assets into infinigen/assets/objects
- minimize pip dependences: remove unused packages & move terrain/gt-vis packages into optional \[terrain,vis\] extras.
- add parameters for object clutter, reduce excessively cluttered / slow indoors scenes
- minorly improve infinigen-indoors performance via logging & asset hiding

v1.4.1
- @David-Yan1 fix placeholder & ocmesher submodule version
- @lahavlipson fix bug in surface normals of wall meshes
- @araistrick bugfix example commands & other typos

v1.4.0 - Infinigen Indoors
- Add library of procedural generators for indoor objects & materials
- Add indoor scene generation system, including constraint language and solver
- Add HelloRoom.md & ExportingToSimulators.md

v1.3.4
- Fixed bug where individual export would fail on objects hidden from viewport
- Fixed Terrain.populated_bounds bad merge

v1.3.3
- Bugfix camera code to allow multiple cameras, prevent all-water frames
- Tweak rendering settings
- Tweak test lists & add timeouts, pass all tests

v1.3.2
- Bugfix USD/OBJ exporter, add export options to generate_individual_assets

v1.3.1
- Fix configuration bug causing massive render slowdown 
- Create noisier video trajectories optimized for training

v1.2.6
- Fix bug where manage_jobs.py would pick a random scene_type config even if one was already loaded
- Fix bug where manage_jobs.py would ignore CUDA_VISIBLE_DEVICES that didnt start at 0
- Add NotImplementedError for dynamic hair.

v1.2.5
- Add Terrain.populated_bounds parameters
- Fix reinitalizing terrain

v1.2.4
- Fix TreeFactory crash for season='winter'

v1.2.0
- Integrate OcMesher terrain option - see https://github.com/princeton-vl/OcMesher

v1.1.0
- Update to blender 3.6, install blender either via pip or standalone
- Restructure project into an `infinigen` python package and `infinigen_examples` directory
- Add unit tests

v1.0.4 
- Tools and docs to download preliminary pre-generated data release, 
- Reformat "frames" folder to be more intuitive / easier to dataload
- ground truth updates
- render throughput improvements

v1.0.3
- Fluid code release
- implementing assets documentation
- render tools improvements
- integration testing script

v1.0.2 - New documentation, plant improvements, disk and reproducibility improvements <br>

v1.0.1 - BSD-3 license, expanded ground-truth docs, show line-credits, miscellaneous fixes <br>

v1.0.0 - Beta code release <br>
