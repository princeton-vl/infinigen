v1.0.0 - Beta code release <br>

v1.0.1 - BSD-3 license, expanded ground-truth docs, show line-credits, miscellaneous fixes <br>

v1.0.2 - New documentation, plant improvements, disk and reproducibility improvements <br>

v1.0.3
- Fluid code release
- implementing assets documentation
- render tools improvements
- integration testing script

v1.0.4 
- Tools and docs to download preliminary pre-generated data release, 
- Reformat "frames" folder to be more intuitive / easier to dataload
- ground truth updates
- render throughput improvements

v1.1.0
- Update to blender 3.6, install blender either via pip or standalone
- Restructure project into an `infinigen` python package and `infinigen_examples` directory
- Add unit tests

v1.2.0
- Integrate OcMesher terrain option - see https://github.com/princeton-vl/OcMesher

v1.2.4
- Fix TreeFactory crash for season='winter'

v1.2.5
- Add Terrain.populated_bounds parameters
- Fix reinitalizing terrain

v1.2.6
- Fix bug where manage_jobs.py would pick a random scene_type config even if one was already loaded
- Fix bug where manage_jobs.py would ignore CUDA_VISIBLE_DEVICES that didnt start at 0
- Add NotImplementedError for dynamic hair.

v1.3.1
- Fix configuration bug causing massive render slowdown 
- Create noisier video trajectories optimized for training

v1.3.2
- Bugfix USD/OBJ exporter, add export options to generate_individual_assets

v1.3.3
- Bugfix camera code to allow multiple cameras, prevent all-water frames
- Tweak rendering settings
- Tweak test lists & add timeouts, pass all tests

v1.3.4
- Fixed bug where individual export would fail on objects hidden from viewport
- Fixed Terrain.populated_bounds bad merge

v1.4.0 - Infinigen Indoors
- Add library of procedural generators for indoor objects & materials
- Add indoor scene generation system, including constraint language and solver
- Add HelloRoom.md & ExportingToSimulators.md

v1.4.1
- bugfix gin file names in example commands
- disable cieling skirtingboard temporarily

v1.5.0
- ruff & auto-lint-fix the entire codebase
- move mesh assets into infinigen/assets/objects
- minimize pip dependences: remove unused packages & move terrain/gt-vis packages into optional \[terrain,vis\] extras.
- add parameters for object clutter, reduce excessively cluttered / slow indoors scenes
- minorly improve infinigen-indoors performance via logging & asset hiding

v1.5.1
- Fix "base.gin" crash in generate_individual_assets
- Fix individual_export in export.py
- Fix Dockerfile
- Remove dependabot
- Add scatter unit tests and fix scatter imports
- Fix black renders due to non-hidden particle emitter

v1.6.0
- Add geometric tile pattern materials
- Tune window parameters and materials