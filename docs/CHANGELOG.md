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

