# "Hello World": Generate your first Infinigen scene 

<p align="center">
  <img src="images/hello_world/Image0048_00_00.png" width="350" />
  <img src="images/hello_world/Depth0048_00_00.png" width="350" />
  <img src="images/hello_world/SurfaceNormal_0001_00_00.png" width="350" />
  <img src="images/hello_world/InstanceSegmentation_0001_00_00.png" width="350" />
</p>

This guide will show you how to generate an image and it's corresponding ground-truth, similar to those shown above.

:warning: **Known issue** : We are actively fixing an issue which causes Infinigen not to be reproducible on many platforms. The same command may produce multiple rearranged scenes with different runtimes and memory requirements.

## Generate a scene step by step
Infinigen generates scenes by running multiple tasks (usually executed automatically, like in [Generate image(s) in one command](#generate-images-in-one-command)). Here we will run them one by one to demonstrate. These commands take approximately 10 minutes and 16GB of memory to execute on an M1 Mac or Linux Desktop.

:exclamation: If you encounter any missing .so files, missing dependencies (such as `gin`), or similar crashes, please check again that all steps of installation ran successfully. If you cannot resolve any issues with installation, please see our README and 'Bug Report' Git Issue template for advice on posting Git Issues to get help quickly - you must include the full installation logs in your issue so that we can help debug.

```
mkdir outputs

# Generate a scene layout
python -m infinigen_examples.generate_nature --seed 0 --task coarse -g desert.gin simple.gin --output_folder outputs/hello_world/coarse

# Populate unique assets
python -m infinigen_examples.generate_nature --seed 0 --task populate fine_terrain -g desert.gin simple.gin --input_folder outputs/hello_world/coarse --output_folder outputs/hello_world/fine

# Render RGB images
python -m infinigen_examples.generate_nature --seed 0 --task render -g desert.gin simple.gin --input_folder outputs/hello_world/fine --output_folder outputs/hello_world/frames

# Render again for accurate ground-truth
python -m infinigen_examples.generate_nature --seed 0 --task render -g desert.gin simple.gin --input_folder outputs/hello_world/fine --output_folder outputs/hello_world/frames -p render.render_image_func=@flat/render_image 
```

:warning: If you installed Infinigen using the "Infinigen as a Blender Python script" option, you will need to replace `python -m infinigen_examples.generate_nature` with `python -m infinigen.launch_blender -m infinigen_examples.generate_nature -- ` in the commands above, e.g. the first command would become `python -m infinigen.launch_blender -m infinigen_examples.generate_nature -- --seed 0 --task coarse -g desert.gin simple.gin --output_folder outputs/hello_world/coarse`. 

Output logs should indicate what the code is working on. Use `--debug` for even more detail. After each command completes you can inspect it's `--output_folder` for results, including running `python -m infinigen.launch_blender outputs/hello_world/coarse/scene.blend` or similar to view blender files. We hide many meshes by default for viewport stability; to view them, click "Render" or use the UI to unhide them.

## Generate image(s) in one command

We provide `infinigen/datagen/manage_jobs.py`, a utility which runs similar steps automatically.

```
python -m infinigen.datagen.manage_jobs --output_folder outputs/hello_world --num_scenes 1 --specific_seed 0 \
--configs desert.gin simple.gin --pipeline_configs local_16GB.gin monocular.gin blender_gt.gin --pipeline_overrides LocalScheduleHandler.use_gpu=False
```

This command will repeatedly print summaries of the status of each stage of the pipeline. Please look in `outputs/hello_world/1/logs` for full output logs of the underlying tasks.

We encourage you to visit [Configuring Infinigen](ConfiguringInfinigen.md) for a breakdown of this command and more advanced usage instructions / example commands.

See [Extended ground-truth](GroundTruthAnnotations.md) for a guide on using our custom ground-truth extraction system.