## Exporting an Indoors Scene for Robotics Simulation in NVIDIA IsaacSim

This documentation details how to run a robotics simulation in an exported Infinigen scene using NVIDIA IsaacSim. For more information on scene generation or export, refer to [Hello Room](HelloRoom.md) or [Exporting to External File Formats](./ExportingToExternalFileFormats.md).

:warning: Exported scenes can be imported to any simulator that supports .usd files. However, we have only extensively tested simulator import on **Indoor** scenes using **IsaacSim**, so quality is not guaranteed for Infinigen Nature scenes and/or other simulators.

First, create and export a scene with the commands below:

```bash
python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse -g overhead_singleroom.gin -p compose_indoors.terrain_enabled=False  restrict_solving.solve_max_rooms=1 
```

```bash
python -m infinigen.tools.export --input_folder outputs/indoors/coarse --output_folder outputs/my_export -f usdc -r 1024 --omniverse
```

Download IsaacSim from [NVIDIA Omniverse](https://developer.nvidia.com/isaac/sim) and set up an IsaacSim conda environment by running the following commands in your IsaacSim Directory (typically ` ~/.local/share/ov/pkg/isaac_sim-2023.1.1`) 

```bash
conda env create -f environment.yml
conda activate isaac-sim
source setup_conda_env.sh
```

Import scene and run a simulation

```bash
python {PATH_TO/isaac_sim.py} --scene-path outputs/my_export/export_scene.blend/export_scene.usdc --json-path outputs/my_export/export_scene.blend/solve_state.json 
```

:warning: Physical properties are applied based on object relations specified in `solve_state.json`. Scenes can be imported without a `solve_state.json`, but all objects will be static colliders. 


