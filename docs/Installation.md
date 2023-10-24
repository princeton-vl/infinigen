
# Installation

## Installation Options & Supported Platforms

You can install Infinigen either as a Python Module or a Blender Python script:
- Python Module (default option)
  - Cannot open a Blender UI - headless execution only
  - Installs the `infinigen` package into the user's own python environment
  - Installs `bpy` as a [pip dependency](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html)
- Blender Python script 
  - Can use Infinigen interactively in the Blender UI
  - Installs the `infinigen` package into *Blender's* built-in python interpreter, not the user's python.
  - Uses a standard standalone installation of Blender.

In either case, certain features have limited support on some operating systems, as shown below:

| Feature Set        | Needed to generate...    | Linux x86_64 | Mac x86_64   | Mac ARM      | Windows x86_64 | Windows WSL2 x86_64 |
|--------------------|--------------------------|--------------|--------------|--------------|----------------|---------------------|
| Minimal Install.   | objects & materials      | yes          | yes          | yes          | experimental   | experimental        |
| Terrain (CPU)      | full scenes              | yes          | yes          | yes          | no             | experimental        |
| Terrain (CUDA)     | speedup, faster videos   | yes          | no           | no           | no             | experimental        |
| OpenGL Annotations | *additional* training GT | yes          | yes          | yes          | no             | experimental        |
| Fluid Simulation   | fires, simulated water   | yes          | experimental | experimental | no             | experimental        |

Users wishing to run our [Hello World Demo](./HelloWorld.md) or generate full scenes should install Infinigen as a Python Module and enable the Terrain (CPU) setting.
Users wishing to use Infinigen assets in the Blender UI, or develop their own assets, can install Infinigen as a Blender-Python script with the "Minimal Install" setting.

See our [Configuring Infinigen](./ConfiguringInfinigen.md), [Ground Truth Annotations ](./GroundTruthAnnotations.md), and [Fluid Simulation](./GeneratingFluidSimulations.md) docs for more information about the various optional features. Note: fields marked "experimental" are feasible but untested and undocumented. Fields marked "no" are largely _possible_ but not yet implemented.

Once you have chosen your configuration, proceed to the relevant section below for instructions.

## Installing Infinigen as a Python Module

### Dependencies

Please install anaconda or miniconda. Platform-specific instructions can be found [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

Then, install the following dependencies using the method of your choice. Examples are shown for Ubuntu, Mac ARM and Mac x86.
```bash
# on Ubuntu / Debian / WSL / etc
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev

# on an Mac ARM (M1/M2/...)
arch -arm64 brew install wget llvm open-mpi libomp glm glew

# on  Mac x86_64 (Intel)
brew install wget llvm open-mpi libomp glm glew
```

### Installation

First, download infinigen and set up your environment.

On Linux / Mac / WSL:
```bash
git clone https://github.com/princeton-vl/infinigen.git -b rc_1.1.1
cd infinigen
conda create --name infinigen python=3.10
conda activate infinigen
```

Then, install the infinigen package using one of the options below:

:warning: Mac-ARM (M1/M2) users should prefix their installation command with `arch -arm64`

```bash
# Default install (includes CPU Terrain, and CUDA Terrain if available)
pip install -e .

# Minimal install (objects/materials only, no terrain or optional features)
INFINIGEN_MINIMAL_INSTALL=True pip install -e .

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True pip install -e .

```

:exclamation: If you encounter any issues with the above, please add `-vv > logs.txt 2>&1` to the end of your command and run again, then provide the resulting logs.txt file as an attachment when making a Github Issue.

## Installing Infinigen as a Blender Python script

On Linux / Mac / WSL:
```bash
git clone https://github.com/princeton-vl/infinigen.git -b rc_1.1.1
cd infinigen
```

Then, install using one of the options below:
```bash

# Minimal installation (recommended setting for use in the Blender UI)
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Normal install (includes CPU Terrain, and CUDA Terrain if available)
bash scripts/install/interactive_blender.sh

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True scripts/install/interactive_blender.sh
```

:exclamation: If you encounter any issues with the above, please add ` > logs.txt 2>&1` to the end of your command and run again, then provide the resulting logs.txt file as an attachment when making a Github Issue.

Once complete, you can use the helper script `python -m infinigen.launch_blender` to launch a blender UI, which will find and execute the `blender` executable in your `infinigen/blender` or `infinigen/Blender.app` folder.

:warning: If you installed Infinigen as a Blender-Python scriptand encounter encounter example commands of the form `python -m <MODULEPATH> <ARGUMENTS>` in our documentation, you should instead run `python -m infinigen.launch_blender -m <MODULEPATH> -- <ARGUMENTS>` to launch them using your standalone blender installation rather than the system python..

## Using Infinigen in a Docker Container

**Docker on Linux**

```
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
make docker-build
make docker-setup
make docker-run
```
To enable CUDA compilation, use `make docker-build-cuda` instead of `make docker-build`

To run without GPU passthrough use `make docker-run-no-gpu`
To run without OpenGL ground truth use `docker-run-no-opengl` 
To run without either, use `docker-run-no-gpu-opengl` 

Note: `make docker-setup` can be skipped if not using OpenGL.

Use `exit` to exit the container and `docker exec -it infinigen bash` to re-enter the container as needed. Remember to `conda activate infinigen` before running scenes.

**Docker on Windows**

Install [WSL2](https://infinigen.org/docs/installation/intro#setup-for-windows) and [Docker Desktop](https://www.docker.com/products/docker-desktop/), with "Use the WSL 2 based engine..." enabled in settings. Keep the Docker Desktop application open while running containers. Then follow instructions as above.
