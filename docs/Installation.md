
# Installation


## Supported Platforms

## Hardware Requirements

Running our [Hello World Example](./HelloWorld.md) requires ~16-20GB of RAM. Generating fully-detailed scenes requires 32+GB of RAM, and we . 

## Installation Options & Supported Platforms

You can install Infinigen either as a Python Module or a Blender Python script:
- Python Module (default option)
  - Cannot open a Blender UI - headless execution only
  - Installs the `infinigen` package into the user's own python environment
  - Installs `bpy` as a [pip dependency](https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html)
- Blender Python script 
  - Can use Infinigen interactively in the Blender UI
  - Installs the `infinigen` package into *Blender's* built-in python interpreter, not the user's python.
  - Uses a standard installation of Blender.

Most technical documentation in this project assumes the user installed infinigen as a Python Module. If you used the Blender-Python option and encounter documentation which instructs you to run `python -m path.to.script ...` you should attempt to instead run `blender/blender -b --python path/to/script.py -- ...`.

In either case, certain features have limited support on some operating systems, as shown below:

| Feature Set        | Needed to generate...    | Linux x86_64 | Mac x86_64   | Mac ARM      | Windows x86_64 | Windows WSL2 x86_64 |
|--------------------|--------------------------|--------------|--------------|--------------|----------------|---------------------|
| Minimal Install.   | objects & materials      | yes          | yes          | yes          | experimental   | experimental        |
| Terrain (CPU)      | full scenes              | yes          | yes          | yes          | no             | experimental        |
| Terrain (CUDA)     | speedup, faster videos   | yes          | no           | no           | no             | no                  |
| OpenGL Annotations | *additional* training GT | yes          | yes          | yes          | no             | experimental        |
| Fluid Simulation   | fires, simulated water   | yes          | experimental | experimental | no             | experimental        |

Users wishing to run our [Hello World Demo](./HelloWorld.md) or generate full scenes should install Infinigen as a Python Module and enable the Terrain (CPU) setting.
Users wishing to use Infinigen assets in the Blender UI, or develop their own assets, can install Infinigen as a Blender-Python script with the "Minimal Install" setting.

See our [Configuring Infinigen](./ConfiguringInfinigen.md), [Ground Truth Annotations ](./GroundTruthAnnotations.md), and [Fluid Simulation](./GeneratingFluidSimulations.md) docs for more information about the various optional features. Note: fields marked "experimental" are feasible but untested. Fields marked "no" are largely _possible_ but not yet implemented.

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
git clone https://github.com/princeton-vl/infinigen.git
conda create --name infinigen python=3.10
conda activate infinigen
```

On Windows:
- Install [Github Desktop](https://desktop.github.com/) and download a copy of this repository
- Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) and [open the Anaconda Prompt](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)
On Windows, we recommend downloading Infinigen via  and using the , then proceeding with the commands above.

Then, install the infinigen package using one of the options below:
```bash
# Default install (includes CPU Terrain, or CUDA if available)
pip install -e .

# Minimal install (objects/materials only, no terrain or optional features)
INFINIGEN_MINIMAL_INSTALL=True pip install -e .

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True pip install -e .

```

## Installing Infinigen as a Blender Python script

On Linux / Mac / WSL:
```
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen

```

On Windows:
- Download a copy of this repository via [Github Desktop](https://desktop.github.com/)
- Install [Mingw-w64](https://www.cygwin.com/install.html) (to allow you to run the bash script in the next step)
- 
First, download a copy of this repository, either through `git clone https://github.com/princeton-vl/infinigen.git` (preferred), via [Github Desktop](), or by downloading a ZIP. 

Then, run the following in your terminal:
```
cd infinigen

```

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
