
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

(install)=

## Installing Infinigen as a Python Package

### Dependencies

Please install anaconda or miniconda. Platform-specific instructions can be found [here](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

Then, install the following dependencies using the method of your choice. Examples are shown for Ubuntu, Mac ARM and Mac x86.
```bash
# on Ubuntu / Debian / WSL / etc
sudo apt-get install wget cmake g++ libgles2-mesa-dev libglew-dev libglfw3-dev libglm-dev zlib1g-dev

# on an Mac ARM (M1/M2/...)
arch -arm64 brew install wget cmake llvm open-mpi libomp glm glew zlib

# on  Mac x86_64 (Intel)
brew install wget cmake llvm open-mpi libomp glm glew zlib

# on Conda. Useful when you don't have sudo permissions
conda install conda-forge::gxx=11.4.0 mesalib glew glm menpo::glfw3
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Developer Install

Install uv: [instructions here](https://docs.astral.sh/uv/getting-started/installation/)

Then install a local copy of infinigen:
```bash
git clone https://github.com/princeton-vl/infinigen.git

cd infinigen
git submodule update --init

uv venv
uv pip install -e ".[dev,terrain,vis]"
uv run pre-commit install
```

### Minimal Install

No terrain or opengl GT, ok for Infinigen-Indoors or single-object generation

This avoids building any C++ or installing dependencies which are sometimes problematic (e.g. landlab)

```bash
INFINIGEN_MINIMAL_INSTALL=True uv pip install -e .
```

### Legacy (1.0) generation

The legacy `infinigen_examples.generate_nature` / `generate_indoors` pipelines need
extra dependencies and native builds (terrain, the `marching_cubes` extension, git
submodules) that the default 2.0 install omits. 1.0 installs via `pip` (not `uv`); make
sure the system dependencies from the [Dependencies](#dependencies) section above are
installed, then:

```bash
git clone --recurse-submodules https://github.com/princeton-vl/infinigen.git
cd infinigen
conda create --name infinigen python=3.11
conda activate infinigen
INFINIGEN_MINIMAL_INSTALL=False INFINIGEN_INSTALL_TERRAIN=True pip install -e ".[infinigen1]"
```

To additionally build the OpenGL ground-truth annotator, also set
`INFINIGEN_INSTALL_CUSTOMGT=True` in the `pip install` command above.

:exclamation: If you encounter any issues with the above, please add `-vv > logs.txt 2>&1` to the end of your command and run again, then provide the resulting logs.txt file as an attachment when making a Github Issue.

## Installing Infinigen into Blender's internal Python interpreter

First, complete normal installation as shown above

On Linux / Mac / WSL:
```bash
git clone https://github.com/princeton-vl/infinigen.git
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh
```

:exclamation: If you encounter any issues with the above, please add ` > logs.txt 2>&1` to the end of your command and run again, then provide the resulting logs.txt file as an attachment when making a Github Issue.

Once complete, you can use the helper script `python -m infinigen.launch_blender` to launch a blender UI, which will find and execute the `blender` executable in your `infinigen/blender` or `infinigen/Blender.app` folder.

:warning: If you installed Infinigen as a Blender-Python script and encounter encounter example commands of the form `python -m <MODULEPATH> <ARGUMENTS>` in our documentation, you should instead run `python -m infinigen.launch_blender -m <MODULEPATH> -- <ARGUMENTS>` to launch them using your standalone blender installation rather than the system python..

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
