
# Installation

Installation is tested and working on the following platforms:

- Ubuntu 22.04.2 LTS 
    - GPUs options tested: CPU only, GTX-1080, RTX-2080, RTX-3090, RTX-4090 (Driver 525, CUDA 12.0)
    - RAM: 16GB
- MacOS Monterey & Ventura, Apple M1 Pro, 16GB RAM

We are working on support for rendering with AMD GPUs. Windows users should use [WSL2](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview). More instructions coming soon.

<details closed>
<summary><b>:warning: Errors with git pull / merge conflicts when migrating from v1.0.0 to v1.0.1</b></summary>
To properly display open-source line by line git credits for our team, we have switched to a new version of the repo which does not share commit history with the the version available from 6/17/2023 to 6/29/2023 date. We hope this will help open source contributors identify the current "code owner" or person best equipped to support you with issues you encounter with any particular lines of the codebase.

You will not be able to pull or merge infinigen v1.0.1 into a v1.0.0 repo without significant git expertise. If you have no ongoing changes, we recommend you clone a new copy of the repo. We apologize for any inconvenience, please make an issue if you have problems updating or need help migrating ongoing changes. We understand this change is disruptive, but it is one-time-only and will not occur in future versions. Now it is complete, we intend to iterate rapidly in the coming weeks, please see our [roadmap](https://infinigen.org/roadmap) and [twitter](https://twitter.com/PrincetonVL) for updates.
</details closed>

**Run these commands to get started**
```
git clone --recursive https://github.com/princeton-vl/infinigen.git
cd infinigen
conda create --name infinigen python=3.10
conda activate infinigen
bash install.sh
```
`install.sh` may take significant time to download Blender and compile all source files.

Ignore non-fatal warnings. See [Getting Help](#getting-help) for guidelines on posting github issues

Run the following or add it to your `~/.bashrc` (Linux/WSL) or `~/.bash_profile` (Mac)
```
# on Linux/WSL
export BLENDER="/PATH/TO/infinigen/blender/blender"
# on MAC
export BLENDER="/PATH/TO/infinigen/Blender.app/Contents/MacOS/Blender"
```

<details closed>
<summary><b>(Optional) Running Infinigen in a Docker Container</b></summary>

**Docker on Linux**

In `/infinigen/`
```
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

</details>