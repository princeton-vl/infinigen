#!/bin/bash

REQUIREMENTS_PATH='./requirements.txt'
PYTHON_WGET_LINK='https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz'
PYTHON_WGET_FILE='Python-3.10.9.tgz'
PYTHON_DIR='Python-3.10.9'
NURBS_SCRIPT="setup_linux.py"

git submodule init
git submodule update

# Install Conda dependencies
pip install -r "${REQUIREMENTS_PATH}"

if [ ! -d "${PYTHON_DIR}" ]; then
    # Install Python include file
    wget -O "${PYTHON_WGET_FILE}" "${PYTHON_WGET_LINK}"
    tar -xf "${PYTHON_WGET_FILE}"
    rm "${PYTHON_WGET_FILE}"
fi

# Install llvm for MacOS
if [ "${OS}" = "Darwin" ]; then
    arch -arm64 brew install llvm open-mpi libomp glm glew
fi

#bash worldgen/tools/install/install_bnurbs.sh
cd ./worldgen/assets/creatures/geometry/cpp_utils
rm -f *.so
rm -rf build
python "${NURBS_SCRIPT}" build_ext --inplace
cd -

# Build terrain
rm -rf *.egg-info
rm -rf __pycache__
rm -rf ./infinigen/terrain/build

cd ./infinigen/terrain
bash install_terrain.sh
python setup.py build_ext --inplace --force
cd -

# Build NURBS
cd ./infinigen/assets/creatures/geometry/cpp_utils
rm -f *.so
rm -rf build
"../../../../../${BLENDER_PYTHON}" "${NURBS_SCRIPT}" build_ext --inplace
cd -

if [ "$1" = "opengl" ]; then
    bash ./infinigen/tools/install/compile_opengl.sh
fi

# Build Flip Fluids addon
if [ "$1" = "flip_fluids" ] || [ "$2" = "flip_fluids" ]; then
    bash ./infinigen/tools/install/compile_flip_fluids.sh
fi