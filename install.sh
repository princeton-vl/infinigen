#!/bin/bash

REQUIREMENTS_PATH='./requirements.txt'
PYTHON_WGET_LINK='https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz'
PYTHON_WGET_FILE='Python-3.10.9.tgz'
PYTHON_DIR='Python-3.10.9'

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

bash worldgen/tools/install/install_bnurbs.sh

# Build terrain
rm -rf *.egg-info
rm -rf __pycache__
rm -rf ./worldgen/terrain/build

cd ./worldgen/terrain
bash install_terrain.sh
python setup.py build_ext --inplace --force
cd -

# Compile process_mesh (i.e. OpenGL-based ground truth)
cd ./process_mesh
/usr/bin/cmake -S . -Bbuild -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build build --target all
./build/process_mesh -in x -out x --height 100 --width 10
if [ $? -eq 174 ]; then
    echo "OpenGL/EGL ground truth is working."
else
    echo "WARNING: OpenGL/EGL is not supported on this machine. If you are running from a cluster head-node, this is likely not an issue."
fi
cd -
<<<<<<< HEAD

# Build NURBS
cd ./worldgen/assets/creatures/geometry/cpp_utils
rm -f *.so
rm -rf build
"../../../../../${BLENDER_PYTHON}" "${NURBS_SCRIPT}" build_ext --inplace
cd -

if [ "$1" = "opengl" ]; then
    bash ./worldgen/tools/install/compile_opengl.sh
fi


# Build Flip Fluids addon
if [ "$1" = "flip_fluids" ] || [ "$2" = "flip_fluids" ]; then
    bash ./worldgen/tools/install/compile_flip_fluids.sh
fi
=======
>>>>>>> 84118268d (Initial buggy 3.5 fixes)
