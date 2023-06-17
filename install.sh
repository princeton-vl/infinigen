#!/bin/bash

if ! command -v wget &> /dev/null
then
    echo "wget could not be found, please 'sudo apt-get install wget' or 'brew install wget'"
    exit
fi

OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" = "Linux" ]; then
    BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz'
    BLENDER_WGET_FILE='blender.tar.xz'

    BLENDER_UNTAR_DIR='blender-3.3.1-linux-x64'
    BLENDER_DIR='blender'
    BLENDER_PYTHON="${BLENDER_DIR}/3.3/python/bin/python3.10"
    BLENDER_INCLUDE="${BLENDER_DIR}/3.3/python/include/python3.10"
    BLENDER_PACKAGES="${BLENDER_DIR}/3.3/python/lib/python3.10/site-packages"

    NURBS_SCRIPT="setup_linux.py"
elif [ "${OS}" = "Darwin" ]; then
    if [ "${ARCH}" = "arm64" ]; then
        BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.3/blender-3.3.1-macos-arm64.dmg'
        NURBS_SCRIPT="setup_macos_as.py"
    else
        BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.3/blender-3.3.1-macos-x64.dmg'
        NURBS_SCRIPT="setup_macos.py"
    fi
    if [ "${ARCH}" = "arm64" ]; then
        HOMEBREW_PREFIX="/opt/homebrew/"
    else
        HOMEBREW_PREFIX="/usr/local"
    fi
    BLENDER_WGET_FILE='blender.dmg'

    BLENDER_VOLM='/Volumes/Blender'
    BLENDER_DIR='./Blender.app'
    BLENDER_PYTHON="${BLENDER_DIR}/Contents/Resources/3.3/python/bin/python3.10"
    BLENDER_INCLUDE="${BLENDER_DIR}/Contents/Resources/3.3/python/include/python3.10"
    BLENDER_PACKAGES="${BLENDER_DIR}/Contents/Resources/3.3/python/lib/python3.10/site-packages"

    export CC="${HOMEBREW_PREFIX}/opt/llvm/bin/clang"
    export CPATH="${HOMEBREW_PREFIX}/include:${CPATH}"
else
    echo "Unsupported OS"
    exit -1
fi
REQUIREMENTS_PATH='./requirements.txt'
PYTHON_WGET_LINK='https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz'
PYTHON_WGET_FILE='Python-3.10.2.tgz'
PYTHON_DIR='Python-3.10.2'

git submodule init
git submodule update

if [ ! -d "${BLENDER_DIR}" ]; then
    # Download Blender
    wget -O "${BLENDER_WGET_FILE}" "${BLENDER_WGET_LINK}"

    # Unzip Blender
    if [ "${OS}" = "Darwin" ]; then
        hdiutil attach "${BLENDER_WGET_FILE}"
        cp -r "${BLENDER_VOLM}/Blender.app" "${BLENDER_DIR}"
        hdiutil detach "${BLENDER_VOLM}"
    else
        tar -xf "${BLENDER_WGET_FILE}"
        mv "${BLENDER_UNTAR_DIR}" "${BLENDER_DIR}"
    fi

    rm "${BLENDER_WGET_FILE}"
fi

# Install llvm for MacOS
if [ "${OS}" = "Darwin" ]; then
    brew install llvm open-mpi libomp glm glew
fi

# Install Conda dependencies
pip install -r "${REQUIREMENTS_PATH}"
pip install fake-bpy-module-latest

if [ ! -d "${PYTHON_DIR}" ]; then
    # Install Python include file
    wget -O "${PYTHON_WGET_FILE}" "${PYTHON_WGET_LINK}"
    tar -xf "${PYTHON_WGET_FILE}"
    rm "${PYTHON_WGET_FILE}"
fi
cp -r "${PYTHON_DIR}/Include/"* "${BLENDER_INCLUDE}"

# Install Blender dependencies
"${BLENDER_PYTHON}" -m ensurepip
CFLAGS="-I$(realpath ${BLENDER_INCLUDE}) ${CFLAGS}" "${BLENDER_PYTHON}" -m pip install -r "${REQUIREMENTS_PATH}"

# Build terrain
rm -rf ${BLENDER_PACKAGES}/terrain-*
rm -rf *.egg-info
rm -rf __pycache__
rm -rf ./worldgen/terrain/build

cd ./worldgen/terrain
if [ -f "/usr/local/cuda/bin/nvcc" ]; then
    USE_CUDA=1 bash install_terrain.sh
else
    USE_CUDA=0 bash install_terrain.sh
fi
"../../${BLENDER_PYTHON}" setup.py build_ext --inplace --force
cd -

# Build NURBS
cd ./worldgen/assets/creatures/geometry/cpp_utils
rm -f *.so
rm -rf build
"../../../../../${BLENDER_PYTHON}" "${NURBS_SCRIPT}" build_ext --inplace
cd -

if [ "$1" = "opengl" ]; then
    # Compile process_mesh (i.e. OpenGL-based ground truth)
    cd ./process_mesh
    /usr/bin/cmake -S . -Bbuild -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=Release
    /usr/bin/cmake --build build --target all
    ./build/process_mesh -in x -out x --height 100 --width 10 --frame 0
    if [ $? -eq 174 ]; then
        echo "OpenGL/EGL ground truth is working."
    else
        echo "WARNING: OpenGL/EGL is not supported on this machine. If you are running from a cluster head-node, this is likely not an issue."
    fi
    cd -
fi