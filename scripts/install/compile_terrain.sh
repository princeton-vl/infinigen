#!/bin/bash

# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


shopt -s expand_aliases
set -e

cd ./infinigen/terrain

elements=(
    "voronoi_rocks"
    "upsidedown_mountains"
    "ground"
    "warped_rocks"
    "mountains"
    "landtiles"
    "atmosphere"
    "waterbody"
)

surfaces=(
    "chunkyrock"
    "cobble_stone"
    "cracked_ground"
    "dirt"
    "ice"
    "mountain"
    "mud"
    "sand"
    "sandstone"
    "snow"
    "soil"
    "stone"
)

nvcc_location="/usr/local/cuda/bin/nvcc"
alias nx="$nvcc_location -O3 -Xcompiler -fPIC -shared "

# cuda part
if ! [ -x "$nvcc_location" ]; then
    echo "skipping cuda"
    rm -rf lib/cuda/utils/FastNoiseLite.so
    for element in "${elements[@]}"; do
        rm -rf lib/cuda/elements/${element}*.so
    done
    for surface in "${surfaces[@]}"; do
        rm -rf lib/cuda/surfaces/${surface}*.so
    done
else
    mkdir -p lib/cuda/utils
    nx -o lib/cuda/utils/FastNoiseLite.so source/cuda/utils/FastNoiseLite.cu
    echo "compiled lib/cuda/utils/FastNoiseLite.so"
    mkdir -p lib/cuda/elements
    for element in "${elements[@]}"; do
        nx -o lib/cuda/elements/${element}.so source/cuda/elements/${element}.cu
        cp lib/cuda/elements/${element}.so lib/cuda/elements/${element}_1.so
        echo "compiled lib/cuda/elements/${element}.so"
    done
    mkdir -p lib/cuda/surfaces
    for surface in "${surfaces[@]}"; do
        nx -o lib/cuda/surfaces/${surface}.so source/cuda/surfaces/${surface}.cu
        echo "compiled lib/cuda/surfaces/${surface}.so"
    done
fi

# cpu part
OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" = "Linux" ]; then
    alias gx1="g++ -O3 -c -fpic -fopenmp "
    alias gx2="g++ -O3 -shared -fopenmp "
elif [ "${OS}" = "Darwin" ]; then
    if [ "${ARCH}" = "arm64" ]; then
        compiler="/opt/homebrew/opt/llvm/bin/clang++"
    else
        compiler="/usr/local/opt/llvm/bin/clang++"
    fi
    alias gx1="${compiler} -O3 -c -fpic -fopenmp "
    alias gx2="${compiler} -O3 -shared -fopenmp "
    alias gx="${compiler} -O3 -fpic -shared -fopenmp "
else
    echo "Unsupported OS"
    exit -1
fi

mkdir -p lib/cpu/utils
gx1 -o lib/cpu/utils/FastNoiseLite.o source/cpu/utils/FastNoiseLite.cpp
gx2 -o lib/cpu/utils/FastNoiseLite.so lib/cpu/utils/FastNoiseLite.o
echo "compiled lib/cpu/utils/FastNoiseLite.so"
mkdir -p lib/cpu/elements
for element in "${elements[@]}"; do
    gx1 -o lib/cpu/elements/${element}.o source/cpu/elements/${element}.cpp
    gx2 -o lib/cpu/elements/${element}.so lib/cpu/elements/${element}.o
    cp lib/cpu/elements/${element}.so lib/cpu/elements/${element}_1.so
    echo "compiled lib/cpu/elements/${element}.so"
done
mkdir -p lib/cpu/surfaces
for surface in "${surfaces[@]}"; do
    gx1 -o lib/cpu/surfaces/${surface}.o source/cpu/surfaces/${surface}.cpp
    gx2 -o lib/cpu/surfaces/${surface}.so lib/cpu/surfaces/${surface}.o
    echo "compiled lib/cpu/surfaces/${surface}.so"
done

mkdir -p lib/cpu/meshing
gx1 -o lib/cpu/meshing/cube_spherical_mesher.o source/cpu/meshing/cube_spherical_mesher.cpp
gx2 -o lib/cpu/meshing/cube_spherical_mesher.so lib/cpu/meshing/cube_spherical_mesher.o
echo "compiled lib/cpu/meshing/cube_spherical_mesher.so"
gx1 -o lib/cpu/meshing/frontview_spherical_mesher.o source/cpu/meshing/frontview_spherical_mesher.cpp
gx2 -o lib/cpu/meshing/frontview_spherical_mesher.so lib/cpu/meshing/frontview_spherical_mesher.o
echo "compiled lib/cpu/meshing/frontview_spherical_mesher.so"
gx1 -o lib/cpu/meshing/uniform_mesher.o source/cpu/meshing/uniform_mesher.cpp
gx2 -o lib/cpu/meshing/uniform_mesher.so lib/cpu/meshing/uniform_mesher.o
echo "compiled lib/cpu/meshing/uniform_mesher.so"
gx1 -o lib/cpu/meshing/utils.o source/cpu/meshing/utils.cpp
gx2 -o lib/cpu/meshing/utils.so lib/cpu/meshing/utils.o
echo "compiled lib/cpu/meshing/utils.so"

if [ "${OS}" = "Darwin" ]; then
    if [ "${ARCH}" = "arm64" ]; then
        alias gx1="CPATH=/opt/homebrew/include:${CPATH} g++ -O3 -c -fpic -std=c++17"
        alias gx2="CPATH=/opt/homebrew/include:${CPATH} g++ -O3 -shared -std=c++17"
    else
        alias gx1="CPATH=/usr/local/include:${CPATH} g++ -O3 -c -fpic -std=c++17"
        alias gx2="CPATH=/usr/local/include:${CPATH} g++ -O3 -shared -std=c++17"
    fi
fi
mkdir -p lib/cpu/soil_machine
gx1 -o lib/cpu/soil_machine/SoilMachine.o source/cpu/soil_machine/SoilMachine.cpp
gx2 -o lib/cpu/soil_machine/SoilMachine.so lib/cpu/soil_machine/SoilMachine.o
echo "compiled lib/cpu/soil_machine/SoilMachine.so"

cd -