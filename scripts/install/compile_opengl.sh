#!/bin/bash

set -e

cd ./infinigen/datagen/customgt

cmake -S . -Bbuild -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build --target all
./build/customgt -in x -out x --frame 0
if [ $? -eq 174 ]; then
	echo "OpenGL/EGL ground truth is working."
else
	echo "WARNING: OpenGL/EGL is not supported on this machine. If you are running from a cluster head-node, this is likely not an issue."
fi
cd -
