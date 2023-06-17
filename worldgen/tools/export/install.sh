#!/bin/bash

OS=$(uname -s)
ARCH=$(uname -m)


if [ "${OS}" = "Linux" ]; then
    BLENDER_DIR='blender_3.6'
    BLENDER_WGET_LINK='https://builder.blender.org/download/daily/archive/blender-3.6.0-beta+v36.962331c256b0-linux.x86_64-release.tar.xz'
    BLENDER_WGET_FILE='blender.tar.xz'
    BLENDER_UNTAR_DIR='blender-3.6.0-beta+v36.962331c256b0-linux.x86_64-release'

elif [ "${OS}" = "Darwin" ]; then
    BLENDER_VOLM='/Volumes/Blender'
    BLENDER_DIR='./Blender.app'
    BLENDER_WGET_FILE='blender.dmg'
    if [ "${ARCH}" = "arm64" ]; then
        BLENDER_WGET_LINK='https://builder.blender.org/download/daily/archive/blender-3.6.0-beta+v36.83ef3bc9232b-darwin.arm64-release.dmg'
    else
        BLENDER_WGET_LINK='https://builder.blender.org/download/daily/archive/blender-3.6.0-beta+v36.83ef3bc9232b-darwin.x86_64-release.dmg'
    fi

else
    echo "Unsupported OS"
    exit -1
fi

if [ ! -d "${BLENDER_DIR}" ]; then
    # Download Blender
    wget -q -O "${BLENDER_WGET_FILE}" "${BLENDER_WGET_LINK}"

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