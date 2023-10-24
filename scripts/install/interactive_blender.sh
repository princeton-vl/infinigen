#!/bin/bash

set -e 

if ! command -v wget &> /dev/null
then
    echo "wget could not be found, please 'sudo apt-get install wget' or 'brew install wget'"
    exit
fi

OS=$(uname -s)
ARCH=$(uname -m)

if [ "${OS}" = "Linux" ]; then
    BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz'
    BLENDER_WGET_FILE='blender.tar.xz'

    BLENDER_UNTAR_DIR='blender-3.6.0-linux-x64'
    BLENDER_DIR='blender'
    BLENDER_PYTHON="${BLENDER_DIR}/3.6/python/bin/python3.10"
    BLENDER_INCLUDE="${BLENDER_DIR}/3.6/python/include/python3.10"
    BLENDER_PACKAGES="${BLENDER_DIR}/3.6/python/lib/python3.10/site-packages"
    BLENDER_ADDONS="${BLENDER_DIR}/3.6/scripts/addons"
    BLENDER_EXE="${BLENDER_DIR}/blender"

elif [ "${OS}" = "Darwin" ]; then

    if [ "${ARCH}" = "arm64" ]; then
        BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.6/blender-3.6.0-macos-arm64.dmg'
    else
        BLENDER_WGET_LINK='https://download.blender.org/release/Blender3.6/blender-3.6.0-macos-x64.dmg'
    fi
    
    BLENDER_WGET_FILE='blender.dmg'

    BLENDER_VOLM='/Volumes/Blender'
    BLENDER_DIR='./Blender.app'
    BLENDER_PYTHON="${BLENDER_DIR}/Contents/Resources/3.6/python/bin/python3.10"
    BLENDER_INCLUDE="${BLENDER_DIR}/Contents/Resources/3.6/python/include/python3.10"
    BLENDER_PACKAGES="${BLENDER_DIR}/Contents/Resources/3.6/python/lib/python3.10/site-packages"
    BLENDER_ADDONS="${BLENDER_DIR}/Contents/Resources/3.6/scripts/addons"
    BLENDER_EXE="${BLENDER_DIR}/Contents/MacOS/Blender"

else
    echo "Unsupported OS"
    exit -1
fi

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

# Install Blender dependencies
"${BLENDER_PYTHON}" -m ensurepip

CFLAGS="-I/usr/include/python3.10 ${CFLAGS}" "${BLENDER_PYTHON}" -m pip install -vv -e .