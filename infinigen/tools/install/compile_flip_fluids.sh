OS=$(uname -s)

if [ "${OS}" = "Linux" ]; then
    BLENDER_DIR='blender'
    BLENDER_ADDONS="${BLENDER_DIR}/3.3/scripts/addons"
    BLENDER_EXE="${BLENDER_DIR}/blender"
elif [ "${OS}" = "Darwin" ]; then
    BLENDER_DIR='./Blender.app'
    BLENDER_ADDONS="${BLENDER_DIR}/Contents/Resources/3.3/scripts/addons"
    BLENDER_EXE="${BLENDER_DIR}/Contents/MacOS/Blender"
else
    echo "Unsupported OS"
    exit -1
fi

FLIP_FLUIDS="https://github.com/rlguy/Blender-FLIP-Fluids"
FLIP_FLUIDS_FOLDER="Blender-FLIP-Fluids"
FLIP_FLUIDS_ADDON_FOLDER="${FLIP_FLUIDS_FOLDER}/build/bl_flip_fluids/flip_fluids_addon"

if [ ! -d "${FLIP_FLUIDS_ADDON_FOLDER}" ]; then
    git clone "${FLIP_FLUIDS}"
    cd "${FLIP_FLUIDS_FOLDER}"
    python build.py
    cd -
    cp -r "${FLIP_FLUIDS_ADDON_FOLDER}" "${BLENDER_ADDONS}"
    "${BLENDER_EXE}" --background -noaudio -P ./infinigen/assets/fluid/flip_init.py
fi