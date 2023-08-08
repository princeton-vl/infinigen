FLIP_FLUIDS="https://github.com/rlguy/Blender-FLIP-Fluids"
FLIP_FLUIDS_FOLDER="Blender-FLIP-Fluids"
FLIP_FLUIDS_ADDON_FOLDER="${FLIP_FLUIDS_FOLDER}/build/bl_flip_fluids/flip_fluids_addon"

if [ ! -d "${FLIP_FLUIDS_ADDON_FOLDER}" ]; then
    git clone "${FLIP_FLUIDS}"
    cd "${FLIP_FLUIDS_FOLDER}"
    python build.py
    cd -
    cp -r "${FLIP_FLUIDS_ADDON_FOLDER}" "${BLENDER_ADDONS}"
    "${BLENDER_EXE}" --background -noaudio -P ./worldgen/fluid/flip_init.py
fi