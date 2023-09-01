OS=$(uname -s)

BLENDER_ADDONS=$(python -c "import bpy; from addon_utils import paths; print(paths()[0])")

FLIP_FLUIDS="https://github.com/rlguy/Blender-FLIP-Fluids"
FLIP_FLUIDS_FOLDER="Blender-FLIP-Fluids"
FLIP_FLUIDS_ADDON_FOLDER="${FLIP_FLUIDS_FOLDER}/build/bl_flip_fluids/flip_fluids_addon"

if [ ! -d "${FLIP_FLUIDS_ADDON_FOLDER}" ]; then
    git clone "${FLIP_FLUIDS}"
    cd "${FLIP_FLUIDS_FOLDER}"
    python build.py
    cd -
fi

echo "Installing Flip Fluids into $BLENDER_ADDONS"
cp -r "${FLIP_FLUIDS_ADDON_FOLDER}" "${BLENDER_ADDONS}"
python ./infinigen/assets/fluid/flip_init.py