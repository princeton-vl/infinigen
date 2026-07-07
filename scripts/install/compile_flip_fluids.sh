OS=$(uname -s)

FLIP_FLUIDS="https://github.com/rlguy/Blender-FLIP-Fluids"
FLIP_FLUIDS_FOLDER="Blender-FLIP-Fluids"
FLIP_FLUIDS_ADDON_FOLDER="${FLIP_FLUIDS_FOLDER}/build/bl_flip_fluids/flip_fluids_addon"

# last release before FLIP Fluids made Alembic a hard build dependency (v1.8.4)
FLIP_FLUIDS_TAG="v0.8.3"

if [ ! -d "${FLIP_FLUIDS_ADDON_FOLDER}" ]; then
    git clone --branch "${FLIP_FLUIDS_TAG}" --depth 1 "${FLIP_FLUIDS}" "${FLIP_FLUIDS_FOLDER}"
    cd "${FLIP_FLUIDS_FOLDER}"
    python build.py
    cd -
fi

# addon_enable needs a real Blender context, so flip_init.py runs inside the
# standalone Blender and copies the built addon into its user addons dir first.
FLIP_FLUIDS_ADDON_SRC="$(pwd)/${FLIP_FLUIDS_ADDON_FOLDER}" \
    python -m infinigen.launch_blender -s ./src/infinigen/assets/fluid/flip_init.py
