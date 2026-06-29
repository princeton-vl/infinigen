#!/bin/bash
set -u

OUTPUT_PATH=$1
PARALLEL_JOBS=$2
REST_ARGS=${@:3}
LIST_ARGS=""

if [ -z "$OUTPUT_PATH" ]; then
    echo "Please provide an output path"
    exit 1
fi

XARGS="-t -I {}"
if ! [ -z "$PARALLEL_JOBS" ]; then
    XARGS="$XARGS -P $PARALLEL_JOBS"
fi

# Materials are tiny sphere renders, so pack more per slot than heavy scenes.
# Peak concurrent material jobs = slots * MATERIAL_PARALLEL.
MATERIAL_PARALLEL=${MATERIAL_PARALLEL:-1}
MATERIAL_XARGS="-t -I {} -P $MATERIAL_PARALLEL"

MATERIALS=${MATERIALS-$(uv run python -m infinigen_v2.list $LIST_ARGS --categories Material --missing_values drop --columns shortname $REST_ARGS)}
OBJECTS=${OBJECTS-$(uv run python -m infinigen_v2.list $LIST_ARGS --categories Object --missing_values drop --columns shortname $REST_ARGS)}
SCENES=${SCENES-$(uv run python -m infinigen_v2.list $LIST_ARGS --categories Scene --missing_values drop --columns shortname $REST_ARGS)}
MASKS=${MASKS-$(uv run python -m infinigen_v2.list $LIST_ARGS --categories Mask --missing_values drop --columns shortname $REST_ARGS)}

# store git info (for display purposes)
mkdir -p "$OUTPUT_PATH"
if [ ! -f "$OUTPUT_PATH/git_info.toml" ]; then
    if [ -n "${GIT_BRANCH_OVERRIDE:-}" ]; then
        GIT_BRANCH="$GIT_BRANCH_OVERRIDE"
    else
        GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    fi
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    {
        echo "# Render metadata"
        echo "branch = \"$GIT_BRANCH\""
        echo "commit = \"$GIT_COMMIT\""
        echo "timestamp = \"$TIMESTAMP\""
    } > "$OUTPUT_PATH/git_info.toml"
fi

# store manifest.json; this is necessary since there may be differences between branches
cp src/infinigen_v2/generators/manifest.json $OUTPUT_PATH

GEN_ARGS="--loglevel WARNING"
if [ -n "${RENDER_RUNNER:-}" ]; then
    read -r -a RENDER_RUNNER_ARGS <<< "$RENDER_RUNNER"
else
    RENDER_RUNNER_ARGS=(uv run infinigen_v2)
fi

# MATERIALS VISUAL CHECK
for i in {0..5}; do
    echo "$MATERIALS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_sphere render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/material-{}-sphere-cycles-$i --seed $i \
        --passes rgb --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128

done

# MATERIALS DISPLACEMENT TEST (Cycles: BUMP, DISPLACEMENT_AND_BUMP, REALIZE_MESH)
for disp in BUMP DISPLACEMENT_AND_BUMP REALIZE_MESH; do
    echo "$MATERIALS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" \
        {} material_sphere render_cycles render_cycles_ground_truth visualize_gt \
        $GEN_ARGS --output $OUTPUT_PATH/material-{}-sphere-cycles-$disp \
        --seed 0 --passes rgb surface-normal --displacement_mode $disp -r 192 192 -s 128
done

# MATERIALS DISPLACEMENT TEST (Eevee: DISPLACEMENT_AND_BUMP only)
echo "$MATERIALS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" \
    {} material_sphere render_eevee render_eevee_ground_truth visualize_gt \
    $GEN_ARGS --output $OUTPUT_PATH/material-{}-sphere-eevee-DISPLACEMENT_AND_BUMP \
    --seed 0 --passes rgb surface-normal --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128

# MASKS VISUAL CHECK (black/white pattern renders on a flat UV plane)
for i in {0..5}; do
    echo "$MASKS" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_plane_uv render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/mask-{}-planeuv-cycles-$i --seed $i \
        --passes rgb -r 384 384 -s 128
done

# OBJECTS VISUAL CHECK
for i in {0..5}; do
    echo "$OBJECTS" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} object_demo render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/object-{}-demo-cycles-$i --seed $i \
        --passes rgb -r 512 512 -s 128
done

# SCENES VISUAL CHECK
for i in {0..5}; do
    echo "$SCENES" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/scene-{}-demo-cycles-$i --seed $i \
        --passes rgb -r 480 480 -s 256
done
