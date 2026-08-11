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

MATERIALS=${MATERIALS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Material --missing_values drop --columns shortname $REST_ARGS)}
OBJECTS=${OBJECTS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Object --missing_values drop --columns shortname $REST_ARGS)}
MASKS=${MASKS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Mask --missing_values drop --columns shortname $REST_ARGS)}
PRESETS=${PRESETS-$(uv run python -m infinigen2.list $LIST_ARGS --presets --missing_values drop --columns shortname $REST_ARGS)}
ENVIRONMENTS=${ENVIRONMENTS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Environment --missing_values drop --columns shortname $REST_ARGS)}
CAMERAS=${CAMERAS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Cameras --missing_values drop --columns shortname $REST_ARGS)}

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
        [ -n "${PR_NUMBER:-}" ] && echo "pr = \"$PR_NUMBER\""
        [ -n "${GITHUB_REPOSITORY:-}" ] && echo "repo = \"$GITHUB_REPOSITORY\""
    } > "$OUTPUT_PATH/git_info.toml"
fi

# store manifest.json; this is necessary since there may be differences between branches
cp src/infinigen2/manifest.json $OUTPUT_PATH

# store the preset -> parent-generator map so the viewer can group presets beneath
# their _rand without importing infinigen2 (the web server runs a minimal env)
uv run python -c "import json, sys; from infinigen2.list import preset_parents; sys.stdout.write(json.dumps(preset_parents()))" > $OUTPUT_PATH/preset_parents.json

GEN_ARGS="--loglevel WARNING"
if [ -n "${RENDER_RUNNER:-}" ]; then
    read -r -a RENDER_RUNNER_ARGS <<< "$RENDER_RUNNER"
    PY_BIN="${RENDER_RUNNER_ARGS[0]}"
    CAM_RUNNER_ARGS=("$PY_BIN" scripts/integration_v2/run_and_index.py --index-root "$OUTPUT_PATH" -- "$PY_BIN" scripts/integration_v2/render_trajectory_video.py)
else
    RENDER_RUNNER_ARGS=(uv run infinigen)
    CAM_RUNNER_ARGS=(uv run python scripts/integration_v2/render_trajectory_video.py)
fi

# render_cycles drops the surface-normal pass; the GT exporter renders it as a second unshaded pass.
NORMAL_STEPS="render_cycles_ground_truth visualize_gt"

# integration_test_string = full command tail; few materials set it, rest default to the cube.
MATERIAL_CMDS=$(uv run python -m infinigen2.list $LIST_ARGS --categories Material \
    --columns shortname integration_test_string --missing_values drop \
    --separator $'\t' $REST_ARGS)

# Presets inherit the demo geometry (2nd token) of the generator that owns them.
PRESET_CMDS=$(awk -F'\t' 'NR==FNR{split($2,a," "); g[$1]=a[2]; next}
    ($2 in g){print $1"\t"$1" "g[$2]" render_cycles"}' \
    <(printf '%s\n' "$MATERIAL_CMDS") \
    <(uv run python -c "from infinigen2.list import preset_parents
for k, v in preset_parents().items():
    print(f'{k}\t{v}')"))

# Shard items ($2) present in a shortname->command map ($1), as "shortname<TAB>cmd".
overrides_in_shard() {
    awk -F'\t' 'NR==FNR{c[$1]=$0;next} NF && ($1 in c){print c[$1]}' \
        <(printf '%s\n' "$1") <(printf '%s\n' $2 | awk 'NF')
}

# Shard items ($2) absent from the map ($1), as bare shortnames.
defaults_in_shard() {
    awk -F'\t' 'NR==FNR{c[$1]=1;next} NF && !($1 in c){print $1}' \
        <(printf '%s\n' "$1") <(printf '%s\n' $2 | awk 'NF')
}

MATERIAL_OVERRIDES=$(overrides_in_shard "$MATERIAL_CMDS" "$MATERIALS")
MATERIAL_DEFAULTS=$(defaults_in_shard "$MATERIAL_CMDS" "$MATERIALS")
PRESET_OVERRIDES=$(overrides_in_shard "$PRESET_CMDS" "$PRESETS")
PRESET_DEFAULTS=$(defaults_in_shard "$PRESET_CMDS" "$PRESETS")

# MATERIALS VISUAL CHECK (cube defaults + own-tail overrides; normals on every seed)
for i in {0..5}; do
    echo "$MATERIAL_DEFAULTS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_cube render_cycles $NORMAL_STEPS \
        $GEN_ARGS --output $OUTPUT_PATH/material-{}-cube-cycles-$i --seed $i \
        --passes rgb surface-normal --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128
    while IFS=$'\t' read -r sn cmd; do
        [ -z "$sn" ] && continue
        "${RENDER_RUNNER_ARGS[@]}" $cmd $NORMAL_STEPS \
            $GEN_ARGS --output $OUTPUT_PATH/material-$sn-cube-cycles-$i --seed $i \
            --passes rgb surface-normal --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128
    done <<< "$MATERIAL_OVERRIDES"
done

# MATERIAL PRESETS VISUAL CHECK (fixed-look variants; deterministic, one seed each)
echo "$PRESET_DEFAULTS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_cube render_cycles \
    $GEN_ARGS --output $OUTPUT_PATH/preset-{}-cube-cycles-0 --seed 0 \
    --passes rgb --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128
while IFS=$'\t' read -r sn cmd; do
    [ -z "$sn" ] && continue
    "${RENDER_RUNNER_ARGS[@]}" $cmd \
        $GEN_ARGS --output $OUTPUT_PATH/preset-$sn-cube-cycles-0 --seed 0 \
        --passes rgb --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128
done <<< "$PRESET_OVERRIDES"

# MATERIALS DISPLACEMENT TEST (Cycles GT is geometry-agnostic; DISPLACEMENT_AND_BUMP from seed loop)
for disp in BUMP REALIZE_MESH; do
    echo "$MATERIALS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" \
        {} material_cube render_cycles $NORMAL_STEPS \
        $GEN_ARGS --output $OUTPUT_PATH/material-{}-cube-cycles-$disp \
        --seed 0 --passes rgb surface-normal --displacement_mode $disp -r 192 192 -s 128
done

# MATERIALS EXPORT TEST (Eevee: DISPLACEMENT_AND_BUMP only)
echo "$MATERIALS" | xargs $MATERIAL_XARGS "${RENDER_RUNNER_ARGS[@]}" \
    {} material_cube render_eevee render_eevee_ground_truth visualize_gt \
    $GEN_ARGS --output $OUTPUT_PATH/material-{}-cube-eevee-DISPLACEMENT_AND_BUMP \
    --seed 0 --passes rgb surface-normal --displacement_mode DISPLACEMENT_AND_BUMP -r 192 192 -s 128

# MASKS VISUAL CHECK (black/white pattern renders on a flat UV plane)
for i in {0..5}; do
    echo "$MASKS" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_plane_uv render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/mask-{}-planeuv-cycles-$i --seed $i \
        --passes rgb -r 384 384 -s 128
done

# OBJECTS VISUAL CHECK
for i in {0..5}; do
    echo "$OBJECTS" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} object_demo render_cycles $NORMAL_STEPS \
        $GEN_ARGS --output $OUTPUT_PATH/object-{}-demo-cycles-$i --seed $i \
        --passes rgb surface-normal -r 384 384 -s 128
done

# save_object_data must follow the GT render, which stamps each object's segmentation index.
SCENE_STEPS="render_cycles_ground_truth save_object_data visualize_gt"
SCENE_PASSES="rgb surface-normal camera semantic-segmentation"

# \x1f-separated: cmd can be an empty middle field, and tab-IFS read collapses those
SCENE_CMDS=${SCENE_CMDS-$(uv run python -m infinigen2.list $LIST_ARGS --categories Scene \
    --columns shortname integration_test_string num_seeds --missing_values keep \
    --separator $'\x1f' $REST_ARGS)}

# launch_andromeda shards scenes via SCENES; unfiltered, every slot renders every scene
if [ -n "${SCENES+set}" ]; then
    SCENE_CMDS=$(awk -F$'\x1f' 'NR==FNR{keep[$1];next} $1 in keep' <(echo "$SCENES") <(echo "$SCENE_CMDS"))
fi

while IFS=$'\x1f' read -r sn cmd num_seeds; do
    [ -z "$sn" ] && continue
    cmd=${cmd:-"$sn render_cycles"}
    num_seeds=${num_seeds%.*}
    num_seeds=${num_seeds:-9}
    for ((i = 0; i < num_seeds; i++)); do
        echo "+ scene $sn -> $cmd $SCENE_STEPS (seed $i)"
        "${RENDER_RUNNER_ARGS[@]}" $cmd $SCENE_STEPS \
            $GEN_ARGS --output $OUTPUT_PATH/scene-$sn-demo-cycles-$i --seed $i \
            --passes $SCENE_PASSES -r 384 384 -s 256
    done
done <<< "$SCENE_CMDS"

# ENVIRONMENTS VISUAL CHECK (each lighting/sky generator lights a demo monkey)
for i in {0..5}; do
    echo "$ENVIRONMENTS" | xargs $XARGS "${RENDER_RUNNER_ARGS[@]}" {} material_monkey render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/environment-{}-monkey-cycles-$i --seed $i \
        --passes rgb -r 512 512 -s 128
done

if grep -Fxq bricks_rand <<< "$MATERIALS"; then
    "${RENDER_RUNNER_ARGS[@]}" bricks_rand material_torus_uv render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/landing-bricks_rand-torus-cycles-0 --seed 0 \
        --passes rgb -r 512 512 -s 128
fi

if grep -Fxq fabric_patterned_rand <<< "$MATERIALS"; then
    "${RENDER_RUNNER_ARGS[@]}" fabric_patterned_rand material_monkey render_cycles \
        $GEN_ARGS --output $OUTPUT_PATH/landing-fabric_patterned_rand-monkey-cycles-0 --seed 0 \
        --passes rgb -r 512 512 -s 128
fi

# CAMERA TRAJECTORIES VISUAL CHECK (48-frame workbench mp4 per camera generator, 3 seeds)
CAM_SCENE=${CAM_SCENE:-livingroom_rand}
for i in {0..2}; do
    echo "$CAMERAS" | xargs $XARGS "${CAM_RUNNER_ARGS[@]}" \
        --output $OUTPUT_PATH/camera-{}-$CAM_SCENE-workbench-traj$i \
        --scene $CAM_SCENE --camera {} --seed $i --frames 0 47 -r 640 360 --fps 3
done
