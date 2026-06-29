#!/bin/bash
# Assemble a clay demo video from a multi-scene render dir. Usage: make_video.sh <root> [out.mp4]
set -euo pipefail

ROOT="$1"
OUT="${2:-clay_pan_video.mp4}"

# default 0.5s ao-flat / 0.5s ao-disp / 1.0s rgb at 24fps; all overridable via env
FPS="${FPS:-24}"
NFLAT="${NFLAT:-12}"
NCLAY="${NCLAY:-12}"
NRGB="${NRGB:-24}"
SWIPE="${SWIPE:-0.3}"
FS="${FS:-12}"

NSW=$(awk "BEGIN{print int($SWIPE*$FPS + 0.5)}")
SW_T=$(awk "BEGIN{print $NSW/$FPS}")

# seamless pose windows over the single pan: each clip takes the next block of frames
S1=$NFLAT                       # swipe1 (flat->clay) start
CC=$((NFLAT + NSW))             # pure clay start
S2=$((NFLAT + NSW + NCLAY))     # swipe2 (clay->rgb) start
RR=$((NFLAT + 2 * NSW + NCLAY)) # pure rgb start

# diagonal line sweeping right->left; reveal new clip (B) from the right, 2px black cut line
SWIPE_EXPR="if(lt(abs(X/W + 0.3*(Y/H) - 1.3*P)\, 0.002)\, 0\, if(gt(X/W + 0.3*(Y/H)\, 1.3*P)\, B\, A))"

flat_t=$(awk "BEGIN{print $NFLAT/$FPS}")
clay_t=$(awk "BEGIN{print $NCLAY/$FPS}")
rgb_t=$(awk "BEGIN{print $NRGB/$FPS}")

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
concat_list="$TMP/concat.txt"
: > "$concat_list"

scene_idx=0
for scene in "$ROOT"/*/; do
    cam="$scene/Camera"
    [ -d "$cam" ] || continue
    meta="$scene/metadata.json"

    seed=$(python3 -c "import json;print(int(json.load(open('$meta'))['seed'],16))" 2>/dev/null || echo "?")
    runtime=$(python3 -c "
import json
m = json.load(open('$meta'))
s = m['stats']
gpus = m.get('hardware', {}).get('gpus_all') or ['?']
gpu = gpus[0].split(',')[0]
for prefix in ('NVIDIA', 'GeForce'):
    gpu = gpu.replace(prefix, '')
gpu = ' '.join(gpu.split())
print(f\"{round(s['blend_build_sec'])}sec CPU for .blend, {round(s['render_sec_per_frame'])}sec per frame on a {gpu}\")
" 2>/dev/null || echo "?")

    # bottom bar: generation command left, runtime right
    BL="fontcolor=white:fontsize=$FS:box=1:boxcolor=black@0.65:boxborderw=6:x=16:y=h-th-16"
    BR="fontcolor=white:fontsize=$FS:box=1:boxcolor=black@0.65:boxborderw=6:x=w-tw-16:y=h-th-16"

    clip="$TMP/scene_$(printf '%03d' "$scene_idx").mp4"
    filt="$TMP/filt_${scene_idx}.txt"
    cat > "$filt" <<EOF
[0:v]format=gbrp,fps=$FPS,setsar=1[f];
[1:v]format=gbrp,fps=$FPS,setsar=1[s1a];
[2:v]format=gbrp,fps=$FPS,setsar=1[s1b];
[3:v]format=gbrp,fps=$FPS,setsar=1[c];
[4:v]format=gbrp,fps=$FPS,setsar=1[s2a];
[5:v]format=gbrp,fps=$FPS,setsar=1[s2b];
[6:v]format=gbrp,fps=$FPS,setsar=1[r];
[s1a][s1b]xfade=transition=custom:duration=$SW_T:offset=0:expr='$SWIPE_EXPR'[s1];
[s2a][s2b]xfade=transition=custom:duration=$SW_T:offset=0:expr='$SWIPE_EXPR'[s2];
[f][s1][c][s2][r]concat=n=5:v=1:a=0[cat];
[cat]drawtext=text='uv run infinigen2 livingroom_with_smallobj_rand linear_pan_camera_rand render_cycles --seed $seed':$BL,drawtext=text='$runtime':$BR[out]
EOF

    ffmpeg -y -loglevel error \
        -framerate $FPS -t "$flat_t" -start_number 0    -i "$cam/ao-flat-%04d.png" \
        -framerate $FPS -t "$SW_T"   -start_number $S1   -i "$cam/ao-flat-%04d.png" \
        -framerate $FPS -t "$SW_T"   -start_number $S1   -i "$cam/ao-disp-%04d.png" \
        -framerate $FPS -t "$clay_t" -start_number $CC   -i "$cam/ao-disp-%04d.png" \
        -framerate $FPS -t "$SW_T"   -start_number $S2   -i "$cam/ao-disp-%04d.png" \
        -framerate $FPS -t "$SW_T"   -start_number $S2   -i "$cam/rgb-%04d.png" \
        -framerate $FPS -t "$rgb_t"  -start_number $RR   -i "$cam/rgb-%04d.png" \
        -filter_complex_script "$filt" -map "[out]" \
        -c:v libx264 -pix_fmt yuv420p -r $FPS "$clip"

    echo "file '$clip'" >> "$concat_list"
    scene_idx=$((scene_idx + 1))
done

[ "$scene_idx" -gt 0 ] || { echo "no scenes with Camera/ found under $ROOT"; exit 1; }

ffmpeg -y -loglevel error -f concat -safe 0 -i "$concat_list" -c:v libx264 -pix_fmt yuv420p -r $FPS "$OUT"
echo "wrote $OUT ($scene_idx scene(s))"
