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

# normalize every segment to one fixed format so the file-level concat never renegotiates
ENC="-c:v libx264 -pix_fmt yuv420p -r $FPS"
NORM="format=gbrp,fps=$FPS,setsar=1,format=yuv420p"

# a single still->clip segment (one image input range)
plain_seg() { # <out> <pattern> <start> <dur>
    ffmpeg -nostdin -y -loglevel error \
        -framerate $FPS -t "$4" -start_number "$3" -i "$2" \
        -vf "$NORM" $ENC "$1"
}

# a swipe transition between two image ranges (two inputs only -> no concat deadlock)
swipe_seg() { # <out> <patternA> <startA> <patternB> <startB>
    local sf="$1.filt"
    cat > "$sf" <<FEOF
[0:v]format=gbrp,fps=$FPS,setsar=1[a];
[1:v]format=gbrp,fps=$FPS,setsar=1[b];
[a][b]xfade=transition=custom:duration=$SW_T:offset=0:expr='$SWIPE_EXPR',format=yuv420p[out]
FEOF
    ffmpeg -nostdin -y -loglevel error \
        -framerate $FPS -t "$SW_T" -start_number "$3" -i "$2" \
        -framerate $FPS -t "$SW_T" -start_number "$5" -i "$4" \
        -filter_complex_script "$sf" -map "[out]" $ENC "$1"
}

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

    # build the five segments as standalone clips, then concat the files
    sd="$TMP/segs_${scene_idx}"
    mkdir -p "$sd"
    plain_seg "$sd/s0.mp4" "$cam/ao-flat-%04d.png" 0    "$flat_t"
    swipe_seg "$sd/s1.mp4" "$cam/ao-flat-%04d.png" $S1  "$cam/ao-disp-%04d.png" $S1
    plain_seg "$sd/s2.mp4" "$cam/ao-disp-%04d.png" $CC  "$clay_t"
    swipe_seg "$sd/s3.mp4" "$cam/ao-disp-%04d.png" $S2  "$cam/rgb-%04d.png" $S2
    plain_seg "$sd/s4.mp4" "$cam/rgb-%04d.png" $RR      "$rgb_t"

    seg_list="$sd/list.txt"
    printf "file '%s'\n" "$sd/s0.mp4" "$sd/s1.mp4" "$sd/s2.mp4" "$sd/s3.mp4" "$sd/s4.mp4" > "$seg_list"

    # bottom bar: generation command left, runtime right
    BL="fontcolor=white:fontsize=$FS:box=1:boxcolor=black@0.65:boxborderw=6:x=16:y=h-th-16"
    BR="fontcolor=white:fontsize=$FS:box=1:boxcolor=black@0.65:boxborderw=6:x=w-tw-16:y=h-th-16"
    DT="drawtext=text='uv run infinigen livingroom_with_smallobj_rand linear_pan_camera_rand render_cycles --seed $seed':$BL,drawtext=text='$runtime':$BR"

    clip="$TMP/scene_$(printf '%03d' "$scene_idx").mp4"
    ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "$seg_list" \
        -vf "$DT" $ENC "$clip"

    echo "file '$clip'" >> "$concat_list"
    scene_idx=$((scene_idx + 1))
done

[ "$scene_idx" -gt 0 ] || { echo "no scenes with Camera/ found under $ROOT"; exit 1; }

ffmpeg -nostdin -y -loglevel error -f concat -safe 0 -i "$concat_list" -c:v libx264 -pix_fmt yuv420p -r $FPS "$OUT"
echo "wrote $OUT ($scene_idx scene(s))"
