#!/bin/bash
# Usage: repeat_sbatch.sh <N> <sbatch command and args...>
# Runs the sbatch command N times. Launches the next round once
# COMPLETION_THRESHOLD (default 0.8 = 80%) of the previous round's array
# tasks have left the queue, so the slow tail doesn't block the next round.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <repeats> <sbatch args...>" >&2
    echo "Example: $0 10 sbatch -J myjob --array=1-999 --partition=allcs script.sh" >&2
    exit 1
fi

REPEATS="$1"
shift

POLL_INTERVAL="${POLL_INTERVAL:-15}"
COMPLETION_THRESHOLD="${COMPLETION_THRESHOLD:-0.8}"

# Parse --array=N-M[:S][%K] from the sbatch command to get total task count.
# Slurm groups un-launched array tasks into a single squeue row like
# id_[100-512], so counting squeue lines under-counts; we need the original
# array spec to know the true total.
parse_array_total() {
    local spec=""
    local prev=""
    for a in "$@"; do
        case "$a" in
            --array=*) spec="${a#--array=}" ;;
            --array)   prev="--array" ;;
            *)
                if [[ "$prev" == "--array" ]]; then
                    spec="$a"
                fi
                prev=""
                ;;
        esac
    done
    if [[ -z "$spec" ]]; then
        echo "0"; return
    fi
    spec="${spec%%%*}"   # strip throttle (%K)
    local total=0
    IFS=',' read -ra parts <<<"$spec"
    for p in "${parts[@]}"; do
        local rng="${p%%:*}"  # strip step (:S)
        if [[ "$rng" == *-* ]]; then
            local lo="${rng%-*}" hi="${rng#*-}"
            total=$(( total + hi - lo + 1 ))
        else
            total=$(( total + 1 ))
        fi
    done
    echo "$total"
}

# Count remaining tasks for a job, expanding bracketed array ranges.
count_remaining() {
    squeue -r -j "$1" -h -O JobID 2>/dev/null | wc -l
}

wait_for_completion() {
    local jid="$1" total="$2"
    if [[ $total -le 0 ]]; then
        echo "  WARN: total task count is 0; falling back to live squeue count" >&2
        total=$(count_remaining "$jid")
        if [[ $total -eq 0 ]]; then return; fi
    fi
    local target_done
    target_done=$(awk -v t=$total -v thr=$COMPLETION_THRESHOLD 'BEGIN { printf "%d", t*thr }')
    while true; do
        local remaining done pct
        remaining=$(count_remaining "$jid")
        done=$((total - remaining))
        if (( done < 0 )); then done=0; fi
        pct=$(awk -v d=$done -v t=$total 'BEGIN { printf "%.0f", 100*d/t }')
        echo "  job $jid: $done/$total done (${pct}%, target ${target_done})"
        if [[ $done -ge $target_done ]]; then
            return
        fi
        sleep "$POLL_INTERVAL"
    done
}

ARRAY_TOTAL=$(parse_array_total "$@")
echo "parsed array total: $ARRAY_TOTAL"

for ((i = 1; i <= REPEATS; i++)); do
    echo "=== Round $i/$REPEATS ==="

    output=$("$@" 2>&1)
    echo "$output"

    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+' | tail -1)

    if [[ -z "$job_id" ]]; then
        echo "ERROR: Could not parse job ID from sbatch output" >&2
        exit 1
    fi

    if [[ $i -lt $REPEATS ]]; then
        echo "  Submitted job $job_id, waiting for $(awk -v t=$COMPLETION_THRESHOLD 'BEGIN{printf "%.0f%%", t*100}') completion..."
        wait_for_completion "$job_id" "$ARRAY_TOTAL"
        echo "  Threshold reached. Launching next round."
    else
        echo "  Final round submitted (job $job_id)."
    fi
done

echo "=== All $REPEATS rounds submitted ==="
