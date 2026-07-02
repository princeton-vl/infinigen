#!/bin/bash
#SBATCH --job-name=if2claypan
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --array=1-999

JOBNAME="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
START_TIME=$(date -Iseconds)

# Non-interactive submits (e.g. over ssh) may inherit a PATH without uv; set
# INFINIGEN_UV_ENV to an env file that puts uv on PATH to recover it.
command -v uv >/dev/null 2>&1 || { [ -n "${INFINIGEN_UV_ENV:-}" ] && source "${INFINIGEN_UV_ENV}"; } 2>/dev/null || true

RENDER_SCRIPT="examples/clay_pan_video/render_clay_pan_video.py"

LOG_BASE="outputs/renderjobs" # presumably matches the --output and --error flags above, but we cant autoconfigure this

SCRATCH_MIN_KB=$((10 * 1024 * 1024))
SCRATCH_CLEANUP_AGE_MINS=180

# Ordered candidates from INFINIGEN_OUTPUT_DIRS (colon-separated, fast first), then
# default outputs/. No site-specific paths committed; set the envvar to override.
CANDIDATES=()
IFS=':' read -ra ENV_CANDIDATES <<< "${INFINIGEN_OUTPUT_DIRS:-}"
for env_dir in "${ENV_CANDIDATES[@]}"; do
    [ -n "${env_dir}" ] && CANDIDATES+=("${env_dir}/${JOBNAME}")
done
CANDIDATES+=("outputs/${JOBNAME}")

can_use_candidate() {
    local candidate="$1"
    mkdir -p "${candidate}" 2>/dev/null || return 1
    local avail
    avail="$(df --output=avail "${candidate}" 2>/dev/null | tail -1)"
    echo "${candidate} available: ${avail} KB"
    [ "${avail}" -gt "${SCRATCH_MIN_KB}" ] 2>/dev/null || return 1
    touch "${candidate}/.writetest" 2>/dev/null || return 1
    rm -f "${candidate}/.writetest"
    return 0
}

# Pick the first candidate that is writable with enough space; error if none work.
JOBFOLDER=""
for candidate in "${CANDIDATES[@]}"; do
    if can_use_candidate "${candidate}"; then
        JOBFOLDER="${candidate}"
        if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
            find "${JOBFOLDER}" -mindepth 1 -mmin +"${SCRATCH_CLEANUP_AGE_MINS}" -delete 2>/dev/null
        fi
        break
    fi
done
[ -n "${JOBFOLDER}" ] || { echo "ERROR: no writable output dir with enough space among: ${CANDIDATES[*]}"; exit 1; }

PROJ_BASE="$(dirname "${JOBFOLDER}")"

# nest raw renders under _raw/ so OUTDIR never collides with FINALDIR/TASKNAME
JOBFOLDER="${JOBFOLDER}/_raw"
echo "JOBFOLDER: ${JOBFOLDER} RETRY COUNT: ${SLURM_RESTART_COUNT}"

TASKNAME="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTDIR="${JOBFOLDER}/${TASKNAME}"
SEED="$(( ${SEED_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))"
uv run --no-sync python "${RENDER_SCRIPT}" --seed "${SEED}" --output "${OUTDIR}" --frames 0 71 --resolution 1280 720 --samples 256 --skip_gt
[ -f "${OUTDIR}/metadata.json" ] || { echo "ERROR: render did not produce ${OUTDIR}/metadata.json, assuming crashed, exiting"; exit 1; }

FINALDIR="${PROJ_BASE}/${JOBNAME}"
mkdir -p "${FINALDIR}"

PACK_MODE="${PACK_MODE:-direct_copy}"
if [ "${PACK_MODE}" = "cvdpack" ]; then
    cp "scripts/cvdpack_infinigen2.json" "${OUTDIR}/cvdpack_config.json"
    ALLOW_LOSSY_RGB_ENCODE=1 uv run --no-sync cvdpack pack \
        --input "${JOBFOLDER}" \
        --output "${FINALDIR}" \
        --config "scripts/cvdpack_infinigen2.json" \
        --subset "scene=${TASKNAME}" \
        --tmp_folder "${JOBFOLDER}/tmp" \
        --n_workers 2 --parallel_mode multiprocess --cpus_per_worker 2
    OUTFILE="${FINALDIR}/${TASKNAME}"
elif [ "${PACK_MODE}" = "targz" ]; then
    OUTFILE="${FINALDIR}/${TASKNAME}.tar.gz"
    tar -czf "${OUTFILE}" -C "${OUTDIR}" .
elif [ "${PACK_MODE}" = "direct_copy" ]; then
    OUTFILE="${FINALDIR}/${TASKNAME}"
    if [ "${OUTDIR}" != "${OUTFILE}" ]; then
        cp -r "${OUTDIR}" "${OUTFILE}"
    fi
else
    echo "ERROR: unknown PACK_MODE=${PACK_MODE}"; exit 1
fi
rm -rf "${OUTDIR}"

END_TIME=$(date -Iseconds)
ERRFILE="${LOG_BASE}/${JOBNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
OUTFILE_LOG="${LOG_BASE}/${JOBNAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "${START_TIME} ${END_TIME} ${SLURM_NODELIST} ${CUDA_VISIBLE_DEVICES} ${OUTFILE} ${ERRFILE} ${OUTFILE_LOG}" >> "${LOG_BASE}/${JOBNAME}_state.log"
