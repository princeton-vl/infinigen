#!/bin/bash
#SBATCH --job-name=if2render
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --array=1-999

JOBNAME="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
START_TIME=$(date -Iseconds)

RENDER_SCRIPT="scripts/render_stereo_video.py"

LOG_BASE="outputs/renderjobs" # presumably matches the --output and --error flags above, but we cant autoconfigure this

SCRATCH_MIN_KB=$((10 * 1024 * 1024))
SCRATCH_CLEANUP_AGE_MINS=180

MAIN_CANDIDATES=(
    "/n/fs/pvl-renders/${USER}/renders"
    "/scratch/gpfs/JIADENG/${USER}/renders"
)
OPTIM_CANDIDATES=(
    "/scratch/${USER}/${JOBNAME}"
    "/tmp/${USER}/${JOBNAME}"
)

# Use first permanent storage location that exists
PROJ_BASE=""
for candidate in "${MAIN_CANDIDATES[@]}"; do
    if [ -d "${candidate}" ] || mkdir -p "${candidate}" 2>/dev/null; then
        PROJ_BASE="${candidate}"
        break
    fi
done
[ -n "${PROJ_BASE}" ] || { echo "ERROR: no permanent storage found"; exit 1; }

can_use_optim_candidate() {
    local candidate="$1"
    local root="${candidate%/${USER}/${JOBNAME}}"
    local avail
    avail="$(df --output=avail "${root}" 2>/dev/null | tail -1)"
    echo "${root} available: ${avail} KB"
    [ "${avail}" -gt "${SCRATCH_MIN_KB}" ] 2>/dev/null || return 1
    mkdir -p "${candidate}" 2>/dev/null && touch "${candidate}/.writetest" 2>/dev/null || return 1
    rm -f "${candidate}/.writetest"
    return 0
}

# Use a fast local dir if it has enough space and we can write there, otherwise fall back to PROJ_BASE
JOBFOLDER="${PROJ_BASE}/${JOBNAME}"
for local_candidate in "${OPTIM_CANDIDATES[@]}"; do
    if can_use_optim_candidate "${local_candidate}"; then
        JOBFOLDER="${local_candidate}"
        if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
            find "${JOBFOLDER}" -mindepth 1 -mmin +"${SCRATCH_CLEANUP_AGE_MINS}" -delete 2>/dev/null
        fi
        break
    fi
done

# nest raw renders under _raw/ so OUTDIR never collides with FINALDIR/TASKNAME
JOBFOLDER="${JOBFOLDER}/_raw"
echo "JOBFOLDER: ${JOBFOLDER} RETRY COUNT: ${SLURM_RESTART_COUNT}"

TASKNAME="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTDIR="${JOBFOLDER}/${TASKNAME}"
SEED="${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID}"
uv run --no-sync python "${RENDER_SCRIPT}" --seed "${SEED}" --output "${OUTDIR}"
[ -f "${OUTDIR}/metadata.json" ] || { echo "ERROR: render did not produce ${OUTDIR}/metadata.json, assuming crashed, exiting"; exit 1; }

FINALDIR="${PROJ_BASE}/${JOBNAME}"
mkdir -p "${FINALDIR}"

PACK_MODE="${PACK_MODE:-cvdpack}"
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
