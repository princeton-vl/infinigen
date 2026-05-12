#!/bin/bash
#SBATCH --job-name=if2render
#SBATCH --account=allcs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --array=1-512

JOBNAME=${SLURM_JOB_NAME}
START_TIME=$(date -Iseconds)

RENDER_CONFIG="\
    -r 1024 1024 -s 512 \
    --save_blend \
    --passes rgb depth surface-normal semantic-segmentation material-segmentation environment"

LOG_BASE="outputs/renderjobs" # presumably matches the --output and --error flags above, but we cant autoconfigure this

PROJ_BASE="/n/fs/scratch/" # default storage space which is guaranteed to exist
OPTIM_BASE="/scratch/" # storage space which is preferable if available but not guaranteed
SCRATCH_MIN_KB=$((50 * 1024 * 1024)) # fallback to /n/fs/scratch if /scratch is unavailable/full/toosmall
SCRATCH_CLEANUP_AGE_MINS=180 # delete old files after this many minutes

if [ "$(df --output=avail ${OPTIM_BASE} 2>/dev/null | tail -1)" -gt "${SCRATCH_MIN_KB}" ] 2>/dev/null; then
    JOBFOLDER="${OPTIM_BASE}/${USER}/${JOBNAME}"
    find "${JOBFOLDER}" -mindepth 1 -mmin +"${SCRATCH_CLEANUP_AGE_MINS}" -delete 2>/dev/null
else
    JOBFOLDER="${PROJ_BASE}/${USER}/${JOBNAME}"
fi

TASKNAME="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUTDIR="${JOBFOLDER}/${TASKNAME}"
SEED="${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID}"
uv run --no-sync python -m infinigen_v2.generate --seed "${SEED}" --output "${OUTDIR}" \
    livingroom_distribution render_cycles render_cycles_ground_truth \
    --frames 0 12 ${RENDER_CONFIG}

TARPATH="${PROJ_BASE}/${USER}/${JOBNAME}/${TASKNAME}.tar.gz"
mkdir -p "$(dirname "${TARPATH}")"
tar -czvf "${TARPATH}" -C "${OUTDIR}" .
rm -rf $OUTDIR

END_TIME=$(date -Iseconds)
echo "${START_TIME} ${END_TIME} ${SLURM_NODELIST} ${CUDA_VISIBLE_DEVICES} ${TARPATH}" >> "${LOG_BASE}/${JOBNAME}_completed.log"
