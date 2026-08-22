#!/bin/bash
#SBATCH --job-name=if2dataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --array=1-200
#SBATCH --no-requeue

[ -f /n/fs/pvl-renders/${USER}/uv/env ] && source /n/fs/pvl-renders/${USER}/uv/env

CONFIG="examples/flying_indoor/cvdpack.json"
NUM_CAMERAS=2
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-4}"

: "${SLURM_ARRAY_JOB_ID:=$(date -u +%Y%m%d%H%M)}"
: "${SLURM_ARRAY_TASK_ID:?must be set (slurm env, or e.g. SLURM_ARRAY_TASK_ID=1)}"

# shards over (scene, trajectory, camera); size the array N_SCENES*NUM_TRAJECTORIES*NUM_CAMERAS
IDX=$((SLURM_ARRAY_TASK_ID - 1))
CAMERA_IDX=$((IDX % NUM_CAMERAS))
TRAJ_IDX=$(((IDX / NUM_CAMERAS) % NUM_TRAJECTORIES))
SCENE_IDX=$((IDX / (NUM_CAMERAS * NUM_TRAJECTORIES)))
SEED="${SLURM_ARRAY_JOB_ID}${SCENE_IDX}"
TRAJECTORY_SEED="${SEED}${TRAJ_IDX}"
printf -v SCENE_ID "%04d" "${SCENE_IDX}"
SCENE="${SLURM_ARRAY_JOB_ID}_${SCENE_ID}"
TRAJ="traj${TRAJ_IDX}"
SHARD="${SCENE}/${TRAJ}"

JOBNAME="${SLURM_JOB_NAME:-if2dataset}_${SLURM_ARRAY_JOB_ID}"
START_TIME=$(date -Iseconds)

LOG_BASE="outputs/renderjobs"
mkdir -p "${LOG_BASE}"
ERRFILE="${LOG_BASE}/${JOBNAME}_${SLURM_ARRAY_TASK_ID}.err"
OUTFILE_LOG="${LOG_BASE}/${JOBNAME}_${SLURM_ARRAY_TASK_ID}.out"

emit_state() {
    local end_time; end_time="$(date -Iseconds)"
    echo "${START_TIME} ${end_time} ${SLURM_NODELIST:-none} ${CUDA_VISIBLE_DEVICES:-none} ${FINALDIR:-none}/${SHARD} ${ERRFILE} ${OUTFILE_LOG} $1" >> "${LOG_BASE}/${JOBNAME}_state.log"
}

on_term() {
    trap - TERM
    sleep 2
    local reason="killed"
    case "$(tail -n 50 "${ERRFILE}" 2>/dev/null)" in
        *"DUE TO PREEMPTION"*) reason="preempted" ;;
        *"DUE TO TIME LIMIT"*) reason="time_limit" ;;
        *"DUE TO NODE FAILURE"*) reason="node_fail" ;;
        *"CANCELLED AT"*) reason="cancelled" ;;
    esac
    emit_state "${reason}"
    exit 1
}
trap on_term TERM

SCRATCH_MIN_KB=$((10 * 1024 * 1024))
SCRATCH_CLEANUP_AGE_MINS=180

MAIN_CANDIDATES=(
    "/n/fs/pvl-renders/${USER}/renders"
    "/scratch/gpfs/JIADENG/${USER}/renders"
    "outputs/renders"
)
TASK_TAG="${SLURM_ARRAY_JOB_ID}/_raw"
OPTIM_CANDIDATES=(
    "/scratch/${USER}/${TASK_TAG}"
    "/tmp/${USER}/${TASK_TAG}"
)

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
    local root="${candidate%/${USER}/${TASK_TAG}}"
    local avail
    avail="$(df --output=avail "${root}" 2>/dev/null | tail -1)"
    echo "${root} available: ${avail} KB"
    [ "${avail}" -gt "${SCRATCH_MIN_KB}" ] 2>/dev/null || return 1
    mkdir -p "${candidate}" 2>/dev/null && touch "${candidate}/.writetest" 2>/dev/null || return 1
    rm -f "${candidate}/.writetest"
    return 0
}

SCRATCH_DIR="${PROJ_BASE}/${SLURM_ARRAY_JOB_ID}/_raw"
for local_candidate in "${OPTIM_CANDIDATES[@]}"; do
    if can_use_optim_candidate "${local_candidate}"; then
        SCRATCH_DIR="${local_candidate}"
        if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]; then
            find "${local_candidate}" -mindepth 1 -maxdepth 1 -mmin +"${SCRATCH_CLEANUP_AGE_MINS}" -exec rm -rf {} + 2>/dev/null
        fi
        break
    fi
done

FINALDIR="${PROJ_BASE}/${SLURM_ARRAY_JOB_ID}"
OUTDIR="${SCRATCH_DIR}/${SHARD}"
mkdir -p "${OUTDIR}" "${FINALDIR}"
echo "SHARD: ${SHARD} SCRATCH_DIR: ${SCRATCH_DIR} RETRY COUNT: ${SLURM_RESTART_COUNT:-0}"

if ! uv run --no-sync python examples/flying_indoor/render.py \
    --seed "${SEED}" --trajectory_seed "${TRAJECTORY_SEED}" \
    --camera_idx "${CAMERA_IDX}" --output "${OUTDIR}"; then
    echo "ERROR: render failed for ${SHARD}"
    emit_state "crashed"
    exit 1
fi
[ -f "${OUTDIR}/metadata.json" ] || { echo "ERROR: no ${OUTDIR}/metadata.json"; emit_state "crashed"; exit 1; }

ALLOW_LOSSY_RGB_ENCODE=1 uv run --no-sync cvdpack pack \
    --input "${SCRATCH_DIR}" \
    --output "${FINALDIR}" \
    --config "${CONFIG}" \
    --subset "scene=${SCENE}" "traj=${TRAJ}" \
    --tmp_folder "${OUTDIR}/tmp" \
    --n_workers 2 --parallel_mode multiprocess --cpus_per_worker 2

# scratch is kept on failure so the render can be repacked instead of re-rendered
PACK_STATUS=$?
if [ "${PACK_STATUS}" -ne 0 ]; then
    echo "ERROR: cvdpack pack failed (exit ${PACK_STATUS}) for ${SHARD}"
    emit_state "pack_failed"
    exit 1
fi

rm -rf "${OUTDIR}"
echo "packed ${SHARD} -> ${FINALDIR}/${SHARD}"

trap - TERM
emit_state "completed"
