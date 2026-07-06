#!/bin/bash
#SBATCH --job-name=if2dataset
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=1-100
#SBATCH --output=outputs/renderjobs/%A_%a.out

# Renders one seed per array task. Customize OUTPUT_PATH below to suit your
# cluster, and adjust --partition/--account/--time/--mem/--gres/--array above.

SEED="${SLURM_ARRAY_TASK_ID}"
OUTPUT_PATH="outputs/${SLURM_ARRAY_JOB_ID}/${SEED}"
uv run python render_floatingobj_stereo.py --seed "${SEED}" --output "${OUTPUT_PATH}"
