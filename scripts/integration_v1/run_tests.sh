#!/bin/bash
cwd=$(pwd)
cd ../..
python -m infinigen.datagen.manage_jobs --output_folder "$1" \
--num_scenes "$2" --pipeline_configs slurm monocular cuda_terrain gt_test \
--pipeline_overrides LocalScheduleHandler.use_gpu=True

cd - 
cd "$1" && mkdir -p test_results

cd -
python manual_integration_check.py  --dir "$1" --num "$2" --time "$3"  >| "$1/test_results/logs.log" 
