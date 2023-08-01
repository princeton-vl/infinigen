#!/bin/bash
cwd=$(pwd)
cd ../..
python -m tools.manage_datagen_jobs --output_folder "$1" \
--num_scenes "$2" --pipeline_configs slurm monocular cuda_terrain gt_test \
--pipeline_overrides LocalScheduleHandler.use_gpu=True

cd - 
cp test_infinigen.py "$1/test_infinigen.py"
cp conftest.py "$1/conftest.py" 

cd "$1" && mkdir -p test_results && pytest -v -s --dir "$1" --num "$2" --days "$3" >| "test_results/logs.log" 
cd "$1" && rm -rf test_infinigen.py conftest.py __pycache__ .pytest_cache

