#!/bin/bash

OUTPUT_PATH=$1
shift

if [ -z "$OUTPUT_PATH" ]; then
    echo "Please provide an output path"
    exit 1
fi

# make outputs group-writable
umask 0002

# Initialize arrays to hold the arguments
arg1_strings=()
arg2_strings=()
arg3_strings=()

# Parse the command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --nature_configs)
      shift  # Move past the option
      # Collect arguments until the next option or end of input
      while [[ $# -gt 0 && $1 != --* ]]; do
        arg1_strings+=("$1")
        shift
      done
      ;;
    --indoors_configs)
      shift  # Move past the option
      # Collect arguments until the next option or end of input
      while [[ $# -gt 0 && $1 != --* ]]; do
        arg2_strings+=("$1")
        shift
      done
      ;;
    --pipeline_configs)
      shift  # Move past the option
      # Collect arguments until the next option or end of input
      while [[ $# -gt 0 && $1 != --* ]]; do
        arg3_strings+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done


# Combine the arrays into single strings
parsed_nature_configs="${arg1_strings[*]}"
parsed_indoors_configs="${arg2_strings[*]}"
parsed_pipeline_configs="${arg3_strings[*]}"

# Environment Variables for Opting In/Out
RUN_INDOOR=${RUN_INDOOR:-1}
RUN_NATURE=${RUN_NATURE:-1}
RUN_OBJECTS=${RUN_OBJECTS:-1}
RUN_MATERIALS=${RUN_MATERIALS:-1}

# Version Info
INFINIGEN_VERSION=$(python -c "import infinigen; print(infinigen.__version__)")
COMMIT_HASH=$(git rev-parse HEAD | cut -c 1-6) 
DATE=$(date '+%Y-%m-%d')
JOBTAG="${DATE}_ifg-int"
BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/_/-/g; s|/|-|g; s/\//_/g')
VERSION_STRING="${DATE}_${BRANCH}_${COMMIT_HASH}_${USER}"

mkdir -p $OUTPUT_PATH
OUTPUT_PATH=$OUTPUT_PATH/$VERSION_STRING

# Run Indoor Scene Generation
if [ "$RUN_INDOOR" -eq 1 ]; then
    for indoor_type in DiningRoom Bathroom Bedroom Kitchen LivingRoom; do
        python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_scene_indoor_$indoor_type \
        --num_scenes 3 --cleanup big_files --configs singleroom.gin fast_solve.gin $parsed_indoors_configs --overwrite \
        --pipeline_configs slurm.gin monocular.gin blender_gt.gin indoor_background_configs.gin $parsed_pipeline_configs \
        --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors sample_scene_spec.seed_range=[0,100] slurm_submit_cmd.slurm_nodelist=$NODECONF \
        --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"$indoor_type\"\] compose_indoors.solve_small_enabled=False &
    done
fi

# Run Nature Scene Generation
if [ "$RUN_NATURE" -eq 1 ]; then
    for nature_type in arctic canyon cave coast coral_reef desert forest kelp_forest mountain plain river snowy_mountain under_water; do
        python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_scene_nature_$nature_type \
        --num_scenes 3 --cleanup big_files --overwrite \
        --configs $nature_type.gin dev.gin $parsed_nature_configs \
        --pipeline_configs slurm.gin monocular.gin blender_gt.gin $parsed_pipeline_configs \
        --pipeline_overrides sample_scene_spec.seed_range=[0,100] &
    done
fi

if [ -n "$parsed_nature_configs" ]; then
    parsed_nature_configs="--configs $parsed_nature_configs "
fi

if [ -n "$parsed_indoors_configs" ]; then
    parsed_indoors_configs="--configs $parsed_indoors_configs "
fi

# Objects
if [ "$RUN_OBJECTS" -eq 1 ]; then

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_nature_meshes.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_nature_meshes \
    $parsed_nature_configs \
    --slurm --n_workers 100 -n 3 --gpu & 

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_indoor_meshes.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_indoor_meshes \
    $parsed_indoors_configs \
    --slurm --n_workers 100 -n 3 --gpu &
fi

# Materials
if [ "$RUN_MATERIALS" -eq 1 ]; then

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_materials.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_new_materials $parsed_indoors_configs \
    --slurm --n_workers 100 -n 3 --gpu & 


    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_materials_deprecated_interface.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_deprec_materials $parsed_nature_configs \
    --slurm --n_workers 100 -n 3 --gpu &
fi

# Wait for all background processes to finish
wait