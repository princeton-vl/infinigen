OUTPUT_PATH=$1

if [ -z "$OUTPUT_PATH" ]; then
    echo "Please provide an output path"
    exit 1
fi

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
BRANCH=$(git rev-parse --abbrev-ref HEAD | sed 's/_/-/g')
VERSION_STRING="${DATE}_${BRANCH}_${COMMIT_HASH}_${USER}"

mkdir -p $OUTPUT_PATH
OUTPUT_PATH=$OUTPUT_PATH/$VERSION_STRING

# Run Indoor Scene Generation
if [ "$RUN_INDOOR" -eq 1 ]; then
    for indoor_type in DiningRoom Bathroom Bedroom Kitchen LivingRoom; do
        python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_scene_indoor_$indoor_type \
        --num_scenes 3 --cleanup big_files --configs singleroom.gin fast_solve.gin --overwrite \
        --pipeline_configs slurm_1h monocular indoor_background_configs.gin \
        --pipeline_overrides get_cmd.driver_script=infinigen_examples.generate_indoors sample_scene_spec.seed_range=[0,100] slurm_submit_cmd.slurm_nodelist=$NODECONF \
        --overrides compose_indoors.terrain_enabled=True restrict_solving.restrict_parent_rooms=\[\"$indoor_type\"\] &
    done
fi

# Run Nature Scene Generation
if [ "$RUN_NATURE" -eq 1 ]; then
    for nature_type in arctic canyon cave coast coral_reef desert forest kelp_forest mountain plain river snowy_mountain under_water; do
        python -m infinigen.datagen.manage_jobs --output_folder $OUTPUT_PATH/${JOBTAG}_scene_nature_$nature_type \
        --num_scenes 3 --cleanup big_files --overwrite \
        --configs $nature_type.gin dev.gin \
        --pipeline_configs slurm_1h monocular \
        --pipeline_overrides sample_scene_spec.seed_range=[0,100] &
    done
fi

# Objects
if [ "$RUN_OBJECTS" -eq 1 ]; then

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_nature_meshes.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_nature_meshes \
    --slurm --n_workers 100 -n 3 --gpu & 

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_indoor_meshes.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_indoor_meshes \
    --slurm --n_workers 100 -n 3 --gpu &
fi

# Materials
if [ "$RUN_MATERIALS" -eq 1 ]; then

    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_indoor_materials.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_indoor_materials \
    --slurm --n_workers 100 -n 3 --gpu & 


    python -m infinigen_examples.generate_individual_assets \
    -f tests/assets/list_nature_materials.txt --output_folder $OUTPUT_PATH/${JOBTAG}_asset_nature_materials \
    --slurm --n_workers 100 -n 3 --gpu &
fi

# Wait for all background processes to finish
wait