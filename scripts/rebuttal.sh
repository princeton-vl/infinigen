python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/rebuttal_figure/kitchen_0 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/rebuttal_figure/kitchen_1 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 2 --task coarse --output_folder outputs/rebuttal_figure/kitchen_2 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 3 --task coarse --output_folder outputs/rebuttal_figure/kitchen_3 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 4 --task coarse --output_folder outputs/rebuttal_figure/kitchen_4 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 5 --task coarse --output_folder outputs/rebuttal_figure/kitchen_5 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 6 --task coarse --output_folder outputs/rebuttal_figure/kitchen_6 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 7 --task coarse --output_folder outputs/rebuttal_figure/kitchen_7 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 8 --task coarse --output_folder outputs/rebuttal_figure/kitchen_8 -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  &

python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/rebuttal_figure/bathroom_0 -p compose_indoors.room_tags=[\"bathroom\"]  --configs overhead_singleroom & 
python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/rebuttal_figure/bathroom_1 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 2 --task coarse --output_folder outputs/rebuttal_figure/bathroom_2 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 3 --task coarse --output_folder outputs/rebuttal_figure/bathroom_3 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 4 --task coarse --output_folder outputs/rebuttal_figure/bathroom_4 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 5 --task coarse --output_folder outputs/rebuttal_figure/bathroom_5 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 6 --task coarse --output_folder outputs/rebuttal_figure/bathroom_6 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 7 --task coarse --output_folder outputs/rebuttal_figure/bathroom_7 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 8 --task coarse --output_folder outputs/rebuttal_figure/bathroom_8 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  &

wait $(jobs -ps)

python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/rebuttal_figure/dining_0 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/rebuttal_figure/dining_1 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 2 --task coarse --output_folder outputs/rebuttal_figure/dining_2 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 3 --task coarse --output_folder outputs/rebuttal_figure/dining_3 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 4 --task coarse --output_folder outputs/rebuttal_figure/dining_4 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 5 --task coarse --output_folder outputs/rebuttal_figure/dining_5 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 6 --task coarse --output_folder outputs/rebuttal_figure/dining_6 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 7 --task coarse --output_folder outputs/rebuttal_figure/dining_7 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 8 --task coarse --output_folder outputs/rebuttal_figure/dining_8 -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  &

python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/rebuttal_figure/living_0 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_0/logs.txt &
python -m infinigen_examples.generate_indoors --seed 1 --task coarse --output_folder outputs/rebuttal_figure/living_1 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_1/logs.txt &
python -m infinigen_examples.generate_indoors --seed 2 --task coarse --output_folder outputs/rebuttal_figure/living_2 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_2/logs.txt &
python -m infinigen_examples.generate_indoors --seed 3 --task coarse --output_folder outputs/rebuttal_figure/living_3 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_3/logs.txt &
python -m infinigen_examples.generate_indoors --seed 4 --task coarse --output_folder outputs/rebuttal_figure/living_4 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_4/logs.txt &
python -m infinigen_examples.generate_indoors --seed 5 --task coarse --output_folder outputs/rebuttal_figure/living_5 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_5/logs.txt &
python -m infinigen_examples.generate_indoors --seed 6 --task coarse --output_folder outputs/rebuttal_figure/living_6 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_6/logs.txt &
python -m infinigen_examples.generate_indoors --seed 7 --task coarse --output_folder outputs/rebuttal_figure/living_7 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_7/logs.txt &
python -m infinigen_examples.generate_indoors --seed 8 --task coarse --output_folder outputs/rebuttal_figure/living_8 -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  > outputs/rebuttal_figure/living_8/logs.txt &

wait $(jobs -ps)

python -m infinigen_examples.generate_indoors --seed 0 --task render --input_folder outputs/rebuttal_figure/kitchen_0 --output_folder outputs/rebuttal_figure/kitchen_0  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 1 --task render --input_folder outputs/rebuttal_figure/kitchen_1 --output_folder outputs/rebuttal_figure/kitchen_1  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 2 --task render --input_folder outputs/rebuttal_figure/kitchen_2 --output_folder outputs/rebuttal_figure/kitchen_2  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 3 --task render --input_folder outputs/rebuttal_figure/kitchen_3 --output_folder outputs/rebuttal_figure/kitchen_3  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/kitchen_4 --output_folder outputs/rebuttal_figure/kitchen_4  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 5 --task render --input_folder outputs/rebuttal_figure/kitchen_5 --output_folder outputs/rebuttal_figure/kitchen_5  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 6 --task render --input_folder outputs/rebuttal_figure/kitchen_6 --output_folder outputs/rebuttal_figure/kitchen_6  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 7 --task render --input_folder outputs/rebuttal_figure/kitchen_7 --output_folder outputs/rebuttal_figure/kitchen_7  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 8 --task render --input_folder outputs/rebuttal_figure/kitchen_8 --output_folder outputs/rebuttal_figure/kitchen_8  -p compose_indoors.room_tags=[\"kitchen\"] --configs overhead_singleroom  

python -m infinigen_examples.generate_indoors --seed 0 --task render --input_folder outputs/rebuttal_figure/bathroom_0 --output_folder outputs/rebuttal_figure/bathroom_0  -p compose_indoors.room_tags=[\"bathroom\"]  --configs overhead_singleroom
python -m infinigen_examples.generate_indoors --seed 1 --task render --input_folder outputs/rebuttal_figure/bathroom_1 --output_folder outputs/rebuttal_figure/bathroom_1  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 2 --task render --input_folder outputs/rebuttal_figure/bathroom_2 --output_folder outputs/rebuttal_figure/bathroom_2  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 3 --task render --input_folder outputs/rebuttal_figure/bathroom_3 --output_folder outputs/rebuttal_figure/bathroom_3  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/bathroom_4 --output_folder outputs/rebuttal_figure/bathroom_4  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 5 --task render --input_folder outputs/rebuttal_figure/bathroom_5 --output_folder outputs/rebuttal_figure/bathroom_5  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 6 --task render --input_folder outputs/rebuttal_figure/bathroom_6 --output_folder outputs/rebuttal_figure/bathroom_6  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 7 --task render --input_folder outputs/rebuttal_figure/bathroom_7 --output_folder outputs/rebuttal_figure/bathroom_7 -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 8 --task render --input_folder outputs/rebuttal_figure/bathroom_8 --output_folder outputs/rebuttal_figure/bathroom_8  -p compose_indoors.room_tags=[\"bathroom\"] --configs overhead_singleroom  

python -m infinigen_examples.generate_indoors --seed 0 --task render --input_folder outputs/rebuttal_figure/dining_0 --output_folder outputs/rebuttal_figure/dining_0  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom &
python -m infinigen_examples.generate_indoors --seed 1 --task render --input_folder outputs/rebuttal_figure/dining_1 --output_folder outputs/rebuttal_figure/dining_1  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom & 
python -m infinigen_examples.generate_indoors --seed 2 --task render --input_folder outputs/rebuttal_figure/dining_2 --output_folder outputs/rebuttal_figure/dining_2  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom & 
python -m infinigen_examples.generate_indoors --seed 3 --task render --input_folder outputs/rebuttal_figure/dining_3 --output_folder outputs/rebuttal_figure/dining_3  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom 
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/dining_4 --output_folder outputs/rebuttal_figure/dining_4  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom 
python -m infinigen_examples.generate_indoors --seed 5 --task render --input_folder outputs/rebuttal_figure/dining_5 --output_folder outputs/rebuttal_figure/dining_5  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom 
python -m infinigen_examples.generate_indoors --seed 6 --task render --input_folder outputs/rebuttal_figure/dining_6 --output_folder outputs/rebuttal_figure/dining_6  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 7 --task render --input_folder outputs/rebuttal_figure/dining_7 --output_folder outputs/rebuttal_figure/dining_7  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 8 --task render --input_folder outputs/rebuttal_figure/dining_8 --output_folder outputs/rebuttal_figure/dining_8  -p compose_indoors.room_tags=[\"dining-room\"] --configs overhead_singleroom  

python -m infinigen_examples.generate_indoors --seed 0 --task render --input_folder outputs/rebuttal_figure/living_0 --output_folder outputs/rebuttal_figure/living_0  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 1 --task render --input_folder outputs/rebuttal_figure/living_1 --output_folder outputs/rebuttal_figure/living_1  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 2 --task render --input_folder outputs/rebuttal_figure/living_2 --output_folder outputs/rebuttal_figure/living_2  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  &
python -m infinigen_examples.generate_indoors --seed 3 --task render --input_folder outputs/rebuttal_figure/living_3 --output_folder outputs/rebuttal_figure/living_3  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/living_4 --output_folder outputs/rebuttal_figure/living_4  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/living_4 --output_folder outputs/rebuttal_figure/living_5  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/living_4 --output_folder outputs/rebuttal_figure/living_6  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/living_4 --output_folder outputs/rebuttal_figure/living_7  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  
python -m infinigen_examples.generate_indoors --seed 4 --task render --input_folder outputs/rebuttal_figure/living_4 --output_folder outputs/rebuttal_figure/living_8  -p compose_indoors.room_tags=[\"living-room\"] --configs overhead_singleroom  