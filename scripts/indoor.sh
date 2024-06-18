#!/bin/bash


for s in $(seq 0 20)
do
    python -m infinigen_examples.generate_indoors --output_folder outputs/room_${s} -s ${s} -g base disable/no_objects -t coarse
    python -m infinigen_examples.generate_indoors --input_folder outputs/room_${s} --output_folder outputs/room_${s}/frames -s ${s} -g base disable/no_objects -t render
done
