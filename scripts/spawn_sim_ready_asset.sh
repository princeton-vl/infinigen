#!/bin/bash

ASSET=$1
SEEDS=$2
EXP=$3
C_FLAG=""

# Parse optional fourth argument
if [[ "$4" == "-c" ]]; then
    C_FLAG="-c"
fi

# Debugging starting commands
echo "Checking GPU status"
nvidia-smi

for ((i=1001; i<1001+SEEDS; i++)); do
    python scripts/spawn_asset.py \
        -exp $EXP \
        -n $ASSET \
        -s $i \
        $C_FLAG
done
