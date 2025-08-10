#!/bin/bash

# Required arguments
ASSET=$1
SEEDS=$2
EXP=$3

# Optional flags
C_FLAG=""
GIN_FILE=""

# Shift past required args
shift 3

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        -c)
            C_FLAG="-c"
            shift
            ;;
        -g)
            GIN_FILE=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Debugging starting commands
echo "Checking GPU status"
nvidia-smi

for ((i=1001; i<1001+SEEDS; i++)); do
    CMD="python scripts/spawn_asset.py -exp $EXP -n $ASSET -s $i $C_FLAG"
    if [[ -n "$GIN_FILE" ]]; then
        CMD="$CMD --gin_config=$GIN_FILE"
    fi
    eval $CMD
done
