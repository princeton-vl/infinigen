#!/bin/bash

# Exit if anything fails
set -e

# Check required arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <blend_path> <blend_obj_name> [--output_name NAME] [--output_dir DIR] [--ignore_from_catalog]"
  exit 1
fi

BLENDER_PATH="$1"
OBJECT_NAME="$2"
shift 2  # Remove first two required arguments

# Initialize optional args
OUTPUT_NAME=""
OUTPUT_DIR=""
IGNORE_FLAG=""

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_name)
      OUTPUT_NAME="--output_name $2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="--output_dir $2"
      shift 2
      ;;
    --ignore_from_catalog)
      IGNORE_FLAG="--ignore_from_catalog"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Construct and run command
CMD="python scripts/transpile_sim.py -bp \"$BLENDER_PATH\" -bon \"$OBJECT_NAME\" $OUTPUT_NAME $OUTPUT_DIR $IGNORE_FLAG"

echo "Running command:"
echo "$CMD"

# Execute
eval "$CMD"
