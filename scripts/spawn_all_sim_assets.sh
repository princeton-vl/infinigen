#!/bin/bash

mkdir -p logs

assets=("door" 
        "toaster" 
        "multifridge" 
        "multidoublefridge" 
        "dishwasher" 
        "lamp")
exporters=("mjcf" "usd" "urdf")

for asset in "${assets[@]}"; do
  for exporter in "${exporters[@]}"; do
    log_file="logs/${asset}_${exporter}.log"
    echo "Spawning $asset with $exporter (logging to $log_file)"
    ./scripts/spawn_sim_ready_asset.sh "$asset" 3 "$exporter" > "$log_file" 2>&1 &
  done
done

wait
echo "All spawn_sim_ready_asset.sh jobs completed."
