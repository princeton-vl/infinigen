# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Karhan Kayan

import submitit
import argparse
from pathlib import Path
import sys
import os
import time 

sys.path.append(str(Path(os.path.split(os.path.abspath(__file__))[0]) / ".."))


def get_slurm_banned_nodes(config_path=None):
    if config_path is None:
        return []
    with Path(config_path).open('r') as f:
        return list(f.read().split())
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--asset_folder', type=str)
parser.add_argument('-a', '--assets', nargs='+', default=[
    'CachedBushFactory',
    'CachedTreeFactory',
    'CachedCactusFactory',
    'CachedCreatureFactory',
    'CachedBoulderFactory'
])
parser.add_argument('-n', '--number', type=int, default=1)
parser.add_argument('-s', '--start_frame', type=int, default=-20)
parser.add_argument('-d', '--simulation_duration', type=int, default=24*20+20)
# parser.add_argument('-r', '--resolution', type=int)
# parser.add_argument('--dissolve_speed', type=int, default=25)
# parser.add_argument('--dom_scale', type=int, default=1)
args = parser.parse_args()

Path(args.asset_folder).mkdir(parents=True, exist_ok=True)


for asset in args.assets:
    for i in range(args.number):
        cmd = f"python -m infinigen.asstes.fluid.run_asset_cache -f {args.asset_folder}/ -a {asset} -s {args.start_frame} -d {args.simulation_duration}".split(" ")
        print(cmd)
        executor = submitit.AutoExecutor(folder=str(Path(args.asset_folder) / "logs"))
        executor.update_parameters(
            mem_gb=16,
            name=f"{asset}_{i}",
            cpus_per_task=4,
            timeout_min=60*24,
            slurm_account="pvl",
            slurm_exclude= "node408,node409",
        )
        render_fn = submitit.helpers.CommandFunction(cmd)
        executor.submit(render_fn)
