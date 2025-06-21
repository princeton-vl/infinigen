# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

import argparse
import time

import pybullet as p
import pybullet_data

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, required=True, help="path to the urdf asset"
)
args = parser.parse_args()

# Connect to PyBullet
p.connect(p.GUI)

# Set the simulation environment
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load an asset (URDF file)
asset_id = p.loadURDF(args.path, basePosition=[0, 0, 1])

# Run the simulation
while True:
    # p.stepSimulation()
    time.sleep(1 / 240)  # Maintain real-time simulation speed


# Disconnect (not reachable due to infinite loop)
p.disconnect()
