# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


from pathlib import Path
import subprocess
import os
import shutil

def listdir(remote_path):
    stdout = subprocess.check_output(f"{shutil.which('rclone')} lsf infinigen_renders:{remote_path}/".split(), text=True)
    return sorted((Path(remote_path) / l) for l in stdout.splitlines())

def download(remote_path, local_folder):
    assert os.path.exists(local_folder) and os.path.isdir(local_folder)
    dest_path = os.path.join(local_folder, os.path.basename(remote_path))
    print(f"Downloading to {dest_path}")
    with Path('/dev/null').open('w') as devnull:
        subprocess.run(f'{shutil.which("rclone")} copy infinigen_renders:{remote_path} {local_folder}/',
        shell=True, check=True, stderr=devnull, stdout=devnull)
    return dest_path