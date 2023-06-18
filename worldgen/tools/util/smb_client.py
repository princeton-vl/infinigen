# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson

# Date Signed: May 2 2023

from pathlib import Path
import subprocess
import gin
import os
import re

def check_exists(folder_path):
    folder_path = str(folder_path).strip("/")
    return run_command(f"ls {folder_path}", False).returncode == 0

def mkdir(folder_path: Path):
    assert isinstance(folder_path, Path)
    for path in list(reversed(folder_path.parents))[1:] + [folder_path]:
        run_command(f"mkdir {path}")

def upload(local_path: Path, dest_folder: Path):
    assert isinstance(local_path, Path) and isinstance(dest_folder, Path)
    assert local_path.exists()
    mkdir(dest_folder)
    data = run_command(f"put {local_path} {dest_folder / local_path.name}")
    assert data.returncode == 0

def remove(remote_path: Path):
    run_command(f"deltree {remote_path}")

def download(remote_path: Path):
    assert isinstance(remote_path, Path)
    if not check_exists(remote_path):
        raise FileNotFoundError(remote_path)
    dest_path = Path.cwd().resolve() / remote_path.name
    print(f"Downloading to {dest_path}")
    data = run_command(f"recurse ON; prompt OFF; cd {remote_path.parent}; mget {remote_path.name}")
    assert data.returncode == 0
    return dest_path

def listdir(remote_path):
    """
    Args: str or Path
    Returns [(path, is_dir), ...]
    """
    remote_path = str(remote_path).strip("/")
    if len(remote_path) > 0 and not check_exists(remote_path):
        raise FileNotFoundError(remote_path)
    data = run_command_stdout(f'ls {remote_path}\*')
    ls_regex = re.compile(" *([A-Za-z0-9_\.]+) +([AD]) .*").fullmatch
    output = [ls_regex(l).groups() for l in data.splitlines() if ls_regex(l)]
    return sorted((remote_path / Path(p), typ == 'D') for p, typ in output if (p not in {'.', '..'}))

def run_command_stdout(command):
    smb_str = os.environ['SMB_AUTH']
    return subprocess.check_output(f'smbclient {smb_str} -c "{command}"', text=True, shell=True)

def run_command(command, check=True):
    smb_str = os.environ['SMB_AUTH']
    with Path('/dev/null').open('w') as devnull:
        return subprocess.run(f'smbclient {smb_str} -c "{command}"', 
            shell=True, stderr=devnull, stdout=devnull, check=check)

def list_files_recursive(base_path):
    """
    Args: str or Path
    Returns [path, ...]
    """
    all_paths=[]
    children = listdir(base_path)
    for child, is_dir in children:
        if is_dir:
            all_paths.extend(list_files_recursive(child))
        else:
            all_paths.append(child)
    return all_paths