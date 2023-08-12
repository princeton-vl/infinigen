# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: 
# - Lahav Lipson: everything except noted below
# - Alex Raistrick: dest_folder options, warnings, SMB_AUTH envvar change



from pathlib import Path
import subprocess
import gin
import os
import re
import logging

logger = logging.getLogger(__file__)
smb_auth_varname = 'SMB_AUTH'

if smb_auth_varname not in os.environ:
    logging.warning(
        f'{smb_auth_varname} envvar is not set, smb_client upload '
        'will not work. Ignore this message if not using upload'
    )

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

def download(remote_path: Path, dest_folder=None, verbose=False):
    assert isinstance(remote_path, Path)
    if not check_exists(remote_path):
        raise FileNotFoundError(remote_path)

    def pathlib_to_smb(p):
        p = str(p).replace('/', '\\')
        if not p.endswith('\\'):
            p += '\\'
        return p

    statements = [
        f"cd {pathlib_to_smb(remote_path.parent)}",
        "recurse ON",
        "prompt OFF",
    ]

    if dest_folder is not None:
        statements.append(f'lcd {str(dest_folder)}')
        print(f"Downloading {remote_path} to {dest_folder}")
    else:
        print(f'Downloading {remote_path} to working directory')

    statements.append(f"mget {remote_path.name}")
    
    command = str.join('; ', statements)

    if verbose:
        print(command)
    data = run_command(command, verbose=verbose)

    dest_path = dest_folder/remote_path.name

    assert data.returncode == 0

    return dest_path

def yield_dirfiles(data, extras, parent):

    for line in data.splitlines():
        if 'blocks of size' in line:
            continue
        parts = line.split()
        if not len(parts):
            continue
        if parts[0].startswith('.'):
            continue
        parts[0] = parent/parts[0]
        
        if extras:
            yield parts
        else:
            yield parts[0]

def globdir(remote_path, extras=False):

    remote_path = Path(remote_path)
    assert '*' in remote_path.parts[-1], remote_path

    search_path = str(remote_path).strip("/")
    search_path = search_path.replace('/', '\\')

    try:
        data = run_command_stdout(f'ls {search_path}')
    except subprocess.CalledProcessError as e:
        return []

    yield from yield_dirfiles(data, extras, parent=remote_path.parent)
    

def listdir(remote_path, extras=False):
    """
    Args: str or Path
    Returns [(path, is_dir), ...]
    """

    search_path = str(remote_path).strip("/")
    search_path = search_path.replace('/', '\\')

    if '*' in search_path:
        raise ValueError(f'Found \"*\" in {search_path=}, use smb_client.globdir instead')

    if len(search_path) > 0 and not check_exists(search_path):
        raise FileNotFoundError(search_path)
    search_path += '\\*'

    data = run_command_stdout(f'ls {search_path}')
    yield from yield_dirfiles(data, extras, parent=remote_path)

def run_command_stdout(command):
    smb_str = os.environ['SMB_AUTH']
    return subprocess.check_output(f'smbclient {smb_str} -c "{command}"', text=True, shell=True)

def run_command(command, check=True, verbose=False):
    smb_str = os.environ['SMB_AUTH']

    with Path('/dev/null').open('w') as devnull:
        outstream = None if verbose else devnull
        return subprocess.run(f'smbclient {smb_str} -c "{command}"', 
            shell=True, stderr=outstream, stdout=outstream, check=check)

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