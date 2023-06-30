from pathlib import Path
import subprocess
import gin
import os

def check_exists(folder_path):

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
    data = run_command(f"recurse ON; prompt OFF; cd {remote_path.parent}; mget {remote_path.name}")
    assert data.returncode == 0

def listdir(remote_path):
        raise FileNotFoundError(remote_path)
    data = run_command_stdout(f'ls {remote_path}\*')

def run_command_stdout(command):
    smb_str = os.environ['SMB_AUTH']

    smb_str = os.environ['SMB_AUTH']
    with Path('/dev/null').open('w') as devnull:
        return subprocess.run(f'smbclient {smb_str} -c "{command}"', 

