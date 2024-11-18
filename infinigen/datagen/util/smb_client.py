# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson: everything except noted below
# - Alex Raistrick: dest_folder options, warnings, SMB_AUTH envvar change


import argparse
import logging
import os
import re
import subprocess
import time
import types
from itertools import product
from multiprocessing import Pool
from pathlib import Path

try:
    import submitit
except ImportError:
    logging.warning(
        f"Failed to import submitit, {Path(__file__).name} will crash if slurm job is requested"
    )
    submitit = None

from tqdm import tqdm

logger = logging.getLogger(__file__)
SMB_AUTH_VARNAME = "SMB_AUTH"

if SMB_AUTH_VARNAME not in os.environ:
    logging.warning(
        f"{SMB_AUTH_VARNAME} envvar is not set, smb_client upload "
        "will not work. Ignore this message if not using upload"
    )

_SMB_RATELIMIT_DELAY = 0.0


def check_exists(folder_path: Path):
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


def pathlib_to_smb(p: Path):
    p = str(p).replace("/", "\\")
    if not p.endswith("\\"):
        p += "\\"
    return p


def remove(remote_path: Path):
    run_command(
        f"recurse ON; cd {pathlib_to_smb(remote_path.parent)}; deltree {remote_path.name}"
    )


def download(remote_path: Path, dest_folder=None, verbose=False):
    assert " " not in str(remote_path), remote_path

    assert isinstance(remote_path, Path)
    if not check_exists(remote_path):
        raise FileNotFoundError(remote_path)

    statements = [
        f"cd {pathlib_to_smb(remote_path.parent)}",
        "recurse ON",
        "prompt OFF",
    ]

    if dest_folder is not None:
        dest_folder.mkdir(exist_ok=True, parents=True)
        statements.append(f"lcd {str(dest_folder)}")
        print(f"Downloading {remote_path} to {dest_folder}")
    else:
        print(f"Downloading {remote_path} to working directory")

    statements.append(f"mget {remote_path.name}")

    command = str.join("; ", statements)

    if verbose:
        print(command)
    data = run_command(command, verbose=verbose)

    if dest_folder:
        dest_path = dest_folder / remote_path.name
    else:
        dest_path = remote_path.name

    assert data.returncode == 0

    return dest_path


def yield_dirfiles(data, extras, parent):
    for line in data.splitlines():
        if "blocks of size" in line:
            continue
        parts = line.split()
        if not len(parts):
            continue
        if parts[0].startswith("."):
            continue
        parts[0] = parent / parts[0]

        if extras:
            yield parts
        else:
            yield parts[0]


def globdir(remote_path: Path, extras=False):
    remote_path = Path(remote_path)
    assert "*" in remote_path.parts[-1], remote_path

    search_path = str(remote_path).strip("/")
    search_path = search_path.replace("/", "\\")

    try:
        data = run_command_stdout(f"ls {search_path}")
    except subprocess.CalledProcessError:
        return []

    yield from yield_dirfiles(data, extras, parent=remote_path.parent)


def listdir(remote_path: Path, extras=False):
    """
    Args: str or Path
    Returns [(path, is_dir), ...]
    """

    search_path = str(remote_path).strip("/")
    search_path = search_path.replace("/", "\\")

    if "*" in search_path:
        raise ValueError(f'Found "*" in {search_path=}, use smb_client.globdir instead')

    if len(search_path) > 0 and not check_exists(search_path):
        raise FileNotFoundError(search_path)
    search_path += "\\*"

    data = run_command_stdout(f"ls {search_path}")
    yield from yield_dirfiles(data, extras, parent=remote_path)


def disk_usage(remote_path: Path):
    data = run_command_stdout(f"recurse ON; du {remote_path}")
    n_blocks, block_size, blocks_avail, file_bytes = map(int, re.findall(r"\d+", data))

    gb_file = file_bytes / 1024**3
    msg = f"{gb_file:.2f} GB"
    return msg


def disk_free():
    data = run_command_stdout("du infinigen")
    data = data.splitlines()[1].strip()
    n_blocks, block_size, blocks_avail = re.findall(r"\d+", data)

    gb_used = int(n_blocks) * int(block_size) / 1024**3
    gb_free = int(blocks_avail) * int(block_size) / 1024**3
    msg = f"Used: {gb_used:.2f} GB, Free: {gb_free:.2f} GB, Total: {gb_used + gb_free:.2f} GB"
    return msg


def run_command_stdout(command: str):
    smb_str = os.environ[SMB_AUTH_VARNAME]
    time.sleep(_SMB_RATELIMIT_DELAY)
    return subprocess.check_output(
        f'smbclient {smb_str} -c "{command}"', text=True, shell=True
    )


def run_command(command: str, check=True, verbose=False):
    smb_str = os.environ[SMB_AUTH_VARNAME]

    time.sleep(_SMB_RATELIMIT_DELAY)

    with Path("/dev/null").open("w") as devnull:
        outstream = None if verbose else devnull
        return subprocess.run(
            f'smbclient {smb_str} -c "{command}"',
            shell=True,
            stderr=outstream,
            stdout=outstream,
            check=check,
        )


def list_files_recursive(base_path: Path):
    """
    Args: str or Path
    Returns [path, ...]
    """
    all_paths = []
    children = listdir(base_path)
    for child, is_dir in children:
        if is_dir:
            all_paths.extend(list_files_recursive(child))
        else:
            all_paths.append(child)
    return all_paths


def mapfunc(f, its, args):
    if args.n_workers == 1:
        return [f(i) for i in its]
    elif args.slurm:
        if submitit is None:
            raise ValueError("submitit not imported, cannot use --slurm")

        executor = submitit.AutoExecutor(folder=args.local_path / "logs")
        executor.update_parameters(
            name=args.local_path.name,
            timeout_min=48 * 60,
            cpus_per_task=8,
            mem_gb=8,
            slurm_partition=os.environ["INFINIGEN_SLURMPARTITION"],
            slurm_array_parallelism=args.n_workers,
        )
        executor.map_array(f, its)
    else:
        with Pool(args.n_workers) as p:
            return list(tqdm(p.imap(f, its), total=len(its)))


def resolve_globs(p: Path, args):
    def resolved(parts):
        if any(x in str(p) for x in args.exclude):
            return

        first_glob = next((i for i, pp in enumerate(parts) if "*" in pp), None)
        if first_glob is None:
            yield p
        else:
            curr_level = p.parts[: first_glob + 1]
            remainder = p.parts[first_glob + 1 :]
            for child in globdir(Path(*curr_level)):
                yield from resolve_globs(child / Path(*remainder), args)

    if args.command == "glob":
        before, after = p.parts[:-1], p.parts[-1:]
        for f in resolved(before):
            yield f / Path(*after)
    else:
        yield from resolved(p.parts)


commands = {
    "ls": listdir,
    "glob": globdir,
    "rm": remove,
    "download": download,
    "upload": upload,
    "mkdir": mkdir,
    "exists": check_exists,
    "du": disk_usage,
    "df": disk_free,
}


def process_one(p: list[Path]):
    p = [Path(pi) for pi in p]

    if args.command not in commands:
        raise ValueError(
            f"Unrecognized command {args.command}, options are {commands.keys()}"
        )

    res = commands[args.command](*p)

    p_summary = " ".join(str(pi) for pi in p)

    def result(r):
        if args.verbose:
            print(f"{p_summary} {r}")
        else:
            print(r)

    if isinstance(res, types.GeneratorType):
        for r in res:
            result(r)
    else:
        result(res)


def main(args):
    n_globs = len([x for x in args.paths if "*" in str(x)])
    if n_globs > 1:
        raise ValueError(f"{args.paths=} had  {n_globs=}, only equipped to handle 1")

    paths = [resolve_globs(p, args) for p in args.paths]

    targets = list(product(*paths))

    mapfunc(process_one, targets, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=list(commands.keys()))
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument("--exclude", type=str, nargs="+", default=[])
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
