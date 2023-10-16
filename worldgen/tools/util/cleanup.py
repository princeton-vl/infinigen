# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Lahav Lipson
# - Karhan Kayan (cleanup fluid files)


import argparse
import shutil
from pathlib import Path

FILES_TO_DELETE = [
    "*.mtl",
    "*.attr",
    "*.normal",
    "*.water_dist",
    "*.obj",
    "*.glb",
    "*.altitude",
    "*.pkl",
    "*.blend",
    "*.blend1",
    "assets/",
]


def cleanup(folder, verbose=False, skip_coarse=False):
    if not verbose:
        print(f"Cleaning up {folder}")

    def check_delete(filepath):
        if skip_coarse and "coarse" in str(file_path):
            return
        elif file_path.is_file() or file_path.is_symlink():
            if verbose:
                print(f"Removing {file_path}")
            file_path.unlink()
        elif file_name_to_del.endswith("/"):  # Just extra precaution
            if verbose:
                print(f"Removing {file_path}")
            shutil.rmtree(file_path)

    for file_name_to_del in FILES_TO_DELETE:
        for file_path in sorted(folder.rglob(file_name_to_del)):
            check_delete(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--fluid_cache", action="store_true")
    parser.add_argument("--skip_coarse", action="store_true")
    args = parser.parse_args()
    folder = args.folder.resolve()
    assert folder.exists() and folder.is_dir()
    if args.fluid_cache:
        FILES_TO_DELETE += ["*.wwp", "*.bobj"]
    cleanup(folder, verbose=args.verbose, skip_coarse=args.skip_coarse)
