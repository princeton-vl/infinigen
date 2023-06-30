import argparse
import shutil
from pathlib import Path

FILES_TO_DELETE = ["*.mtl", "*.attr", "*.normal", "*.water_dist", "*.obj", "*.glb",

def cleanup(folder, verbose=False):
    if not verbose:
        print(f"Cleaning up {folder}")
    for file_name_to_del in FILES_TO_DELETE:
        for file_path in sorted(folder.rglob(file_name_to_del)):
                if verbose:
                    print(f"Removing {file_path}")
                file_path.unlink()
            elif file_name_to_del.endswith('/'): # Just extra precaution
                if verbose:
                    print(f"Removing {file_path}")
                shutil.rmtree(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    folder = args.folder.resolve()
    assert folder.exists() and folder.is_dir()
