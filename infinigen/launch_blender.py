import subprocess
import argparse
from pathlib import Path

root = Path(__file__).parent.parent

BLENDER_BINARY_RELATIVE = [
    root/"blender/blender",
    root/"Blender.app/Contents/MacOS/Blender"
]
IMPORT_INFINIGEN_SCRIPT = root/'infinigen/tools/blendscript_import_infinigen.py'
APPEND_SYSPATH_SCRIPT = root/'infinigen/tools/blendscript_path_append.py'


HEADLESS_ARGS = [
    '-noaudio',
    '--background',
]

def get_standalone_blender_path():
    try:
        return next(x for x in BLENDER_BINARY_RELATIVE if x.exists())
    except StopIteration:
        raise ValueError(
            "Could not find blender binary - please check you have completed "
            "'Infinigen as a Blender-Python script' section of docs/Installation.md" 
            f" and that one of {BLENDER_BINARY_RELATIVE} exists"
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', type=str, default=None)
    parser.add_argument('-s', '--script', type=str, default=None)
    args, unknown_args = parser.parse_known_args()

    cmd_args = [str(get_standalone_blender_path())]

    if args.module is not None:

        cmd_args += HEADLESS_ARGS

        cmd_args += [
            '--python',
            str(APPEND_SYSPATH_SCRIPT)
        ]       

        relpath = '/'.join(args.module.split('.')) + '.py'
        path = root/relpath
        if not path.exists():
            raise FileNotFoundError(f'Could not find python script {path}')
        
        cmd_args += ['--python', str(path)]
        
    elif args.script is not None:
        cmd_args += HEADLESS_ARGS + ['--python', args.script]
    else:
        cmd_args += [
            '--python',
            str(IMPORT_INFINIGEN_SCRIPT)
        ]        

    if len(unknown_args):
        cmd_args += unknown_args

    print(' '.join(cmd_args))

    subprocess.run(cmd_args, cwd=root)