import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=Path, nargs='+')
parser.add_argument('--output_folder', type=Path, default=None)
parser.add_argument('--image_type', default='Image')
parser.add_argument('--overlay', type=int, default=1)
parser.add_argument('--join', type=int, default=1)
args = parser.parse_args()

for input_folder in args.input_folder:

    if not input_folder.is_dir():
        continue

    if args.output_folder is None:
        output_folder = input_folder
    else:
        output_folder = args.output_folder
        output_folder.mkdir()

    for seed_folder in input_folder.iterdir():
        if not seed_folder.is_dir():
            continue
        if len(list(seed_folder.glob('frames*'))) == 0:
            print(f'{seed_folder=} has no frames*')
            continue
        filters = " "
        if args.overlay:
            filters += f"-vf drawtext='text={seed_folder.absolute()}' "
        print(cmd.split())
        subprocess.run(cmd.split())

    if args.join:
        cat_output = output_folder/f'{output_folder.name}_{args.image_type}.mp4'
        videos = [x for x in output_folder.glob(f'*_{args.image_type}.mp4') if x != cat_output ]

        instructions = (output_folder/'videos.txt')
        instructions.write_text('\n'.join([f"file '{x.absolute()}'" for x in videos]))
        
        cmd = f"ffmpeg -y -f concat -safe 0 -i {instructions} -c copy {cat_output}"
        subprocess.run(cmd.split())
        instructions.unlink()

        print(cat_output.absolute())
