# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Lingjie Mei


import imageio
from pathlib import Path
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('folder', type=Path, nargs='+')
parser.add_argument('thresh', type=float, default=0.05)


def main(thresh, folder):
    assert thresh > 1, f'Images are 0-255 you probably didnt want {thresh=}'

    for folder in folder:
        out_folder = folder.parent / (folder.stem + f'_thresh_{thresh}')
        out_folder.mkdir(exist_ok=True, parents=True)

        for imgpath in folder.iterdir():
            try:
                img = imageio.imread(imgpath)
            except:
                continue

            pixs = img.reshape(-1, 4)
            mask = pixs[:, -1] < thresh
            pixs[mask] = 0
            img = pixs.reshape(img.shape)

            print(f'Stripped {100 * mask.mean()}% from {imgpath}')
            imageio.imwrite(out_folder / imgpath.name, img)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.thresh, args.folder)
