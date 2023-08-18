## Copyright (c) Princeton University.
## This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

## Authors: Lahav Lipson


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from imageio import imread

np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=Path)
    parser.add_argument('pos', type=int, nargs='*')
    args = parser.parse_args()

    if args.input_path.suffix == ".npy":
        image = np.load(args.input_path)
    else:
        image = imread(args.input_path)

    if len(args.pos) > 0:
        assert len(args.pos) == 4
        x1,y1,x2,y2 = args.pos
        image = image[y1:y2+1, x1:x2+1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    textvar: Text = ax.text(0, -15, "", style='italic')

    def hover(event):
        if event.xdata is not None:
            x, y = round(event.xdata), round(event.ydata)
            val = image[y,x]
            if len(args.pos) > 0:
                x += x1
                y += y1
            if val.dtype in [np.float32, np.float64]:
                val = np.around(val, 3)
            textvar.set_text(f"({x}, {y}) -> {val}")
        else:
            textvar.set_text("")
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)

    plt.show()