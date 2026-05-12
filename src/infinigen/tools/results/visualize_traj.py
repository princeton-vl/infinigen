# Code from https://github.com/aharley/pips2
# MIT License

# Copyright (c) 2022 Adam W. Harley

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

sys.path.append(str(Path(__file__).parent.parent))
import argparse

import imageio
from suffixes import parse_suffix


def gif_and_tile(ims, just_gif=False):
    S = len(ims)
    # each im is B x H x W x C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im


COLORMAP_FILE = "bremm.png"


class ColorMap2d:
    def __init__(self, filename=None):
        self._colormap_file = filename or COLORMAP_FILE
        self._img = plt.imread(self._colormap_file)

        self._height = self._img.shape[0]
        self._width = self._img.shape[1]

    def __call__(self, X):
        assert len(X.shape) == 2
        output = np.zeros((X.shape[0], 3))
        for i in range(X.shape[0]):
            x, y = X[i, :]
            xp = int((self._width - 1) * x)
            yp = int((self._height - 1) * y)
            xp = np.clip(xp, 0, self._width - 1)
            yp = np.clip(yp, 0, self._height - 1)
            output[i, :] = self._img[yp, xp]
        return output


class Summ_writer(object):
    def __init__(
        self,
        writer=None,
        global_step=0,
        log_freq=10,
        fps=8,
        scalar_freq=100,
        just_gif=False,
    ):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 10000
        self.save_this = self.global_step % self.log_freq == 0
        self.scalar_freq = max(scalar_freq, 1)

    def summ_boxlist2d(
        self,
        name,
        rgb,
        boxlist,
        scores=None,
        tids=None,
        frame_id=None,
        only_return=False,
        linewidth=2,
    ):
        B, C, H, W = list(rgb.shape)
        boxlist_vis = self.draw_boxlist2d_on_image(
            rgb, boxlist, scores=scores, tids=tids, linewidth=linewidth
        )
        return self.summ_rgb(
            name, boxlist_vis, frame_id=frame_id, only_return=only_return
        )

    def summ_rgbs(
        self, name, ims, frame_ids=None, blacken_zeros=False, only_return=False
    ):
        if self.save_this:
            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims
            assert vis.dtype in {torch.uint8, torch.float32}

            B, S, C, H, W = list(vis.shape)

            if int(W) > self.maxwidth:
                vis = vis[:, :, :, : self.maxwidth]

            if only_return:
                return vis
            else:
                pass

    def draw_traj_on_image_py(
        self,
        rgb,
        traj,
        S=50,
        linewidth=1,
        show_dots=False,
        show_lines=True,
        cmap="coolwarm",
        val=None,
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgb is 3 x H x W
        # traj is S x 2

        H, W, C = rgb.shape
        assert C == 3

        rgb = rgb.astype(np.uint8).copy()

        S1, D = traj.shape
        assert D == 2

        color_map = cm.get_cmap(cmap)
        S1, D = traj.shape

        for s in range(S1):
            if val is not None:
                color = np.array(color_map(val[s])[:3]) * 255  # rgb
            else:
                if maxdist is not None:
                    val = (np.sqrt(np.sum((traj[s] - traj[0]) ** 2)) / maxdist).clip(
                        0, 1
                    )
                    color = np.array(color_map(val)[:3]) * 255  # rgb
                else:
                    color = (
                        np.array(color_map((s) / max(1, float(S - 2)))[:3]) * 255
                    )  # rgb

            if show_lines and s < (S1 - 1):
                cv2.line(
                    rgb,
                    (int(traj[s, 0]), int(traj[s, 1])),
                    (int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                    color,
                    linewidth,
                    cv2.LINE_AA,
                )
            if show_dots:
                cv2.circle(
                    rgb,
                    (int(traj[s, 0]), int(traj[s, 1])),
                    linewidth,
                    np.array(color_map(1)[:3]) * 255,
                    -1,
                )

        # if maxdist is not None:
        #     val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        #     color = np.array(color_map(val)[:3]) * 255 # rgb
        # else:
        #     # draw the endpoint of traj, using the next color (which may be the last color)
        #     color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb

        # # emphasize endpoint
        # cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)

        return rgb

    def summ_traj2ds_on_rgbs(
        self,
        name,
        trajs,
        rgbs,
        valids=None,
        frame_ids=None,
        only_return=False,
        show_dots=False,
        cmap="coolwarm",
        vals=None,
        linewidth=1,
    ):
        # trajs is B, S, N, 2
        # rgbs is B, S, C, H, W
        B, S, C, H, W = rgbs.shape
        B, S2, N, D = trajs.shape
        assert S == S2

        rgbs = rgbs[0]  # S, C, H, W
        trajs = trajs[0]  # S, N, 2
        if valids is None:
            valids = torch.ones_like(trajs[:, :, 0])  # S, N
        else:
            valids = valids[0]

        # print('trajs', trajs.shape)
        # print('valids', valids.shape)

        if vals is not None:
            vals = vals[0]  # N
            # print('vals', vals.shape)

        rgbs_color = []
        for rgb in rgbs:
            rgb = rgb.numpy()
            rgb = np.transpose(rgb, [1, 2, 0])  # put channels last
            rgbs_color.append(rgb)  # each element 3 x H x W

        for i in range(N):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i].long().detach().cpu().numpy()  # S, 2
            valid = valids[:, i].long().detach().cpu().numpy()  # S

            # print('traj', traj.shape)
            # print('valid', valid.shape)

            for t in range(S):
                if valid[t]:
                    # traj_seq = traj[max(t-16,0):t+1]
                    traj_seq = traj[max(t - 8, 0) : t + 1]
                    val_seq = np.linspace(0, 1, len(traj_seq))
                    # if t<2:
                    #     val_seq = np.zeros_like(val_seq)
                    # print('val_seq', val_seq)
                    # val_seq = 1.0
                    # val_seq = np.arange(8)/8.0
                    # val_seq = val_seq[-len(traj_seq):]
                    # rgbs_color[t] = self.draw_traj_on_image_py(rgbs_color[t], traj_seq, S=S, show_dots=show_dots, cmap=cmap_, val=val_seq, linewidth=linewidth)
                    rgbs_color[t] = self.draw_traj_on_image_py(
                        rgbs_color[t],
                        traj_seq,
                        S=S,
                        show_dots=show_dots,
                        cmap=cmap_,
                        val=val_seq,
                        linewidth=linewidth,
                    )
            # input()

        for i in range(N):
            if cmap == "onediff" and i == 0:
                cmap_ = "spring"
            elif cmap == "onediff":
                cmap_ = "winter"
            else:
                cmap_ = cmap
            traj = trajs[:, i]  # S,2
            # vis = visibles[:,i] # S
            vis = torch.ones_like(traj[:, 0])  # S
            valid = valids[:, i]  # S
            rgbs_color = self.draw_circ_on_images_py(
                rgbs_color,
                traj,
                vis,
                S=0,
                show_dots=show_dots,
                cmap=cmap_,
                linewidth=linewidth,
            )

        rgbs = []
        for rgb in rgbs_color:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            rgbs.append(rgb)

        return self.summ_rgbs(name, rgbs, only_return=only_return, frame_ids=frame_ids)

    def draw_traj_on_images_py(
        self,
        rgbs,
        traj,
        S=50,
        linewidth=1,
        show_dots=False,
        cmap="coolwarm",
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgbs is a list of H,W,3
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert C == 3

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert D == 2

        x = int(np.clip(traj[0, 0], 0, W - 1))
        y = int(np.clip(traj[0, 1], 0, H - 1))
        color = rgbs[0][y, x]
        color = (int(color[0]), int(color[1]), int(color[2]))
        for s in range(S):
            # bak_color = np.array(color_map(1.0)[:3]) * 255 # rgb
            # cv2.circle(rgbs[s], (traj[s,0], traj[s,1]), linewidth*4, bak_color, -1)
            cv2.polylines(
                rgbs[s], [traj[: s + 1]], False, color, linewidth, cv2.LINE_AA
            )
        return rgbs

    def draw_circ_on_images_py(
        self,
        rgbs,
        traj,
        vis,
        S=50,
        linewidth=1,
        show_dots=False,
        cmap=None,
        maxdist=None,
    ):
        # all inputs are numpy tensors
        # rgbs is a list of 3,H,W
        # traj is S,2
        H, W, C = rgbs[0].shape
        assert C == 3

        rgbs = [rgb.astype(np.uint8).copy() for rgb in rgbs]

        S1, D = traj.shape
        assert D == 2

        if cmap is None:
            bremm = ColorMap2d()
            traj_ = traj[0:1].astype(np.float32)
            traj_[:, 0] /= float(W)
            traj_[:, 1] /= float(H)
            color = bremm(traj_)
            # print('color', color)
            color = (color[0] * 255).astype(np.uint8)
            # color = (int(color[0]),int(color[1]),int(color[2]))
            color = (int(color[2]), int(color[1]), int(color[0]))

        for s in range(S1):
            if cmap is not None:
                color_map = cm.get_cmap(cmap)
                # color = np.array(color_map(s/(S-1))[:3]) * 255 # rgb
                color = (
                    np.array(color_map((s + 1) / max(1, float(S - 1)))[:3]) * 255
                )  # rgb
                # color = color.astype(np.uint8)
                # color = (color[0], color[1], color[2])
                # print('color', color)
            # import ipdb; ipdb.set_trace()

            cv2.circle(
                rgbs[s], (int(traj[s, 0]), int(traj[s, 1])), linewidth + 1, color, -1
            )
            # vis_color = int(np.squeeze(vis[s])*255)
            # vis_color = (vis_color,vis_color,vis_color)
            # cv2.circle(rgbs[s], (int(traj[s,0]), int(traj[s,1])), linewidth+1, vis_color, -1)

        return rgbs


def visualize_folder(folder):
    assert folder.name == "frames"
    images = glob.glob(str(folder / "Image/*/*.png"))

    lists = {}
    frames = set()
    for image in images:
        suffix = parse_suffix(Path(image))
        key = dict(suffix)
        frames.add(key["frame"])
        del key["frame"]
        key = tuple(sorted(key.items()))
        if key in lists:
            lists[key].append(image)
        else:
            lists[key] = [image]
    frame_start = min(frames)

    for key in lists:
        sub_lists = sorted(lists[key])
        rgbs = np.stack([cv2.imread(file).transpose((2, 0, 1)) for file in sub_lists])[
            :, ::-1
        ].copy()
        S, C, H, W = rgbs.shape
        # pick N points to track; we'll use a uniform grid
        N = 1024
        N_ = np.sqrt(N).round().astype(np.int32)
        grid_y, grid_x = np.meshgrid(range(N_), range(N_), indexing="ij")
        grid_y = (8 + grid_y.reshape(1, -1) / float(N_ - 1) * (H - 16)).astype(np.int32)
        grid_x = (8 + grid_x.reshape(1, -1) / float(N_ - 1) * (W - 16)).astype(np.int32)
        xy0 = np.stack([grid_x, grid_y], axis=-1)  # B, N_*N_, 2
        trajs_e = np.zeros((1, S, N_ * N_, 2))
        for file in sub_lists:
            file = Path(file)
            frame = parse_suffix(file.name)["frame"]
            traj_path = (
                folder
                / "PointTraj3D"
                / file.parent.name
                / ("PointTraj3D" + file.name[5:-4] + ".npy")
            )
            traj = np.load(traj_path)
            trajs_e[:, frame - frame_start] = (
                xy0 + traj[grid_y * 2, grid_x * 2][..., :2]
            )

        trajs_e = trajs_e.transpose(0, 1, 3, 2)  # 1, S, 2, N_*N_
        trajs_e = trajs_e.reshape(S * 2, -1).transpose()  # N_*N_, S*2
        mask = ~np.isnan(trajs_e).any(axis=1)
        trajs_e = trajs_e[mask].transpose().reshape(1, S, 2, -1)  # 1, S, 2, K
        trajs_e = trajs_e.transpose(0, 1, 3, 2)

        summ = Summ_writer(just_gif=True)
        results = summ.summ_traj2ds_on_rgbs(
            None,
            torch.from_numpy(trajs_e),
            torch.from_numpy(rgbs).unsqueeze(0),
            only_return=True,
        )
        results = results.numpy()
        frames = [results[0, i].transpose((1, 2, 0)) for i in range(results.shape[1])]

        suffix = Path(sub_lists[0]).name[5:-4]
        imageio.mimsave(
            str(
                folder
                / "PointTraj3D"
                / Path(sub_lists[0]).parent.name
                / ("PointTrajVis" + suffix + ".gif")
            ),
            frames,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    visualize_folder(args.folder)
