# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma, Alexander Raistrick


from infinigen.core.util.color import color_category
import gin
import numpy as np
import random
import json
import json5
import colorsys
import mathutils
from matplotlib import colors
from numpy.random import normal, uniform

from infinigen.core.util.math import md5_hash, clip_gaussian
from infinigen.core.init import repo_root


def log_uniform(low, high, size=1):
    return np.exp(uniform(np.log(low), np.log(high), size))

def sample_json_palette(pallette_name, n_sample=1):
    
    rel = f"infinigen_examples/configs/palette/{pallette_name}.json"

    with (repo_root()/rel).open('r') as f:
        color_template = json5.load(f)

    colors = color_template["color"]
    means = np.array(color_template["hsv"])
    stds = np.array(color_template["std"])
    probs = np.array(color_template["prob"])
    selected = np.zeros(len(means), dtype=bool)
    for c in colors:
        selected[int(c)] = 1
    means = means[selected]
    stds = stds[selected]
    probs = probs[selected]
    i = np.random.choice(range(len(colors)), 1, p=probs / np.sum(probs))[0]
    color_samples = []
    for j in range(n_sample):
        color = np.array(means[i]) + np.matmul(np.array(stds[i]).reshape((3, 3)), np.clip(np.random.randn(3), a_min=-1, a_max=1))
        color[2] = max(min(color[2], 0.9), 0.1)
        color = colorsys.hsv_to_rgb(*color)
        color = np.clip(color, a_min=0, a_max=1)
        color = np.where(color >= 0.04045,((color+0.055)/1.055) ** 2.4, color / 12.92)
        color = np.concatenate((color, np.ones(1)))
        color_samples.append(color)
    if n_sample == 1:
        return color
    return color_samples

def random_general(var):
    if not (isinstance(var, tuple) or isinstance(var, list)):
        return var

    func, *args = var
    if func == "weighted_choice":
        weights, recargs = zip(*args)
        p = np.array(weights)/sum(weights)
        i = np.random.choice(np.arange(len(recargs)), p=p)
        return random_general(recargs[i])
    elif func == "spherical_sample":
        min_elevation, max_elevation = args
        while True:
            # angle distribution from uniform sphere
            P = np.random.randn(3)
            x = np.arctan2(np.abs(P[2]), (P[0] ** 2 + P[1] ** 2) ** 0.5)
            if (min_elevation is None or x > np.radians(min_elevation)) and (max_elevation is None or x < np.radians(max_elevation)):
                break
        return np.degrees(x)
    elif func == "uniform":
        return np.random.uniform(*args)
    elif func == "normal":
        return np.random.normal(*args)
    elif func == "clip_gaussian":
        return clip_gaussian(*args)
    elif func == "power_uniform":
        return 10 ** np.random.uniform(*args)
    elif func == "log_uniform":
        return log_uniform(*args)[0]
    elif func == "discrete_uniform":
        return np.random.randint(args[0], args[1] + 1)
    elif func == "bool":
        return np.random.uniform() < args[0]
    elif func == "choice":
        return np.random.choice(args[0], 1, p=args[1])[0]
    elif func == "palette":
        return sample_json_palette(*args)
    elif func == "color_category":
        return color_category(*args)
    else:
        return var


def random_vector3():
    return mathutils.Vector((np.random.randint(999), np.random.randint(999), np.random.randint(999)))


def _rgb_to_hsv(rgb):
    """
    returns (h, s, v) form either (r, g, b) or (r, g, b, a) tuple
    """
    a = None
    if len(rgb) == 4:
        a = rgb[-1]
        rgb = rgb[:3]
    hsv = colors.rgb_to_hsv(rgb)
    return hsv, a


def _hsv_to_rgb(hsv, a):
    """
    returns (r, g, b) or (r, g, b, a) form (h, s, v) and a
    """
    rgb = list(colors.hsv_to_rgb(hsv))
    if a is not None:
        rgb.append(a)
    return rgb


def random_color_neighbour(
    rgb, hue_diff=0.0, sat_diff=0.0, val_diff=0.0,
    only_less_hue=False, only_less_sat=False, only_less_val=False,
    only_more_hue=False, only_more_sat=False, only_more_val=False,
):
    """
    returns a random color in the neighbourhood of the given one
    :param color: (r, g, b) or (r, g, b, a)
    :param hue_diff: maximum change in hue, if none all hue is allowed
    :param sat_diff: maximum change in saturation
    :param val_diff: maximum change in value
    :param only_less_hue: only small hue values smaller than the one provided
        in rgb
    :param only_less_sat:
    :param only_less_val:
    :param only_more_hue:
    :param only_more_sat:
    :param only_more_val:
    :return: (r, g, b) or (r, g, b, a) depending in the input
    """
    assert not (only_less_hue and only_more_hue)
    assert not (only_less_sat and only_more_sat)
    assert not (only_less_val and only_more_val)

    hsv, a = _rgb_to_hsv(rgb)

    def sample(x, diff, low=0, high=1, only_less=False, only_more=False):
        """
        sample a float from max(0, x-diff) and min(1, x+diff) if diff is not
        None. Else sample a number from low to high
        :param only_less: valid when diff is not None. only samples a small
            value
        :param only_more:
        """
        if diff is None:
            out = np.random.uniform(low, high)
        else:
            lb = max(0, x) if only_more else max(0, x-diff)
            ub = min(1, x) if only_less else min(1, x+diff)
            out = np.random.uniform(lb, ub)
        return out

    hsv[0] = sample(
        hsv[0], hue_diff, only_less=only_less_hue,
        only_more=only_more_hue)
    hsv[1] = sample(
        hsv[1], sat_diff, only_less=only_less_sat,
        only_more=only_more_sat)
    hsv[2] = sample(
        hsv[2], val_diff, only_less=only_less_val,
        only_more=only_more_val)

    rgb = _hsv_to_rgb(hsv, a)

    return rgb


def clip_hsv(rgb, max_h=None, max_s=None, max_v=None):
    """
    reduces the hue, saturation and value of an rgb color
    """
    hsv, a = _rgb_to_hsv(rgb)
    if max_h is not None:
        assert 0 <= max_h <= 1
        hsv[0] = hsv[0] * max_h
    if max_s is not None:
        assert 0 <= max_s <= 1
        hsv[1] = hsv[1] * max_s
    if max_v is not None:
        assert 0 <= max_v <= 1
        hsv[2] = hsv[2] * max_v
    rgb = _hsv_to_rgb(hsv, a)

    return rgb

def random_color(brightness_lim=1):
    return (np.random.randint(256) / 256. * brightness_lim, np.random.randint(256) / 256. * brightness_lim, np.random.randint(256) / 256. * brightness_lim, 1)

def sample_registry(reg):
    classes, weights = zip(*reg)
    weights = np.array(weights)
    return np.random.choice(classes, p=weights/weights.sum())