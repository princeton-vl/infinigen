# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick, Zeyu Ma, Lingjie Mei, Jia Deng


import hashlib
import math
import random
import warnings
import sys

import numpy as np
import gin
import cv2

@gin.configurable
class FixedSeed:

    def __init__(self, seed):
        
        self.seed = int(seed)
        self.py_state = None
        self.np_state = None

    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, *_):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)


@gin.configurable
class AddedSeed:

    def __init__(self, added_seed):
        self.added_seed = added_seed
        self.py_state = None
        self.np_state = None

    def __enter__(self):
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()

        random.randbytes(self.added_seed)
        np.random.rand(self.added_seed)

    def __exit__(self, *_):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)


class BBox:

    def __init__(self, mins, maxs):
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

    def __repr__(self):
        return f'{self.__class__}({self.mins}, {self.maxs})'

    def __contains__(self, p):
        p = np.array(p)
        return np.all((self.mins <= p) * (self.maxs >= p))

    def uniform(self):
        return np.random.uniform(0, 1, len(self.mins)) * (self.maxs - self.mins) + self.mins

    def union(self, other):

        if isinstance(other, BBox):
            return BBox(
                mins=np.minimum(self.mins, other.mins),
                maxs=np.maximum(self.maxs, other.maxs)
            )
        elif isinstance(other, np.ndarray) and other.shape[-1] == len(self.mins):
            return BBox(
                mins=np.minimum(self.mins, other),
                maxs=np.maximum(self.maxs, other)
            )
        else:
            raise ValueError(f'Unrecognized arg {other} in BBox.union')

    def intersect(self, other):
        return BBox(
            mins=np.maximum(self.mins, other.mins),
            maxs=np.minimum(self.maxs, other.maxs)
        )

    def empty(self):
        return np.any(self.mins >= self.maxs)

    def subset(self, idx):
        return BBox(self.mins[idx], self.maxs[idx])

    def linspace(self, n):
        if isinstance(n, int):
            n = [n] * len(self.mins)
        lins = [np.linspace(self.mins[i], self.maxs[i], n[i]) for i in range(len(self.mins))]
        return np.meshgrid(*lins)

    def to_local_coords(self, p):
        return (p - self.mins) / self.dims()

    def to_global_coords(self, p):
        assert p.min() >= 0
        assert p.max() <= 1
        return self.mins + p * self.dims()

    def __len__(self):
        return len(self.mins)

    def dims(self):
        return self.maxs - self.mins

    def measure(self):
        return math.prod(self.dims())

    def root(self):
        return self.mins

    def center(self):
        return (self.mins + self.maxs) / 2

    def eroded(self, margin):
        if not isinstance(margin, np.ndarray):
            margin = np.array([margin] * len(self))
        return BBox(
            mins=self.mins + margin,
            maxs=self.maxs - margin
        )

    def inflated(self, margin):
        if not isinstance(margin, np.ndarray):
            margin = np.array([margin] * len(self))
        return BBox(
            mins=self.mins - margin,
            maxs=self.maxs + margin
        )

    @classmethod
    def from_center_dims(cls, center, dims):
        return cls(
            mins=center - dims / 2,
            maxs=center + dims / 2
        )

    @classmethod
    def from_bpy_box(cls, bpy_obj):
        if not (
                hasattr(bpy_obj, 'empty_display_type') and
                bpy_obj.empty_display_type == 'CUBE'
        ):
            raise ValueError(f'BBox.from_bpy_box expected a CUBE type blender empty')

        center = bpy_obj.location
        dims = bpy_obj.scale * bpy_obj.empty_display_size / 2  # default has a RADIUS of 1

        return cls.from_center_dims(center, dims)

    @classmethod
    def empty_box(cls, dim):
        return cls(np.zeros(dim), np.zeros(dim))

    def to_limits(self):
        return np.stack([self.mins, self.maxs], axis=-1).T


def md5_hash(x):
    if isinstance(x, (tuple, list)):
        m = hashlib.md5()
        for s in x:
            assert isinstance(s, (int, str))
            m.update(str(s).encode('utf-8'))
        return m
    elif isinstance(x, (int, str)):
        x = str(x).encode('utf-8')
        return hashlib.md5(x)
    else:
        raise ValueError(f'util.md5_hash doesnt currently support type({type(x)}')


def int_hash(x, max=(2**32 - 1)):
    md5 = int(md5_hash(x).hexdigest(), 16)
    h = abs(md5) % max
    return h

def round_to_nearest(x, step):
    return step * np.round(x / step)


def lerp_sample(vec, ts: np.array):
    vec = np.array(vec, dtype=np.float32)
    ts = np.array(ts, dtype=np.float32)
    assert ts.min() >= 0 and ts.max() <= len(vec)

    # compute integer and fractional indexes
    idx_int = np.floor(ts).astype(np.int32)
    idx_int = np.clip(idx_int, 0, len(vec) - 1)
    idx_rem = ts - idx_int

    # do fractional indexing only where not already at last elt
    res = vec[idx_int]
    m = idx_int < (len(vec) - 1)
    res[m] = (1 - idx_rem[m, None]) * res[m] + idx_rem[m, None] * vec[idx_int[m] + 1]

    return res


def inverse_interpolate(vals, ds):
    '''
    Find ts such that lerp_sample(vals, ts) = ds
    '''

    assert (ds >= vals.min()).all()
    assert (ds <= vals.max()).all()

    idx = (ds[:, None] <= vals[None]).argmax(axis=-1)
    m = idx > 0

    assert (vals[idx] >= ds).all()
    assert (vals[idx[m] - 1] < ds[m]).all()

    ts = np.zeros_like(ds)
    bucket_sizes = vals[idx[m]] - vals[idx[m] - 1]
    ts[m] = idx[m] - 1 + (ds[m] - vals[idx[m] - 1]) / bucket_sizes
    return ts


def cross_matrix(v):
    o = np.zeros(v.shape[0])

    cross_mat = np.stack([
        np.stack([o, -v[:, 2], v[:, 1]], axis=-1),
        np.stack([v[:, 2], o, -v[:, 0]], axis=-1),
        np.stack([-v[:, 1], v[:, 0], o], axis=-1),
    ], axis=-1).transpose(0, 2, 1)

    return cross_mat


def rodrigues(angle, axi):
    axi = axi / np.linalg.norm(axi, axis=-1, keepdims=True)

    id = np.zeros((axi.shape[0], 3, 3))
    id[:, [0, 1, 2], [0, 1, 2]] = 1
    th = angle[:, None, None]
    K = cross_matrix(axi)

    return id + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rotate_match_directions(a, b):
    assert a.shape == b.shape
    norm = np.linalg.norm

    axes = np.cross(a, b, axis=-1)
    m = np.linalg.norm(axes, axis=-1) > 1e-4
    rots = np.empty((len(a), 3, 3))
    rots[~m] = np.eye(3)[None]

    if np.all(~m): # needed to prevent exceptions if continued
        return rots

    dots = (a[m] * b[m]).sum(axis=-1)
    dots /= norm(a[m], axis=-1) * norm(b[m], axis=-1)
    rots[m] = rodrigues(np.arccos(dots), axes[m])

    return rots


def lerp(a, b, x):
    "linear interpolation"
    return (1 - x) * a + x * b

def dict_lerp(a, b, t):
    assert list(a.keys()) == list(b.keys())
    return {k: lerp(va, b[k], t) for k, va in a.items()}

def dict_convex_comb(dicts, weights):
    assert all(d.keys == dicts[0].keys() for d in dicts[1:])
    weights = np.array(weights)
    vals = {k: np.array([d[k] for d in dicts]) for k in dicts[0]}
    return {k: (v * weights).sum() for k, v in vals.items()}

def randomspacing(min, max, n, margin):
    assert 0 <= margin and margin <= 0.5

    pos = np.linspace(min, max, n, endpoint=False)
    bucket_size = (max - min) / n

    if margin < 0.5:
        pos += np.random.uniform(margin * bucket_size, (1 - margin) * bucket_size, n)

    return pos


def linvec(n, low, high):
    return np.full(n, low) + np.linspace(0, 1, n) * (high - low)


def homogenize(points):
    ones = np.ones(points.shape[:-1])[..., None]
    return np.concatenate([points, ones], axis=-1)


def dehomogenize(points):
    return points[..., :-1] / points[..., [-1]]

def clip_gaussian(mean, std, min, max, max_tries=20):
    assert min <= max
    i = 0
    while True:
        val = np.random.normal(mean, std)
        if min <= val and val <= max:
            return val

        if i == max_tries:
            warnings.warn(f'clip_gaussian({mean=}, {std=}, {min=}, {max=}) reached {max_tries=}')
            return np.clip(val, min, max)

        i += 1

def normalize(v, disallow_zero_norm=False, in_place=True):
    n = np.linalg.norm(v, axis=-1)
    if disallow_zero_norm and np.any(n == 0):
        raise ValueError("zero norm")
    res = v if in_place else np.copy(v)
    res[n > 0] /= n[n > 0, None]
    return res


def project_to_unit_vector(k, v):
    return (k * v).sum(axis=-1)[..., None] * v


def wrap_around_cyclic_coord(u, u_start, u_end):
    _, r = np.divmod(u - u_start, u_end - u_start)
    return r + u_start

def new_domain_from_affine(old_domain, a=1.0, b=0.0):
    """
    old domain: domain of u(t)
    new domain: domain of u(f(t)), f(t) = a * t + b
    """
    if a == 0:
        raise ValueError("a cannot be zero")
    new_domain = (np.array(old_domain) - b) / a
    if a < 0:
        new_domain = new_domain[::-1]
    return tuple(new_domain)


def affine_from_new_domain(old_domain, new_domain):
    """
    Get affine parameters a, b such that u(f(at+b)) has new_domain
    """
    s = old_domain
    t = new_domain
    a = (s[1] - s[0]) / (t[1] - t[0])
    b = s[0] - a * t[0]
    return (a, b)

def resize(arr, shape):
    return cv2.resize(arr, shape) #, interpolation=cv2.INTER_LANCZOS4)
