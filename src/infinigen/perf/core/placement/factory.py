# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: `AssetFactory` and `make_asset_collection` in the original Infinigen asset factory (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/factory.py)
# - Lahav Lipson: `quickly_resample` in the original Infinigen asset factory (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/core/placement/factory.py)
# - David Yan, Jia Deng: optimized implementation

import logging
from contextlib import nullcontext

import bpy
import numpy as np
from tqdm import trange

from infinigen.assets.utils.object import center
from infinigen.core.placement import detail
from infinigen.core.placement import factory as _default
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.perf import options
from infinigen.perf.core.util import isolation as _isolated_scene

# these run on instances of the default classes, so they log as the default module
logger = logging.getLogger(_default.__name__)


def _isolation(factory, *link, batch=False):
    """Context building `factory`'s asset in a scratch view layer, or a null context if
    isolation is off or the factory sets `isolate_scene = False`. A factory whose build
    reads scene collections lists them in `isolation_collections()`."""
    if not options.isolation_enabled() or not getattr(factory, "isolate_scene", True):
        return nullcontext()
    collections = getattr(factory, "isolation_collections", None)
    # the placeholder and everything parented under it must be visible to the
    # operators
    objs = [o for p in link if p is not None for o in (p, *p.children_recursive)]
    return _isolated_scene.IsolatedScene(
        link=objs,
        collections=collections() if collections is not None else [],
        tag=repr(factory),
        batch=batch,
    )


class AssetFactory(_default.AssetFactory):
    def spawn_placeholder(self, i, loc, rot):
        # Not intended to be overridden - override create_placeholder instead

        logger.debug(f"{self}.spawn_placeholder({i}...)")

        with FixedSeed(int_hash((self.factory_seed, i))), _isolation(self):
            obj = self.create_placeholder(i=i, loc=loc, rot=rot)

        has_sensitive_constraint = any(
            c.type in ["FOLLOW_PATH"] for c in obj.constraints
        )

        if not has_sensitive_constraint:
            obj.location = loc
            obj.rotation_euler = rot
        else:
            logger.debug(
                f"Not assigning placeholder {obj.name=} location due to presence of"
                "location-sensitive constraint, typically a follow curve"
            )
        obj.name = f"{repr(self)}.spawn_placeholder({i})"

        if obj.parent is not None:
            logger.warning(
                f"{obj.name=} has no-none parent {obj.parent.name=}, this may cause it not to get populated"
            )

        return obj

    def spawn_asset(
        self,
        i,
        placeholder=None,
        distance=None,
        vis_distance=0,
        loc=(0, 0, 0),
        rot=(0, 0, 0),
        **kwargs,
    ):
        if not isinstance(i, int):
            raise TypeError(f"{i=} {type(i)=}, expected int")
        # Not intended to be overridden - override create_asset instead

        logger.debug(f"{self}.spawn_asset({i}...)")

        if distance is None:
            distance = detail.scatter_res_distance()

        if self.coarse:
            raise ValueError(
                "Attempted to spawn_asset() on an AssetFactory(coarse=True)"
            )

        user_provided_placeholder = placeholder is not None

        if user_provided_placeholder:
            assert loc == (0, 0, 0) and rot == (0, 0, 0)
        else:
            placeholder = self.spawn_placeholder(i=i, loc=loc, rot=rot)
            self.finalize_placeholders([placeholder])

        gc_targets = [
            bpy.data.meshes,
            bpy.data.textures,
            bpy.data.node_groups,
            bpy.data.materials,
        ]

        export_path = None
        with (
            FixedSeed(int_hash((self.factory_seed, i))),
            butil.GarbageCollect(gc_targets, verbose=False),
            _isolation(self, placeholder),
        ):
            params = self.asset_parameters(distance, vis_distance)
            params.update(kwargs)
            obj = self.create_asset(i=i, placeholder=placeholder, **params)
            # TODO: clean this up
            if "export" in params and params["export"]:
                obj, export_path, semantic_mapping = obj
                assert export_path

        obj.name = f"{repr(self)}.spawn_asset({i})"

        if user_provided_placeholder:
            if obj is not placeholder:
                if obj.parent is None:
                    butil.parent_to(obj, placeholder, no_inverse=True)
            else:
                obj.hide_render = False
        else:
            obj.parent = None
            obj.location = placeholder.location
            obj.rotation_euler = placeholder.rotation_euler
            # the delete operator, not bpy.data.objects.remove, which would free
            # a placeholder that a node group still references
            butil.delete(placeholder)

        if export_path is not None:
            return obj, export_path, semantic_mapping
        return obj


def _batch_isolation(spawn_fns):
    """Batch isolation for a loop of spawns, from the first spawn function bound to an
    `AssetFactory`; a null context if there is none."""
    for f in spawn_fns:
        owner = getattr(f, "__self__", None)
        if isinstance(owner, _default.AssetFactory):
            return _isolation(owner, batch=True)
    return nullcontext()


def make_asset_collection(
    spawn_fns,
    n,
    name=None,
    weights=None,
    as_list=False,
    verbose=True,
    centered=False,
    **kwargs,
):
    if not isinstance(spawn_fns, list):
        spawn_fns = [spawn_fns]
    if weights is None:
        weights = np.ones(len(spawn_fns))
    weights /= sum(weights)

    if name is None:
        name = ",".join([repr(f) for f in spawn_fns])

    if verbose:
        logger.info(f"Generating collection of {n} assets from {name}")

    objs = [[] for _ in range(len(spawn_fns))]
    r = trange(n) if verbose else range(n)
    # one batch isolation context for the whole loop
    with _batch_isolation(spawn_fns):
        for i in r:
            fn_idx = np.random.choice(np.arange(len(spawn_fns)), p=weights)
            obj = spawn_fns[fn_idx](i=i, **kwargs)
            if centered:
                obj.location = -center(obj)
                butil.apply_transform(obj, True)
            objs[fn_idx].append(obj)

    for os, f in zip(objs, spawn_fns):
        if hasattr(f, "finalize_assets"):
            f.finalize_assets(os)

    objs = sum(objs, start=[])

    if as_list:
        return objs
    else:
        col = butil.group_in_collection(objs, name=f"assets:{name}", reuse=False)
        col.hide_viewport = True
        col.hide_render = True
        return col
