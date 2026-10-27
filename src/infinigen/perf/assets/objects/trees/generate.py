# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick, Yiming Zuo, Alejandro Newell, Lingjie Mei: original Infinigen tree factories (https://github.com/princeton-vl/infinigen/blob/224598d4a2ff95b4fc928e72c222598b7122c698/src/infinigen/assets/objects/trees/generate.py)
# - David Yan, Jia Deng: optimized implementation

import logging

import bpy
import numpy as np

from infinigen import _perf_selection
from infinigen.assets.objects.trees import generate as _default
from infinigen.assets.objects.trees import tree
from infinigen.assets.objects.trees.generate import (
    BushFactory,
    random_season,
    random_tree_child_factory,
)
from infinigen.assets.utils.misc import toggle_hide, toggle_show
from infinigen.core import surface
from infinigen.core.placement import detail
from infinigen.core.placement.factory import make_asset_collection
from infinigen.core.placement.split_in_view import split_inview
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, int_hash
from infinigen.perf import options
from infinigen.perf.assets.objects.trees import tessellate
from infinigen.perf.assets.objects.trees import tree as _tree
from infinigen.perf.assets.utils import surface_fields
from infinigen.perf.assets.utils.heightfield import mesh_route
from infinigen.perf.core.util import blender as _blender_ext

# the methods here run on instances of the default classes, so log as its module
logger = logging.getLogger(_default.__name__)


def _detail_transfers_nearest_point() -> bool:
    """Whether the selected placement can transfer attributes by nearest point. The
    default one interpolates every attribute across a remesh, discrete ones included."""
    return _perf_selection.selected("PLACEMENT")


_PLACEHOLDER_CUBE_VERTICES = (
    -2,
    -2,
    -2,
    -2,
    -2,
    2,
    -2,
    2,
    -2,
    -2,
    2,
    2,
    2,
    -2,
    -2,
    2,
    -2,
    2,
    2,
    2,
    -2,
    2,
    2,
    2,
)
_PLACEHOLDER_CUBE_EDGES = (
    2,
    0,
    0,
    1,
    1,
    3,
    3,
    2,
    6,
    2,
    3,
    7,
    7,
    6,
    4,
    6,
    7,
    5,
    5,
    4,
    0,
    4,
    5,
    1,
)
_PLACEHOLDER_CUBE_LOOP_VERTICES = (
    0,
    1,
    3,
    2,
    2,
    3,
    7,
    6,
    6,
    7,
    5,
    4,
    4,
    5,
    1,
    0,
    2,
    6,
    4,
    0,
    7,
    3,
    1,
    5,
)
_PLACEHOLDER_CUBE_LOOP_EDGES = (
    1,
    2,
    3,
    0,
    3,
    5,
    6,
    4,
    6,
    8,
    9,
    7,
    9,
    11,
    1,
    10,
    4,
    7,
    10,
    0,
    5,
    2,
    11,
    8,
)
_PLACEHOLDER_CUBE_POLYGON_STARTS = (0, 4, 8, 12, 16, 20)
_PLACEHOLDER_CUBE_UVS = (
    0.375,
    0.0,
    0.625,
    0.0,
    0.625,
    0.25,
    0.375,
    0.25,
    0.375,
    0.25,
    0.625,
    0.25,
    0.625,
    0.5,
    0.375,
    0.5,
    0.375,
    0.5,
    0.625,
    0.5,
    0.625,
    0.75,
    0.375,
    0.75,
    0.375,
    0.75,
    0.625,
    0.75,
    0.625,
    1.0,
    0.375,
    1.0,
    0.125,
    0.5,
    0.375,
    0.5,
    0.375,
    0.75,
    0.125,
    0.75,
    0.625,
    0.5,
    0.875,
    0.5,
    0.875,
    0.75,
    0.625,
    0.75,
)


def _new_unlinked_placeholder_cube():
    """Build Blender's size-4 primitive cube without invoking an operator."""

    cube_mesh = bpy.data.meshes.new("Cube")
    cube_mesh.vertices.add(8)
    cube_mesh.vertices.foreach_set("co", _PLACEHOLDER_CUBE_VERTICES)
    cube_mesh.edges.add(12)
    cube_mesh.edges.foreach_set("vertices", _PLACEHOLDER_CUBE_EDGES)
    cube_mesh.loops.add(24)
    cube_mesh.loops.foreach_set("vertex_index", _PLACEHOLDER_CUBE_LOOP_VERTICES)
    cube_mesh.loops.foreach_set("edge_index", _PLACEHOLDER_CUBE_LOOP_EDGES)
    cube_mesh.polygons.add(6)
    cube_mesh.polygons.foreach_set("loop_start", _PLACEHOLDER_CUBE_POLYGON_STARTS)
    cube_mesh.polygons.foreach_set("loop_total", (4,) * 6)

    uv_layer = cube_mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", _PLACEHOLDER_CUBE_UVS)
    # the operator's cube is flat shaded (sharp_face set on every face)
    cube_mesh.shade_flat()
    cube_mesh.update()
    return bpy.data.objects.new("Cube", cube_mesh)


def _delete_leaf_templates(objs):
    """`butil.delete`, skipping objects that cannot be selected (such as hidden ones),
    on which the operator does nothing but still updates the view layer."""
    for obj in objs:
        butil.select(obj)
        if not obj.select_get():
            continue
        butil.delete(obj)


# A coarse bush placeholder builds its skeleton only once it survives fine
# population culling; this property holds its index, which seeds the skeleton.
_DEFERRED_SKELETON_INDEX = "infinigen_bush_skeleton_index"


def _create_skeleton(factory):
    return tree.tree_skeleton(
        factory.genome.skeleton,
        factory.genome.trunk_spacecol,
        factory.genome.roots_spacecol,
        init_pos=(0, 0, 0),
        scale=factory.scale,
    )


def _spawn_placeholder_cube(factory):
    """Copy a template cube for each placeholder. Bushes share its mesh, as their
    coarse cubes are never edited and fine geometry is built as a separate child."""
    template = getattr(factory, "_placeholder_cube_template", None)
    try:
        template_is_live = template is not None and template.name in bpy.data.objects
    except ReferenceError:
        template_is_live = False
    if not template_is_live:
        pholder = _new_unlinked_placeholder_cube()
        bpy.context.collection.objects.link(pholder)
        factory._placeholder_cube_template = pholder
    elif isinstance(factory, BushFactory):
        pholder = template.copy()
        bpy.context.collection.objects.link(pholder)
    else:
        pholder = deep_clone_obj(template)
    factory._last_placeholder_cube = pholder
    return pholder


def native_meshing(factory):
    """Whether this factory's trunks are meshed natively, each stem retessellated as
    its own tube instead of remeshing the whole trunk. Used where the default would
    remesh unless `native_tree_meshing.enabled = False` or the cylinder kernel is not
    built; `adapt_mesh_method="stem_tessellate"` forces it."""
    method = factory.adapt_mesh_method
    if method == "stem_tessellate":
        return True
    return (
        method == "remesh"
        and options.native_tree_meshing()
        and tessellate.KERNEL_AVAILABLE
    )


def _coarse_mesh(factory, skeleton_obj):
    """`GenericTreeFactory._create_coarse_mesh`, with the stems left unwelded when
    they are to be meshed natively."""
    coarse_mesh = deep_clone_obj(skeleton_obj)
    surface.add_geomod(
        coarse_mesh,
        _tree.skin_tree_unmerged if native_meshing(factory) else tree.skin_tree,
        input_kwargs={"params": factory.genome.skinning},
        apply=True,
    )
    if factory.decimate_placeholder_levels > 0:
        butil.modify_mesh(
            coarse_mesh,
            "DECIMATE",
            decimate_type="UNSUBDIV",
            iterations=factory.decimate_placeholder_levels,
        )
    return coarse_mesh


def _materialize_deferred_skeleton(factory, placeholder):
    if _DEFERRED_SKELETON_INDEX not in placeholder:
        return

    if len(placeholder.children) != 0:
        raise RuntimeError(
            f"Deferred bush placeholder {placeholder.name!r} must not "
            f"already have children, found {len(placeholder.children)}"
        )

    original_index = int(placeholder[_DEFERRED_SKELETON_INDEX])
    with FixedSeed(int_hash((factory.factory_seed, original_index))):
        materialized = _create_skeleton(factory)

    butil.parent_to(materialized, placeholder, no_inverse=True)
    del placeholder[_DEFERRED_SKELETON_INDEX]


class GenericTreeFactory(_default.GenericTreeFactory):
    def create_placeholder(self, i, loc, rot):
        if (
            isinstance(self, BushFactory)
            and self.coarse
            and not self.coarse_mesh_placeholder
        ):
            # Bush visibility is computed from the four-unit parent cube, not
            # from its child skeleton. Materialize that child only if this
            # placeholder survives fine-population culling.
            pholder = _spawn_placeholder_cube(self)
            pholder[_DEFERRED_SKELETON_INDEX] = i
            return pholder

        logger.debug("generating tree skeleton")
        skeleton_obj = _create_skeleton(self)
        if self.coarse_mesh_placeholder:
            pholder = _coarse_mesh(self, skeleton_obj)
        else:
            pholder = _spawn_placeholder_cube(self)

        butil.parent_to(skeleton_obj, pholder, no_inverse=True)
        return pholder

    def finalize_placeholders(self, placeholders):
        cube = getattr(self, "_last_placeholder_cube", None)
        if not self.coarse_mesh_placeholder and cube is not None:
            butil.select_none()
            cube.select_set(True)
            bpy.context.view_layer.objects.active = cube
        return _default.GenericTreeFactory.finalize_placeholders.default(
            self, placeholders
        )

    def create_asset(
        self, placeholder, face_size, distance, **kwargs
    ) -> bpy.types.Object:
        if isinstance(self, BushFactory):
            _materialize_deferred_skeleton(self, placeholder)

        skeleton_obj = placeholder.children[0]

        native = native_meshing(self)
        if not self.coarse_mesh_placeholder:
            skin_obj = _coarse_mesh(self, skeleton_obj)
            self.trunk_surface.apply(self, skin_obj)
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)
        else:
            skin_obj = butil.deep_clone_obj(placeholder)

        if self.child_col is not None:
            assert self.genome.child_placement is not None

            max_needed_child_fs = (
                detail.target_face_size(self.min_dist, global_multiplier=1)
                if self.min_dist is not None
                else None
            )

            logger.debug(f"adding tree children using {self.child_col=}")
            butil.select_none()
            surface.add_geomod(
                skeleton_obj,
                tree.add_tree_children,
                input_kwargs=dict(
                    child_col=self.child_col,
                    params=self.genome.child_placement,
                    realize=self.realize,
                    merge_dist=max_needed_child_fs,
                ),
            )

        if self.cameras is not None and distance < self.cam_meshing_max_dist:
            assert self.adapt_mesh_method != "remesh"

            skin_obj_cleanup = skin_obj
            skin_obj, outofview, vert_dists, _ = split_inview(
                skin_obj, cameras=self.cameras, vis_margin=0.15
            )
            butil.parent_to(outofview, skin_obj, no_inverse=True, no_transform=True)
            if native:
                # the deferred bark goes with the pieces; the out-of-view one
                # is not retessellated, so it is displaced now
                surface_fields.transfer_pending(skin_obj_cleanup, skin_obj, outofview)
                surface_fields.apply_pending(outofview)

            butil.delete(skin_obj_cleanup)
            face_size = detail.target_face_size(vert_dists.min())

        skin_obj.hide_render = False

        with butil.DisableModifiers(skin_obj):
            # The trunk surface's modifiers stay disabled through the subdivision
            # too, so they are not evaluated on the subdivided mesh; they are
            # applied at the end.
            if native:
                tessellate.retessellate_stems(
                    skin_obj,
                    face_size,
                    profile_resolution=int(self.genome.skinning.get("Profile res", 20)),
                    edge_length_ceiling_factor=options.native_tree_edge_length(),
                )
            else:
                if self.adapt_mesh_method == "remesh":
                    # one extra level to smooth things out or remesh is jaggedy
                    butil.modify_mesh(
                        skin_obj,
                        "SUBSURF",
                        levels=self.decimate_placeholder_levels + 1,
                    )
                adapt_kwargs = {"apply": True}
                if (
                    self.adapt_mesh_method in {"remesh", "sharp_remesh"}
                    and _detail_transfers_nearest_point()
                ):
                    adapt_kwargs["nearest_point_attributes"] = [
                        "depth",
                        "level",
                        "parent_idx",
                        "rev_depth",
                        "stem_id",
                    ]
                detail.adapt_mesh_resolution(
                    skin_obj,
                    face_size,
                    method=self.adapt_mesh_method,
                    **adapt_kwargs,
                )

        if native:
            # The bark displacement deferred by BarkRandom.apply, on the trunk
            # alone: before realize joins the children into skin_obj, or the
            # leaves would be displaced and given the bark material too.
            surface_fields.apply_pending(skin_obj)

        butil.parent_to(skin_obj, placeholder, no_inverse=True, no_transform=True)

        if self.realize:
            logger.debug("realizing tree children")
            _blender_ext.apply_modifiers_via_evaluated_mesh(skin_obj)
            _blender_ext.apply_modifiers_via_evaluated_mesh(skeleton_obj)

            butil.join_objects([skin_obj, skeleton_obj])
            assert len(skin_obj.children) == 0
        else:
            butil.parent_to(skeleton_obj, skin_obj, no_inverse=True)

        if native:
            _blender_ext.apply_modifiers_via_evaluated_mesh(skin_obj)
            tag_object(skin_obj, "tree")
        else:
            tag_object(skin_obj, "tree")
            _blender_ext.apply_modifiers_via_evaluated_mesh(skin_obj)

        return skin_obj


def make_leaf_collection(
    seed, leaf_params, n_leaf, leaf_types, decimate_rate=0.0, season=None
):
    logger.debug(f"Starting make_leaf_collection({seed=}, {n_leaf=} ...)")

    if season is None:
        season = random_season()

    weights = []

    if not isinstance(leaf_types, list):
        leaf_types = [leaf_types]

    child_factories = []
    for leaf_type in leaf_types:
        if leaf_type is not None:
            leaf_factory, _ = random_tree_child_factory(
                seed, leaf_params, leaf_type=leaf_type, season=season
            )
            child_factories.append(leaf_factory)
            weights.append(1.0)

    weights = np.array(weights)
    weights /= np.sum(weights)  # normalize to 1

    with mesh_route.leaf_collection_decimation(decimate_rate):
        col = make_asset_collection(
            child_factories, n_leaf, verbose=True, weights=weights
        )
    # if leaf_surface is not None:
    #     leaf_surface.apply(list(col.objects))
    toggle_show(col)
    for obj in col.objects:
        if decimate_rate > 0 and not mesh_route.is_final_mesh(obj):
            butil.modify_mesh(obj, "DECIMATE", ratio=1.0 - decimate_rate, apply=True)
        butil.apply_transform(obj, rot=True, scale=True)
        butil.apply_modifiers(obj)
    toggle_hide(col)
    return col
