# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Build assets in a scratch view layer, since each operator call updates the view
layer at a cost that grows with the objects in it."""

import logging

import bpy

logger = logging.getLogger(__name__)


class IsolatedScene:
    """Context in which operators see only the asset being built.

    The window switches to a second view layer of the same scene (switching scenes
    re-evaluates the target), so new objects and the object order `join` follows match
    the main layer. It holds the active collection and a link collection for `link`,
    with older objects hidden; others are exposed on demand. Contexts nest, and in a
    `batch` one earlier spawns are parked outside the layer while the next one builds.
    """

    VL_NAME = "asset_scratch (no gc)"
    LINK_NAME = "asset_scratch:link (no gc)"
    PARK_NAME = "asset_scratch:done (no gc)"
    active = []  # stack of entered instances

    def __init__(self, link=(), collections=(), tag=None, batch=False):
        self.link = [o for o in link if o is not None]
        self.collections = [c for c in collections if c is not None]
        self.unexcluded = []
        self.tag = tag
        self.batch = bool(batch)
        self.parked = {}  # batch contexts: session_uid -> object, creation order
        self.parent = None
        self.a_base = None  # batch: session_uids in the active collection at entry
        self.l_before = None
        self.park_col = None

    @classmethod
    def ensure_selectable(cls, obj):
        """Expose `obj` to operators in the current isolated build; False if none."""
        if not cls.active:
            return False
        cls.active[-1]._expose(obj)
        return True

    @classmethod
    def is_parked(cls, obj):
        """True if `obj` is a finished spawn of the enclosing batch, parked outside the
        scratch view layer with no base there until exposed."""
        if not cls.active:
            return False
        park = cls.active[-1].park_col
        return park is not None and obj.name in park.objects

    @classmethod
    def ensure_evaluable(cls, obj):
        """Make the scratch view layer evaluate `obj` before its evaluated data is read,
        since `evaluated_get` silently returns un-evaluated data outside it. A parked
        spawn goes back into the active collection; another object with modifiers or a
        non-mesh type gets its collections included with their objects hidden, which
        still evaluates them but keeps them out of selection-based operators."""
        if not cls.active:
            return False
        iso = cls.active[-1]
        if obj.name not in iso.scene.objects or iso._included(obj):
            return False
        if obj.type == "MESH" and not obj.modifiers:
            return False
        if cls.is_parked(obj):
            iso._expose(obj)
            return True
        logger.warning(
            f"IsolatedScene: including the collections of {obj.name!r} for "
            f"depsgraph evaluation ({[c.name for c in obj.users_collection]})"
        )
        for c in obj.users_collection:
            iso._include(c)
        return True

    def _layer_collection_path(self, collection):
        """Layer collections from the view layer root down to `collection`."""

        def walk(lc, path):
            for child in lc.children:
                if child.collection == collection:
                    return path + [child]
                found = walk(child, path + [child])
                if found:
                    return found
            return None

        return walk(self.vl.layer_collection, [])

    def _include(self, collection):
        """Include `collection` and its ancestors in the scratch view layer, hiding the
        objects they bring in; False if it is not part of the scene."""
        path = self._layer_collection_path(collection)
        if path is None:
            return False
        for lc in path:
            if lc.exclude:
                before = set(self.vl.objects)
                lc.exclude = False
                self.unexcluded.append(lc)
                self._hide_all(self.vl, skip=before)
        return True

    def _expose(self, obj, adopt=False):
        """Give `obj` a base in the scratch view layer and unhide it there. A parked
        spawn returns to the active collection, `link` objects (`adopt`) and objects
        outside the scene go into the link collection, and other scene objects get
        their collections included, since those set the object order `join` follows."""
        if obj.name not in self.vl.objects:
            park = self.park_col
            if park is not None and obj.name in park.objects:
                park.objects.unlink(obj)
                self.main_active_col.objects.link(obj)
            elif adopt or obj.name not in self.scene.objects:
                if obj.name not in self.link_col.objects:
                    self.link_col.objects.link(obj)
            else:
                included = False
                for c in obj.users_collection:
                    if c == self.scene.collection or self._include(c):
                        included = True
                if not included:
                    self.link_col.objects.link(obj)
        obj.hide_set(False, view_layer=self.vl)

    def _included(self, obj):
        """True if `obj` is in a collection included in the scratch view layer, going by
        the collection tree, as `view_layer.objects` can lag behind (un)link calls."""
        if obj.name in self.scene.collection.objects:
            return True
        for c in obj.users_collection:
            path = self._layer_collection_path(c)
            if path is not None and not any(lc.exclude for lc in path):
                return True
        return False

    @classmethod
    def _link_collection(cls, scene):
        col = bpy.data.collections.get(cls.LINK_NAME)
        if col is None:
            col = bpy.data.collections.new(cls.LINK_NAME)
        if col.name not in scene.collection.children:
            scene.collection.children.link(col)
        return col

    @classmethod
    def _park_collection(cls, scene, main_vl):
        """The parking collection, linked to the scene and excluded from the main view
        layer (it is empty whenever that layer is in use)."""
        col = bpy.data.collections.get(cls.PARK_NAME)
        if col is None:
            col = bpy.data.collections.new(cls.PARK_NAME)
        if col.name not in scene.collection.children:
            scene.collection.children.link(col)
        lc = main_vl.layer_collection.children.get(col.name)
        if lc is not None and not lc.exclude:
            lc.exclude = True
        return col

    def _prepare(self):
        """The current scene's scratch view layer, with its exclusions, active
        collection and hidden pre-existing objects brought up to date."""
        scene = self.scene
        vl = scene.view_layers.get(self.VL_NAME)
        if vl is None:
            vl = scene.view_layers.new(self.VL_NAME)
        # never rendered: the layer is saved with the scene, and Cycles would
        # otherwise render it as a second view layer of every frame
        vl.use = False
        keep = {self.main_active_col, self.link_col, *self.collections}
        for lc in vl.layer_collection.children:
            exclude = lc.collection not in keep
            if lc.exclude != exclude:
                lc.exclude = exclude
        if self.main_active_col == scene.collection:
            vl.active_layer_collection = vl.layer_collection
        else:
            vl.active_layer_collection = vl.layer_collection.children[
                self.main_active_col.name
            ]
        # hide what already exists in the scratch view layer (the master
        # collection's objects, earlier assets in the active collection)
        self._hide_all(vl)
        return vl

    @staticmethod
    def _hide_all(vl, skip=()):
        """Hide every object of `vl` not in `skip`, leaving alone any that reports it is
        not in the layer, since the object list can be stale after (un)excluding."""
        for o in vl.objects:
            if o in skip:
                continue
            try:
                if not o.hide_get(view_layer=vl):
                    o.hide_set(True, view_layer=vl)
            except RuntimeError:
                pass

    def _park_new(self):
        """Batch context: park what the spawns so far added to the active collection,
        in creation order. The view layer is updated once first, as the next operator
        call would do, so cached world matrices match an unparked build."""
        A, T = self.main_active_col, self.park_col
        # session_uid, not the pointer: Blender reuses a freed object's address
        # for the next object it allocates
        new = [o for o in A.objects if o.session_uid not in self.a_base]
        if not new:
            return
        self.vl.update()
        for o in new:
            A.objects.unlink(o)
            T.objects.link(o)
            self.parked.setdefault(o.session_uid, o)

    def _unpark(self):
        """Batch context: put every parked object back into the active collection in
        creation order, moving any exposed again since then to its place as well."""
        A, T = self.main_active_col, self.park_col
        for o in self.parked.values():
            try:
                name = o.name
            except ReferenceError:
                continue  # removed by a build
            if name in T.objects:
                T.objects.unlink(o)
                A.objects.link(o)
            elif name in A.objects:
                A.objects.unlink(o)
                A.objects.link(o)
        self.parked.clear()
        if not any(c.batch for c in IsolatedScene.active):
            # outermost batch: nothing may stay parked
            left = list(T.objects)
            if left:
                logger.warning(
                    f"IsolatedScene: {len(left)} object(s) still parked after the batch "
                    f"({[o.name for o in left[:5]]}); moving them back"
                )
                for o in left:
                    T.objects.unlink(o)
                    A.objects.link(o)

    def __enter__(self):
        self.nested = bool(IsolatedScene.active)
        self.parent = IsolatedScene.active[-1] if self.nested else None
        if self.nested:
            outer = IsolatedScene.active[0]
            self.scene, self.vl, self.link_col = outer.scene, outer.vl, outer.link_col
            self.park_col = outer.park_col
            self.main_active_col = outer.main_active_col
        else:
            self.scene = bpy.context.scene
            self.main_vl = bpy.context.view_layer
            self.main_active_col = self.main_vl.active_layer_collection.collection
            self.link_col = self._link_collection(self.scene)
            self.park_col = self._park_collection(self.scene, self.main_vl)
            self.vl = self._prepare()
        if self.batch and self.park_col is not None:
            self.a_base = {o.session_uid for o in self.main_active_col.objects}
        elif (
            self.parent is not None and self.parent.batch and self.park_col is not None
        ):
            # a spawn inside a batch: the earlier spawns leave the view layer
            self.parent._park_new()
        self.l_before = set(self.link_col.objects.keys())
        if self.nested and self.collections:
            # a nested build reading collections the outer one did not
            for c in self.collections:
                self._include(c)
        for o in self.link:
            self._expose(o, adopt=True)
        if not self.nested:
            # Switch the window's view layer so every context member the
            # operators read (view layer, selection, active object) agrees.
            self.window = getattr(bpy.context, "window", None)
            self.override = None
            if self.window is not None:
                self.window.view_layer = self.vl
            else:
                self.override = bpy.context.temp_override(view_layer=self.vl)
                self.override.__enter__()
        IsolatedScene.active.append(self)
        return self.vl

    def __exit__(self, *exc):
        IsolatedScene.active.remove(self)
        # whatever was linked into the link collection during this context
        for o in list(self.link_col.objects):
            if o.name not in self.l_before:
                self.link_col.objects.unlink(o)
        for lc in reversed(self.unexcluded):
            try:
                lc.exclude = True
            except ReferenceError:
                pass  # the collection was removed by the build
        if self.batch and self.park_col is not None:
            self._park_new()
            self._unpark()
        if not self.nested:
            if self.override is not None:
                self.override.__exit__(*exc)
            else:
                self.window.view_layer = self.main_vl
        return False
