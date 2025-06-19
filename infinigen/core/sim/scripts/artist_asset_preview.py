from typing import Any, Dict, Optional

import bpy
import mathutils
import numpy as np

frame = 1
obj = bpy.context.active_object

debug_frame = 79


def select_none():
    if hasattr(bpy.context, "active_object") and bpy.context.active_object is not None:
        bpy.context.active_object.select_set(False)
    if hasattr(bpy.context, "selected_objects"):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)


def select(objs: bpy.types.Object | list[bpy.types.Object]):
    select_none()
    if not isinstance(objs, list):
        objs = [objs]
    for o in objs:
        if o.name not in bpy.context.scene.objects:
            raise ValueError(f"Object {o.name=} not in scene and cant be selected")
        o.select_set(True)


class ViewportMode:
    def __init__(self, obj, mode):
        self.obj = obj
        self.mode = mode

    def __enter__(self):
        self.orig_active = bpy.context.active_object
        bpy.context.view_layer.objects.active = self.obj
        self.orig_mode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode=self.mode)

    def __exit__(self, *args):
        bpy.context.view_layer.objects.active = self.obj
        bpy.ops.object.mode_set(mode=self.orig_mode)
        bpy.context.view_layer.objects.active = self.orig_active


class SelectObjects:
    def __init__(self, objects, active=0):
        self.objects = list(objects) if hasattr(objects, "__iter__") else [objects]
        self.active = active

        self.saved_objs = None
        self.saved_active = None

    def _check_selectable(self):
        unlinked = [o for o in self.objects if o.name not in bpy.context.scene.objects]
        if len(unlinked) > 0:
            raise ValueError(
                f"{SelectObjects.__name__} had objects {unlinked=} which are not in bpy.context.scene.objects and cannot be selected"
            )

        hidden = [o for o in self.objects if o.hide_viewport]
        if len(hidden) > 0:
            raise ValueError(
                f"{SelectObjects.__name__} had objects {hidden=} which are hidden and cannot be selected"
            )

    def _get_intended_active(self):
        if isinstance(self.active, int):
            if self.active >= len(self.objects):
                return None
            else:
                return self.objects[self.active]
        else:
            return self.active

    def _validate(self, error=False):
        if error:
            raise ValueError(str)

        difference = set(self.objects) - set(bpy.context.selected_objects)
        if len(difference):
            print(
                f"{SelectObjects.__name__} failed to select {self.objects=}, result was {bpy.context.selected_objects=}. "
                "The most common cause is that the objects are in a collection with col.hide_viewport=True"
            )

        intended = self._get_intended_active()
        if intended is not None and bpy.context.active_object != intended:
            print(
                f"{SelectObjects.__name__} failed to set active object to {intended=}, result was {bpy.context.active_object=}"
            )

    def __enter__(self):
        self.saved_objects = list(bpy.context.selected_objects)
        self.saved_active = bpy.context.active_object

        select_none()
        select(self.objects)

        intended = self._get_intended_active()
        if intended is not None:
            bpy.context.view_layer.objects.active = intended

        self._validate()

    def __exit__(self, *_):
        # our saved selection / active objects may have been deleted, update them to only include valid ones
        def enforce_not_deleted(o):
            try:
                return o if o.name in bpy.data.objects else None
            except ReferenceError:
                return None

        self.saved_objects = [enforce_not_deleted(o) for o in self.saved_objects]
        self.saved_objects = [o for o in self.saved_objects if o is not None]

        select_none()
        select(self.saved_objects)
        if self.saved_active is not None:
            bpy.context.view_layer.objects.active = enforce_not_deleted(
                self.saved_active
            )


def is_join(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a join node
    """
    return node.bl_idname == "GeometryNodeJoinGeometry"


def is_joint(node: bpy.types.Node) -> bool:
    """
    Returns true if the current node is a joint node.
    """
    if node.bl_idname == "GeometryNodeGroup":
        if (
            "Hinge Joint" in node.node_tree.name
            or "Sliding Joint" in node.node_tree.name
        ):
            return True
    return False


def deep_clone_obj(obj, keep_modifiers=False, keep_materials=False):
    """
    Deep clones the given object.
    """
    new_obj = obj.copy()
    new_obj.data = obj.data.copy()
    if not keep_modifiers:
        for mod in new_obj.modifiers:
            new_obj.modifiers.remove(mod)
    if not keep_materials:
        while len(new_obj.data.materials) > 0:
            new_obj.data.materials.pop()
    bpy.context.collection.objects.link(new_obj)
    return new_obj


def inject_store_named_attr(
    link: bpy.types.NodeLink,
    data_type: str = "INT",
    default_name: Optional[str] = None,
    default_value: Optional[Any] = None,
):
    """
    Injects a store named attribute node of type integer between
    two nodes given the link connecting them.
    """
    parent_socket, child_socket = link.to_socket, link.from_socket

    # remove the link between parent and child node
    node_tree = link.id_data
    node_tree.links.remove(link)

    # create a new store attribute node
    store_named_attr_node = node_tree.nodes.new("GeometryNodeStoreNamedAttribute")
    store_named_attr_node.data_type = data_type
    if default_name:
        store_named_attr_node.inputs["Name"].default_value = default_name
    if default_value:
        store_named_attr_node.inputs["Value"].default_value = default_value

    # connects the new node in between the parent and child nodes
    node_tree.links.new(child_socket, store_named_attr_node.inputs["Geometry"])
    node_tree.links.new(store_named_attr_node.outputs["Geometry"], parent_socket)

    return store_named_attr_node


def read_attr(obj: bpy.types.Object, name: str, dtype=int):
    """
    Reads the attribute data from an object.
    """
    attr = obj.data.attributes[name]
    n = len(obj.data.vertices)
    data = np.empty(n, dtype=dtype)
    attr.data.foreach_get("value", data)
    return data


def spawn_point_cloud(name, pts, edges=None):
    if edges is None:
        edges = []

    mesh = bpy.data.meshes.new(name=name)
    mesh.from_pydata(pts, edges, [])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def spawn_vert(name="vert"):
    return spawn_point_cloud(name, np.zeros((1, 3)))


def extract_vertex_mask(
    obj: bpy.types.Object, vertex_mask: np.array, nonempty=False
) -> bpy.types.Object:
    if not vertex_mask.any():
        if nonempty:
            raise ValueError(f"extract_vertex_mask({obj.name=}) got empty mask")
        return spawn_vert()

    orig_hide_viewport = obj.hide_viewport
    obj.hide_viewport = False

    # Switch to Edit mode, duplicate the selection, and separate it
    with SelectObjects(obj, active=0):
        with ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type="VERT")
            bpy.ops.mesh.select_all(action="DESELECT")

        # select vertices based on the mask
        for vert in obj.data.vertices:
            vert.select = vertex_mask[vert.index]
        if nonempty and len([v for v in obj.data.vertices if v.select]) == 0:
            raise ValueError(
                f"extract_vertex_mask({obj.name=}, {nonempty=}) failed to select vertices"
            )

        with ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")

        res = next((o for o in bpy.context.selected_objects if o != obj), None)

    obj.hide_viewport = orig_hide_viewport

    if nonempty:
        if res is None:
            raise ValueError(
                f"extract_vertex_mask({obj.name=} got {res=} for {vertex_mask.mean()=})"
            )
        if len(res.data.vertices) == 0:
            raise ValueError(
                f"extract_vertex_mask({obj.name=}) got {res=} with {len(res.data.vertices)=}"
            )

    return res


def get_mesh_geometry(obj: bpy.types.Object, attrs: Dict) -> bpy.types.Object:
    """
    Returns the mesh geometry corresponding to the given attributes.
    """
    # get the vertex mask based on the attributes
    vertex_mask = np.ones(len(obj.data.vertices), dtype=bool)
    for attr, val in attrs.items():
        # handles switch cases where certain joints may not be a
        # part of the final geometry
        if attr not in obj.data.attributes:
            continue
        attr = obj.data.attributes[attr]
        n = len(obj.data.vertices)
        data = np.empty(n, dtype=int)
        attr.data.foreach_get("value", data)
        # data = read_attr_data(obj, attr)
        vertex_mask = vertex_mask & (data == int(val))

    # extract the mesh based on the vertex mask
    obj_clone = deep_clone_obj(obj, keep_modifiers=True, keep_materials=True)
    mesh_obj = extract_vertex_mask(obj_clone, vertex_mask)
    bpy.data.objects.remove(bpy.data.objects[obj_clone.name], do_unlink=True)
    return mesh_obj


def is_point_inside_bvh(bvh, point, obj, epsilon=1e-4):
    """
    Determines if a point is inside a BVH mesh (obj) using ray casting.

    A point is inside an object if the number of intersections along a ray is odd.

    Parameters:
        bvh: BVH tree of the target object
        point: The 3D point to test
        obj: The object whose BVH tree we are testing against
        epsilon: A small threshold to filter out numerical precision errors

    Returns:
        bool: True if the point is inside the object, False otherwise.
    """
    global frame
    directions = [
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, 0, 1)),
        mathutils.Vector((-1, 0, 0)),
        mathutils.Vector((0, -1, 0)),
        mathutils.Vector((0, 0, -1)),
    ]

    inside_count = 0
    for direction in directions:
        hit_loc, normal, index, distance = bvh.ray_cast(point, direction)

        if hit_loc is not None:
            # Make sure the hit point is not too close to the original point (avoiding surface hits)
            if (hit_loc - point).length > epsilon:
                inside_count += 1

    collides = inside_count == len(directions)

    if collides >= 5:
        print(point, obj.name)

    return collides


def check_mesh_penetration(obj1, obj2, epsilon=1e-5):
    """
    Checks if two Blender objects truly penetrate each other (not just touch).

    Parameters:
        obj1, obj2: Blender objects (must have mesh data)
        epsilon: Small threshold for numerical precision

    Returns:
        bool: True if objects actually **penetrate**, False if they only touch.
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Generate BVH trees for both objects
    bvh1 = mathutils.bvhtree.BVHTree.FromObject(obj1, depsgraph)
    bvh2 = mathutils.bvhtree.BVHTree.FromObject(obj2, depsgraph)

    # Step 1: Check if their bounding volumes overlap
    overlapping_faces = bvh1.overlap(bvh2)
    if not overlapping_faces:
        return False  # No intersection at all

    # Step 2: Check if at least one vertex of obj1 is inside obj2
    for vert in obj1.data.vertices:
        world_vert = obj1.matrix_world @ vert.co  # Convert to world coordinates
        if is_point_inside_bvh(bvh2, world_vert, obj2, epsilon):
            return True  # obj1 has a vertex inside obj2 -> intersection detected

    # Step 3: Check if at least one vertex of obj2 is inside obj1
    for vert in obj2.data.vertices:
        world_vert = obj2.matrix_world @ vert.co
        if is_point_inside_bvh(bvh1, world_vert, obj1, epsilon):
            return True  # obj2 has a vertex inside obj1 -> intersection detected

    return False


def clone_orig_and_inject():
    """
    Clones the object and injects the attributes
    """
    obj_clone = deep_clone_obj(obj, keep_modifiers=True, keep_materials=True)
    assert len(obj_clone.modifiers) == 1
    obj_clone.modifiers[0].node_group = obj.modifiers[0].node_group.copy()

    def apply_to_node_tree(node_tree: bpy.types.NodeTree):
        joint_nodes = []
        join_nodes = []
        for node in node_tree.nodes:
            if is_joint(node):
                joint_nodes.append(node)
            elif is_join(node):
                join_nodes.append(node)
            elif node.bl_idname == "GeometryNodeGroup" and not is_joint(node):
                node.node_tree = node.node_tree.copy()
                apply_to_node_tree(node.node_tree)

        # adds attributes for joint nodes
        for node in joint_nodes:
            node.node_tree = node.node_tree.copy()
            parent_link = node.inputs["Parent"].links[0]
            child_link = node.inputs["Child"].links[0]

            # adds attribute nodes for the parent and child links
            inject_store_named_attr(
                parent_link, default_name=f"{node.name}_joint", default_value=0
            )
            inject_store_named_attr(
                child_link, default_name=f"{node.name}_joint", default_value=1
            )

        # adds attributes for join nodes
        for node in join_nodes:
            for i, link in enumerate(node.inputs["Geometry"].links):
                inject_store_named_attr(
                    link, default_name=f"{node.name}_join", default_value=i
                )

    apply_to_node_tree(obj_clone.modifiers[0].node_group)
    return obj_clone


injected_obj = clone_orig_and_inject()


def clone_and_apply_mod(obj_to_clone: bpy.types.Object):
    """
    Clones the object and applies the modifier.
    """
    obj_clone = deep_clone_obj(injected_obj, keep_modifiers=True, keep_materials=True)
    bpy.context.view_layer.objects.active = obj_clone
    bpy.ops.object.select_all(action="DESELECT")
    obj_clone.select_set(state=True)
    bpy.ops.object.modifier_apply(modifier=obj_clone.modifiers[0].name)

    return obj_clone


def get_body_attrs(obj: bpy.types.Object):
    """
    Gets the attributes associated with different bodies.
    """
    attr_names = []
    for attr in obj.data.attributes:
        if "join" in attr.name:  # includes both "join" and "joint"
            attr_names.append(attr.name)

    attr_vals = []
    for name in attr_names:
        data = read_attr(obj, name)
        attr_vals.append(data)

    geom_data = np.column_stack(attr_vals)
    body_attrs = np.unique(geom_data, axis=0)

    return attr_names, body_attrs


def checking_matching_joint_attrs(obj1: bpy.types.Object, obj2: bpy.types.Object):
    attr_names = []
    for attr in obj1.data.attributes:
        if "joint" in attr.name:
            attr_names.append(attr.name)
    obj1_attr_vals = []
    obj2_attr_vals = []
    for name in attr_names:
        obj1_attr_vals.append(read_attr(obj1, name))
        obj2_attr_vals.append(read_attr(obj2, name))

    obj1_data = np.column_stack(obj1_attr_vals)
    obj2_data = np.column_stack(obj2_attr_vals)

    obj1_data = np.unique(obj1_data, axis=0)
    obj2_data = np.unique(obj2_data, axis=0)

    return (obj1_data == obj2_data).all()


def check_frame_collision():
    """
    Check if any body's geometries intersect. Returns true if no collision, false otherwise.
    """
    global frame, injected_obj
    obj_clone = clone_and_apply_mod(injected_obj)
    attr_names, body_attrs = get_body_attrs(obj_clone)

    meshes = []
    for ba in body_attrs:
        attrs = {key: ba[i] for i, key in enumerate(attr_names)}
        mesh = get_mesh_geometry(obj_clone, attrs)
        meshes.append(mesh)

    for i in range(len(meshes)):
        for j in range(i + 1, len(meshes)):
            mesh1 = meshes[i]
            mesh2 = meshes[j]

            if checking_matching_joint_attrs(mesh1, mesh2):
                continue

            do_overlap = check_mesh_penetration(mesh1, mesh2)
            if do_overlap:
                bpy.ops.outliner.orphans_purge()
                for mesh in meshes:
                    if mesh == mesh1 or mesh == mesh2:
                        continue
                    bpy.data.objects.remove(bpy.data.objects[mesh.name], do_unlink=True)
                bpy.data.objects.remove(
                    bpy.data.objects[injected_obj.name], do_unlink=True
                )
                raise ValueError(
                    f"Meshes are overlapping! Please clean all newly created meshes when fixed! The overlapping meshes are {mesh1.name} and {mesh2.name}"
                )

    bpy.data.objects.remove(bpy.data.objects[obj_clone.name], do_unlink=True)
    for mesh in meshes:
        bpy.data.objects.remove(bpy.data.objects[mesh.name], do_unlink=True)


def check_animation_collisions():
    """
    Step through animation and check if objects intersect.
    """
    global frame
    bpy.ops.screen.frame_offset(delta=1)

    if frame == bpy.data.scenes["Scene"].frame_end:
        # clear all unused data at the end of the animation
        bpy.data.objects.remove(bpy.data.objects[injected_obj.name], do_unlink=True)
        bpy.ops.outliner.orphans_purge()
        return None

    print(f"Frame {frame}. Stepping...")

    check_frame_collision()
    bpy.context.view_layer.objects.active = obj

    frame += 1

    return 0.02


bpy.ops.screen.frame_jump(end=False)
bpy.app.timers.register(check_animation_collisions)
