from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import List, Optional


class KinematicType(Enum):
    NONE = -1
    JOINT = 0
    ASSET = 1
    SWITCH = 2
    DUPLICATE = 3


class JointType(Enum):
    NONE = -1
    WELD = 0
    HINGE = 1
    SLIDING = 2
    BALL = 3


class KinematicNode:
    """
    Node part of a directed acyclic graph representing a kinematic
    tree.
    """

    joint_count = 0
    asset_count = 0
    switch_count = 0
    duplicate_count = 0

    def __init__(
        self,
        kinematic_type: KinematicType,
        joint_type: Optional[JointType] = None,
        idn: Optional[str] = None,
    ) -> None:
        """
        Initializes a KinematicNode instance given the type.
        """
        self.kinematic_type = kinematic_type
        self.joint_type = joint_type

        self.idn = ""
        if kinematic_type == KinematicType.JOINT:
            self.idn = f"joint{KinematicNode.joint_count}"
            KinematicNode.joint_count += 1
        elif kinematic_type == KinematicType.ASSET:
            self.idn = f"visasset{KinematicNode.asset_count}"
            KinematicNode.asset_count += 1
        elif kinematic_type == KinematicType.DUPLICATE:
            self.idn = f"duplicate{KinematicNode.duplicate_count}"
            KinematicNode.duplicate_count += 1
        elif kinematic_type == KinematicType.SWITCH:
            self.idn = f"switch{KinematicNode.switch_count}"
            KinematicNode.switch_count += 1

        if idn is not None:
            self.idn = idn

        self.children = defaultdict()

    @staticmethod
    def reset_counts():
        KinematicNode.joint_count = 0
        KinematicNode.asset_count = 0
        KinematicNode.switch_count = 0
        KinematicNode.duplicate_count = 0

    def set_idn(self, idn: str):
        """
        Sets the node identifier
        """
        self.idn = idn

    def add_child(self, attr_value: int, node: "KinematicNode") -> None:
        """
        Adds a child to the node. The key is equal to the value of the named
        attribute on the path from the current node to the child node.
        """
        self.children[attr_value] = node

    def get_all_children(self) -> List["KinematicNode"]:
        """
        Retruns a list of all the node's chidren.
        """
        res = []
        for key, child in self.children.items():
            res.append(child)
        return res

    def get_graph(self):
        """
        Returns the dictionary representation of a DAG starting at the node.
        This is a easy-to-parse verison of the kinematic graph.
        """
        graph = defaultdict(dict)
        graph[self.idn]["kinematic_type"] = self.kinematic_type.value
        graph[self.idn]["joint_type"] = (
            self.joint_type.value if self.joint_type is not None else -1
        )
        graph[self.idn]["children"] = {}
        for path_idx, child in self.children.items():
            graph[self.idn]["children"][path_idx] = child.idn
        for path_idx, child in self.children.items():
            graph.update(child.get_graph())
        return graph

    def __str__(self):
        res = f"--- {hex(id(self))} {self.idn, self.kinematic_type} ({self.joint_type if self.kinematic_type == KinematicType.JOINT else ''}) ---\n"
        for key, child in self.children.items():
            res += f"\t{key}: {hex(id(child))} ({child.idn} {child.kinematic_type})\n"
        return res


def kinematic_node_factory(
    kinematic_type: KinematicType,
    joint_type: JointType = JointType.NONE,
    idn: Optional[str] = None,
) -> KinematicNode:
    """
    Creates a node with the given kinematic type and optional joint type.
    """
    return KinematicNode(kinematic_type=kinematic_type, joint_type=joint_type, idn=idn)


def print_subgraph(node: KinematicNode) -> None:
    """
    Prints the acyclic graph starting at the node
    """
    print(node)
    for _, child in node.children.items():
        print_subgraph(child)
