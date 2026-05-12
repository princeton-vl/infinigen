import os
import sys
import threading
from contextlib import contextmanager

import bpy

SHADER_NODE_COUNT_FAIL = 1000


class ShaderTooComplexError(RuntimeError):
    pass


def count_material_nodes(material: bpy.types.Material) -> int:
    if not material.use_nodes or material.node_tree is None:
        return 0
    seen = set()

    def walk(tree):
        n = len(tree.nodes)
        for node in tree.nodes:
            if (
                node.type == "GROUP"
                and node.node_tree
                and node.node_tree.name not in seen
            ):
                seen.add(node.node_tree.name)
                n += walk(node.node_tree)
        return n

    return walk(material.node_tree)


def check_scene_shader_complexity(fail: int = SHADER_NODE_COUNT_FAIL):
    counts = {m.name: count_material_nodes(m) for m in bpy.data.materials if m.users}
    offenders = [(n, c) for n, c in counts.items() if c >= fail]
    if offenders:
        raise ShaderTooComplexError(
            f"materials exceed {fail} flattened nodes: {offenders}"
        )
    return counts


CYCLES_ERROR_PATTERNS = (
    "SVM stack",
    "shader graph",
    "shader compile",
    "Error: ",
)


class CyclesShaderError(RuntimeError):
    pass


@contextmanager
def detect_cycles_errors(replay: bool = True):
    sys.stdout.flush()
    sys.stderr.flush()

    old1, old2 = os.dup(1), os.dup(2)
    r, w = os.pipe()
    captured = bytearray()

    def pump():
        while chunk := os.read(r, 4096):
            if replay:
                os.write(old1, chunk)
            captured.extend(chunk)

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    try:
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        # the dup2's above replaced the pipe writers on fd 1+2;
        # reader thread hits EOF and exits — wait before closing old1/old2
        # since pump still writes to old1
        t.join(timeout=5)
        os.close(old1)
        os.close(old2)
        os.close(r)

    text = captured.decode(errors="replace")
    hits = [
        ln for ln in text.splitlines() if any(p in ln for p in CYCLES_ERROR_PATTERNS)
    ]
    if hits:
        raise CyclesShaderError("\n".join(hits))
