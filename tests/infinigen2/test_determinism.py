"""Determinism guards for v2 generation.

Two concerns:
  1. A static lint ban on nondeterministic sources (uuid, stdlib random, global
     numpy RNG, wall-clock, ...) in the generator tree.
  2. A dynamic scan: build a generator twice from the same seed and assert the
     resulting geometry is byte-identical (run, capture, flush, run, compare).

Geometry is compared name-independently: object *names* are only deterministic
once procfunc's bpy_nocollide_data_name fix (drop uuid4) lands via the 0.33
migration; geometry/transforms are deterministic today.
"""

import hashlib
import random
import re
from pathlib import Path

import bpy
import numpy as np
import procfunc as pf
import pytest
from procfunc.util.manifest import import_item

import infinigen2 as _generators_pkg
from infinigen2 import GENERATORS_MANIFEST

SEED = 42

_GENERATORS_DIR = Path(_generators_pkg.__file__).parent

# Driver modules that legitimately use os.urandom for the top-level seed and time
# for profiling; they are not generator/asset sources so are exempt from the scan.
_EXCLUDE = {"generate.py"}

# A line carrying this marker in a comment is exempted from the scan.
_IGNORE_MARKER = "validate-ignore: test_determinism"

# (regex, human reason). Scoped to the generator tree only — drivers (generate.py)
# legitimately use os.urandom for the top-level seed and time for profiling.
_BANNED = [
    (r"^\s*import\s+uuid(\s|$|\.)", "uuid is random; reproduce from the seeded rng"),
    (r"^\s*from\s+uuid\s+import", "uuid is random; reproduce from the seeded rng"),
    (r"\buuid\.", "uuid is random; reproduce from the seeded rng"),
    (
        r"^\s*import\s+random(\s|$)",
        "stdlib random has global state; use the seeded rng",
    ),
    (
        r"^\s*from\s+random\s+import",
        "stdlib random has global state; use the seeded rng",
    ),
    (r"^\s*import\s+secrets(\s|$)", "secrets is cryptographically random"),
    (r"\bos\.urandom\b", "os.urandom is random; reproduce from the seeded rng"),
    (
        r"\bnp\.random\.(seed|rand|randn|randint|random_sample|random|ranf|sample"
        r"|choice|bytes|shuffle|permutation|permuted|integers|uniform|normal"
        r"|standard_normal|beta|binomial|gamma|poisson)\b",
        "numpy global RNG; thread the seeded np.random.Generator (rng) instead",
    ),
    (
        r"\b(default_rng|RandomState|np\.random\.Generator)\s*\(",
        "fresh rng stream; take a seeded rng argument and spawn from it",
    ),
    (
        r"^\s*(from\s+numpy\.random\s+import|import\s+numpy\.random)",
        "numpy.random import bypasses the rng scan; thread the seeded rng",
    ),
    (
        r"(?<![\w.])id\s*\(",
        "id() addresses are reused after free and vary run to run; key by the "
        "object itself or by bpy session_uid",
    ),
    (
        r"(?<![\w.])hash\s*\(",
        "str hash() is randomized per process (PYTHONHASHSEED)",
    ),
    (
        r"\btime\.(time|perf_counter|monotonic|process_time)\s*\(",
        "wall-clock time is nondeterministic",
    ),
    (r"\bdatetime\.(now|today|utcnow)\s*\(", "wall-clock datetime is nondeterministic"),
]


def _scan_line(compiled, path, lineno, line):
    if _IGNORE_MARKER in line:
        return None
    code = line.split("#", 1)[0]
    for rx, msg in compiled:
        if not rx.search(code):
            continue
        rel = path.relative_to(_GENERATORS_DIR.parent.parent)
        return f"{rel}:{lineno}: {line.strip()}  -> {msg}"
    return None


def test_no_nondeterministic_sources():
    compiled = [(re.compile(p), msg) for p, msg in _BANNED]
    violations = []
    for path in sorted(_GENERATORS_DIR.rglob("*.py")):
        if path.name in _EXCLUDE:
            continue
        lines = enumerate(path.read_text().splitlines(), start=1)
        found = (_scan_line(compiled, path, n, line) for n, line in lines)
        violations.extend(f for f in found if f)
    assert not violations, "Nondeterministic sources in generators:\n" + "\n".join(
        violations
    )


# Coords are rounded before hashing: Blender's noise-texture geonode eval is not
# bit-reproducible (~1e-6 jitter, rng-independent), the codegen-fidelity noise floor.
_ROUND = 5


def _scene_geometry_signature() -> list[tuple]:
    """Name-independent signature of all mesh geometry currently in bpy."""
    sigs = []
    for obj in bpy.data.objects:
        mesh = obj.data
        if obj.type != "MESH" or mesh is None or len(mesh.vertices) == 0:
            continue
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        world = np.array(obj.matrix_world, dtype=np.float64)
        co_hash = hashlib.sha256(np.round(co, _ROUND).tobytes()).hexdigest()
        sigs.append((len(mesh.vertices), len(mesh.polygons), co_hash, world))
    return sorted(sigs, key=lambda sig: sig[:3])


def _signatures_match(first: list[tuple], second: list[tuple]) -> bool:
    if len(first) != len(second):
        return False
    for a, b in zip(first, second):
        if a[:3] != b[:3]:
            return False
        if not np.allclose(a[3], b[3], atol=10.0**-_ROUND):
            return False
    return True


def _global_rng_state() -> tuple:
    np_state = np.random.get_state()
    return random.getstate(), np_state[1].tobytes(), np_state[2:]


def _build(pathspec: str, kind: str, room_count: int | None) -> list[tuple]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before = _global_rng_state()
    _build_into_scene(pathspec, kind, room_count)
    assert _global_rng_state() == before, (
        f"{pathspec}: consumed the global random/np.random stream instead of the "
        f"seeded rng argument"
    )
    return _scene_geometry_signature()


def _build_into_scene(pathspec: str, kind: str, room_count: int | None) -> None:
    func = import_item(pathspec)
    if kind == "Material":
        res = func(
            rng=np.random.default_rng(SEED), vector=pf.nodes.shader.coord().object
        )
        plane = pf.ops.primitives.mesh_plane(size=1)
        pf.ops.object.set_material(plane, material=res)
    elif room_count is None:
        func(rng=np.random.default_rng(SEED))
    else:
        func(rng=np.random.default_rng(SEED), room_count=room_count)


# Generators we cannot assert deterministic yet, with the blocking reason.
# These flip to expected-pass once the dependency lands.
_DETERMINISM_BLOCKED = {}


_SKIP_TESTS = GENERATORS_MANIFEST.set_index("name")["skip_tests"].dropna().to_dict()


def _determinism_params():
    for category in ("Object", "Scene"):
        df = pf.util.manifest.filter_manifest(
            GENERATORS_MANIFEST,
            filter={"category": category},
            exclude={"name": ["LATER", "DECLINE"]},
            require_nonempty=["name"],
            min_entries=0,
        )
        for name in df["name"].values:
            short = name.rsplit(".", 1)[-1]
            reason = _SKIP_TESTS.get(name, _DETERMINISM_BLOCKED.get(short))
            marks = () if reason is None else pytest.mark.skip(reason=reason)
            room_count = 1 if short == "house_furnished_rand" else None
            yield pytest.param(name, category, room_count, id=short, marks=marks)


@pytest.mark.parametrize("pathspec, kind, room_count", _determinism_params())
def test_generator_determinism(pathspec, kind, room_count):
    first = _build(pathspec, kind, room_count)
    assert first, f"{pathspec}: produced no mesh geometry"
    second = _build(pathspec, kind, room_count)
    assert _signatures_match(first, second), (
        f"{pathspec}: geometry differs between two builds of seed {SEED} "
        f"(nondeterministic generation)"
    )
