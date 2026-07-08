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
        r"\btime\.(time|perf_counter|monotonic|process_time)\s*\(",
        "wall-clock time is nondeterministic",
    ),
    (r"\bdatetime\.(now|today|utcnow)\s*\(", "wall-clock datetime is nondeterministic"),
]


def test_no_nondeterministic_sources():
    compiled = [(re.compile(p), msg) for p, msg in _BANNED]
    violations = []
    for path in sorted(_GENERATORS_DIR.rglob("*.py")):
        if path.name in _EXCLUDE:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            for rx, msg in compiled:
                if rx.search(code):
                    rel = path.relative_to(_GENERATORS_DIR.parent.parent)
                    violations.append(f"{rel}:{lineno}: {line.strip()}  -> {msg}")
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
        sigs.append(
            (
                len(mesh.vertices),
                len(mesh.polygons),
                hashlib.sha256(np.round(co, _ROUND).tobytes()).hexdigest(),
                hashlib.sha256(np.round(world, _ROUND).tobytes()).hexdigest(),
            )
        )
    return sorted(sigs)


def _build(pathspec: str, kind: str) -> list[tuple]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    func = import_item(pathspec)
    if kind == "Material":
        res = func(
            rng=np.random.default_rng(SEED), vector=pf.nodes.shader.coord().object
        )
        plane = pf.ops.primitives.mesh_plane(size=1)
        pf.ops.object.set_material(plane, material=res)
    else:
        func(rng=np.random.default_rng(SEED))
    return _scene_geometry_signature()


# Generators we cannot assert deterministic yet, with the blocking reason.
# These flip to expected-pass once the dependency lands.
_DETERMINISM_BLOCKED = {}


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
            marks = ()
            if short in _DETERMINISM_BLOCKED:
                marks = pytest.mark.skip(reason=_DETERMINISM_BLOCKED[short])
            yield pytest.param(name, category, id=short, marks=marks)


@pytest.mark.parametrize("pathspec, kind", _determinism_params())
def test_generator_determinism(pathspec, kind):
    first = _build(pathspec, kind)
    assert first, f"{pathspec}: produced no mesh geometry"
    second = _build(pathspec, kind)
    assert first == second, (
        f"{pathspec}: geometry differs between two builds of seed {SEED} "
        f"(nondeterministic generation)"
    )
