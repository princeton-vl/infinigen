# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import ast
import types
from typing import Callable, TypeVar

import bpy
import numpy as np
import procfunc as pf
import pytest
from procfunc import codegen
from procfunc import compute_graph as cg
from procfunc.codegen import to_python
from procfunc.compute_graph.operators_info import OPERATORS_TO_FUNCTIONS, OperatorType
from procfunc.nodes import NODE_OPERATOR_TABLE
from procfunc.tracer import TraceLevel
from procfunc.util.manifest import import_item

from infinigen2 import GENERATORS_MANIFEST
from infinigen2.exporters.render_error_check import (
    assert_displacement_coords_safe,
    assert_shader_complexity_ok,
    assert_uv_coords_satisfied,
    degenerate_geometry_defects,
)
from infinigen2.util.codestats.setup import build_model_from_compute_graph

T = TypeVar("T")
SEED = 0


def _assert_render_valid(objects: list[pf.MeshObject]):
    assert_displacement_coords_safe(objects)
    assert_shader_complexity_ok(objects)
    assert_uv_coords_satisfied(objects)


_SKIP_TESTS = GENERATORS_MANIFEST.set_index("name")["skip_tests"].dropna().to_dict()


def _manifest_params(df, defaults: dict):
    sub = df[["name"]].copy()
    for col, val in defaults.items():
        if col in df.columns:
            sub[col] = df[col].fillna(val).astype(type(val))
        else:
            sub[col] = val
    for row in sub.itertuples(index=False):
        reason = _SKIP_TESTS.get(row[0])
        marks = () if reason is None else pytest.mark.skip(reason=reason)
        yield pytest.param(*row, id=row[0], marks=marks)


def _private_function_calls(graph: cg.ComputeGraph) -> list[str]:
    nodes = [
        node
        for subgraph in cg.traverse_nested_graphs(graph)
        for node in cg.traverse_depth_first(subgraph)
        if isinstance(node, cg.FunctionCallNode)
    ]
    names = {
        f"{node.func.__module__}.{node.func.__name__}"
        for node in nodes
        if node.func.__name__.startswith("_")
    }
    return sorted(names)


def validate_trace_generator(
    generator_func: Callable[..., T],
    rng: np.random.Generator,
    min_parameters: int = 2,
    **generator_kwargs,
):
    graph = pf.trace(
        generator_func,
        trace_level=TraceLevel.RANDOM_CONTROL,
        rng=rng,
        **generator_kwargs,
    )
    private_calls = _private_function_calls(graph)
    assert not private_calls, (
        f"Trace resolved to private functions {private_calls}; "
        "decorate their public caller with @pf.tracer.generator"
    )
    _code = to_python(graph, toplevel_as_maincall=False)
    try:
        ast.parse(_code)
    except SyntaxError as e:
        raise ValueError(f"Generated code has syntax error: {e}") from e
    model = build_model_from_compute_graph(graph)
    assert model["n_continuous_params"] >= min_parameters


_MATERIAL_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Material"},
    exclude={"name": ["LATER", "DECLINE"]},
    require_nonempty=["name"],
    min_entries=1,
)


@pytest.mark.parametrize(
    "pathspec, min_parameters",
    _manifest_params(_MATERIAL_FUNCS, {"min_parameters": 2}),
)
def test_generators_material(rng, pathspec, min_parameters):
    material_sample = import_item(pathspec)

    vector = pf.nodes.shader.coord().object
    res = material_sample(rng=rng, vector=vector)

    sockets = [
        getattr(res, socket, None)
        for socket in ["surface", "displacement", "volume"]
        if hasattr(res, socket)
    ]
    assert len(sockets) > 0, f"No sockets in {res=}"

    plane = pf.ops.primitives.mesh_plane(size=1)
    pf.ops.object.set_material(plane, material=res)
    _assert_render_valid([plane])

    validate_trace_generator(material_sample, rng, min_parameters=min_parameters)


_DISPLACEMENT_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Displacement"},
    require_nonempty=["name"],
    min_entries=1,
)


@pytest.mark.parametrize(
    "pathspec, min_parameters",
    _manifest_params(_DISPLACEMENT_FUNCS, {"min_parameters": 2}),
)
def test_generators_displacement(rng, pathspec, min_parameters):
    displacement_sample = import_item(pathspec)
    vector = pf.nodes.shader.coord().object
    displacement = displacement_sample(rng=rng, vector=vector)
    material = pf.Material(
        surface=pf.nodes.shader.diffuse_bsdf(color=(0.35, 0.3, 0.25, 1.0)),
        displacement=displacement,
    )
    plane = pf.ops.primitives.mesh_plane(size=1)
    pf.ops.object.set_material(plane, material=material)
    _assert_render_valid([plane])
    validate_trace_generator(displacement_sample, rng, min_parameters=min_parameters)


def test_displacement_integration_commands_present() -> None:
    assert _DISPLACEMENT_FUNCS["integration_test_string"].notna().all()


_OBJECT_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Object"},
    exclude={"name": ["LATER", "DECLINE"]},
    require_nonempty=["name"],
    min_entries=None,
)

_PRIMITIVES_TRACE_EXCLUDES = {
    "infinigen2.objects.bed.bed_rand",
    "infinigen2.objects.chair.chair_back_rand",
    "infinigen2.objects.desk.desk_rand",
    "infinigen2.objects.table.table_cocktail_rand",
    "infinigen2.objects.table.table_coffee_rand",
    "infinigen2.objects.table.table_dining_rand",
    "infinigen2.objects.table.table_side_rand",
}
# curtain cloth is a thin open tube booleaned against its rod, non-manifold by design
_DEGENERATE_GEOMETRY_EXCLUDES = {
    "infinigen2.objects.window.curtain_rand",
    "infinigen2.objects.window.window_composite_rand",
    "infinigen2.objects.window.window_rectangular_composite_rand",
}
_PRIMITIVES_TRACE_FUNCS = _OBJECT_FUNCS[
    ~_OBJECT_FUNCS["name"].isin(_PRIMITIVES_TRACE_EXCLUDES | set(_SKIP_TESTS))
]


def _primitives_trace_params():
    for row in _PRIMITIVES_TRACE_FUNCS[["name"]].itertuples(index=False):
        pathspec = row[0]
        marks = []
        if pathspec == "infinigen2.objects.toilet.toilet_rand":
            marks.append(
                pytest.mark.xfail(
                    reason=(
                        "Linux CI and a local current-develop merge expose a direct/traced "
                        "execution parity gap for toilet_rand (7367 vs 7655 evaluated faces)"
                    ),
                    strict=True,
                )
            )
        yield pytest.param(pathspec, id=pathspec, marks=marks)


@pytest.mark.parametrize(
    "pathspec, min_parameters",
    _manifest_params(_OBJECT_FUNCS, {"min_parameters": 2}),
)
def test_generators_object(rng, pathspec, min_parameters):
    func = import_item(pathspec)
    assert callable(func)
    res = func(rng=rng)
    assert not isinstance(res, pf.MeshObject), (
        f"Expected NamedTuple with mesh field, got bare MeshObject from {pathspec}"
    )
    assert hasattr(res, "mesh"), f"Result missing .mesh from {pathspec}"
    assert isinstance(res.mesh, pf.MeshObject), (
        f".mesh should be MeshObject, got {type(res.mesh)} from {pathspec}"
    )
    _assert_render_valid([res.mesh])
    defects = degenerate_geometry_defects([res.mesh])

    validate_trace_generator(func, rng, min_parameters=min_parameters)

    if pathspec in _DEGENERATE_GEOMETRY_EXCLUDES:
        return
    assert defects == {}, f"{pathspec} produced degenerate geometry: {defects}"


def _evaluated_mesh_geometry(
    res: object,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    obj = res if isinstance(res, pf.MeshObject) else res.mesh
    bpy_obj = obj.item()
    evaluated = bpy_obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    polygons = [tuple(p.vertices) for p in mesh.polygons]
    matrix = np.array(bpy_obj.matrix_world)
    evaluated.to_mesh_clear()
    return coords.reshape(-1, 3), polygons, matrix


def _build_func_resolution_map(toplevel_graph) -> tuple[dict, list[str]]:
    func_resolution = {
        op_func: op_type for op_type, op_func in OPERATORS_TO_FUNCTIONS.items()
    }
    for oprow in NODE_OPERATOR_TABLE:
        if oprow.operator_type is not OperatorType.NOOP:
            func_resolution[oprow.pf_func] = oprow.operator_type
    for name in dir(pf):
        obj = getattr(pf, name)
        if not name.startswith("_") and isinstance(obj, type):
            if not isinstance(obj, types.ModuleType):
                func_resolution[obj] = f"pf.{name}"
    default_resolution, import_lines = codegen.default_func_resolution_map(
        toplevel_graph, skip_funcs=set(func_resolution)
    )
    func_resolution.update(default_resolution)
    specs = [toplevel_graph.outputs.spec]
    while specs:
        spec = specs.pop()
        specs.extend(spec.items)
        if spec.container is None or spec.container in func_resolution:
            continue
        module = spec.container.__module__
        name = spec.container.__name__
        import_lines.append(f"from {module} import {name}")
        func_resolution[spec.container] = name
    return func_resolution, import_lines


def _trace_roundtrip(func: Callable, trace_level: TraceLevel) -> object:
    def generate(rng: object) -> object:
        return func(rng=rng)

    rng_node = cg.InputPlaceholderNode(
        name="rng", default_value=None, metadata={"varname": "rng"}
    )
    rng = pf.tracer.RngProxy(rng_node, np.random.default_rng(SEED), dirty=False)
    graph = pf.trace(generate, trace_level=trace_level, rng=rng)
    func_resolution, import_lines = _build_func_resolution_map(graph)
    import_lines.append("from numpy.random import Generator")
    code = to_python(
        graph,
        func_resolution=func_resolution,
        import_lines=import_lines,
        toplevel_as_maincall=False,
    )
    namespace = {}
    exec(compile(code, f"<{trace_level.name}>", "exec"), namespace)  # noqa: S102
    return namespace[graph.name](rng=np.random.default_rng(SEED))


@pytest.mark.slow
@pytest.mark.parametrize(
    "trace_level", [TraceLevel.PRIMITIVES], ids=lambda level: level.name
)
@pytest.mark.parametrize("pathspec", _primitives_trace_params())
def test_generators_object_trace_reproduces_geometry(
    pathspec: str, trace_level: TraceLevel
) -> None:
    func = import_item(pathspec)

    direct = func(rng=np.random.default_rng(SEED))
    direct_coords, direct_polygons, direct_matrix = _evaluated_mesh_geometry(direct)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    traced = _trace_roundtrip(func, trace_level)
    traced_coords, traced_polygons, traced_matrix = _evaluated_mesh_geometry(traced)

    same_topology = direct_polygons == traced_polygons
    assert same_topology, (
        f"{pathspec}: {trace_level.name} changed evaluated mesh topology "
        f"({len(direct_polygons)} direct faces, {len(traced_polygons)} traced faces)"
    )
    assert direct_coords.shape == traced_coords.shape
    np.testing.assert_allclose(
        direct_coords,
        traced_coords,
        atol=1e-5,
        err_msg=f"{pathspec}: {trace_level.name} changed evaluated vertex positions",
    )
    np.testing.assert_allclose(
        direct_matrix,
        traced_matrix,
        atol=1e-5,
        err_msg=f"{pathspec}: {trace_level.name} changed the world transform",
    )


_MANIFEST_NAMES = [n for n in GENERATORS_MANIFEST["name"].values if isinstance(n, str)]


@pytest.mark.parametrize("name", _MANIFEST_NAMES, ids=_MANIFEST_NAMES)
def test_generators_naming_validate(name: str) -> None:
    func_name = name.rsplit(".", 1)[-1]
    suggested = func_name[: -len("_distribution")] + "_rand"
    assert not func_name.endswith("_distribution"), (
        f"Manifest entry {name!r} ends in '_distribution'; "
        f"use the '_rand' suffix instead (e.g. rename to '{suggested}')."
    )
