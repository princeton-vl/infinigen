import ast
import inspect
from typing import Callable, TypeVar

import numpy as np
import procfunc as pf
import pytest
from procfunc import compute_graph as cg
from procfunc.codegen import to_python
from procfunc.tracer import TraceLevel
from procfunc.util.manifest import import_item

from infinigen_v2.generators import GENERATORS_MANIFEST
from infinigen_v2.util.codestats import compute_stats

T = TypeVar("T")


def _manifest_params(df, defaults: dict):
    sub = df[["name", *defaults.keys()]].copy()
    for col, val in defaults.items():
        sub[col] = sub[col].fillna(val).astype(type(val))
    for row in sub.itertuples(index=False):
        yield pytest.param(*row, id=row[0])


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
    _code = to_python(graph, toplevel_as_maincall=False)
    try:
        ast.parse(_code)
    except SyntaxError as e:
        raise ValueError(f"Generated code has syntax error: {e}") from e
    stats = compute_stats(graph)
    assert stats["continuous_params"] >= min_parameters


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

    validate_trace_generator(material_sample, rng, min_parameters=min_parameters)


_OBJECT_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Object"},
    exclude={"name": ["LATER", "DECLINE"]},
    require_nonempty=["name"],
    min_entries=None,
)


@pytest.mark.parametrize(
    "pathspec, min_parameters",
    _manifest_params(_OBJECT_FUNCS, {"min_parameters": 2}),
)
def test_generators_object(rng, pathspec, min_parameters):
    func = import_item(pathspec)
    assert callable(func)
    res = func(rng=rng)
    assert isinstance(res, pf.MeshObject) or hasattr(res, "mesh"), res

    validate_trace_generator(func, rng, min_parameters=min_parameters)


_SCENE_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Scene"},
    exclude={"name": ["LATER", "DECLINE"]},
    require_nonempty=["name"],
    min_entries=0,
)


@pytest.mark.skip(reason="Scene generators are not implemented yet")
@pytest.mark.parametrize("pathspec", _SCENE_FUNCS["name"].values)
def test_generators_scene(pathspec, rng):
    raise NotImplementedError("Scene generators are not implemented yet")

    SceneClass = import_item(pathspec)
    SceneClass()

    dummy_material = pf.Material(
        surface=pf.nodes.shader.principled_bsdf(
            base_color=(0.8, 0.8, 0.8, 1.0), roughness=0.5
        )
    )

    def assets_to_dummies(node: cg.Node) -> cg.Node:
        if not isinstance(node, cg.FunctionCallNode):
            return node
        func_output_type = inspect.signature(node.func).return_annotation
        if func_output_type is pf.Object:
            return cg.FunctionCallNode(
                func=pf.primitive.cube,
                args=(),
                kwargs={},
            )
        elif func_output_type is pf.Material:
            return cg.FunctionCallNode(
                func=dummy_material,
                args=(),
                kwargs={},
            )
        return node

    # dummied_scenegen = gtr.transform_generator(
    #     pf.trace(generator, rng=rng), assets_to_dummies
    # )

    # _res = dummied_scenegen(rng=rng)
    # validate_generator(generator, rng)


_OBJECT_FUNCS = pf.util.manifest.filter_manifest(
    GENERATORS_MANIFEST,
    filter={"category": "Object"},
    exclude={"name": ["LATER", "DECLINE"]},
    require_nonempty=["name"],
    min_entries=None,
)


@pytest.mark.parametrize("pathspec", _OBJECT_FUNCS["name"].values)
def test_generators_mesh_object(pathspec, rng):
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

    # validate_generator(func, rng)
