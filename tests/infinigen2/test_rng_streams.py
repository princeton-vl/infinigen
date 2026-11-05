import ast
from pathlib import Path
from typing import NamedTuple

import infinigen2 as _generators_pkg

_GENERATORS_DIR = Path(_generators_pkg.__file__).parent
_EXCLUDE = {"generate.py"}
_IGNORE_MARKER = "validate-ignore: test_rng_streams"

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMP_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


class Ctx(NamedTuple):
    loops: tuple[int, ...]
    branches: tuple[tuple[int, str], ...]
    shadowed: frozenset[str]


class RngUse(NamedTuple):
    name: str
    lineno: int
    col: int
    kind: str
    gated: bool
    ctx: Ctx


class RngBinding(NamedTuple):
    name: str
    end: tuple[int, int]
    loops: tuple[int, ...]


class Scan(NamedTuple):
    rng_names: set[str]
    bindings: list[RngBinding]
    uses: list[RngUse]


def _is_spawn_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    return isinstance(node.func, ast.Attribute) and node.func.attr == "spawn"


def _target_names(target: ast.AST) -> list[ast.Name]:
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _params(fn: ast.AST) -> list[ast.arg]:
    return fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs


def _is_rng_param(fn: ast.AST, arg: ast.arg) -> bool:
    if isinstance(fn, ast.Lambda):
        return arg.arg == "r" or arg.arg.startswith("rng")
    return arg.annotation is not None and "RNG" in ast.unparse(arg.annotation)


def _rng_params(fn: ast.AST) -> set[str]:
    return {a.arg for a in _params(fn) if _is_rng_param(fn, a)}


def _stmt_end(node: ast.AST) -> tuple[int, int]:
    return (node.end_lineno, node.end_col_offset)


def _use_kind(call: ast.Call, name: str) -> tuple[str, bool]:
    func_src = ast.unparse(call.func)
    if func_src.startswith("pf.random."):
        return "draw", False
    if func_src == "pf.control.choice":
        return "choice", True
    owner = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(owner, ast.Name) and owner.id == name:
        return call.func.attr, False
    return func_src.split(".")[-1], False


def _call_rng_names(call: ast.Call) -> list[ast.Name]:
    names = [a for a in call.args if isinstance(a, ast.Name)]
    names += [k.value for k in call.keywords if isinstance(k.value, ast.Name)]
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        names.append(call.func.value)
    return names


def _bind(
    scan: Scan,
    target: ast.AST,
    spawned: bool,
    ctx: Ctx,
    end: tuple[int, int] | None = None,
) -> None:
    for name in _target_names(target):
        if name.id in ctx.shadowed:
            continue
        if spawned:
            scan.rng_names.add(name.id)
        name_end = end or (name.end_lineno, name.end_col_offset)
        scan.bindings.append(RngBinding(name.id, name_end, ctx.loops))


def _record_call(scan: Scan, call: ast.Call, ctx: Ctx) -> None:
    for arg in _call_rng_names(call):
        if arg.id in ctx.shadowed:
            continue
        kind, gated = _use_kind(call, arg.id)
        scan.uses.append(RngUse(arg.id, arg.lineno, arg.col_offset, kind, gated, ctx))


def _walk_block(scan: Scan, stmts: list[ast.stmt], ctx: Ctx) -> None:
    for stmt in stmts:
        _walk(scan, stmt, ctx)
        exits = (
            isinstance(stmt, ast.If) and stmt.body and isinstance(stmt.body[-1], _EXITS)
        )
        if exits:
            ctx = ctx._replace(branches=ctx.branches + ((id(stmt), "orelse"),))


def _walk_part(scan: Scan, part: ast.AST | list, ctx: Ctx) -> None:
    if isinstance(part, list) and part and isinstance(part[0], ast.stmt):
        _walk_block(scan, part, ctx)
        return
    children = part if isinstance(part, list) else [part]
    for child in children:
        if isinstance(child, ast.AST):
            _walk(scan, child, ctx)


def _walk_fields(scan: Scan, node: ast.AST, ctx: Ctx) -> None:
    for _, value in ast.iter_fields(node):
        _walk_part(scan, value, ctx)


def _walk_loop(scan: Scan, node: ast.AST, ctx: Ctx) -> None:
    inner = ctx._replace(loops=ctx.loops + (id(node),))
    if isinstance(node, (ast.For, ast.AsyncFor)):
        _walk(scan, node.iter, ctx)
        spawned = _is_spawn_call(node.iter)
        _bind(scan, node.target, spawned, inner, _stmt_end(node.iter))
        _walk_block(scan, node.body, inner)
        _walk_block(scan, node.orelse, ctx)
        return
    if isinstance(node, ast.While):
        _walk(scan, node.test, inner)
        _walk_block(scan, node.body, inner)
        _walk_block(scan, node.orelse, ctx)
        return
    start = (node.lineno, node.col_offset)
    for i, gen in enumerate(node.generators):
        _walk(scan, gen.iter, ctx if i == 0 else inner)
        _bind(scan, gen.target, _is_spawn_call(gen.iter), inner, start)
        for cond in gen.ifs:
            _walk(scan, cond, inner)
    for field in ("elt", "key", "value"):
        if hasattr(node, field):
            _walk(scan, getattr(node, field), inner)


def _walk(scan: Scan, node: ast.AST, ctx: Ctx) -> None:
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While) + _COMP_TYPES):
        _walk_loop(scan, node, ctx)
        return
    if isinstance(node, _FUNC_TYPES):
        shadowed = ctx.shadowed | {a.arg for a in _params(node)}
        _walk_fields(scan, node, ctx._replace(shadowed=shadowed))
        return
    if isinstance(node, (ast.If, ast.IfExp)):
        _walk(scan, node.test, ctx)
        body_ctx = ctx._replace(branches=ctx.branches + ((id(node), "body"),))
        else_ctx = ctx._replace(branches=ctx.branches + ((id(node), "orelse"),))
        _walk_part(scan, node.body, body_ctx)
        _walk_part(scan, node.orelse, else_ctx)
        return
    if isinstance(node, ast.Assign):
        _walk(scan, node.value, ctx)
        for target in node.targets:
            _bind(scan, target, _is_spawn_call(node.value), ctx, _stmt_end(node))
        return
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
        _walk(scan, node.value, ctx)
        _bind(scan, node.target, False, ctx, _stmt_end(node))
        return
    if isinstance(node, ast.Call):
        _record_call(scan, node, ctx)
    _walk_fields(scan, node, ctx)


def _exclusive(a: RngUse, b: RngUse) -> bool:
    a_branches = dict(a.ctx.branches)
    return any(a_branches.get(k, side) != side for k, side in b.ctx.branches)


def _version(binds: list[RngBinding], use: RngUse) -> RngBinding | None:
    prior = [b for b in binds if b.end <= (use.lineno, use.col)]
    return prior[-1] if prior else None


def _group_violation(uses: list[RngUse], version: RngBinding | None) -> bool:
    bound_loops = version.loops if version is not None else ()
    for use in uses:
        if not use.gated:
            continue
        if len(use.ctx.loops) > len(bound_loops):
            return True
        others = [
            u
            for u in uses
            if u is not use and u.kind != "spawn" and not _exclusive(use, u)
        ]
        if others:
            return True
    return False


def _function_violations(fn: ast.AST, lines: list[str], rel: Path) -> list[str]:
    scan = Scan(_rng_params(fn), [], [])
    _walk_part(scan, fn.body, Ctx((), (), frozenset()))
    fn_name = getattr(fn, "name", "<lambda>")
    groups: dict[tuple, list[RngUse]] = {}
    for use in scan.uses:
        if use.name not in scan.rng_names:
            continue
        if _IGNORE_MARKER in lines[use.lineno - 1]:
            continue
        version = _version([b for b in scan.bindings if b.name == use.name], use)
        groups.setdefault((use.name, version), []).append(use)

    violations = []
    for (name, version), uses in groups.items():
        if not _group_violation(uses, version):
            continue
        uses = [u for u in uses if u.kind != "spawn"]
        desc = ", ".join(f"{u.lineno}:{u.kind}" for u in uses)
        violations.append(f"{rel}:{uses[0].lineno}: {fn_name} [{name}] {desc}")
    return violations


def rng_stream_violations(path: Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    rel = path.relative_to(_GENERATORS_DIR.parent.parent)
    fns = [n for n in ast.walk(ast.parse(source)) if isinstance(n, _FUNC_TYPES)]
    return [v for fn in fns for v in _function_violations(fn, lines, rel)]


def test_choices_and_callees_use_clean_rng_streams() -> None:
    violations = []
    for path in sorted(_GENERATORS_DIR.rglob("*.py")):
        if path.name in _EXCLUDE:
            continue
        violations.extend(rng_stream_violations(path))
    assert not violations, (
        "RNG stream shared between a choice or stochastic callee and another use; "
        "give each its own child from one spawn at the top of the function:\n"
        + "\n".join(violations)
    )
