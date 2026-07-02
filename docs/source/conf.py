# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import html
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from docutils import nodes as docutils_nodes
from sphinx import addnodes
from sphinx.ext import apidoc
from sphinx.ext.autodoc import ModuleDocumenter
from sphinx.util.typing import stringify_annotation

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import _render_commands
except ImportError:
    _render_commands = None

project = "Infinigen"
copyright = "2025, Princeton Vision and Learning Lab"
author = "Princeton Vision and Learning Lab"
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxarg.ext",
]

# Hotlink types in autodoc-rendered signatures/params (Material, ProcNode,
# Color, Generator, ...) to their defining project's docs. Missing entries
# just stay as plain text (nitpicky is off, so this never fails the build).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "procfunc": ("https://procfunc.readthedocs.io/en/latest/", None),
    "blender": ("https://docs.blender.org/api/current", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = []


# [source] links resolve to the GitHub repo+ref the docs are built from: env
# overrides, else the origin remote / current commit. Public release builds (tag
# on main) thus point at the public infinigen repo, internal builds at internal.
def _github_repo() -> str:
    env = os.environ.get("INFINIGEN_DOCS_GITHUB_REPO")
    if env:
        return env
    try:
        here = str(Path(__file__).resolve().parent)
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        match = re.search(r"github\.com[:/](.+?/.+?)(?:\.git)?$", url)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "princeton-vl/infinigen"


GITHUB_REPO = _github_repo()


def _git_ref() -> str:
    ref = os.environ.get("INFINIGEN_DOCS_GIT_REF")
    if ref:
        return ref
    try:
        here = str(Path(__file__).resolve().parent)
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "develop2"


GITHUB_REF = _git_ref()


def linkcode_resolve(domain, info):
    if domain != "py" or not info.get("module"):
        return None
    try:
        obj = importlib.import_module(info["module"])
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        src = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (ImportError, AttributeError, TypeError, OSError):
        return None
    if not src:
        return None
    try:
        rel = Path(src).resolve().relative_to(Path(__file__).resolve().parents[2])
    except ValueError:
        return None
    end = start + len(lines) - 1
    return f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_REF}/{rel}#L{start}-L{end}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "Infinigen"
html_static_path = ["_static"]
html_css_files = ["copy-command.css"]
html_js_files = ["copy-command.js"]

# Expand the whole left-nav TOC flat (no collapsible dropdown arrows).
# show_navbar_depth expands sections down to that depth; a large value plus
# max_navbar_depth lists every section and its pages without any collapsing.
html_theme_options = {
    "show_navbar_depth": 6,
    "max_navbar_depth": 6,
    "collapse_navbar": False,
    "show_toc_level": 2,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_typehints = (
    "description"  # Show type hints in the description, not the signature
)

# _shorten_signature reconstructs the return type into the signature; show it
# as the leaf name (`-> BookcaseResult`) but keep the fully-qualified xref link.
python_use_unqualified_type_names = True

# Long signatures (dozens of params) wrap one-parameter-per-line instead of
# one unreadable line. Types stay out of the signature (autodoc_typehints
# above), so this wrap is driven by param names/defaults only.
python_maximum_signature_line_length = 88

# One scrollable page per package: render members inline instead of a TOC of
# per-member sub-pages (paired with dropping apidoc's --separate below).
# member-order bysource: no preset puts `*_rand` first/helpers last by name.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}


# -- Example renders inlined next to manifest generator functions ------------
# Integration renders seeds 0..5 per generator for the visual-check categories
# (scripts/integration_v2/launch.sh) at <base>/<slug>/images/<name>/<seed>.png.

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "src" / "infinigen2" / "manifest.json"

# Seed renders per manifest category; categories absent here get no images.
_CATEGORY_IMAGE_COUNT = {
    "Material": 6,
    "Mask": 6,
    "Object": 6,
    "Scene": 6,
}

# Image URLs: <base>/<slug>/images/<name>/<seed>.png; empty base = local paths.
IMAGE_URL_BASE = os.environ.get(
    "INFINIGEN_DOCS_IMAGE_BASE", "https://infinigen.cs.princeton.edu/docs"
)


def _version_slug() -> str:
    env = os.environ.get("READTHEDOCS_VERSION") or os.environ.get(
        "INFINIGEN_DOCS_VERSION"
    )
    if env:
        return env
    init = (REPO_ROOT / "src" / "infinigen2" / "__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    return f"v{match.group(1)}" if match else "vunknown"


VERSION_SLUG = _version_slug()


def _manifest_entries() -> tuple[dict[str, int], dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}, {}
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        return {}, {}
    counts = {}
    categories = {}
    for e in data:
        category = e.get("category")
        count = _CATEGORY_IMAGE_COUNT.get(category, 0)
        if e.get("name") and count:
            counts[e["name"]] = count
            categories[e["name"]] = category
    return counts, categories


_IMAGE_COUNTS, _IMAGE_CATEGORIES = _manifest_entries()


def _manifest_entrypoints() -> frozenset[str]:
    if not MANIFEST_PATH.exists():
        return frozenset()
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except json.JSONDecodeError:
        return frozenset()
    return frozenset(e["name"] for e in data if e.get("name"))


_ENTRYPOINTS = _manifest_entrypoints()


def _image_urls(name: str) -> list[str]:
    rels = [f"images/{name}/{i}.png" for i in range(_IMAGE_COUNTS[name])]
    if not IMAGE_URL_BASE:
        return rels
    return [f"{IMAGE_URL_BASE}/{VERSION_SLUG}/assets/{rel}" for rel in rels]


def _replicate_command(category: str, name: str, seed: int) -> str | None:
    if _render_commands is None:
        return None
    return _render_commands.replicate_command(category, name, seed)


def _figure_html(url: str, name: str, seed: int, cmd: str | None) -> list[str]:
    alt = html.escape(f"{name} seed {seed}", quote=True)
    lines = [
        ".. raw:: html",
        "",
        '   <figure class="example-render">',
        f'     <img class="example-render__img" src="{url}" loading="lazy" alt="{alt}">',
    ]
    if cmd is not None:
        esc = html.escape(cmd, quote=True)
        lines += [
            '     <div class="example-render__cmd">',
            f'       <code class="example-render__code">{esc}</code>',
            f'       <button class="example-render__copy" type="button" data-command="{esc}" aria-label="Copy command">📋</button>',
            "     </div>",
        ]
    lines += ["   </figure>", ""]
    return lines


def _inject_images(app, what, name, obj, options, lines):  # noqa: ARG001
    if name not in _IMAGE_COUNTS:
        return
    category = _IMAGE_CATEGORIES.get(name)
    lines += ["", ".. rubric:: Example renders", ""]
    for seed, url in enumerate(_image_urls(name)):
        cmd = _replicate_command(category, name, seed)
        lines += _figure_html(url, name, seed, cmd)


def _clean_namedtuple(app, what, name, obj, options, lines):  # noqa: ARG001
    if what == "attribute" and lines and lines[0].startswith("Alias for field number"):
        del lines[:]
        return
    if what != "class" or not isinstance(obj, type):
        return
    fields = getattr(obj, "_fields", None)
    if fields is None or not issubclass(obj, tuple):
        return
    auto = f"{obj.__name__}({', '.join(fields)})"
    if lines and lines[0].strip() == auto:
        del lines[:]


_LONG_FLOAT = re.compile(r"-?\d+\.\d{5,}")


def _round_floats(text: str) -> str:
    return _LONG_FLOAT.sub(lambda m: repr(round(float(m.group(0)), 3)), text)


def _format_default(value: object) -> str:
    text = _round_floats(repr(value))
    return re.sub(r" at 0x[0-9A-Fa-f]+", "", text)


def _param_defaults(app, what, name, obj, options, lines):  # noqa: ARG001
    if what not in ("function", "method"):
        return
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    documented = set(
        re.findall(r"^:param\s+([A-Za-z_]\w*)\s*:", "\n".join(lines), re.M)
    )
    fields = []
    for pname, param in params.items():
        if pname in ("self", "cls") or pname in documented or param.kind in variadic:
            continue
        if param.default is param.empty:
            fields.append(f":param {pname}:")
        else:
            fields.append(
                f":param {pname}: (default: {_format_default(param.default)})"
            )
    if fields:
        lines += ["", *fields]


def _return_annotation(obj: object) -> str | None:
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return None
    if ann is inspect.Signature.empty or ann is None:
        return None
    try:
        return stringify_annotation(ann, "fully-qualified-except-typing")
    except Exception:
        return getattr(ann, "__name__", None) or str(ann)


def _param_count(obj: object) -> int:
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return 0
    return sum(1 for p in params if p not in ("self", "cls"))


def _shorten_signature(app, what, name, obj, options, signature, return_annotation):  # noqa: ARG001
    # Return type moves into the signature (below), so drop the duplicate
    # "Return type" line autodoc_typehints="description" records for the body.
    # Runs after record_typehints, so the annotation is already recorded here.
    try:
        recorded = app.env.current_document.autodoc_annotations.get(name)
    except AttributeError:
        recorded = None
    if recorded:
        recorded.pop("return", None)
    ret = return_annotation
    if not ret and what in ("function", "method"):
        ret = _return_annotation(obj)
    ret = _round_floats(ret) if ret else ret
    # Many-param functions collapse to `(...)`; every param + default lands in
    # the Parameters block (_param_defaults) instead of a wall-of-args signature.
    if what in ("function", "method") and _param_count(obj) > 3:
        return "(...)", ret
    sig = _round_floats(signature) if signature else signature
    return sig, ret


# apidoc titles pages "infinigen2.objects package" / "...foo module"; strip the
# trailing word from every heading so the sidebar shows the dotted name only.
_HEADING_SUFFIX = re.compile(r"^(?P<title>.+?) (?:package|module)$")

# Section-underline characters, deepest last; used to demote inlined subpackages.
_LADDER = '=-~^"+*'


def _is_underline(text: str) -> bool:
    return len(text) > 0 and len(set(text)) == 1 and text[0] in _LADDER


def _strip_api_suffixes(api_dir: Path) -> None:
    for rst in api_dir.glob("*.rst"):
        raw = rst.read_text().splitlines()
        lines = []
        i = 0
        while i < len(raw):
            drop = (
                raw[i].strip() == "Submodules"
                and i + 1 < len(raw)
                and _is_underline(raw[i + 1])
            )
            if drop:
                i += 2
                continue
            lines.append(raw[i])
            i += 1
        for i in range(len(lines) - 1):
            if not (_is_underline(lines[i + 1]) and lines[i].strip()):
                continue
            match = _HEADING_SUFFIX.match(lines[i])
            if not match:
                continue
            lines[i] = match.group("title")
            lines[i + 1] = lines[i + 1][0] * len(lines[i])
        rst.write_text("\n".join(lines) + "\n")


# One friendly section title per subpackage (keyed by its dotted path under
# infinigen2). Anything not listed falls back to a title-cased leaf name.
_SUBPKG_NAMES = {
    "animations": "Animations",
    "cameras": "Cameras",
    "curves": "Curves",
    "exporters": "Exporters",
    "exporters.util": "Exporter Utilities",
    "lighting": "Lighting",
    "objects": "Objects",
    "scenes": "Scenes",
    "scenes.room": "Rooms",
    "shaders": "Shaders",
    "shaders.composites": "Composite Materials",
    "shaders.displacements": "Displacements",
    "shaders.masks": "Masks",
    "shaders.materials": "Materials",
    "shaders.util": "Shader Utilities",
    "util": "Utilities",
    "util.codestats": "Code Statistics",
    "uv_surface": "UV Surface",
}

# Top-level subpackage pages that lead the API Reference toctree, in this order;
# the rest follow in apidoc's (alphabetical) order.
_TOP_LEAD = ["shaders", "objects", "scenes"]

# Same friendly-name treatment for the v1 `infinigen` tree, keyed by dotted path
# under `infinigen`. infinigen_examples is a sibling top-level package.
_V1_SUBPKG_NAMES = {
    "assets": "Assets",
    "assets.materials": "Materials",
    "assets.objects": "Objects",
    "assets.fluid": "Fluid",
    "assets.lighting": "Lighting",
    "assets.scatters": "Scatters",
    "assets.static_assets": "Static Assets",
    "assets.utils": "Asset Utilities",
    "assets.weather": "Weather",
    "core": "Core",
    "datagen": "Datagen",
    "terrain": "Terrain",
    "tools": "Tools",
    "infinigen_examples": "Examples",
}

# Lead the v1 root toctree with Assets, then Core; the rest follow alphabetically.
_V1_TOP_LEAD = ["assets", "core"]

# Inside the folded Assets page, lead its subpackage sub-sections with Materials,
# then Objects; the rest follow alphabetically. Keyed relative to infinigen.assets.
_V1_ASSETS_LEAD = ["materials", "objects"]

# Shaders keeps its child subpackages as their own pages (nested in the sidebar);
# this leads its toctree, the rest (shader utilities) follow.
_SHADERS_LEAD = [
    "shaders.materials",
    "shaders.masks",
    "shaders.composites",
    "shaders.displacements",
]

# Packages that stay hub pages: they keep a toctree to child pages (which nest in
# the left sidebar) instead of being folded into one scrollable page.
_HUBS = {"infinigen2", "infinigen2.shaders"}


def _english_name(sub: str, names: dict = _SUBPKG_NAMES) -> str:
    if sub in names:
        return names[sub]
    return sub.rsplit(".", 1)[-1].replace("_", " ").title()


def _subpackage_children(name: str, files: dict) -> list[str]:
    prefix = name + "."
    depth = name.count(".") + 1
    return sorted(n for n in files if n.startswith(prefix) and n.count(".") == depth)


def _end_of_subpackages(raw: list[str], start: int) -> int:
    i = start + 2
    while i < len(raw):
        line = raw[i]
        if line.strip() == "" or line[:1] in (" ", "\t") or line.startswith(".."):
            i += 1
            continue
        break
    return i


def _package_lines(name: str, files: dict, offset: int) -> list[str]:
    raw = files[name].read_text().splitlines()
    out = []
    i = 0
    while i < len(raw):
        line = raw[i]
        header = line.strip() == "Subpackages" and _is_underline(raw[i + 1])
        if header:
            i = _end_of_subpackages(raw, i)
            for child in _subpackage_children(name, files):
                out += _package_lines(child, files, offset + 1)
                out.append("")
            continue
        if _is_underline(line) and out and out[-1].strip():
            level = _LADDER.index(line[0]) + offset
            out.append(_LADDER[min(level, len(_LADDER) - 1)] * len(out[-1]))
        else:
            out.append(line)
        i += 1
    return out


# Every dotted path apidoc generated a subpackage page for (objects,
# shaders.materials, ...), captured before _build_pages deletes the
# nested ones. Lets the heading rename tell subpackages from plain modules.
def _real_subpackages(api_dir: Path, prefix: str = "infinigen2") -> set:
    return {p.stem[len(prefix) + 1 :] for p in api_dir.glob(prefix + ".*.rst")}


# Hub pages keep their toctree so their children nest in the left sidebar as their
# own pages. Every other surviving page folds its own descendant subpackages into
# one scrollable page; pages under `prefix` that got folded away are deleted.
def _build_pages(api_dir: Path, prefix: str, hubs: set) -> None:
    files = {p.stem: p for p in api_dir.glob("*.rst")}
    survivors = set(hubs)
    for hub in hubs:
        survivors.update(_subpackage_children(hub, files))
    for name in survivors - set(hubs):
        files[name].write_text("\n".join(_package_lines(name, files, 0)) + "\n")
    for name, path in files.items():
        if (name == prefix or name.startswith(prefix + ".")) and name not in survivors:
            path.unlink()


# Rewrite dotted subpackage headings (page titles and inlined sub-sections) to
# their English name plus the dotted path, keeping the heading's level.
def _rename_subpackage_headings(
    rst_path: Path,
    subpkgs: set,
    prefix: str = "infinigen2",
    names: dict = _SUBPKG_NAMES,
) -> None:
    lines = rst_path.read_text().splitlines()
    for i in range(len(lines) - 1):
        if not (lines[i].strip() and _is_underline(lines[i + 1])):
            continue
        title = lines[i].strip()
        if not title.startswith(prefix + "."):
            continue
        key = title[len(prefix) + 1 :].replace("\\", "")
        if key not in subpkgs:
            continue
        heading = f"{_english_name(key, names)} (.{key})"
        lines[i] = heading
        lines[i + 1] = lines[i + 1][0] * len(heading)
    rst_path.write_text("\n".join(lines) + "\n")


# Reorder the root page's subpackage toctree so the lead pages come first.
def _reorder_root_toctree(
    rst_path: Path, lead: list, prefix: str = "infinigen2"
) -> None:
    lines = rst_path.read_text().splitlines()
    entries = [
        i
        for i, line in enumerate(lines)
        if line[:1] in (" ", "\t") and line.strip().startswith(prefix + ".")
    ]
    if not entries:
        return
    block = lines[entries[0] : entries[-1] + 1]

    def rank(entry: str) -> int:
        key = entry.strip()[len(prefix) + 1 :].replace("\\", "")
        return lead.index(key) if key in lead else len(lead)

    lines[entries[0] : entries[-1] + 1] = sorted(block, key=rank)
    rst_path.write_text("\n".join(lines) + "\n")


# Reorder the inlined subpackage sub-sections of a folded page (e.g. Assets) so
# the lead children come first. Each `-`-underlined child heading starts a block
# spanning to the next one. Runs on dotted titles, before the English rename.
def _reorder_inline_subsections(rst_path: Path, prefix: str, lead: list) -> None:
    if not rst_path.exists():
        return
    lines = rst_path.read_text().splitlines()
    starts = []
    stop = len(lines)
    for i in range(2, len(lines) - 1):
        under = lines[i + 1]
        if not (lines[i].strip() and _is_underline(under) and under[0] == "-"):
            continue
        if lines[i].strip().startswith(prefix + "."):
            starts.append(i)
        elif starts:
            stop = i
            break
    if len(starts) < 2:
        return
    bounds = starts + [stop]
    blocks = [lines[bounds[k] : bounds[k + 1]] for k in range(len(starts))]

    def rank(block: list) -> int:
        key = block[0].strip()[len(prefix) + 1 :].replace("\\", "")
        return lead.index(key) if key in lead else len(lead)

    lines[starts[0] : stop] = [ln for block in sorted(blocks, key=rank) for ln in block]
    rst_path.write_text("\n".join(lines) + "\n")


# Rewrite a page's first heading (its H1 / sidebar label) to a plain title.
def _rename_root_heading(rst_path: Path, title: str) -> None:
    if not rst_path.exists():
        return
    lines = rst_path.read_text().splitlines()
    for i in range(len(lines) - 1):
        if not (lines[i].strip() and _is_underline(lines[i + 1])):
            continue
        lines[i] = title
        lines[i + 1] = lines[i + 1][0] * len(title)
        break
    rst_path.write_text("\n".join(lines) + "\n")


# The package-root page's H1 is the sidebar label; make it read "API Reference"
# and keep the dotted package name as a one-line subtitle below it.
def _retitle_root(rst_path: Path, dotted: str) -> None:
    if not rst_path.exists():
        return
    lines = rst_path.read_text().splitlines()
    for i in range(len(lines) - 1):
        if not (lines[i].strip() and _is_underline(lines[i + 1])):
            continue
        lines[i] = "API Reference"
        lines[i + 1] = "=" * len(lines[i])
        lines[i + 2 : i + 2] = ["", f"The ``{dotted}`` package.", ""]
        break
    rst_path.write_text("\n".join(lines) + "\n")


def _run_apidoc(_app):
    here = os.path.dirname(__file__)
    src = os.path.join(here, "..", "..", "src")
    api_dir = Path(here) / "api"
    apidoc.main(
        ["--force", "--no-toc", "-o", str(api_dir), os.path.join(src, "infinigen2")]
    )
    _strip_api_suffixes(api_dir)
    subpkgs = _real_subpackages(api_dir)
    _build_pages(api_dir, "infinigen2", _HUBS)
    for page in api_dir.glob("infinigen2.*.rst"):
        _rename_subpackage_headings(page, subpkgs)
    _reorder_root_toctree(api_dir / "infinigen2.rst", _TOP_LEAD)
    _reorder_root_toctree(api_dir / "infinigen2.shaders.rst", _SHADERS_LEAD)
    _retitle_root(api_dir / "infinigen2.rst", "infinigen2")
    # v1 docs regenerate here (the api dir is gitignored, so nothing is stale).
    # v1 has only a root hub: each top-level subpackage folds into one page so the
    # Infinigen 1.0 sidebar stays two levels deep (root -> assets/core/...).
    apidoc.main(
        ["--force", "--no-toc", "-o", str(api_dir), os.path.join(src, "infinigen")]
    )
    apidoc.main(
        [
            "--force",
            "--no-toc",
            "-o",
            str(api_dir),
            os.path.join(src, "infinigen_examples"),
        ]
    )
    _strip_api_suffixes(api_dir)
    v1_subpkgs = _real_subpackages(api_dir, "infinigen")
    _build_pages(api_dir, "infinigen", {"infinigen"})
    _reorder_inline_subsections(
        api_dir / "infinigen.assets.rst", "infinigen.assets", _V1_ASSETS_LEAD
    )
    _reorder_root_toctree(api_dir / "infinigen.rst", _V1_TOP_LEAD, "infinigen")
    for page in api_dir.glob("infinigen.*.rst"):
        _rename_subpackage_headings(page, v1_subpkgs, "infinigen", _V1_SUBPKG_NAMES)
    _retitle_root(api_dir / "infinigen.rst", "infinigen")
    _rename_root_heading(api_dir / "infinigen_examples.rst", "Examples")


# Per-file member order: `*_rand`, plain funcs, `*_presets` chooser, individual
# `*_preset`, classes last. No autodoc event for this, so wrap sort_members.
# ModuleDocumenter overrides sort_members, so patch it (not the base Documenter)
# or automodule member order is untouched.
_orig_sort_members = ModuleDocumenter.sort_members


def _member_rank(entry) -> int:
    documenter = entry[0]
    if getattr(documenter, "objtype", "") == "class":
        return 4
    fn = getattr(documenter, "fullname", "") or getattr(documenter, "name", "")
    fn = fn.replace("::", ".")
    short = fn.rsplit(".", 1)[-1]
    if short.endswith("_rand"):
        return 0
    if short.endswith("_presets"):
        return 2
    if short.endswith("_preset"):
        return 3
    return 1


def _sort_members(self, documenters, order):
    ordered = _orig_sort_members(self, documenters, order)
    return sorted(ordered, key=_member_rank)


# mathutils C-types report __module__ == "builtins", so intersphinx never
# resolves them; link them straight to the Blender mathutils reference.
_MATHUTILS_TYPES = {"Vector", "Color", "Matrix", "Euler", "Quaternion"}


def _resolve_mathutils(app, env, node, contnode):  # noqa: ARG001
    short = node.get("reftarget", "").rsplit(".", 1)[-1]
    if short not in _MATHUTILS_TYPES:
        return None
    uri = f"https://docs.blender.org/api/current/mathutils.html#mathutils.{short}"
    ref = docutils_nodes.reference("", "", internal=False, refuri=uri)
    ref.append(contnode)
    return ref


# procfunc's core types read with a `pf.` prefix wherever they show up in
# signatures/params/returns. Match on the xref target leaf and rewrite the display
# text only, so the hotlink (intersphinx or the mathutils resolver) is preserved.
_PF_TYPES = {
    "Material",
    "MeshObject",
    "Texture",
    "Collection",
    "ProcNode",
    "RNG",
    "Vector",
    "Color",
    "Euler",
    "Quaternion",
    "Matrix",
}


def _prefix_pf_types(app, doctree):  # noqa: ARG001
    for node in doctree.findall(addnodes.pending_xref):
        leaf = node.get("reftarget", "").rsplit(".", 1)[-1]
        if leaf not in _PF_TYPES:
            continue
        for text in list(node.findall(docutils_nodes.Text)):
            text.parent.replace(text, docutils_nodes.Text("pf." + leaf))


# v1 modules lack __all__, so automodule's undoc-members otherwise documents
# imported names (numpy.random.randint, ...). Skip any member defined outside the
# infinigen packages so only real module members are documented.
def _skip_imported(app, what, name, obj, skip, options):  # noqa: ARG001
    if skip:
        return None
    module = getattr(obj, "__module__", None)
    if module and not module.startswith("infinigen"):
        return True
    return None


def setup(app):
    ModuleDocumenter.sort_members = _sort_members
    app.connect("builder-inited", _run_apidoc)
    app.connect("autodoc-process-docstring", _inject_images)
    app.connect("autodoc-process-docstring", _clean_namedtuple)
    app.connect("autodoc-process-docstring", _param_defaults)
    app.connect("autodoc-process-signature", _shorten_signature)
    app.connect("autodoc-skip-member", _skip_imported)
    app.connect("doctree-read", _prefix_pf_types)
    app.connect("missing-reference", _resolve_mathutils)
