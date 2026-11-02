# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""Page of shaded and wireframe renders of both sides, with triangle counts."""

from __future__ import annotations

import base64
import json
import pathlib

from infinigen.tools.perf_validation import geometry

CSS = """
:root{--bg:#0f1115;--panel:#171a21;--rule:#262b36;--ink:#e6e9ef;--muted:#8b93a7;
--accent:#7aa2f7;--warn:#e0af68;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px} .sub{color:var(--muted);margin-bottom:20px}
.flags{font-family:var(--mono);font-size:12px;color:var(--accent)}
table{border-collapse:collapse;width:100%}
th{text-align:left;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
color:var(--muted);font-weight:500;padding:8px 10px;border-bottom:1px solid var(--rule)}
td{padding:10px;border-bottom:1px solid var(--rule);vertical-align:top}
img{width:100%;display:block;border-radius:4px;background:#000}
.cell{width:22%}
.count{font-family:var(--mono);font-size:13px}
.delta{font-family:var(--mono);font-size:13px;font-weight:600}
.same{color:var(--muted)} .down{color:var(--warn)} .up{color:var(--accent)}
.note{margin-top:22px;color:var(--muted);font-size:13px;max-width:70ch}
"""


def _uri(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _row(base_dir, opt_dir, a, b, mesh, trunk=False) -> str:
    ta, tb = a["triangles"], b["triangles"] if b else None
    if tb is None:
        delta = '<span class="same">—</span>'
    elif ta == tb:
        delta = '<span class="same">identical</span>'
    else:
        ratio = ta / tb if tb else float("inf")
        cls = "down" if tb < ta else "up"
        delta = f'<span class="{cls}">{ratio:.2f}×</span>'

    def img(d, row, mode, count):
        """One render with its triangle count."""
        if not row or mode not in row.get("images", {}):
            return ""
        return (
            f'<img src="{_uri(d / row["images"][mode])}" alt="{mode}">'
            f'<div class="count">{count:,} tris</div>'
        )

    ba = a.get("build_seconds") or a.get("render_seconds")
    bb = (b.get("build_seconds") or b.get("render_seconds")) if b else None
    if ba and bb:
        speed = ba / bb if bb else float("inf")
        cls = "up" if speed >= 1 else "down"
        timing = (
            f'<span class="count">{ba:.2f}s -> {bb:.2f}s</span><br>'
            f'<span class="{cls} delta">{speed:.2f}x</span>'
        )
    elif ba:
        timing = f'<span class="count">{ba:.2f}s</span>'
    else:
        timing = '<span class="same">&mdash;</span>'

    trunk_cells = (
        f"""
  <td class="cell">{img(base_dir, a, "trunk", ta)}</td>
  <td class="cell">{img(opt_dir, b, "trunk", tb)}</td>"""
        if trunk
        else ""
    )
    return f"""<tr>
  <td><b>{a["target"]}</b><br><span class="count">seed {a["seed"]}</span></td>
  <td class="cell">{img(base_dir, a, "shaded", ta)}</td>
  <td class="cell">{img(base_dir, a, "wireframe", ta)}</td>
  <td class="cell">{img(opt_dir, b, "shaded", tb)}</td>
  <td class="cell">{img(opt_dir, b, "wireframe", tb)}</td>{trunk_cells}
  <td class="count">{ta:,}<br>{f"{tb:,}" if tb is not None else "—"}<br>{delta}</td>
  <td>{timing}</td>
  <td class="count">{mesh}</td>
</tr>"""


def build(manifests: list[pathlib.Path], out: pathlib.Path) -> list[str]:
    """Assemble one page from consecutive (default, optimized) manifest pairs, one pair
    per slice when a run is split by seed range."""
    if len(manifests) % 2:
        raise ValueError(f"expected manifests in pairs, got {len(manifests)}")

    chunks = []
    summary = []
    any_trunk = False
    kinds = []
    selected = None
    for i in range(0, len(manifests), 2):
        base_m, opt_m = manifests[i], manifests[i + 1]
        base_part = json.loads(base_m.read_text())
        opt_part = json.loads(opt_m.read_text())
        kinds.append(base_part["kind"])
        selected = opt_part.get("optimized") or opt_part.get("flags") or selected
        base_dir = base_m.parent / base_m.stem
        opt_dir = opt_m.parent / opt_m.stem
        index = {(r["target"], r["seed"]): r for r in opt_part["rows"]}
        trunk = any("trunk" in r.get("images", {}) for r in base_part["rows"])
        any_trunk = any_trunk or trunk
        for a in base_part["rows"]:
            b = index.get((a["target"], a["seed"]))
            mesh = geometry.describe(a, b, base_dir, opt_dir)
            summary.append(f"{a['target']} seed {a['seed']}: {mesh}")
            chunks.append(_row(base_dir, opt_dir, a, b, mesh, trunk))

    base = {"kind": kinds[0]}
    trunk_headers = (
        "<th>default trunk</th><th>optimized trunk</th>" if any_trunk else ""
    )
    rows = "\n".join(chunks)
    flags = f"--optimized={selected}" if selected else "nothing selected"

    out.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Visual check — {base["kind"]}</title><style>{CSS}</style>
<h1>Visual check — {base["kind"]}</h1>
<div class="sub">left pair: default implementation, nothing selected &nbsp;·&nbsp;
right pair: <span class="flags">{flags}</span></div>
<table>
<tr><th>target</th><th>default</th><th>default wireframe</th>
<th>optimized</th><th>optimized wireframe</th>{trunk_headers}<th>triangles</th>
<th>time</th><th>mesh</th></tr>
{rows}
</table>
<p class="note">Counts are triangles, not polygons. A quad mesh and a
triangulated mesh of the same surface differ by a factor of two in polygon
count while agreeing in triangles, so the wireframe and the number have to
be read together: a wireframe that looks denser at the same triangle count
is a difference in topology, not in detail. The same count is not the same
mesh either: the mesh column compares the geometry itself.</p>
""")
    return summary
