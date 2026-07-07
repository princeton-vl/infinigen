# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Generate an interactive HTML visualization of a graph.json trace file."""

import argparse
import json
import re
import webbrowser
from pathlib import Path

# Categories for slider control: (name, func_prefix_list, color, default_scale)
CATEGORIES = [
    ("rng", [], "#7f8c8d", 0.0),
    ("random", ["procfunc.random", "numpy.random"], "#7f8c8d", 0.0),
    (
        "math",
        [
            "_operator.",
            "operator.",
            "builtins.min",
            "builtins.max",
            "mathutils.Vector",
            "numpy.array",
        ],
        "#95a5a6",
        0.0,
    ),
    ("color", ["procfunc.color"], "#e74c3c", 0.0),
    ("nodes", ["procfunc.nodes"], "#4a90d9", 0.0),
    ("ops", ["procfunc.ops"], "#e8923f", 0.0),
    ("choice", ["procfunc.control"], "#9b59b6", 1.0),
    ("generators", ["infinigen2"], "#f39c12", 1.0),
]

MODULE_COLORS = [
    ("procfunc.nodes", "#4a90d9"),
    ("procfunc.ops", "#e8923f"),
    ("procfunc.control", "#9b59b6"),
    ("procfunc.color", "#e74c3c"),
    ("infinigen2.objects", "#2ecc71"),
    ("infinigen2.shaders", "#f39c12"),
    ("infinigen2.scenes", "#1abc9c"),
    ("infinigen2.cameras", "#3498db"),
    ("infinigen2.lighting", "#f1c40f"),
    ("infinigen2.exporters", "#e67e22"),
    ("procfunc.random", "#7f8c8d"),
    ("_operator.", "#95a5a6"),
    ("operator.", "#95a5a6"),
]
SPECIAL_COLORS = {
    "Constant": "#7f8c8d",
    "InputPlaceholder": "#17a2b8",
    "SubgraphCall": "#9b59b6",
    "Procedural": "#e74c3c",
    "__outputs__": "#17a2b8",
}
DEFAULT_COLOR = "#bdc3c7"


def _color_for_node(node: dict) -> str:
    ntype = node["type"]
    if ntype in SPECIAL_COLORS:
        return SPECIAL_COLORS[ntype]
    func = node.get("func", "")
    for prefix, color in MODULE_COLORS:
        if func.startswith(prefix):
            return color
    return DEFAULT_COLOR


def _node_category(node: dict) -> str | None:
    ntype = node["type"]
    if ntype == "Constant" and isinstance(node.get("value"), dict):
        if node["value"].get("$type") == "rng":
            return "rng"
    func = node.get("func", "")
    for cat_name, prefixes, _, _ in CATEGORIES:
        for prefix in prefixes:
            if func.startswith(prefix):
                return cat_name
    return None


def _constant_label(value) -> str:
    if isinstance(value, dict):
        t = value.get("$type")
        if t == "rng":
            return "rng"
        elif t == "Path":
            return value.get("value", "").rsplit("/", 1)[-1]
        elif t:
            return t
        return json.dumps(value)[:20]
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value[:20]
    elif isinstance(value, list):
        return f"[{len(value)} items]"
    s = json.dumps(value)
    return s[:20] + ("..." if len(s) > 20 else "")


def _node_label(node: dict) -> str:
    ntype = node["type"]
    if ntype == "FunctionCall":
        func = node.get("func", "")
        return func.rsplit(".", 1)[-1]
    elif ntype == "MethodCall":
        return f".{node.get('method', '?')}"
    elif ntype == "GetAttribute":
        return f".{node.get('attribute', '?')}"
    elif ntype == "Constant":
        return _constant_label(node.get("value"))
    elif ntype == "SubgraphCall":
        return node.get("subgraph", "subgraph")
    elif ntype == "Procedural":
        return node.get("node_type", "procedural")
    elif ntype == "InputPlaceholder":
        return node.get("name", "input")
    elif ntype == "MutatedArgument":
        return "mutated"
    return ntype


def _find_refs(obj, path=""):
    """Recursively find all $ref entries in a JSON structure, yielding (path, ref_id)."""
    if isinstance(obj, dict):
        if "$ref" in obj and len(obj) == 1:
            yield path, obj["$ref"]
        else:
            for k, v in obj.items():
                yield from _find_refs(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_refs(v, f"{path}[{i}]")


def build_cytoscape_elements(graph_data: dict) -> list[dict]:
    if "graphs" in graph_data:
        all_nodes = []
        for g in graph_data["graphs"]:
            all_nodes.extend(g["nodes"])
    else:
        all_nodes = graph_data["nodes"]

    # Build passthrough maps for node types we want to skip in the viewer.
    passthrough_ids: dict[str, str] = {}
    for n in all_nodes:
        refs = list(_find_refs(n.get("args", [])))
        if n["type"] == "MutatedArgument" and len(refs) >= 2:
            passthrough_ids[n["id"]] = refs[1][1]
        elif n["type"] == "MethodCall" and refs:
            passthrough_ids[n["id"]] = refs[0][1]
        elif n["type"] == "GetAttribute" and refs:
            passthrough_ids[n["id"]] = refs[0][1]

    getattr_nodes = {n["id"]: n for n in all_nodes if n["type"] == "GetAttribute"}

    def _resolve_ref(ref_id: str) -> tuple[str, str]:
        attrs = []
        seen = set()
        while ref_id in passthrough_ids and ref_id not in seen:
            seen.add(ref_id)
            if ref_id in getattr_nodes:
                attrs.append(getattr_nodes[ref_id]["attribute"])
            ref_id = passthrough_ids[ref_id]
        attrs.reverse()
        return ref_id, ".".join(attrs)

    skip_ids = set(passthrough_ids.keys())

    # Assign categories
    node_cats: dict[str, str | None] = {}
    for n in all_nodes:
        node_cats[n["id"]] = _node_category(n)

    # Build dependency map: for each node, which visible nodes does it depend on?
    deps: dict[str, list[str]] = {}
    for n in all_nodes:
        if n["id"] in skip_ids:
            continue
        for _, ref_id in _find_refs(n.get("args", [])):
            resolved_id, _ = _resolve_ref(ref_id)
            if resolved_id not in skip_ids:
                deps.setdefault(n["id"], []).append(resolved_id)
        for _, ref_id in _find_refs(n.get("kwargs", {})):
            resolved_id, _ = _resolve_ref(ref_id)
            if resolved_id not in skip_ids:
                deps.setdefault(n["id"], []).append(resolved_id)

    # For each node, compute the union of categories of itself + all descendants.
    # A node stays visible (possibly at min size) if ANY of these categories is > 0.
    # Process in reverse topological order (nodes list is already topo-sorted leaves first).
    visible_ids = {n["id"] for n in all_nodes if n["id"] not in skip_ids}
    node_all_cats: dict[str, set[str]] = {}
    for n in all_nodes:
        nid = n["id"]
        if nid not in visible_ids:
            continue
        cats = set()
        own_cat = node_cats.get(nid)
        if own_cat:
            cats.add(own_cat)
        # Union in categories from all dependencies (children in the DAG)
        for dep_id in deps.get(nid, []):
            if dep_id in node_all_cats:
                cats |= node_all_cats[dep_id]
        node_all_cats[nid] = cats

    elements = []

    for node in all_nodes:
        nid = node["id"]
        ntype = node["type"]

        if nid in skip_ids:
            continue

        cat = node_cats.get(nid)
        all_cats = sorted(node_all_cats.get(nid, set()))
        node_data = {
            "id": nid,
            "label": _node_label(node),
            "type": ntype,
            "color": _color_for_node(node),
            "cat": cat,
            "cats": all_cats,
            "details": node,
        }
        # Tag choice nodes with their chosen_idx
        func = node.get("func", "")
        is_choice = func.endswith(".choice") and "choice_options" in node.get(
            "kwargs", {}
        )
        if is_choice:
            chosen_idx = node.get("kwargs", {}).get("chosen_idx")
            node_data["chosen_idx"] = chosen_idx if isinstance(chosen_idx, int) else 0

        elements.append({"data": node_data})

        for path, ref_id in _find_refs(node.get("args", [])):
            resolved_id, attr_suffix = _resolve_ref(ref_id)
            if resolved_id in skip_ids:
                continue
            arg_idx = path.split("]")[0].lstrip("[") if path else ""
            edge_label = f".{attr_suffix}" if attr_suffix else arg_idx
            edge_cat = node_cats.get(resolved_id) or cat
            elements.append(
                {
                    "data": {
                        "id": f"{resolved_id}->{nid}@args.{path}",
                        "source": resolved_id,
                        "target": nid,
                        "label": edge_label,
                        "cat": edge_cat,
                    }
                }
            )

        for path, ref_id in _find_refs(node.get("kwargs", {})):
            resolved_id, attr_suffix = _resolve_ref(ref_id)
            if resolved_id in skip_ids:
                continue
            kwarg_name = path.split("[")[0].split(".")[0]
            edge_label = f".{attr_suffix}" if attr_suffix else kwarg_name
            edge_cat = node_cats.get(resolved_id) or cat
            edge_data = {
                "id": f"{resolved_id}->{nid}@{path}",
                "source": resolved_id,
                "target": nid,
                "label": edge_label,
                "cat": edge_cat,
            }
            # Tag choice_options edges with their option index
            if is_choice and path.startswith("choice_options["):
                m = re.match(r"choice_options\[(\d+)\]", path)
                if m:
                    edge_data["choice_option_idx"] = int(m.group(1))
            elements.append({"data": edge_data})

    # Add outputs node
    outputs_data = None
    if "graphs" in graph_data:
        outputs_data = graph_data["graphs"][0].get("outputs")
    else:
        outputs_data = graph_data.get("outputs")

    if outputs_data:
        elements.append(
            {
                "data": {
                    "id": "__outputs__",
                    "label": "outputs",
                    "type": "__outputs__",
                    "color": "#17a2b8",
                    "details": {"id": "__outputs__", "type": "outputs", **outputs_data},
                }
            }
        )
        for path, ref_id in _find_refs(outputs_data):
            resolved_id, attr_suffix = _resolve_ref(ref_id)
            if resolved_id in skip_ids:
                continue
            edge_label = f".{attr_suffix}" if attr_suffix else path
            elements.append(
                {
                    "data": {
                        "id": f"{resolved_id}->__outputs__@{path}",
                        "source": resolved_id,
                        "target": "__outputs__",
                        "label": edge_label,
                    }
                }
            )

    # --- Precompute gating_choice / gating_option_idx per edge ---
    # For each choice node, BFS backward from each choice_options edge's source,
    # tagging every edge visited with which choice gates it.
    # Stop when hitting another choice node (it's gated by its own parent).

    # Build adjacency: node_id -> list of edge elements whose target is that node
    incoming_edges: dict[str, list[dict]] = {}
    choice_option_edges: list[dict] = []
    for el in elements:
        d = el["data"]
        if "source" in d:
            incoming_edges.setdefault(d["target"], []).append(el)
            if "choice_option_idx" in d:
                choice_option_edges.append(el)

    # Set of choice node IDs
    choice_node_ids = {
        el["data"]["id"]
        for el in elements
        if "source" not in el["data"] and el["data"].get("chosen_idx") is not None
    }

    for choice_edge in choice_option_edges:
        choice_node_id = choice_edge["data"]["target"]
        option_idx = choice_edge["data"]["choice_option_idx"]
        # Tag the choice_options edge itself
        choice_edge["data"]["gating_choice"] = choice_node_id
        choice_edge["data"]["gating_option_idx"] = option_idx
        # BFS backward from the source of this choice_options edge
        queue = [choice_edge["data"]["source"]]
        visited = set()
        while queue:
            nid = queue.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for edge_el in incoming_edges.get(nid, []):
                ed = edge_el["data"]
                # Only tag if not already gated by a more specific (inner) choice
                if "gating_choice" not in ed:
                    ed["gating_choice"] = choice_node_id
                    ed["gating_option_idx"] = option_idx
                src = ed["source"]
                # Stop at other choice nodes — they are gated by their own parent
                if src not in choice_node_ids:
                    queue.append(src)

    return elements


# Category defaults passed to JS
CATEGORY_DEFAULTS = json.dumps(
    {name: default for name, _, color, default in CATEGORIES}
)
CATEGORY_COLORS = json.dumps({name: color for name, _, color, _ in CATEGORIES})

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trace Graph Viewer</title>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; display: flex; height: 100vh; background: #1e1e1e; color: #ccc; }
  #cy { flex: 1; }
  #sidebar {
    width: 380px; background: #252526; border-left: 1px solid #3c3c3c;
    overflow-y: auto; padding: 16px; font-size: 13px;
  }
  #sidebar h2 { color: #ddd; font-size: 15px; margin-bottom: 8px; }
  #sidebar .node-id { color: #888; font-size: 12px; margin-bottom: 12px; }
  #sidebar .field { margin-bottom: 10px; }
  #sidebar .field-name { color: #9cdcfe; font-weight: 600; margin-bottom: 2px; }
  #sidebar .field-value { color: #ce9178; white-space: pre-wrap; word-break: break-all; font-family: "Cascadia Code", "Fira Code", monospace; font-size: 12px; }
  #sidebar .hint { color: #666; font-style: italic; }
  #search-box {
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: #333; border: 1px solid #555; color: #ddd; padding: 6px 10px;
    border-radius: 4px; font-size: 13px; width: 250px;
  }
  #search-box::placeholder { color: #777; }
  #sliders {
    position: absolute; top: 44px; left: 10px; z-index: 10;
    background: #2a2a2a; border: 1px solid #444; border-radius: 6px;
    padding: 8px 12px; display: flex; flex-direction: column; gap: 4px;
  }
  .slider-row {
    display: flex; align-items: center; gap: 8px; font-size: 11px;
  }
  .slider-row .dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  .slider-row label { width: 70px; }
  .slider-row input[type=range] { width: 80px; accent-color: #888; }
  .slider-row .val { width: 30px; text-align: right; color: #999; }
  .section-label { font-size: 10px; color: #888; margin-top: 4px; border-top: 1px solid #444; padding-top: 4px; }
  #update-btn {
    margin-top: 6px; padding: 4px 0; background: #4a90d9; border: none;
    color: #fff; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: bold;
  }
  #update-btn:hover { background: #5ba0e9; }
</style>
</head>
<body>
<input id="search-box" type="text" placeholder="Search nodes (func, id, type...)">
<div id="sliders"></div>
<div id="cy"></div>
<div id="sidebar">
  <h2>Node Details</h2>
  <p class="hint">Click a node to inspect it.</p>
</div>
<script>
const graphElements = __ELEMENTS_JSON__;
const catDefaults = __CATEGORY_DEFAULTS__;
const catColors = __CATEGORY_COLORS__;
const catScales = Object.assign({}, catDefaults);

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: graphElements,
  style: [
    {
      selector: 'node',
      style: {
        'label': 'data(label)',
        'background-color': 'data(color)',
        'color': '#ddd',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': '14px',
        'font-weight': 'bold',
        'width': function(ele) { return Math.max(30, ele.data('label').length * 9 + 12); },
        'height': 32,
        'shape': 'round-rectangle',
        'text-max-width': function(ele) { return Math.max(30, ele.data('label').length * 9 + 12) + 'px'; },
        'text-wrap': 'ellipsis',
        'border-width': 0,
      }
    },
    {
      selector: 'node[type = "__outputs__"]',
      style: { 'shape': 'diamond', 'font-weight': 'bold', 'width': 80, 'height': 40 }
    },
    {
      selector: 'node[type = "InputPlaceholder"]',
      style: { 'shape': 'diamond' }
    },
    {
      selector: 'node[cat = "choice"]',
      style: { 'shape': 'diamond' }
    },
    {
      selector: 'node.highlighted',
      style: { 'border-width': 3, 'border-color': '#fff' }
    },
    {
      selector: 'node.faded',
      style: { 'opacity': 0.15 }
    },
    {
      selector: 'edge',
      style: {
        'width': 1.5, 'line-color': '#555', 'target-arrow-color': '#555',
        'target-arrow-shape': 'triangle', 'arrow-scale': 0.8,
        'curve-style': 'bezier', 'font-size': '8px', 'color': '#777',
        'text-rotation': 'autorotate', 'text-margin-y': -8,
      }
    },
    {
      selector: 'edge.highlighted',
      style: { 'line-color': '#aaa', 'target-arrow-color': '#aaa', 'width': 2.5, 'label': 'data(label)' }
    },
    {
      selector: 'edge.faded',
      style: { 'opacity': 0.1 }
    },
    {
      selector: '.hidden',
      style: { 'display': 'none' }
    },
    {
      selector: '.choice-path',
      style: { 'line-color': '#9b59b6', 'target-arrow-color': '#9b59b6', 'width': 3.5, 'z-index': 10 }
    },
    {
      selector: 'node.choice-path',
      style: { 'border-width': 3, 'border-color': '#9b59b6', 'background-opacity': 1.0 }
    },
    {
      selector: 'node.choice-dimmed',
      style: { 'background-opacity': 0.45 }
    },
  ],
  layout: { name: 'preset' },
  wheelSensitivity: 0.3,
});

// Layout parameters
const layoutParams = { nodeSep: 10, rankSep: 50, edgeSep: 5, nodeSize: 1, edgeWidth: 1.5 };

function getLayoutOpts() {
  return {
    name: 'dagre', rankDir: 'LR',
    nodeSep: layoutParams.nodeSep,
    rankSep: layoutParams.rankSep,
    edgeSep: layoutParams.edgeSep,
    spacingFactor: 1.0, nodeDimensionsIncludeLabels: true,
    minLen: function(edge) {
      var src = edge.source(), tgt = edge.target();
      if (src.hasClass('hidden') || tgt.hasClass('hidden')) return 0;
      if (src.hasClass('scaled-down') && tgt.hasClass('scaled-down')) return 1;
      return 2;
    },
  };
}

// Build slider UI
const slidersDiv = document.getElementById('sliders');

// Category scale sliders
for (const [cat, def] of Object.entries(catDefaults)) {
  const row = document.createElement('div');
  row.className = 'slider-row';
  row.innerHTML =
    '<span class="dot" style="background:' + catColors[cat] + '"></span>' +
    '<label>' + cat + '</label>' +
    '<input type="range" min="0" max="2" step="0.1" value="' + def + '" data-cat="' + cat + '">' +
    '<span class="val">' + def + '</span>';
  slidersDiv.appendChild(row);
}

// Layout section
var layoutSection = document.createElement('div');
layoutSection.className = 'section-label';
layoutSection.textContent = 'Layout';
slidersDiv.appendChild(layoutSection);

var layoutSliders = [
  { name: 'nodeSep', label: 'V spacing', min: 1, max: 60, step: 1, def: 10 },
  { name: 'rankSep', label: 'H spacing', min: 1, max: 150, step: 1, def: 50 },
  { name: 'edgeSep', label: 'Edge gap', min: 0, max: 30, step: 1, def: 5 },
  { name: 'nodeSize', label: 'Node size', min: 0.3, max: 5, step: 0.1, def: 1 },
  { name: 'edgeWidth', label: 'Edge width', min: 0.5, max: 5, step: 0.5, def: 1.5 },
];
for (var ls of layoutSliders) {
  var row = document.createElement('div');
  row.className = 'slider-row';
  row.innerHTML =
    '<span class="dot" style="background:transparent"></span>' +
    '<label>' + ls.label + '</label>' +
    '<input type="range" min="' + ls.min + '" max="' + ls.max + '" step="' + ls.step + '" value="' + ls.def + '" data-layout="' + ls.name + '">' +
    '<span class="val">' + ls.def + '</span>';
  slidersDiv.appendChild(row);
}

// Update button
var updateBtn = document.createElement('button');
updateBtn.id = 'update-btn';
updateBtn.textContent = 'Update Layout';
slidersDiv.appendChild(updateBtn);

function readSliders() {
  slidersDiv.querySelectorAll('[data-layout]').forEach(function(inp) {
    layoutParams[inp.dataset.layout] = parseFloat(inp.value);
  });
  slidersDiv.querySelectorAll('[data-cat]').forEach(function(inp) {
    catScales[inp.dataset.cat] = parseFloat(inp.value);
  });
}

function getNodeScale(node) {
  var cat = node.data('cat');
  var ownScale = cat ? (catScales[cat] || 0) : 1;
  // If own category is at 0 but node is kept visible by descendants, show at min size
  if (ownScale === 0 && !shouldHide(node)) return 0.1;
  return ownScale;
}

function shouldHide(node) {
  var cats = node.data('cats') || [];
  // Node has no categories — never hide
  if (cats.length === 0) return false;
  // Hidden only if ALL categories (own + descendants) are at 0
  return cats.every(function(c) { return catScales[c] === 0; });
}

function applyVisuals() {
  readSliders();
  cy.batch(function() {
    cy.elements().removeClass('hidden scaled-down');
    cy.nodes().forEach(function(n) {
      if (shouldHide(n)) {
        n.addClass('hidden');
        n.connectedEdges().filter(function(e) { return e.data('source') === n.id(); }).addClass('hidden');
        return;
      }
      var s = getNodeScale(n);
      var ns = layoutParams.nodeSize;
      var total = s * ns;
      if (total < 1) {
        n.addClass('scaled-down');
      }
      var baseW = Math.max(20, n.data('label').length * 9 + 12);
      n.style({
        'width': baseW * total,
        'height': 32 * total,
        'font-size': (14 * total) + 'px',
        'text-max-width': (baseW * total) + 'px',
      });
    });
    // Hide edges whose source is hidden, and apply edge width
    cy.edges().forEach(function(e) {
      if (e.source().hasClass('hidden') || e.target().hasClass('hidden')) {
        e.addClass('hidden');
      } else {
        e.style('width', layoutParams.edgeWidth);
      }
    });
  });
}

function applyLayout() {
  readSliders();
  cy.layout(getLayoutOpts()).run();
}

// Initial apply
applyVisuals();
applyLayout();

// Live scale/hide update on slider change
slidersDiv.addEventListener('input', function(e) {
  if (e.target.type !== 'range') return;
  e.target.nextElementSibling.textContent = e.target.value;
  var ln = e.target.dataset.layout;
  if (e.target.dataset.cat || ln === 'nodeSize' || ln === 'edgeWidth') applyVisuals();
});

// Layout reflow only on button click
updateBtn.addEventListener('click', function() {
  applyVisuals();
  applyLayout();
});

const sidebar = document.getElementById('sidebar');

function showDetails(nodeData) {
  const details = nodeData.details;
  let html = '<h2>' + details.type + '</h2>';
  html += '<div class="node-id">' + details.id + '</div>';
  const skip = new Set(['id', 'type']);
  for (const [key, val] of Object.entries(details)) {
    if (skip.has(key)) continue;
    html += '<div class="field"><div class="field-name">' + key + '</div>';
    html += '<div class="field-value">' + JSON.stringify(val, null, 2) + '</div></div>';
  }
  sidebar.innerHTML = html;
}

cy.on('tap', 'node', function(evt) {
  const node = evt.target;
  showDetails(node.data());
  cy.elements().removeClass('highlighted faded');
  const neighborhood = node.neighborhood().add(node);
  neighborhood.addClass('highlighted');
  cy.elements().not(neighborhood).addClass('faded');
});

cy.on('tap', function(evt) {
  if (evt.target === cy) {
    cy.elements().removeClass('highlighted faded');
    sidebar.innerHTML = '<h2>Node Details</h2><p class="hint">Click a node to inspect it.</p>';
  }
});

const searchBox = document.getElementById('search-box');
searchBox.addEventListener('input', function() {
  const q = this.value.toLowerCase().trim();
  if (!q) { cy.elements().removeClass('highlighted faded'); return; }
  cy.elements().removeClass('highlighted faded');
  const matched = cy.nodes().filter(function(n) {
    const d = n.data().details;
    const searchable = [d.id, d.type, d.func, d.method, d.attribute, d.name, n.data().label]
      .filter(Boolean).join(' ').toLowerCase();
    return searchable.includes(q);
  });
  if (matched.length > 0) {
    const neighborhood = matched.neighborhood().add(matched);
    neighborhood.addClass('highlighted');
    cy.elements().not(neighborhood).addClass('faded');
  }
});

// --- Choice path highlighting ---
// choiceIdxMap: { choiceNodeId -> currently selected option index }
const choiceIdxMap = {};
cy.nodes().forEach(function(n) {
  if (n.data('chosen_idx') !== undefined) {
    choiceIdxMap[n.id()] = n.data('chosen_idx');
  }
});

// BFS backward from __outputs__. For each edge:
// - If it has gating_choice, only follow if choiceIdxMap matches gating_option_idx
// - If no gating, always follow
// Highlight followed edges + their source/target nodes
function applyChoicePath() {
  cy.elements().removeClass('choice-path choice-dimmed');
  var outputNode = cy.getElementById('__outputs__');
  if (!outputNode.length) return;

  var visited = {};
  var queue = [outputNode.id()];
  visited[outputNode.id()] = true;
  var pathEdges = cy.collection();
  var pathNodes = cy.collection();
  pathNodes.merge(outputNode);

  while (queue.length > 0) {
    var nodeId = queue.shift();
    var node = cy.getElementById(nodeId);
    node.incomers('edge').forEach(function(e) {
      if (e.hasClass('hidden')) return;
      var gc = e.data('gating_choice');
      if (gc !== undefined) {
        if (choiceIdxMap[gc] !== e.data('gating_option_idx')) return;
      }
      pathEdges.merge(e);
      pathNodes.merge(e.source());
      pathNodes.merge(e.target());
      var src = e.source();
      if (!visited[src.id()]) {
        visited[src.id()] = true;
        queue.push(src.id());
      }
    });
  }

  pathEdges.addClass('choice-path');
  pathNodes.addClass('choice-path');
  cy.nodes().not(pathNodes).addClass('choice-dimmed');
}

if (Object.keys(choiceIdxMap).length > 0) {
  applyChoicePath();
}

// Edge click: activate this choice and all ancestor choices on the path to __outputs__
cy.on('tap', 'edge', function(evt) {
  var edge = evt.target;
  var gc = edge.data('gating_choice');
  if (gc === undefined) return;
  choiceIdxMap[gc] = edge.data('gating_option_idx');
  // Activate ancestor choices: find outgoing edges from the gating choice node
  // that are themselves gated by a parent choice, and set that parent accordingly.
  // Repeat until we reach ungated edges (the root path to __outputs__).
  var cur = gc;
  var seen = {};
  while (cur && !seen[cur]) {
    seen[cur] = true;
    var node = cy.getElementById(cur);
    var parentGc = null;
    node.outgoers('edge').forEach(function(e) {
      var eGc = e.data('gating_choice');
      if (eGc !== undefined && !seen[eGc]) {
        choiceIdxMap[eGc] = e.data('gating_option_idx');
        parentGc = eGc;
      }
    });
    cur = parentGc;
  }
  applyChoicePath();
});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Visualize a graph.json trace file")
    parser.add_argument("input", type=Path, help="Path to graph.json")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output HTML path"
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    args = parser.parse_args()

    graph_data = json.loads(args.input.read_text())
    elements = build_cytoscape_elements(graph_data)

    html = HTML_TEMPLATE.replace("__ELEMENTS_JSON__", json.dumps(elements))
    html = html.replace("__CATEGORY_DEFAULTS__", CATEGORY_DEFAULTS)
    html = html.replace("__CATEGORY_COLORS__", CATEGORY_COLORS)

    out_path = args.output or args.input.with_suffix(".html")
    out_path.write_text(html)
    print(f"Written to {out_path}")

    if not args.no_open:
        webbrowser.open(str(out_path.resolve()))


if __name__ == "__main__":
    main()
