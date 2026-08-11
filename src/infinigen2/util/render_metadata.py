import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import bpy

from infinigen2.exporters.util.format import ExportType
from infinigen2.util.hardware_info import get_hardware_info


@contextmanager
def time_step(times: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()  # validate-ignore: test_determinism
    yield
    end = time.perf_counter()  # validate-ignore: test_determinism
    times[name] = times.get(name, 0.0) + (end - start)


def triangle_counts(objects: list) -> tuple[int, int]:
    """Return base-cage and render-level evaluated triangle counts."""
    states = []
    for obj in objects:
        item = obj.item()
        states.extend(
            (modifier, modifier.show_viewport, modifier.levels)
            for modifier in getattr(item, "modifiers", [])
            if modifier.type == "SUBSURF"
        )

    counts = []
    try:
        for visible in (False, True):
            for modifier, _show_viewport, _levels in states:
                modifier.show_viewport = visible
                if visible:
                    modifier.levels = modifier.render_levels

            deps = bpy.context.evaluated_depsgraph_get()
            total = 0
            for obj in objects:
                item = obj.item()
                if item.type != "MESH":
                    continue
                evaluated = item.evaluated_get(deps)
                mesh = evaluated.to_mesh()
                try:
                    mesh.calc_loop_triangles()
                    total += len(mesh.loop_triangles)
                finally:
                    evaluated.to_mesh_clear()
            counts.append(total)
    finally:
        for modifier, show_viewport, levels in states:
            modifier.show_viewport = show_viewport
            modifier.levels = levels
    return tuple(counts)


def write_render_metadata(
    output: str | Path,
    seed: int,
    times: dict[str, float],
    exports: dict[ExportType, list[Path]],
    build_keys: set[str],
    render_keys: set[str],
    n_frames: int,
) -> dict:
    output = Path(output)
    blend_build_sec = sum(v for k, v in times.items() if k in build_keys)
    render_total = sum(v for k, v in times.items() if k in render_keys)

    metadata = {
        "seed": hex(seed),
        "hardware": get_hardware_info(),
        "generator_times": times,
        "stats": {
            "blend_build_sec": blend_build_sec,
            "render_sec_per_frame": render_total / max(n_frames, 1),
            "n_frames": n_frames,
        },
        "exports": {str(k): [str(p) for p in v] for k, v in exports.items()},
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
