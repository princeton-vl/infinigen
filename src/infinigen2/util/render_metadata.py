import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from infinigen2.exporters.util.format import ExportType


@contextmanager
def time_step(times: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()  # validate-ignore: test_determinism
    yield
    end = time.perf_counter()  # validate-ignore: test_determinism
    times[name] = times.get(name, 0.0) + (end - start)


def _export_path(path: Path, output: Path) -> str:
    try:
        return str(Path(path).relative_to(output))
    except ValueError:
        return Path(path).name


def write_render_metadata(
    output: str | Path,
    seed: int,
    times: dict[str, float],
    exports: dict[ExportType, list[Path]],
    build_keys: set[str],
    render_keys: set[str],
    n_frames: int,
    trajectory_seed: int | None = None,
) -> dict:
    """Write metadata.json, which ships verbatim in datareleases.

    Nothing here may identify the machine or user that rendered the scene, and export
    paths are relative to output so they do not leak the render tree's layout."""
    output = Path(output)
    blend_build_sec = sum(v for k, v in times.items() if k in build_keys)
    render_total = sum(v for k, v in times.items() if k in render_keys)

    metadata = {
        "seed": hex(seed),
        "trajectory_seed": hex(
            trajectory_seed if trajectory_seed is not None else seed
        ),
        "generator_times": times,
        "stats": {
            "blend_build_sec": blend_build_sec,
            "render_sec_per_frame": render_total / max(n_frames, 1),
            "n_frames": n_frames,
        },
        "exports": {
            str(k): [_export_path(p, output) for p in v] for k, v in exports.items()
        },
    }
    with open(output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
