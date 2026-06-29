import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from infinigen_v2.exporters.util.format import ExportType
from infinigen_v2.util.hardware_info import get_hardware_info


@contextmanager
def time_step(times: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    times[name] = times.get(name, 0.0) + (time.perf_counter() - start)


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
