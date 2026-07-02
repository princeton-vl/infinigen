#!/usr/bin/env python3
"""Collect example renders per manifest generator for the CS webspace.

Reads the integration render archive's ``render_index/events/*.json`` files and
copies every successful distribution render for each manifest generator into
``<output>/images/<generator dotted name>/<seed>.png``, matching the URL template
the Sphinx ``conf.py`` emits (``<base>/<slug>/images/<name>/<seed>.png``). The
docs build derives how many seeds to inline from the manifest category alone, so
no index is written here.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "manifest_images"
DEFAULT_MANIFEST = REPO_ROOT / "src" / "infinigen2" / "manifest.json"


def _basename_to_full(manifest_path: Path) -> dict[str, str]:
    """Map each manifest entry's last path component to its full dotted name.

    Render events identify the generator by basename only, so we resolve it via
    the manifest. Basenames that resolve to multiple full names are dropped,
    matching what the rendering side itself can disambiguate.
    """
    data = json.loads(manifest_path.read_text())
    base_to_full: dict[str, str] = {}
    duplicates: set[str] = set()
    for entry in data:
        if entry.get("category") == "Exporter":
            continue
        name = entry.get("name", "")
        if not name:
            continue
        base = name.split(".")[-1]
        if base in base_to_full and base_to_full[base] != name:
            duplicates.add(base)
        else:
            base_to_full[base] = name
    for base in duplicates:
        base_to_full.pop(base, None)
        print(
            f"Warning: duplicate manifest basename '{base}'; skipping.",
            file=sys.stderr,
        )
    return base_to_full


def _is_renderable_image(image_rel: str) -> bool:
    p = Path(image_rel)
    if p.suffix.lower() != ".png":
        return False
    parts = {part.lower() for part in p.parts}
    if "surface-normal" in p.stem.lower() or "error" in p.stem.lower():
        return False
    if "surface-normal" in parts or "error" in parts:
        return False
    return True


def _event_image(event: dict, renders_root: Path) -> Path | None:
    for image_rel in event.get("images", []):
        if not _is_renderable_image(image_rel):
            continue
        src = renders_root / image_rel
        if src.is_file():
            return src
    return None


def collect(renders_root: Path, manifest_path: Path, output_root: Path) -> Path:
    if not renders_root.is_dir():
        print(f"ERROR: renders root not found: {renders_root}", file=sys.stderr)
        sys.exit(1)

    events_dir = renders_root / "render_index" / "events"
    if not events_dir.is_dir():
        print(
            f"ERROR: missing render_index/events dir under {renders_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    base_to_full = _basename_to_full(manifest_path)
    images_dir = output_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True)

    # full dotted name -> list of (seed, source image path)
    per_gen: dict[str, list[tuple[int, Path]]] = {}

    for event_file in sorted(events_dir.glob("*.json")):
        try:
            event = json.loads(event_file.read_text())
        except json.JSONDecodeError:
            continue

        if event.get("status") != "success":
            continue
        base = event.get("generator")
        full = base_to_full.get(base) if base else None
        if full is None:
            continue
        # Random-seed distribution variants have a trailing integer (e.g.
        # "sphere-cycles-3"). Skip displacement/mesh variants ("...-BUMP", etc.).
        variant_key = event.get("variant_key", "")
        trailing = variant_key.rsplit("-", 1)[-1] if variant_key else ""
        if not trailing.isdigit():
            continue

        src = _event_image(event, renders_root)
        if src is not None:
            per_gen.setdefault(full, []).append((int(trailing), src))

    total = 0
    for full, seed_srcs in sorted(per_gen.items()):
        dst_dir = images_dir / full
        dst_dir.mkdir(parents=True)
        for seed, src in sorted(seed_srcs):
            shutil.copyfile(src, dst_dir / f"{seed}.png")
            total += 1

    print(f"Collected {total} images for {len(per_gen)} generators -> {images_dir}")
    return images_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renders-root",
        type=Path,
        required=True,
        help="Path to a single integration render version dir (contains render_index/events/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output dir; images land in <output-root>/images/<name>/<i>.png.",
    )
    args = parser.parse_args()
    collect(args.renders_root, args.manifest, args.output_root)


if __name__ == "__main__":
    main()
