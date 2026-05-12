"""Collect integration render data from render_index events."""

import json
import re
import shlex
from pathlib import Path
from typing import NamedTuple


class ImageInfo(NamedTuple):
    path: str
    variant_key: str
    renderer: str
    variant_type: str
    category: str
    pass_type: str
    filename: str
    label: str
    command_text: str
    stderr_text: str


class AssetData(NamedTuple):
    asset_name: str
    asset_type: str
    version_name: str
    images: list[ImageInfo]


class CollectionResult(NamedTuple):
    version_name: str
    version_path: Path
    assets: dict[str, AssetData]
    matched_assets: int
    no_output_events: int


def classify_image(variant_key: str, filename: str) -> dict[str, str]:
    variant_match = re.match(r"(\w+)-(\w+)-(.*)", variant_key)
    if variant_match:
        renderer = variant_match.group(2)
        variant_type = variant_match.group(3)
    else:
        renderer = "unknown"
        variant_type = variant_key

    if variant_type in {
        "BUMP",
        "DISPLACEMENT",
        "DISPLACEMENT_AND_BUMP",
        "REALIZE_MESH",
    }:
        category = "displacement"
    else:
        category = "distribution"

    if filename.startswith("image_") or filename == "Image" or filename == "image":
        pass_type = "image"
    elif filename.startswith("surface-normal_"):
        pass_type = "surface-normal"
    elif filename == "error":
        pass_type = "error"
    else:
        pass_type = "other"

    return {
        "renderer": renderer,
        "variant_type": variant_type,
        "category": category,
        "pass_type": pass_type,
    }


def _normalize_stem(path_str: str) -> str:
    rel = Path(path_str)
    stem = rel.stem
    if stem.isdigit() and rel.parts and rel.parts[-2].lower().startswith("camera"):
        return f"image_{rel.parts[-2]}_{stem}"
    return stem


def _merge_asset(assets: dict[str, AssetData], new_asset: AssetData):
    if new_asset.asset_name not in assets:
        assets[new_asset.asset_name] = new_asset
        return

    existing = assets[new_asset.asset_name]
    assets[new_asset.asset_name] = AssetData(
        asset_name=existing.asset_name,
        asset_type=existing.asset_type,
        version_name=existing.version_name,
        images=existing.images + new_asset.images,
    )


def _command_text(cmd_field) -> str:
    if isinstance(cmd_field, list):
        return shlex.join(cmd_field)
    return str(cmd_field or "")


def collect_images_structured(
    version_path: Path,
    version_name: str,
    material_files=None,
    manifest_file=None,
) -> CollectionResult:
    del material_files, manifest_file

    events_dir = version_path / "render_index" / "events"
    if not events_dir.is_dir():
        raise FileNotFoundError(
            f"Missing index directory: {events_dir}. This viewer now requires render_index events."
        )

    assets: dict[str, AssetData] = {}
    no_output_events = 0

    for event_file in sorted(events_dir.glob("*.json")):
        event = json.loads(event_file.read_text())
        asset_name = event.get("generator", "unknown")
        asset_type = event.get("asset_type", "unknown")
        variant_key = event.get("variant_key", "unknown")
        status = event.get("status", "no_outputs")
        cmd_text = _command_text(event.get("cmd"))

        images: list[ImageInfo] = []

        for image_path in event.get("images", []):
            stem = _normalize_stem(image_path)
            meta = classify_image(variant_key, stem)
            images.append(
                ImageInfo(
                    path=f"{version_name}/{image_path}",
                    variant_key=f"{variant_key}/{stem}",
                    renderer=meta["renderer"],
                    variant_type=meta["variant_type"],
                    category=meta["category"],
                    pass_type=meta["pass_type"],
                    filename=image_path,
                    label=f"{meta['renderer']} / {meta['variant_type']} / {meta['pass_type']}",
                    command_text=cmd_text,
                    stderr_text="",
                )
            )

        if status == "no_outputs":
            no_output_events += 1
            stderr_text = ""
            stderr_path = event.get("stderr_path", "")
            if stderr_path:
                stderr_file = version_path / stderr_path
                if stderr_file.exists():
                    stderr_text = stderr_file.read_text()

            meta = classify_image(variant_key, "error")
            images.append(
                ImageInfo(
                    path="",
                    variant_key=f"{variant_key}/error",
                    renderer=meta["renderer"],
                    variant_type=meta["variant_type"],
                    category=meta["category"],
                    pass_type="error",
                    filename="",
                    label=f"{meta['renderer']} / {meta['variant_type']} / error",
                    command_text=cmd_text,
                    stderr_text=stderr_text,
                )
            )

        _merge_asset(
            assets,
            AssetData(
                asset_name=asset_name,
                asset_type=asset_type,
                version_name=version_name,
                images=images,
            ),
        )

    return CollectionResult(
        version_name=version_name,
        version_path=version_path,
        assets=assets,
        matched_assets=len(assets),
        no_output_events=no_output_events,
    )


def print_collection_summary(results: list[CollectionResult]):
    print("\n=== Collection Summary ===")
    for result in results:
        print(
            f"{result.version_name}: assets {result.matched_assets}; no_outputs {result.no_output_events}"
        )
    print()
