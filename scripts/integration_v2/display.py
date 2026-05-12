"""Display module: transform collected data into template-ready structures."""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from collection import classify_image
from PIL import Image


@lru_cache(maxsize=2048)
def _load_image_array(path_str: str):
    with Image.open(path_str) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def _get_field(image_info, key: str, default=""):
    if isinstance(image_info, dict):
        return image_info.get(key, default)
    return getattr(image_info, key, default)


def _alignment_key(variant_key: str) -> str:
    # Align logical passes, so errors pair with RGB image slots.
    # Examples:
    #   cycles-0/error -> cycles-0/image
    #   cycles-0/image_camera-0_0000 -> cycles-0/image
    #   cycles-0/surface-normal_camera-0_0000 -> cycles-0/surface-normal
    if "/" not in variant_key:
        return variant_key

    variant_part, suffix = variant_key.rsplit("/", 1)
    if (
        suffix == "error"
        or suffix == "image"
        or suffix == "Image"
        or suffix.startswith("image_")
    ):
        return f"{variant_part}/image"
    if suffix.startswith("surface-normal_") or suffix == "surface-normal":
        return f"{variant_part}/surface-normal"
    return variant_key


def _pairwise_mse(img_a, img_b, root_a: Path, root_b: Path) -> float:
    pass_a = _get_field(img_a, "pass_type")
    pass_b = _get_field(img_b, "pass_type")
    if pass_a == "error" and pass_b == "error":
        return math.inf

    file_a = _get_field(img_a, "filename")
    file_b = _get_field(img_b, "filename")
    has_a = bool(file_a)
    has_b = bool(file_b)

    if has_a != has_b:
        return math.inf
    if not has_a and not has_b:
        return 0.0

    path_a = root_a / file_a
    path_b = root_b / file_b
    if not path_a.exists() or not path_b.exists():
        return math.inf

    arr_a = _load_image_array(str(path_a.resolve()))
    arr_b = _load_image_array(str(path_b.resolve()))
    if arr_a.shape != arr_b.shape:
        return math.inf
    return float(np.mean((arr_a - arr_b) ** 2))


def _format_mse_value(mse: float) -> str:
    if math.isinf(mse):
        return "inf"
    return f"{mse:.2f}"


def _to_image_dict(image_info):
    if isinstance(image_info, dict):
        return dict(image_info)
    return image_info._asdict()


def _has_image_path(image_info) -> bool:
    return bool(_get_field(image_info, "path"))


def _compute_pairwise_mse_map(
    variant_keys: list[str],
    version_images_by_key: dict,
    version_a: str,
    version_b: str,
    root_a: Path,
    root_b: Path,
) -> dict[str, float]:
    if not variant_keys:
        return {}

    def compute_one(key: str):
        mse = _pairwise_mse(
            version_images_by_key[version_a].get(key),
            version_images_by_key[version_b].get(key),
            root_a,
            root_b,
        )
        return key, mse

    # Default to CPU parallelism; allow override with COMPARE_MSE_WORKERS.
    default_workers = min(32, os.cpu_count() or 1)
    workers = int(os.environ.get("COMPARE_MSE_WORKERS", default_workers))
    if workers <= 1 or len(variant_keys) < 8:
        return dict(compute_one(key) for key in variant_keys)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(compute_one, variant_keys))


def render_row(
    asset_name: str, asset_type: str, collection_results: list, version_names: list[str]
) -> dict:
    all_variant_keys = set()
    version_images_by_key = {}

    for result in collection_results:
        version_name = result.version_name
        version_images_by_key[version_name] = {}

        if asset_name in result.assets:
            for img_info in result.assets[asset_name].images:
                key = _alignment_key(img_info.variant_key)
                all_variant_keys.add(key)
                existing = version_images_by_key[version_name].get(key)
                if existing is None:
                    version_images_by_key[version_name][key] = img_info
                elif (not _has_image_path(existing)) and _has_image_path(img_info):
                    # Prefer a real render image over a no-output error marker.
                    version_images_by_key[version_name][key] = img_info

    sorted_variant_keys = sorted(all_variant_keys)
    mse_by_variant_key = {}
    avg_mse = None
    if len(version_names) == 2:
        root_by_version = {
            result.version_name: result.version_path for result in collection_results
        }
        version_a, version_b = version_names
        root_a = root_by_version[version_a]
        root_b = root_by_version[version_b]
        mse_by_variant_key = _compute_pairwise_mse_map(
            list(all_variant_keys),
            version_images_by_key,
            version_a,
            version_b,
            root_a,
            root_b,
        )
        if mse_by_variant_key:
            avg_mse = sum(mse_by_variant_key.values()) / len(mse_by_variant_key)

    objects = []
    for result in collection_results:
        version_name = result.version_name
        aligned_images = []

        for variant_key in sorted_variant_keys:
            if variant_key in version_images_by_key[version_name]:
                image_data = _to_image_dict(
                    version_images_by_key[version_name][variant_key]
                )
            else:
                if "/" in variant_key:
                    variant_part, suffix = variant_key.rsplit("/", 1)
                else:
                    variant_part, suffix = variant_key, ""
                info = classify_image(variant_part, suffix)
                info["variant_type"] = "missing"
                info["label"] = (
                    f"{info['renderer']} / {info['variant_type']} / {info['pass_type']}"
                )
                image_data = {
                    "variant_key": variant_key,
                    "path": "",
                    "command_text": "",
                    "stderr_text": "",
                    **info,
                }

            aligned_images.append(image_data)

        objects.append({"version": version_name, "images": aligned_images})

    if avg_mse is not None:
        asset_label = f"{asset_name} avg. MSE {_format_mse_value(avg_mse)}"
    else:
        asset_label = asset_name

    return {
        "asset": asset_name,
        "asset_label": asset_label,
        "asset_type": asset_type,
        "avg_mse": avg_mse,
        "objects": objects,
    }


def build_comparison_data(
    collection_results: list, version_names: list[str]
) -> list[dict]:
    all_assets = {}
    for result in collection_results:
        for asset_name, asset_data in result.assets.items():
            if asset_name not in all_assets:
                all_assets[asset_name] = asset_data.asset_type

    rows = []
    for asset_name in sorted(all_assets.keys()):
        asset_type = all_assets[asset_name]
        rows.append(
            render_row(asset_name, asset_type, collection_results, version_names)
        )

    if len(version_names) == 2:
        rows.sort(
            key=lambda row: row["avg_mse"]
            if row["avg_mse"] is not None
            else float("-inf"),
            reverse=True,
        )

    return rows
