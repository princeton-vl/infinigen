# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jack Nugent

"""Display module: transform collected data into template-ready structures."""

import json
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


# AFTER-version red/green threshold; matches the perf gate in perf_diff.py.
PERF_THRESHOLD = 0.05


def _fmt_tris(n: int | None) -> str | None:
    if n is None:
        return None
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


def _fmt_time(s: float | None) -> str | None:
    if s is None:
        return None
    return f"{s:.1f}s"


def _aggregate_stats(stats: list[dict]) -> dict:
    tris = [s["tris"] for s in stats if s.get("tris") is not None]
    cpu = [s["cpu_time_sec"] for s in stats if s.get("cpu_time_sec") is not None]
    gpu = [s["gpu_time_sec"] for s in stats if s.get("gpu_time_sec") is not None]
    return {
        "tris": max(tris) if tris else None,
        "cpu": sum(cpu) if cpu else None,
        "gpu": sum(gpu) if gpu else None,
    }


def _sum_stats(stats: list[dict]) -> dict:
    tris = [s["tris"] for s in stats if s.get("tris") is not None]
    cpu = [s["cpu_time_sec"] for s in stats if s.get("cpu_time_sec") is not None]
    gpu = [s["gpu_time_sec"] for s in stats if s.get("gpu_time_sec") is not None]
    return {
        "tris": sum(tris) if tris else None,
        "cpu": sum(cpu) if cpu else None,
        "gpu": sum(gpu) if gpu else None,
    }


def _category_aggregates(stats: list[dict]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for s in stats:
        groups.setdefault(s.get("category", "distribution"), []).append(s)
    return {cat: _aggregate_stats(items) for cat, items in groups.items()}


def _delta_class(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == 0:
        return "same"
    ratio = after / before
    if ratio > 1 + PERF_THRESHOLD:
        return "worse"
    if ratio < 1 - PERF_THRESHOLD:
        return "better"
    return "same"


def _format_metrics(agg: dict, before: dict | None) -> dict:
    out = {}
    for key, formatter in (("tris", _fmt_tris), ("cpu", _fmt_time), ("gpu", _fmt_time)):
        out[key] = formatter(agg[key])
        base = before[key] if before is not None else None
        out[f"{key}_cls"] = _delta_class(base, agg[key])
    return out


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


def _stat_dict(st) -> dict:
    return {
        "variant_key": st.variant_key,
        "tris": st.tris,
        "cpu_time_sec": st.cpu_time_sec,
        "gpu_time_sec": st.gpu_time_sec,
        "category": classify_image(st.variant_key, "")["category"],
    }


def _collect_asset_stats(result, asset_name: str) -> list[dict]:
    if asset_name not in result.assets:
        return []
    return [_stat_dict(st) for st in result.assets[asset_name].stats]


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

        asset_stats = _collect_asset_stats(result, asset_name)
        objects.append(
            {"version": version_name, "images": aligned_images, "stats": asset_stats}
        )

    ran_versions = [
        result.version_name
        for result in collection_results
        if asset_name in result.assets
    ]
    missing_versions = [v for v in version_names if v not in ran_versions]

    after_version = version_names[-1] if version_names else None
    # Rendered on the after/pr side but absent earlier: a new asset, never fold it.
    is_new = after_version in ran_versions and len(missing_versions) > 0
    # Not rendered on the after/pr side, so nothing new to review there.
    not_run = after_version not in ran_versions and len(ran_versions) > 0

    if not_run or is_new:
        asset_label = asset_name
    elif avg_mse is not None:
        asset_label = f"{asset_name} avg. MSE {_format_mse_value(avg_mse)}"
    else:
        asset_label = asset_name

    return {
        "asset": asset_name,
        "asset_label": asset_label,
        "asset_type": asset_type,
        "avg_mse": avg_mse,
        "not_run": not_run,
        "is_new": is_new,
        "missing_versions": missing_versions,
        "objects": objects,
    }


def _archive_preset_parents(collection_results: list) -> dict | None:
    # Union across versions: a branch knows presets its baseline predates.
    merged = {}
    found = False
    for result in collection_results:
        path = result.version_path / "preset_parents.json"
        if not path.exists():
            continue
        found = True
        try:
            merged.update(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return merged if found else None


def _preset_parents(collection_results: list) -> dict:
    """Preset -> parent-generator map. Prefer the `preset_parents.json` written into
    a render archive at launch time (the web server runs a minimal env without
    infinigen2); fall back to importing infinigen2 when running against a checkout,
    then to no grouping."""
    archived = _archive_preset_parents(collection_results)
    if archived is not None:
        return archived
    try:
        from infinigen2.list import preset_parents

        return preset_parents()
    except Exception:
        return {}


def _append_preset_images(target: dict, source: dict, label: str) -> None:
    for target_obj, preset_obj in zip(target["objects"], source["objects"]):
        target_obj["images"].extend(
            {**img, "is_preset": True, "label": label} for img in preset_obj["images"]
        )
        target_obj["stats"].extend(
            {**stat, "category": "preset"} for stat in preset_obj.get("stats", [])
        )


def _fold_presets(rows: list[dict], parents: dict) -> list[dict]:
    """Fold each `*_preset` asset row into its parent generator row as `is_preset`
    images, so presets render on one line beneath their _rand instead of as their
    own rows. Presets whose parent isn't present stay as standalone rows."""
    by_name = {row["asset"]: row for row in rows}
    folded = set()
    for row in rows:
        if row["asset_type"] != "preset":
            continue
        target = by_name.get(parents.get(row["asset"]))
        if target is None:
            continue
        _append_preset_images(target, row, row["asset"].removesuffix("_preset"))
        folded.add(row["asset"])
    return [row for row in rows if row["asset"] not in folded]


def _attach_obj_metrics(
    obj: dict, before_overall: dict, before_cats: dict, is_after: bool
) -> None:
    stats = obj.get("stats", [])
    overall_base = before_overall if is_after else None
    obj["metrics"] = _format_metrics(_aggregate_stats(stats), overall_base)
    obj["section_metrics"] = {}
    for cat, agg in _category_aggregates(stats).items():
        cat_base = before_cats.get(cat) if is_after else None
        obj["section_metrics"][cat] = _format_metrics(agg, cat_base)


def _attach_metrics(rows: list[dict], version_names: list[str]) -> None:
    """Aggregate each version's render stats onto its object cell (overall +
    per-section), and flag the AFTER version's values worse/better vs BEFORE."""
    if not version_names:
        return
    before_ver = version_names[0]
    after_ver = version_names[-1]
    for row in rows:
        by_ver = {obj["version"]: obj for obj in row["objects"]}
        before_stats = by_ver.get(before_ver, {}).get("stats", [])
        before_overall = _aggregate_stats(before_stats)
        before_cats = _category_aggregates(before_stats)
        for obj in row["objects"]:
            is_after = obj["version"] == after_ver and after_ver != before_ver
            _attach_obj_metrics(obj, before_overall, before_cats, is_after)


def _row_sort_key(row: dict) -> tuple:
    # not_run rows last; within a group highest MSE first (inf error rows at top).
    not_run_rank = 1 if row["not_run"] else 0
    mse = row["avg_mse"]
    mse_rank = math.inf if mse is None else -mse
    return (not_run_rank, mse_rank, row["asset"])


def build_version_totals(rows: list[dict], version_names: list[str]) -> dict[str, dict]:
    """Page-level Tris/CPU/GPU totals per version (summed over every render),
    with the AFTER version flagged worse/better vs BEFORE."""
    if not version_names:
        return {}
    before_ver = version_names[0]
    after_ver = version_names[-1]
    per_version: dict[str, list[dict]] = {v: [] for v in version_names}
    for row in rows:
        for obj in row["objects"]:
            per_version.setdefault(obj["version"], []).extend(obj.get("stats", []))
    before_total = _sum_stats(per_version.get(before_ver, []))
    totals = {}
    for version in version_names:
        is_after = version == after_ver and after_ver != before_ver
        totals[version] = _format_metrics(
            _sum_stats(per_version[version]), before_total if is_after else None
        )
    return totals


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

    rows = _fold_presets(rows, _preset_parents(collection_results))
    _attach_metrics(rows, version_names)

    if len(version_names) == 2:
        rows.sort(key=_row_sort_key)

    return rows
