# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse

import numpy as np
import pandas as pd
from procfunc.util.teardown import skip_teardown_on_exit

from infinigen2 import GENERATORS_MANIFEST

CATEGORIES = sorted(GENERATORS_MANIFEST["category"].unique())
_CANONICAL_CATEGORY = {c.lower(): c for c in CATEGORIES}


def _category(value: str) -> str:
    return _CANONICAL_CATEGORY.get(value.lower(), value)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        type=_category,
        choices=CATEGORIES,
    )
    parser.add_argument(
        "--presets",
        action="store_true",
        help="List the `*_preset` variants of the generator modules instead of the "
        "manifest entries; each renders like a material via `material_cube`.",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Drop entries whose `skip_tests` gives a reason not to test them.",
    )
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("--columns", nargs="+", default=["name"])
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument(
        "--missing_values", choices=["error", "drop", "keep"], default="drop"
    )
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--tail", type=int, default=None)
    return parser


def _owned_presets() -> list[tuple[str, str, str, list[str]]]:
    """Generator dotted name, shortname, category, and owned preset shortnames."""
    owned = []
    for record in GENERATORS_MANIFEST.to_dict("records"):
        presets = record.get("presets")
        if isinstance(presets, list) and presets:
            owned.append(
                (
                    record["name"],
                    record["name"].rsplit(".", 1)[-1],
                    record["category"],
                    presets,
                )
            )
    return owned


def preset_dotted_names() -> list[str]:
    """Dotted names of every preset the manifest declares, resolved in its owning
    generator's module (presets live alongside the generator that owns them)."""
    names = []
    for gen_name, _, _, presets in _owned_presets():
        module_path = gen_name.rsplit(".", 1)[0]
        names.extend(f"{module_path}.{preset}" for preset in presets)
    return sorted(names)


def preset_parents() -> dict[str, str]:
    """Map each preset shortname to the generator shortname that owns it, read
    directly from the manifest `presets` lists (no name-based inference)."""
    parents = {}
    for _, owner, _, presets in _owned_presets():
        for preset in presets:
            parents[preset] = owner
    return parents


def preset_categories() -> dict[str, str]:
    categories = {}
    for _, _, category, presets in _owned_presets():
        for preset in presets:
            categories[preset] = category
    return categories


def _preset_manifest() -> pd.DataFrame:
    names = []
    categories = []
    for gen_name, _, category, presets in _owned_presets():
        module_path = gen_name.rsplit(".", 1)[0]
        names.extend(f"{module_path}.{preset}" for preset in presets)
        categories.extend(f"{category}Preset" for _ in presets)
    return pd.DataFrame({"name": names, "category": categories}).sort_values("name")


def _format_cell(value: object) -> str:
    if not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value):
        return ""
    return str(value)


def _main():
    parser = get_parser()
    args = parser.parse_args()

    items = _preset_manifest() if args.presets else GENERATORS_MANIFEST.copy()
    items["_shortname"] = items["name"].str.split(".").str[-1]

    if args.categories is not None:
        items = items[items["category"].isin(args.categories)]

    if args.integration and "skip_tests" in items.columns:
        items = items[items["skip_tests"].isna()]

    if "shortname" in args.columns:
        items["shortname"] = items["_shortname"]

    for column in args.columns:
        if column not in items.columns:
            items[column] = pd.NA

    match args.missing_values:
        case "error":
            for column in args.columns:
                missing_idxs = items[column].isnull()
                if missing_idxs.any():
                    raise ValueError(
                        f"{column} had missing rows {missing_idxs.index[missing_idxs]}"
                    )
        case "drop":
            items = items.dropna(subset=args.columns)
        case "keep":
            pass

    if args.k is not None:
        items = items[items["name"].str.contains(args.k)]

    items = items.sort_values(["category", "_shortname"], kind="stable")

    if args.head is not None:
        items = items.head(args.head)
    if args.tail is not None:
        items = items.tail(args.tail)

    for row in items.itertuples():
        values = [_format_cell(getattr(row, col)) for col in args.columns]
        print(args.separator.join(values))


def main():
    with skip_teardown_on_exit():
        _main()


if __name__ == "__main__":
    main()
