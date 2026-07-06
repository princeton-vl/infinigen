# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse

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
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("--columns", nargs="+", default=["name"])
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--missing_values", choices=["error", "drop"], default="drop")
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--tail", type=int, default=None)
    return parser


def _owned_presets() -> list[tuple[str, str, list[str]]]:
    """(generator dotted name, generator shortname, owned preset shortnames) for each
    manifest entry that declares a `presets` list. Ownership is explicit in the
    manifest — nothing is inferred from names."""
    owned = []
    for record in GENERATORS_MANIFEST.to_dict("records"):
        presets = record.get("presets")
        if isinstance(presets, list) and presets:
            owned.append((record["name"], record["name"].rsplit(".", 1)[-1], presets))
    return owned


def preset_dotted_names() -> list[str]:
    """Dotted names of every preset the manifest declares, resolved in its owning
    generator's module (presets live alongside the generator that owns them)."""
    names = []
    for gen_name, _, presets in _owned_presets():
        module_path = gen_name.rsplit(".", 1)[0]
        names.extend(f"{module_path}.{preset}" for preset in presets)
    return sorted(names)


def preset_parents() -> dict[str, str]:
    """Map each preset shortname to the generator shortname that owns it, read
    directly from the manifest `presets` lists (no name-based inference)."""
    parents = {}
    for _, owner, presets in _owned_presets():
        for preset in presets:
            parents[preset] = owner
    return parents


def _preset_manifest() -> pd.DataFrame:
    return pd.DataFrame({"name": preset_dotted_names(), "category": "MaterialPreset"})


def _main():
    parser = get_parser()
    args = parser.parse_args()

    items = _preset_manifest() if args.presets else GENERATORS_MANIFEST.copy()

    if args.categories is not None:
        items = items[items["category"].isin(args.categories)]

    if "shortname" in args.columns:
        items["shortname"] = items["name"].str.split(".").str[-1]

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

    if args.k is not None:
        items = items[items["name"].str.contains(args.k)]

    if args.head is not None:
        items = items.head(args.head)
    if args.tail is not None:
        items = items.tail(args.tail)

    for row in items.itertuples():
        values = [str(getattr(row, col)) for col in args.columns]
        print(args.separator.join(values))


def main():
    with skip_teardown_on_exit():
        _main()


if __name__ == "__main__":
    main()
