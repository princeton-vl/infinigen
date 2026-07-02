# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse

from infinigen2 import GENERATORS_MANIFEST


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=["Material", "Object", "Exporter", "Scene", "Mask"],
    )
    parser.add_argument("-k", type=str, default=None)
    parser.add_argument("--columns", nargs="+", default=["name"])
    parser.add_argument("--separator", type=str, default=" ")
    parser.add_argument("--missing_values", choices=["error", "drop"], default="drop")
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--tail", type=int, default=None)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    items = GENERATORS_MANIFEST.copy()

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


if __name__ == "__main__":
    main()
