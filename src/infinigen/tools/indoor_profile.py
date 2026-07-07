# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: David Yan

import argparse
import re
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

"""
The following function s attributed to FObersteiner from Stack Overflow at https://stackoverflow.com/a/64662985
and is licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/deed.en#ref-appropriate-credit). 
David Yan used this code WITHOUT modification. 
"""


def td_to_str(td):
    """
    convert a timedelta object td to a string in HH:MM:SS format.
    """
    if pd.isnull(td):
        return td
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def main(dir: Path):
    coarse_data = defaultdict(list)
    render_data = defaultdict(list)
    with open(dir / "finished_seeds.txt") as f:
        seeds = f.read().splitlines()

    for seed in seeds:
        try:
            coarse_log = open(dir / seed / "logs" / "coarse.err").read()
            render_log = open(
                next((dir / seed / "logs").glob("shortrender*.err"))
            ).read()
        except FileNotFoundError:
            continue

        for name, h, m, s in re.findall(
            r"\[INFO\] \| \[(.*?)\] finished in ([0-9]+):([0-9]+):([0-9]+)", coarse_log
        ):
            timedelta_obj = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
            if timedelta_obj.total_seconds() < 1:
                continue
            coarse_data[name].append(timedelta_obj)

        for name, h, m, s in re.findall(
            r"\[INFO\] \| \[(.*?)\] finished in ([0-9]+):([0-9]+):([0-9]+)", render_log
        ):
            timedelta_obj = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
            if timedelta_obj.total_seconds() < 1:
                continue
            render_data[name].append(timedelta_obj)

    coarse_stats = make_stats(pd.DataFrame.from_dict(coarse_data, orient="index"))
    render_stats = make_stats(pd.DataFrame.from_dict(render_data, orient="index"))

    for column in coarse_stats:
        coarse_stats[column] = (
            coarse_stats[column].dt.round("1s").map(lambda x: td_to_str(x))
        )

    for column in coarse_stats:
        render_stats[column] = (
            render_stats[column].dt.round("1s").map(lambda x: td_to_str(x))
        )

    print(coarse_stats.sort_values("median", ascending=False))
    print(render_stats.sort_values("median", ascending=False))


def make_stats(data_df):
    stats = pd.DataFrame()
    stats["mean"] = data_df.mean(axis=1)
    stats["median"] = data_df.median(axis=1)
    stats["90%"] = data_df.quantile(0.9, axis=1)
    stats["95%"] = data_df.quantile(0.95, axis=1)
    stats["99%"] = data_df.quantile(0.99, axis=1)
    return stats


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    main(args.dir)
