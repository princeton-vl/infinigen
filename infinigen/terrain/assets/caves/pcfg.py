# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import re
from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np

CONFIG_FILE = Path(__file__).parent / "cfg.txt"
assert CONFIG_FILE.exists(), CONFIG_FILE.resolve()

STARTING_SYMBOL = "Q"


def create_pcfg():
    config_lines = CONFIG_FILE.read_text().splitlines()
    rule = re.compile("(.+) \(([0-9].[0-9][0-9]?)\) -> (.+)")
    PCFG = defaultdict(lambda: dict(a=[], p=[]))  # 1 levels
    for line in config_lines:
        regex = rule.fullmatch(line)
        if regex is not None and len(regex.groups()) == 3:
            LHS, prob, RHS = regex.groups()
            PCFG[LHS]["a"].append(RHS)
            PCFG[LHS]["p"].append(float(prob))

    for k, v in PCFG.items():
        assert abs(np.sum(v["p"]) - 1) < 1e-4, (k, v["p"])
    return dict(PCFG)


def generate_string(max_len=10000):
    PCFG = create_pcfg()

    # print(f"PCFG Keys: {' '.join(list(PCFG.keys()))}")
    def expand(s):
        return list(np.random.choice(**PCFG[s]).split()) if (s in PCFG) else s

    def terminate_expand(s):
        return ["n"] if (s in PCFG) else s

    symbols = [STARTING_SYMBOL]
    for steps in range(1000):
        symbols = list(chain(*map(expand, symbols)))
        assert all([isinstance(e, str) for e in symbols])
        if not any((s in PCFG for s in symbols)) and len(symbols) < max_len:
            symbols = [STARTING_SYMBOL]

        if len(symbols) >= max_len:
            # print(f"Done making symbols. There are {len(symbols)}")
            symbols = list(chain(*map(terminate_expand, symbols)))
            assert "P" not in symbols, terminate_expand("P")
            return symbols

    raise Exception("Too many steps")


if __name__ == "__main__":
    print(generate_string())
