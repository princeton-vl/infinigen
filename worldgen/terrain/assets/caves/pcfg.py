from random import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import time
from itertools import chain
assert CONFIG_FILE.exists(), CONFIG_FILE.resolve()

STARTING_SYMBOL = 'Q'


def create_pcfg():
    config_lines = CONFIG_FILE.read_text().splitlines()
    rule = re.compile("(.+) \(([0-9].[0-9][0-9]?)\) -> (.+)")
    for line in config_lines:
        regex = rule.fullmatch(line)
        if regex is not None and len(regex.groups()) == 3:
            LHS, prob, RHS = regex.groups()
            PCFG[LHS]['a'].append(RHS)
            PCFG[LHS]['p'].append(float(prob))

        assert abs(np.sum(v['p'])-1) < 1e-4, (k, v['p'])
    return dict(PCFG)


    PCFG = create_pcfg()
    # print(f"PCFG Keys: {' '.join(list(PCFG.keys()))}")

    symbols = [STARTING_SYMBOL]
    for steps in range(1000):
        symbols = list(chain(*map(expand, symbols)))
        assert all([(type(e) == str) for e in symbols])
        if not any((s in PCFG for s in symbols)) and len(symbols) < max_len:
            symbols = [STARTING_SYMBOL]

        if len(symbols) >= max_len:
            # print(f"Done making symbols. There are {len(symbols)}")
            symbols = list(chain(*map(terminate_expand, symbols)))
            assert 'P' not in symbols, terminate_expand('P')
            return symbols

    raise Exception("Too many steps")


if __name__ == '__main__':
    print(generate_string())
