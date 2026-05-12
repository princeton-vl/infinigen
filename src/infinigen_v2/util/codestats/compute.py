"""
The model is already indexed into topological order by setup.py.

- choice_probs[choice_id] gives branch probabilities for one discrete choice
- choice_branches[choice_id] gives the branch ids for that choice
- branch_vars[branch_id] is a bitmask of variables activated by that branch
- var_kind[var_id] is "choice" or "continuous"
"""

from __future__ import annotations

from functools import lru_cache
from math import log2


def shannon_entropy_bits(probabilities) -> float:
    total = 0.0
    for probability in probabilities:
        probability = float(probability)
        if probability <= 0.0:
            continue
        total += -probability * log2(probability)
    return total


def cyclomatic_complexity(choice_probs: dict[int, list[float]]) -> int:
    """Cyclomatic complexity where only control flow is pf.control.hcoice
    Return 1 + sum(branch_count - 1) over discrete choices.
    """
    total = 1
    for probabilities in choice_probs.values():
        total += max(len(probabilities) - 1, 0)
    return total


def _lowest_set_bit_index(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def solve_tree(indexed_model: dict) -> dict:
    """Tree-style recursion.

    child subtrees inside one branch are treated as disjoint and summed independently.
    """

    cache: dict[int, tuple[int, float, float, float]] = {}

    def solve_var(var_id: int) -> tuple[int, float, float, float]:
        cached = cache.get(var_id)
        if cached is not None:
            return cached

        if indexed_model["var_kind"][var_id] == "continuous":
            result = (1, 0.0, 1.0, 0.0)
            cache[var_id] = result
            return result

        probabilities = indexed_model["choice_probs"][var_id]
        branch_ids = indexed_model["choice_branches"][var_id]
        combinations = 0
        discrete_mean = 1.0
        continuous_mean = 0.0
        entropy_bits = shannon_entropy_bits(probabilities)

        for probability, branch_id in zip(probabilities, branch_ids, strict=True):
            child_combinations = 1
            child_discrete = 0.0
            child_continuous = 0.0
            child_entropy = 0.0
            branch_mask = indexed_model["branch_vars"][branch_id]

            while branch_mask:
                child_var = _lowest_set_bit_index(branch_mask)
                branch_mask &= ~(1 << child_var)
                child_stats = solve_var(child_var)
                child_combinations *= child_stats[0]
                child_discrete += child_stats[1]
                child_continuous += child_stats[2]
                child_entropy += child_stats[3]

            combinations += child_combinations
            discrete_mean += probability * child_discrete
            continuous_mean += probability * child_continuous
            entropy_bits += probability * child_entropy

        result = (combinations, discrete_mean, continuous_mean, entropy_bits)
        cache[var_id] = result
        return result

    combinations = 1
    discrete_mean = 0.0
    continuous_mean = 0.0
    entropy_bits = 0.0
    root_mask = indexed_model["root_mask"]
    while root_mask:
        root_var = _lowest_set_bit_index(root_mask)
        root_mask &= ~(1 << root_var)
        root_stats = solve_var(root_var)
        combinations *= root_stats[0]
        discrete_mean += root_stats[1]
        continuous_mean += root_stats[2]
        entropy_bits += root_stats[3]

    return {
        "discrete_combinations": combinations,
        "discrete_count_mean": discrete_mean,
        "continuous_count_mean": continuous_mean,
        "entropy_discrete_bits": entropy_bits,
    }


def solve_dp(indexed_model: dict) -> dict:
    """Exact DP"""

    @lru_cache(maxsize=None)
    def solve(frontier_mask: int) -> tuple[int, float, float, float]:
        if frontier_mask == 0:
            return 1, 0.0, 0.0, 0.0

        next_var = _lowest_set_bit_index(frontier_mask)
        remaining = frontier_mask & ~(1 << next_var)

        if indexed_model["var_kind"][next_var] == "continuous":
            combinations, discrete_mean, continuous_mean, entropy_bits = solve(
                remaining
            )
            return combinations, discrete_mean, continuous_mean + 1.0, entropy_bits

        probabilities = indexed_model["choice_probs"][next_var]
        branch_ids = indexed_model["choice_branches"][next_var]
        combinations = 0
        discrete_mean = 1.0
        continuous_mean = 0.0
        entropy_bits = shannon_entropy_bits(probabilities)

        for probability, branch_id in zip(probabilities, branch_ids, strict=True):
            next_frontier = remaining | indexed_model["branch_vars"][branch_id]
            (
                child_combinations,
                child_discrete,
                child_continuous,
                child_entropy,
            ) = solve(next_frontier)
            combinations += child_combinations
            discrete_mean += probability * child_discrete
            continuous_mean += probability * child_continuous
            entropy_bits += probability * child_entropy

        return combinations, discrete_mean, continuous_mean, entropy_bits

    combinations, discrete_mean, continuous_mean, entropy_bits = solve(
        indexed_model["root_mask"]
    )
    return {
        "discrete_combinations": combinations,
        "discrete_count_mean": discrete_mean,
        "continuous_count_mean": continuous_mean,
        "entropy_discrete_bits": entropy_bits,
        "dp_states": solve.cache_info().currsize,
    }
