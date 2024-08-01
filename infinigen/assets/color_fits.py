# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Stamatis Alexandropoulos, Meenal Parakh


import numpy as np

from infinigen.core.util.color import hsv2rgba

manual_fits = {
    "sofa_fabric": {
        "means": np.array([[0.1, 0.25], [0.5, 0.7], [0.65, 0.15]]),
        "covariances": [
            0.7 * np.array([[0.01, 0], [0, 0.04]]),
            0.7 * np.array([[0.02, 0], [0, 0.02]]),
            0.7 * np.array([[0.03, 0], [-0.01, 0.012]]),
        ],
        "probabilities": [0.5, 0.3, 0.2],
        "n_components": 3,
    },
    "sofa_leather": {
        "means": np.array([[0.07, 0.45], [0.6, 0.3]]),
        "covariances": [
            0.7 * np.array([[0.005, 0], [0, 0.09]]),
            0.7 * np.array([[0.015, 0], [0, 0.04]]),
        ],
        "min_val": [0.04, 0.04],
        "max_val": [0.75, 0.85],
        "probabilities": [0.7, 0.3],
        "n_components": 2,
    },
    "sofa_linen": {
        "means": np.array([[0.12, 0.5], [0.6, 0.4], [0.9, 0.2]]),
        "covariances": [
            0.7 * np.array([[0.01, 0], [0, 0.12]]),
            0.7 * np.array([[0.01, 0], [0, 0.09]]),
            0.7 * np.array([[0.01, 0], [0, 0.02]]),
        ],
        "probabilities": [0.8, 0.15, 0.05],
        "n_components": 3,
    },
    "sofa_velvet": {
        "means": np.array([[0.52, 0.45]]),
        "covariances": [np.array([[0.2, 0], [0, 0.2]])],
        "probabilities": [1.0],
        "n_components": 1,
    },
    "bedding_sheet": {
        "means": np.array([[0.1, 0.4], [0.6, 0.2]]),
        "covariances": [
            0.7 * np.array([[0.01, 0], [0, 0.1]]),
            0.7 * np.array([[0.03, 0], [-0.01, 0.02]]),
        ],
        "probabilities": [0.9, 0.1],
        "n_components": 2,
    },
}

val_params = {
    "bedding_sheet": {"min_val": 0.15, "max_val": 0.94, "mu": 0.66, "std": 0.17},
    "sofa_fabric": {"min_val": 0.10, "max_val": 0.88, "mu": 0.47, "std": 0.23},
    "sofa_leather": {"min_val": 0.06, "max_val": 0.93, "mu": 0.40, "std": 0.2},
    "sofa_linen": {"min_val": 0.15, "max_val": 0.86, "mu": 0.55, "std": 0.2},
    "sofa_velvet": {"min_val": 0.11, "max_val": 0.70, "mu": 0.35, "std": 0.18},
}


def get_val(mu=0.5, std=0.2, min_val=0.1, max_val=0.9):
    val = np.random.normal(mu, std)
    val = np.clip(val, min_val, max_val)
    return val


def real_color_distribution(name):
    params = manual_fits[name]

    num_gaussians = params["n_components"]
    idx = np.random.choice(num_gaussians, p=params["probabilities"])

    mu = params["means"][idx]
    cov = params["covariances"][idx]

    h, s = np.random.multivariate_normal(mu, cov)
    min_val = params.get("min_val", 0.0)
    max_val = params.get("max_val", 1.0)

    h, s = np.clip([h, s], min_val, max_val)

    v = get_val(**(val_params[name])) * 0.1
    rgba = hsv2rgba([h, s, v])

    return rgba
