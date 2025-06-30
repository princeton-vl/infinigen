# Copyright (C) 2025, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Abhishek Joshi: primary author

"""
Factory method for spawn sim ready assets. Current sim ready file formats include mjcf, usd, and urdf.
"""

from pathlib import Path
from typing import Dict, Tuple

from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP
from infinigen.core.sim.exporters.factory import sim_exporter_factory
from infinigen.core.sim import kinematic_compiler
from infinigen.core.util import blender as butil

def spawn_simready(
    name: str,
    exporter: str = "mjcf",
    export_dir: Path = Path("./sim_exports"),
    image_res: int = 512,
    seed: int = 0,
    **kwargs,
) -> Tuple[Path, Dict]:
    """
    Spawns a sim ready asset in the given format.
    """

    if name not in OBJECT_CLASS_MAP:
        raise KeyError(
            f"Asset name {name} not found in the following list of assets: {list(OBJECT_CLASS_MAP.keys())}"
        )

    # gets the desired asset class
    asset_class = OBJECT_CLASS_MAP[name]
    asset = asset_class(seed)
    obj = asset.spawn_asset(i=0, **kwargs)

    sim_blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)

    sim_blueprint["name"] = name

    export_dir = export_dir / exporter

    export_func = sim_exporter_factory(exporter=exporter)
    export_path, semantic_mapping = export_func(
        blend_obj=obj,
        sim_blueprint=sim_blueprint,
        seed=asset.factory_seed,
        sample_joint_params_fn=asset_class.sample_joint_parameters,
        export_dir=export_dir,
        image_res=image_res,
        **kwargs,
    )

    return export_path, semantic_mapping
