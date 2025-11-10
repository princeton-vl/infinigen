import argparse

import numpy as np

from infinigen.tools.sim.build_html import generate_html
from infinigen.tools.sim.vis_mujoco import MujocoAssetInitializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render MuJoCo asset animations")
    parser.add_argument(
        "--asset_name", default="door", help="Name of the asset to render"
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1,
        help="Number of seeds to use. Will generate [0:nr-1] by default.",
    )

    parser.add_argument(
        "--rand_seeds",
        action="store_true",
        help="Whether to use random seeds. Instead of [0:nr-1] will generate random seeds.",
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        help="Specific seeds to use. Use instead of --nr/--rand_seeds",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp",
        help="Directory to save rendered outputs",
    )

    parser.add_argument(
        "--parent_alpha",
        type=float,
        default=0.3,
        help="Transparency level for parent geom (0-1)",
    )

    parser.add_argument(
        "--collision_mesh",
        action="store_true",
        help="Whether to load asset in visual-only mode. By default will generate visual only mujoco.",
    )

    parser.add_argument(
        "--remove_existing",
        action="store_true",
        help="Whether to remove existing output directory. Should only be used to remove existing generated assets/videos.",
    )

    parser.add_argument(
        "--use_cached_xml",
        action="store_true",
        help="Re-spawns assets. By default will use cached xml if available.",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="512x512",
        help="Resolution of the rendered output (WxH)",
    )

    parser.add_argument(
        "--add_ground",
        action="store_true",
        help="Whether to add ground plane to the scene.",
    )

    args = parser.parse_args()

    if args.seeds is not None and args.nr != -1:
        raise ValueError("Cannot specify both --seeds and --nr. Please choose one.")

    if args.seeds is not None and len(args.seeds) > 0:
        seeds = args.seeds
    else:
        if args.nr <= 0:
            raise ValueError(
                "Please specify a positive --nr or provide --seeds values."
            )
        if args.rand_seeds:
            seeds = np.random.randint(0, 100000, size=args.nr)
        else:
            seeds = [i for i in range(args.nr)]

    seeds = sorted([int(s) for s in seeds])
    remove = args.remove_existing if args.remove_existing else False
    failure_seeds = []
    for seed in seeds:
        print(f"Rendering {args.asset_name} with seed {seed}")
        try:
            vis = MujocoAssetInitializer(
                asset_name=args.asset_name,
                seed=seed,
                output_dir=args.output_dir,
                parent_alpha=args.parent_alpha,
                collision_mesh=args.collision_mesh,
                remove_existing=remove,
                use_cached=args.use_cached_xml,
                resolution=args.resolution,
                add_ground=args.add_ground,
            )
            vis.render_all_animations()
            remove = False  # Only remove existing on the first iteration
        except Exception as e:
            print(f"Error rendering asset {args.asset_name} with seed {seed}: {e}")
            failure_seeds.append(seed)

    vis_file_name = (
        "_".join(["vis"] + [args.asset_name] + [str(s) for s in seeds]) + ".html"
    )

    seeds_to_show = [i for i in seeds if i not in failure_seeds]
    generate_html(
        args.asset_name, args.output_dir, vis_file_name, seeds_to_show, failure_seeds
    )
