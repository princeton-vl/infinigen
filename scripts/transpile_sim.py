import argparse

from infinigen.core.nodes.node_transpiler.transpiler_dev_sim import transpile_simready

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bp",
        "--blend_path",
        type=str,
        required=True,
        help="blend file to convert to mjcf scene",
    )
    parser.add_argument(
        "-bon",
        "--blend_obj_name",
        type=str,
        required=True,
        help="name of the objects to transpile and make sim ready",
    )
    parser.add_argument(
        "--output_name", type=str, required=False, help="name of the object"
    )
    args = parser.parse_args()
    transpile_simready(args)
