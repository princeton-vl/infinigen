# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

import argparse
import os
import re
from datetime import timedelta
from pathlib import Path

import jinja2
import pandas as pd


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def td_to_str(td):
    """
    convert a timedelta object td to a string in HH:MM:SS format.
    """
    if pd.isnull(td):
        return td
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def parse_scene_log(
    scene_path: Path,
):
    ret_dict = {
        "coarse_tris": "NAN",
        "fine_tirs": "NAN",
        "obj_count": "NAN",
        "gen_time": "NAN",
        "gen_mem_gb": "NAN",
        "render_time": "NAN",
        "gt_time": "NAN",
    }

    step_times = {
        "fineterrain": [],
        "coarse": [],
        "populate": [],
        "rendershort": [],
        "shortrender": [],
        "blendergt": [],
    }

    log_folder = scene_path / "logs"
    coarse_folder = scene_path / "coarse"
    fine_folder = scene_path / "fine"

    if not (log_folder.exists() and coarse_folder.exists() and fine_folder.exists()):
        return ret_dict

    for filepath in log_folder.glob("*.err"):
        step = ""
        for stepName in step_times:
            if filepath.stem.startswith(stepName):
                step = stepName
                break
        else:
            continue

        text = filepath.read_text()
        if "[MAIN TOTAL] finished in" not in text:
            continue
        search = re.search(
            r"\[MAIN TOTAL\] finished in ([0-9]+):([0-9]+):([0-9]+)", text
        )
        d = None
        if search is None:
            search = re.search(
                r"\[MAIN TOTAL\] finished in ([0-9]) day.*, ([0-9]+):([0-9]+):([0-9]+)",
                text,
            )
            d, h, m, s = search.group(1, 2, 3, 4)
        else:
            h, m, s = search.group(1, 2, 3)
        if d is None:
            step_timedelta = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
        else:
            step_timedelta = timedelta(
                days=int(d), hours=int(h), minutes=int(m), seconds=int(s)
            )
        step_times[step].append(step_timedelta)

    coarse_stage_df = pd.read_csv(os.path.join(coarse_folder, "pipeline_coarse.csv"))

    if len(step_times["coarse"]) >= 1:
        coarse_time = step_times["coarse"][0]
    else:
        coarse_time = timedelta(seconds=0)
    if len(step_times["populate"]) >= 1:
        pop_time = step_times["populate"][0]
    else:
        pop_time = timedelta(seconds=0)

    if len(step_times["fineterrain"]) >= 1:
        fine_time = step_times["fineterrain"][0]
    else:
        fine_time = timedelta(seconds=0)
    if len(step_times["rendershort"]) >= 1:
        render_time = step_times["rendershort"][0]
    elif len(step_times["shortrender"]) >= 1:
        render_time = step_times["shortrender"][0]
    else:
        render_time = timedelta(seconds=0)
    if len(step_times["blendergt"]) >= 1:
        gt_time = step_times["blendergt"][0]
    else:
        gt_time = timedelta(seconds=0)

    mem = coarse_stage_df["mem_at_finish"].iloc[-1]
    obj_count = coarse_stage_df["obj_count"].iloc[-1]

    ret_dict = {
        "obj_count": obj_count,
        "gen_time": coarse_time + pop_time + fine_time,
        "gen_mem_gb": sizeof_fmt(mem),
        "render_time": render_time,
        "gt_time": gt_time,
    }

    fine_poly = parse_poly_file(fine_folder / "polycounts.txt")
    ret_dict["gen_triangles"] = fine_poly.get("Triangles", "NAN")

    return ret_dict


def parse_poly_file(path):
    res = {}

    if not path.exists():
        return res

    for l in path.read_text().splitlines():
        fields = l.split(":")
        if len(fields) != 2:
            continue
        k, v = fields
        res[k] = v

    return res


def parse_asset_log(asset_path):
    poly = parse_poly_file(asset_path / "polycounts.txt")

    return {
        "triangles": poly.get("Tris", "NAN"),
        "gen_mem": poly.get("Memory", "NAN"),
    }


def format_stats(d):
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def parse_run_df(run_path: Path):
    runs = {
        "_".join((x.name.split("_")[2:])): x for x in run_path.iterdir() if x.is_dir()
    }
    for k, v in runs.items():
        print(k, v)

    records = []

    def scene_folders(type):
        scenes = []

        for name, path in runs.items():
            if not name.startswith(type):
                continue
            for scene in path.iterdir():
                if not scene.is_dir():
                    continue
                if scene.name == "logs":
                    continue
                scenes.append(scene)

        return sorted(scenes)

    IMG_NAME = "Image_0_0_0048_0.png"
    NORMAL_NAME = "SurfaceNormal_0_0_0048_0.png"

    for scene in scene_folders("scene_nature"):
        stats = parse_scene_log(scene)
        scenetype = "_".join(scene.parent.name.split("_")[2:])
        img = scene / "frames" / "Image" / "camera_0" / IMG_NAME
        normal = scene / "frames" / "SurfaceNormal" / "camera_0" / NORMAL_NAME
        records.append(
            {
                "name": scenetype + "/" + scene.name,
                "category": "scene_nature",
                "img_path": img,
                "normal_path": normal,
                "stats": format_stats(stats),
            }
        )

    for scene in scene_folders("scene_indoor"):
        stats = parse_scene_log(scene)
        scenetype = "_".join(scene.parent.name.split("_")[2:])
        img = scene / "frames" / "Image" / "camera_0" / IMG_NAME
        normal = scene / "frames" / "SurfaceNormal" / "camera_0" / NORMAL_NAME
        records.append(
            {
                "name": scenetype + "/" + scene.name,
                "category": "scene_indoor",
                "img_path": img,
                "normal_path": normal,
                "stats": format_stats(stats),
            }
        )

    for scene in scene_folders("asset"):
        category = "_".join(scene.parent.name.split("_")[2:])
        record = {
            "category": category,
            "name": category + "/" + scene.name,
            "img_path": scene / "Image.png",
            "stats": format_stats(parse_asset_log(scene)),
        }

        records.append(record)

    print(f"{run_path=} found {len(records)} records")

    return pd.DataFrame.from_records(records)


def find_run(base_path: str, run: str) -> Path:
    base_path = Path(base_path)

    run_path = base_path / run
    if run_path.exists():
        return run_path

    options = [x for x in base_path.iterdir() if run in x.name]
    if len(options) == 1:
        return options[0]
    elif len(options) > 1:
        raise ValueError(f"Multiple runs found for {run}, {options}")
    else:
        raise FileNotFoundError(f"Could not find match for {run=} in {base_path=}")


def fuzzy_merge(dfA, dfB, keyA, keyB, threshold=1):
    from rapidfuzz import fuzz, process

    matches_A = []
    matches_B = []

    def preproc(x):
        x = x.split("/")[-1]
        x = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", x)
        x = x.lower()
        return x

    b_names_list = dfB[keyB].apply(preproc)
    print(list(b_names_list))

    for i, rowA in dfA.iterrows():
        match = process.extractOne(
            preproc(rowA[keyA]), b_names_list, scorer=fuzz.ratio, score_cutoff=threshold
        )

        if match:
            matched_rowB = dfB.loc[match[2]].to_dict()
            # print(f"Matched {rowA[keyA].split('/')[-1]} with {matched_rowB[keyB].split('/')[-1]} with score {match[1]:.2f}")
        else:
            matched_rowB = {col: pd.NA for col in dfB.columns}
            matched_rowB[keyB] = "No Matching Scene"
            print(f"No match found for {rowA[keyA].split('/')[-1]}")

        matches_A.append(rowA.to_dict())
        matches_B.append(matched_rowB)

    dfA_matched = pd.DataFrame(matches_A).add_suffix("_A")
    dfB_matched = pd.DataFrame(matches_B).add_suffix("_B")

    merged_df = pd.concat([dfA_matched, dfB_matched], axis=1)

    return merged_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compare_runs", type=Path, nargs="+")
    parser.add_argument("--nearest", action="store_true")
    args = parser.parse_args()

    for run in args.compare_runs:
        if not run.exists():
            raise FileNotFoundError(f"Could not find {run}")

    runs_folder = args.compare_runs[0].parent
    views_folder = runs_folder.parent / "views"

    runs = [find_run(runs_folder, run) for run in args.compare_runs]

    dfs = {run: parse_run_df(runs_folder / run) for run in runs}

    version_names = [r.name for r in runs]

    if len(runs) == 2:
        lhs = dfs[runs[0]]
        rhs = dfs[runs[1]]
        print(lhs.columns)
        print(rhs.columns)
    elif len(runs) == 1:
        lhs = rhs = dfs[runs[0]]
        print(lhs.columns)
    else:
        raise ValueError("Only 1 or 2 runs supported")

    if args.nearest:
        main_df = fuzzy_merge(lhs, rhs, keyA="name", keyB="name", threshold=80)
    else:
        main_df = lhs.merge(rhs, on="name", suffixes=("_A", "_B"), how="outer")
        main_df["name_A"] = main_df["name"]
        main_df["name_B"] = main_df["name"]

    assert "name_A" in main_df.columns
    assert "name_B" in main_df.columns

    for col in main_df:
        if col.startswith("img_path"):
            main_df[col] = main_df[col].apply(
                lambda x: "../" + str(x.relative_to(runs_folder.parent))
                if isinstance(x, Path)
                else x
            )

    print(main_df.columns)

    categories = [
        "scene_nature",
        "scene_indoor",
        "asset_nature_meshes",
        "asset_indoor_meshes",
        "asset_nature_materials",
        "asset_indoor_materials",
    ]
    path_lookups = {
        category: main_df[main_df["category_A"] == category].to_dict(orient="records")
        for category in categories
    }

    for category, records in path_lookups.items():
        print(category, len(records))

    jenv = jinja2.Environment(loader=jinja2.FileSystemLoader("."))
    template = jenv.get_template("tests/integration/template.html")

    # Render the template with the data
    html_content = template.render(
        title="Version Comparison", version_names=version_names, **path_lookups
    )

    # Save the rendered HTML to a file
    name = "_".join([p.name for p in args.compare_runs]) + ".html"
    output_path = views_folder / name
    print("Writing to ", output_path)
    output_path.write_text(html_content)


if __name__ == "__main__":
    main()
