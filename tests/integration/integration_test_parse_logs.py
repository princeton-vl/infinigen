import os
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd

SCENES = ["dining"]


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def make_stats(data_df):
    stats = pd.DataFrame()
    stats["mean"] = data_df.mean(axis=1)
    stats["median"] = data_df.min(axis=1)
    stats["90%"] = data_df.quantile(0.9, axis=1)
    stats["95%"] = data_df.quantile(0.95, axis=1)
    stats["99%"] = data_df.quantile(0.99, axis=1)
    return stats


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
    scene_path,
    step_times,
    poly_data,
):
    log_folder = os.path.join(scene_path, "logs")
    coarse_folder = os.path.join(scene_path, "coarse")
    fine_folder = next(Path(scene_path).glob("fine*"))
    # seed = Path(scene_path).stem
    # scene_times = []
    if os.path.isdir(log_folder):
        for filepath in Path(log_folder).glob("*.err"):
            step = ""
            for stepName in step_times:
                if filepath.stem.startswith(stepName):
                    step = stepName
                    break
            else:
                continue
            errFile = open(filepath)
            text = errFile.read()
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

    coarse_poly = os.path.join(coarse_folder, "polycounts.txt")
    fine_poly = os.path.join(fine_folder, "polycounts.txt")
    if os.path.isfile(coarse_poly) and os.path.isfile(fine_poly):
        coarse_text = open(coarse_poly).read().replace(",", "")
        fine_text = open(fine_poly).read().replace(",", "")
        for faces, tris in re.findall("Faces:([0-9]+)Tris:([0-9]+)", coarse_text):
            poly_data["[Coarse] Faces"].append(int(faces))
            poly_data["[Coarse] Tris"].append(int(tris))

        for faces, tris in re.findall("Faces:([0-9]+)Tris:([0-9]+)", fine_text):
            poly_data["[Fine] Faces"].append(int(faces))
            poly_data["[Fine] Tris"].append(int(tris))

    coarse_stage_df = pd.read_csv(os.path.join(coarse_folder, "pipeline_coarse.csv"))

    coarse_time = step_times["coarse"][0]
    pop_time = step_times["populate"][0]
    fine_time = step_times["fineterrain"][0]
    render_time = step_times["rendershort"][0]
    gt_time = step_times["blendergt"][0]

    coarse_face = poly_data["[Coarse] Faces"][0]
    coarse_tris = poly_data["[Coarse] Tris"][0]
    fine_face = poly_data["[Fine] Faces"][0]
    fine_tris = poly_data["[Fine] Tris"][0]
    mem = coarse_stage_df["mem_at_finish"].iloc[-1]
    obj_count = coarse_stage_df["obj_count"].iloc[-1]
    instance_count = coarse_stage_df["instance_count"].iloc[-1]

    ret_dict = {
        "coarse_face": coarse_face,
        "coarse_tris": coarse_tris,
        "fine_face": fine_face,
        "fine_tirs": fine_tris,
        "mem": mem,
        "obj_count": obj_count,
        "instance_count": instance_count,
        "coarse_time": coarse_time,
        "pop_time": pop_time,
        "fine_time": fine_time,
        "render_time": render_time,
        "gt_time": gt_time,
    }

    return ret_dict


def test_logs(dir):
    print("")
    step_times = {
        "fineterrain": [],
        "coarse": [],
        "populate": [],
        "rendershort": [],
        "blendergt": [],
    }

    poly_data = {
        "[Coarse] Faces": [],
        "[Coarse] Tris": [],
        "[Fine] Faces": [],
        "[Fine] Tris": [],
    }
    completed_seeds = os.path.join(dir, "finished_seeds.txt")
    # num_lines = sum(1 for _ in open(completed_seeds))
    for scene in os.listdir(dir):
        if scene not in open(completed_seeds).read():
            continue
        scene_path = os.path.join(dir, scene)
        dict = parse_scene_log(
            scene_path,
            step_times,
            poly_data,
        )

    print(dict)


test_logs("/n/fs/scratch/dy2617/system_test/dining/")
