# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import seaborn as sns

plt = sns.mpl.pyplot

def parse_mem(s: str):
    for c, m in [('K', 1e6), ('M', 1e3), ('G', 1)]:
        if s.endswith(c):
            return (float(s.rstrip(c))) / m
    raise Exception()

def make_plot(data, bin_width, path: os.PathLike):
    data = np.asarray(data)
    data = data[data < np.quantile(data, 0.99)]

    suffix = ''
    if data.max() > 1e6:
        data /= 1e6
        suffix = 'M'

    ax = sns.histplot(data, bins=int(data.max() / bin_width))
    plt.xticks(fontsize=16)
    plt.ylabel("")
    plt.yticks([])
    plt.xlim(left=0)

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, lbl in enumerate(labels):
        labels[i] = f"{lbl}{suffix}"
    ax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(str(path))
    plt.clf()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipeline_dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--assume_resample', action='store_true')
    args = parser.parse_args()

    print("Running sacct...")
    sacct = subprocess.check_output('sacct --allusers --starttime 2022-10-31 --endtime 2022-11-07 -o JobID,JobName%100,MaxRSS,Elapsed,AllocTres%100,State%30'.split()).decode().splitlines()
    # sacct = Path("sacct_saved.txt").read_text().splitlines()
    print("Processing")

    scene_lookup = defaultdict(set)
    relevant_jobs = {}
    num_tri_counts = []

    for folder in args.pipeline_dirs:
        run_name = folder.stem
        finished_seeds = set((folder / "finished_seeds.txt").read_text().splitlines())
        print(f"Total number of scenes: {len(finished_seeds)}")

        for line in sacct:
            match = re.fullmatch(f"(\d+) +(\S*{run_name}\S*) +" + "(\d{2}:\d{2}:\d{2}) +\S*cpu=(.*),mem=\S+ +COMPLETED *", line)
            if match:
                jobid, name, elapsed, tres = match.groups()
                cpus, *_ = tres.replace('gres/gpu:', '').split(',')
                gpu = re.fullmatch(".*gres/gpu:(.*=\d+),gres/gpu=.*", tres)
                hours, mins, secs = map(float, elapsed.split(':'))
                relevant_jobs[jobid] = {"name": name, "elapsed": (hours + mins/60 + secs/3600), "cpus": int(cpus), "gpus": gpu.group(1).split('=') if gpu else None}

        for line in sacct:
            memory_match = re.fullmatch("(\d+).0 +python +([\.\d]+[K|M|G]) .*COMPLETED *", line)
            if memory_match:
                jobid, mem = memory_match.groups()
                if jobid in relevant_jobs:
                    relevant_jobs[jobid]['max_memory'] = parse_mem(mem)

        for k,v in relevant_jobs.items():
            seed = re.fullmatch(".*_([A-Z]+)_.*", v["name"]).group(1)
            if seed in finished_seeds:
                scene_lookup[seed].add(k)

        for seed in finished_seeds:
            stats_txt = folder / seed / "logs" / "fine_polycounts.txt"
            text = stats_txt.read_text().replace(',','')
            num_tris, = map(int, re.findall("Tris:(\d+)\n", text))
            num_tri_counts.append(float(num_tris))


    all_max_mems = []
    all_elapsed = []
    all_cpu_hours = []
    all_gpu_hours = []

    for scene_seed, job_id_list in scene_lookup.items():
        jobs = [relevant_jobs[j] for j in sorted(job_id_list)]
        if args.assume_resample:
            while 'render_2' in jobs[-1]["name"]:
                jobs.pop()
            while 'render_1' in jobs[-1]["name"]:
                jobs.pop()
        max_mem = max(j['max_memory'] for j in jobs)
        all_max_mems.append(max_mem)
        total_elapsed = sum(j['elapsed'] for j in jobs)
        all_elapsed.append(total_elapsed)
        cpu_hours = sum((j['elapsed']*j["cpus"]) for j in jobs)
        all_cpu_hours.append(cpu_hours)
        gpu_hours = sum((j['elapsed']*int(j["gpus"][1])) for j in jobs if (j["gpus"] is not None))
        all_gpu_hours.append(gpu_hours)

    make_plot(all_max_mems, 3, 'plots/all_max_mems.png')
    make_plot(all_elapsed, 0.5, 'plots/all_elapsed.png')
    make_plot(all_cpu_hours, 1, 'plots/all_cpu_hours.png')
    make_plot(all_gpu_hours, 0.1, 'plots/all_gpu_hours.png')
    make_plot(num_tri_counts, 2, 'plots/all_tricounts.png')
