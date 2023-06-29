# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import datetime
import os
import re
import subprocess
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

REGEX_PATTERN = f'(\[.*\]) *([^ ]+) -> ([^ ]+) \| ([0-9\.]+)h:([0-9\.]+)m:([0-9\.]+)s'

if __name__ == "__main__":
    *_, pvl_users = subprocess.check_output("/usr/bin/getent group pvl".split()).decode().rstrip('\n').split(":")
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stage', required=True, choices=['coarse', 'fine', 'fine_terrain'], type=str)
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help="Output directory of experiments.")
    parser.add_argument('--user', type=str, default=os.environ['USER'], choices=pvl_users.split(','), help="User who ran the jobs.")
    parser.add_argument('-d', '--days_since', type=int, default=14, help="At least how long ago were the jobs run? Smaller values are faster.")
    args = parser.parse_args()

    date_since = (datetime.datetime.now() - datetime.timedelta(days=args.days_since)).strftime("%Y-%m-%d")
    cmd = ['sacct', '--noheader', '--starttime', date_since, '-u', args.user, '-o', 'jobname%50,ElapsedRaw%50,stat%50']
    out = subprocess.check_output(cmd).decode('utf-8')
    job_times = {}
    for l in out.splitlines():
        job_name, job_sec, status, *_ = l.strip().split()
        regex = re.compile(f"{args.output_folder.stem}_({'[A-Z]'*8}_.+)").fullmatch(job_name)
        if regex is not None:
            seed_stage, = regex.groups()
            job_times[seed_stage]  = (int(job_sec), (status == "COMPLETED"))

    data_dict = defaultdict(list)
    all_time_logs = list(args.output_folder.rglob(f"{args.stage}_times.txt"))
    for log in all_time_logs:
        job_times_key = f"{log.parent.parent.stem}_{args.stage}"
        job_time, finished = job_times[job_times_key]
        if not finished:
            continue
        for name, start, end, h, m, s in re.findall(REGEX_PATTERN, log.read_text()):
            chunk_time = float(h)*3600 + float(m)*60 + float(s)
            percent = chunk_time * 100 / job_time
            data_dict[name].append((chunk_time, percent))

    to_print = []
    for key, data_list in data_dict.items():
        average_time = round(np.mean([t for t,_ in data_list]))
        average_percent = round(np.mean([p for _,p in data_list]))
        to_print.append((average_percent, f"{key.ljust(40)} {average_time//3600:02d}h:{((average_time%3600)//60):02d}m ({average_percent}%) [#{len(data_list)}]"))
    
    for _,s in sorted(to_print):
        print(s)
        