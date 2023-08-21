# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import argparse
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


@dataclass
class Job:
    job_id: str
    name: str
    current_status: str
    memory: str = None
    cpu: str = None
    gpu: str = None
    node: str = None
    start_time: datetime = None
    time_elapsed: timedelta = None

    def end_time(self):
        return self.start_time + self.time_elapsed

    def __lt__(self, other):
        return (int(self.job_id) < int(other.job_id))

    def __str__(self):
        if self.memory is not None:
            return f"{self.job_id}     {self.name.ljust(40)}     {self.gpu.ljust(10)} {self.cpu.ljust(2)}  {self.start_time.strftime('%m/%d/%Y, %H:%M:%S')}  {str(self.time_elapsed).ljust(20)}  {self.current_status.ljust(10)}  {self.node}"
        else:
            return f"{self.job_id}     {self.name.ljust(40)}     {self.current_status.ljust(10)}"

sacct_line_regex = re.compile("([0-9]+) +([^ ]+) +([^ ]+) +([0-9]+) +([A-Z]+) +(node[0-9]+) +([^ ]+).*").fullmatch

def parse_sacct_line(line):
    if sacct_line_regex(line) is None:
        return
    job_id, job_name, resources, elapsed_raw, current_status, node, start_time = sacct_line_regex(line).groups()
    request = dict(e.split('=') for e in resources.split(','))
    start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    elapsed = timedelta(seconds=int(elapsed_raw))
    return Job(job_id=job_id, name=job_name, memory=request['mem'], cpu=request['cpu'], gpu=request.get('gpu', '0'), current_status=current_status, node=node, start_time=start_time, time_elapsed=elapsed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--days_since', type=int, default=14)
    parser.add_argument('-o', '--output_folder', type=Path, required=True)
    args = parser.parse_args()
    assert args.output_folder.exists()
    squeue_out = subprocess.check_output('squeue --user=llipson -o "%.24i%.40j%.14R" -h'.split()).decode()
    non_started_jobs = [Job(job_id=i, name=n, current_status=s) for i,n,s in re.findall("([0-9]+) +([^ ]+) +(\([a-zA-Z]+\))", squeue_out) if n.startswith(args.output_folder.stem)]
    sacct_start_date = (datetime.now() - timedelta(days=args.days_since)).strftime('%Y-%m-%d')# 2022-05-07
    sacct_command = f"sacct --starttime {sacct_start_date} -u {os.environ['USER']} --noheader -o jobid,jobname%40,AllocTRES%80,ElapsedRaw,stat,NodeList,Start"
    sacct_output = subprocess.check_output(sacct_command.split()).decode()
    relevant_started_jobs = []
    for sacct_line in sacct_output.splitlines():
        parsed_job = parse_sacct_line(sacct_line)
        if (parsed_job is not None) and (parsed_job.name.startswith(args.output_folder.stem)):
            relevant_started_jobs.append(parsed_job)

    all_jobs = (relevant_started_jobs + non_started_jobs)
    seed_dict = defaultdict(list)
    for j in all_jobs:
        seed, = re.compile(f"{args.output_folder.stem}_([^ _]+)_.*").fullmatch(j.name).groups()
        seed_dict[seed].append(j)
    all_times = {"fine": defaultdict(list), "coarse": defaultdict(list), "full": []}
    for k,v in seed_dict.items():
        coarse_job, fine_job, *render_jobs = sorted(v)
        pipeline_start = coarse_job.start_time
        pipeline_end = max(j.end_time() for j in render_jobs)
        total_pipeline_time = pipeline_end - pipeline_start
        assert pipeline_end > fine_job.end_time()
        assert pipeline_start < fine_job.start_time
        assert fine_job.time_elapsed.total_seconds() <= total_pipeline_time.total_seconds(), (pipeline_start, pipeline_end, total_pipeline_time)
        coarse_job_percentage = (100 * coarse_job.time_elapsed.total_seconds()) / total_pipeline_time.total_seconds()
        fine_job_percentage = (100 * fine_job.time_elapsed.total_seconds()) / total_pipeline_time.total_seconds()
        all_times["fine"]["elapsed"].append(fine_job.time_elapsed.total_seconds())
        all_times["fine"]["percentage"].append(fine_job_percentage)
        all_times["coarse"]["elapsed"].append(coarse_job.time_elapsed.total_seconds())
        all_times["coarse"]["percentage"].append(coarse_job_percentage)
        all_times["full"].append(total_pipeline_time.total_seconds())
    fine_status_freq = Counter([j.current_status for j in all_jobs if j.name.endswith("_fine")])
    print(fine_status_freq)

    avg_fine_elapsed = timedelta(seconds=np.mean(all_times["fine"]["elapsed"]))
    print(f'{avg_fine_elapsed} {round(np.mean(all_times["fine"]["percentage"])):02d}%')

    avg_coarse_elapsed = timedelta(seconds=np.mean(all_times["coarse"]["elapsed"]))
    print(f'{avg_fine_elapsed} {round(np.mean(all_times["coarse"]["percentage"])):02d}%')
    vg_full_elapsed = timedelta(seconds=np.mean(all_times["full"]))
    print(vg_full_elapsed)
