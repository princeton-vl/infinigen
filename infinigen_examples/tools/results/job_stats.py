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


@dataclass
class Job:
    job_id: str
    name: str
    current_status: str
    req_memory: str = None
    max_memory_gb: float = -1
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
        if self.req_memory is not None:
            return f"{self.job_id}     {self.name.ljust(60)}     {self.gpu.ljust(10)} {self.cpu.ljust(5)}  {f'{self.max_memory_gb:.3f}G'.ljust(9)}  {self.start_time.strftime('%m/%d/%Y, %H:%M:%S')}  {str(self.time_elapsed).ljust(20)}  {self.current_status.ljust(10)}  {self.node}"
        else:
            return f"{self.job_id}     {self.name.ljust(60+73)}     {self.current_status.ljust(10)}"

sacct_line_regex = re.compile("([0-9]+) +(\S+) +(\S+) +([0-9]+) +([A-Z_]+) +(node[0-9]+) +(\S+).*").fullmatch

HEADER = "Job ID       Job name                                                         GPU        CPU    Max Mem    Start date, time      Elapsed               Status      Node"

MEM_FACTOR = {"G": 1, "M": 1e3, "K": 1e6}

def parse_sacct_line(line):
    if sacct_line_regex(line) is None:
        return
    job_id, job_name, resources, elapsed_raw, current_status, node, start_time = sacct_line_regex(line).groups()
    request = dict(e.split('=') for e in resources.split(','))
    start_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S')
    elapsed = timedelta(seconds=int(elapsed_raw))
    return Job(job_id=job_id, name=job_name, req_memory=request['mem'], cpu=request['cpu'], gpu=request.get('gpu', '0'), current_status=current_status, node=node, start_time=start_time, time_elapsed=elapsed)

def get_node_info():
    sinfo_out = subprocess.check_output('/usr/bin/sinfo --Node --format=%12N%12P%C --noheader'.split()).decode()
    nodes_info = defaultdict(dict)
    for node, group, allocated_cpus, total_cpus in re.findall("(\S+) +(\S+) +([0-9]+)/[0-9]+/[0-9]+/([0-9]+)", sinfo_out):
        if group != "all":
            assert node not in nodes_info
            nodes_info[node]['cpus'] = int(total_cpus)
            nodes_info[node]['allocated_cpus'] = int(allocated_cpus)
            nodes_info[node]['group'] = group
    return dict(nodes_info)

if __name__ == '__main__':
    *_, pvl_users = subprocess.check_output("/usr/bin/getent group pvl".split()).decode().rstrip('\n').split(":")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--days_since', type=int, default=14, help="At least how long ago were the jobs run? Smaller values are faster.")
    parser.add_argument('-o', '--output_folder', type=Path, required=True, help="Output directory of experiments.")
    parser.add_argument('--user', type=str, default=os.environ['USER'], choices=pvl_users.split(','), help="User who ran the jobs.")
    args = parser.parse_args()
    assert args.output_folder.exists()
    squeue_out = subprocess.check_output(f'squeue --user={args.user} -o "%.24i%.40j%.14R" -h'.split()).decode()
    non_started_jobs = [Job(job_id=i, name=n, current_status=s) for i,n,s in re.findall("([0-9]+) +(\S+) +(\([a-zA-Z]+\))", squeue_out) if n.startswith(args.output_folder.stem)]
    sacct_start_date = (datetime.now() - timedelta(days=args.days_since)).strftime('%Y-%m-%d')# 2022-05-07
    sacct_command = f"sacct --starttime {sacct_start_date} -u {args.user} --noheader -o jobid,jobname%80,AllocTRES%80,ElapsedRaw,stat%30,NodeList,Start,MaxRSS"
    print(f"Running command: {sacct_command}")
    sacct_output = subprocess.check_output(sacct_command.split()).decode()
    relevant_started_jobs = []
    mem_dict = dict(re.findall("([0-9]+)\.0 +.* +([0-9]*.?[0-9]*[KMG])", sacct_output))

    for sacct_line in sacct_output.splitlines():
        parsed_job = parse_sacct_line(sacct_line)
        if (parsed_job is not None) and (parsed_job.name.startswith(args.output_folder.stem)):
            if parsed_job.job_id in mem_dict:
                max_memory = mem_dict[parsed_job.job_id]
                parsed_job.max_memory_gb = float(max_memory[:-1]) / MEM_FACTOR[max_memory[-1]]
            relevant_started_jobs.append(parsed_job)

    all_jobs = (relevant_started_jobs + non_started_jobs)
    seed_dict = defaultdict(list)
    for j in all_jobs:
        seed, = re.compile(f"{args.output_folder.stem}_([^ _]+)_.*").fullmatch(j.name).groups()
        seed_dict[seed].append(j)
    node_info = get_node_info()
    for k,v in sorted(seed_dict.items()):
        print("-"*len(HEADER) + "\n" + HEADER)
        for j in v:
            if j.node is not None:
                print(f"{j}({node_info[j.node]['group']})")
            else:
                print(j)
