# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lahav Lipson


import subprocess
import argparse
import re
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--with_substring", required=True)
    parser.add_argument('-wo', "--without_substring", default=None)
    parser.add_argument("--not_running", action="store_true")
    args = parser.parse_args()
    job_cmd = f'/usr/bin/squeue --user={os.environ["USER"]} -o %.24i%.40j%.14R -h'
    squeue_output = subprocess.check_output(job_cmd.split()).decode()
    matches = re.findall("([0-9]+) *([^ ]+) *([^ ]+) *\n", squeue_output)
    for job_id, job_name, job_status in tqdm(matches):
        should_cancel = ((args.with_substring in job_name) and 
        ((args.without_substring is None) or args.without_substring not in job_name) and 
        ((not args.not_running) or 'node' not in job_status))
        if should_cancel:
            subprocess.check_output(["/usr/bin/scancel", job_id])
    