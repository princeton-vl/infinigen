#!/usr/bin/env python3
"""Audit infinigen2 render jobs by GPU type.

For every batch under ``--finished-root``, enumerate the full slurm job array
via the .err files in ``--renderjobs-dir``, then cross-reference with:

  * metadata.json under the finished folder (canonical "ok")
  * the state.log line for that job, if any (gives wall-clock runtime)
  * the .err file's contents, classified into timeout / preempt / vram / other

Slurm-killed jobs (TIME LIMIT / preemption) **never write to state.log**, so
state.log alone undercounts attempts. .err files are the only complete index.

Run on soak. Default args target the 2026-04-28 batch.
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re
import statistics
import subprocess
import sys

# ---------- failure classification --------------------------------------

# Order matters: first match wins. Patterns chosen by sampling err logs.
FAILURE_PATTERNS = [
    ("preempt", "DUE TO PREEMPTION"),
    ("timeout", "TIME LIMIT"),
    ("vram", "out of GPU and shared host memory"),
    ("optix", "Failed to build OptiX"),
    ("noframe", "No frames found"),  # right-camera silent render failure
    ("traceback", "Traceback"),
]


def classify_err_files(renderjobs_dir):
    """Return {basename: category} by running one grep -lF per pattern."""
    out_map = {}
    for label, pattern in FAILURE_PATTERNS:
        try:
            out = subprocess.check_output(
                [
                    "grep",
                    "-lF",
                    pattern,
                    "-r",
                    "--include=*.err",
                    renderjobs_dir,
                ],
                text=True,
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:  # nothing matched
                continue
            out = e.output or ""
        for path in out.splitlines():
            base = os.path.basename(path)
            out_map.setdefault(base, label)  # first match wins
    return out_map


def extract_err_nodes(renderjobs_dir):
    """Return {basename: short_node} parsed from slurm header lines like
    'JOB 28450848 ON node808 CANCELLED ...'. Used to recover GPU type for
    jobs that never wrote to state.log."""
    try:
        # Print path:line for every match, so we know which file.
        out = subprocess.check_output(
            [
                "grep",
                "-rHE",
                r"JOB [0-9]+ ON node[0-9]+",
                "--include=*.err",
                renderjobs_dir,
            ],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return {}
        out = e.output or ""
    rx = re.compile(r"JOB \d+ ON (node\d+)")
    result = {}
    for line in out.splitlines():
        path, _, content = line.partition(":")
        m = rx.search(content)
        if not m:
            continue
        base = os.path.basename(path)
        result.setdefault(base, m.group(1))
    return result


# ---------- slurm GPU mapping -------------------------------------------

SLURM_GPU_MAP = {
    "rtx_2080": "NVIDIA GeForce RTX 2080 Ti",
    "rtx_3090": "NVIDIA GeForce RTX 3090",
    "rtx_a5000": "NVIDIA RTX A5000",
    "rtx_a6000": "NVIDIA RTX A6000",
    "rtx_6000": "NVIDIA RTX 6000 (Ada)",
    "a40": "NVIDIA A40",
    "l40": "NVIDIA L40",
    "gtx_1080": "NVIDIA GeForce GTX 1080 Ti",
}


def fetch_slurm_node_gpus():
    try:
        out = subprocess.check_output(["scontrol", "show", "node"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    result = {}
    current = None
    for line in out.splitlines():
        m = re.search(r"NodeName=(\S+)", line)
        if m:
            current = m.group(1)
        m = re.search(r"Gres=gpu:([^:\s,]+):", line)
        if m and current:
            token = m.group(1).lower()
            result[current] = SLURM_GPU_MAP.get(token, f"slurm:{token}")
    return result


# ---------- input parsing -----------------------------------------------


def parse_state_log(path):
    """state.log line:
        start_iso  end_iso  node  status  scratch_path  err_log  out_log
    status is always 0; jobs killed by slurm don't write here at all."""
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            scratch_path = parts[4]
            job_key = os.path.basename(scratch_path.rstrip("/"))
            batch = os.path.basename(os.path.dirname(scratch_path.rstrip("/")))
            yield {
                "start": datetime.datetime.fromisoformat(parts[0]),
                "end": datetime.datetime.fromisoformat(parts[1]),
                "node": parts[2],
                "batch": batch,
                "job_key": job_key,
                "idx": job_key.rsplit("_", 1)[-1],
            }


def gpu_label(gpus_all):
    if not gpus_all:
        return "unknown"
    return gpus_all[0].split(",")[0].strip()


def short_node(hostname):
    return hostname.split(".")[0]


def load_metadata(folder):
    p = os.path.join(folder, "metadata.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


# ---------- main --------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-glob",
        default="/u/ar8564/projects/infinigen2/outputs/renderjobs/*_state.log",
    )
    ap.add_argument(
        "--finished-root", default="/n/fs/scratch/ar8564/renders_batch/2026-04-28"
    )
    ap.add_argument(
        "--renderjobs-dir", default="/u/ar8564/projects/infinigen2/outputs/renderjobs"
    )
    ap.add_argument("--per-batch", action="store_true")
    args = ap.parse_args()

    # 1) Enumerate batches and the metadata.json for each finished folder.
    batches = sorted(
        d
        for d in os.listdir(args.finished_root)
        if os.path.isdir(os.path.join(args.finished_root, d))
    )
    if not batches:
        print(f"no batches under {args.finished_root}", file=sys.stderr)
        sys.exit(1)

    finished_meta = {}  # (batch, idx) -> meta or None
    node_to_gpu = collections.Counter()  # observations from real metadata
    for batch in batches:
        for entry in os.listdir(os.path.join(args.finished_root, batch)):
            folder = os.path.join(args.finished_root, batch, entry)
            if not os.path.isdir(folder):
                continue
            idx = entry.rsplit("_", 1)[-1]
            meta = load_metadata(folder)
            finished_meta[(batch, idx)] = meta
            if meta is None:
                continue
            hw = meta.get("hardware", {})
            node = short_node(hw.get("hostname", ""))
            gpu = gpu_label(hw.get("gpus_all", []))
            if node:
                node_to_gpu[(node, gpu)] += 1

    # 2) Build node -> GPU map (observed first, then fall back to slurm).
    node_best_gpu = {}
    by_node = collections.defaultdict(list)
    for (node, gpu), n in node_to_gpu.items():
        by_node[node].append((n, gpu))
    for node, hits in by_node.items():
        hits.sort(reverse=True)
        node_best_gpu[node] = hits[0][1]
    slurm_filled = 0
    for node, gpu in fetch_slurm_node_gpus().items():
        if node not in node_best_gpu:
            node_best_gpu[node] = gpu
            slurm_filled += 1

    # 3) Index state.log by (batch, idx).
    state_lookup = {}
    for state_log in glob.glob(args.state_glob):
        for e in parse_state_log(state_log):
            state_lookup[(e["batch"], e["idx"])] = e

    # 4) Classify all err files (one grep per pattern, then dict lookup),
    #    plus extract slurm-recorded node from the err header.
    err_class = classify_err_files(args.renderjobs_dir)
    err_node = extract_err_nodes(args.renderjobs_dir)

    # 5) Walk every (batch, idx) attempted: enumerate via err files.
    by_gpu = collections.defaultdict(list)
    by_batch_gpu = collections.defaultdict(list)
    err_pattern = re.compile(r"^(?P<batch>.+)_(?P<idx>\d+)\.err$")

    seen_keys = set()
    for batch in batches:
        prefix = batch + "_"
        for fname in os.listdir(args.renderjobs_dir):
            if not fname.startswith(prefix) or not fname.endswith(".err"):
                continue
            m = err_pattern.match(fname)
            if not m or m.group("batch") != batch:
                continue
            idx = m.group("idx")
            key = (batch, idx)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            meta = finished_meta.get(key)
            state = state_lookup.get(key)

            # category
            if meta is not None:
                category = "ok"
            elif fname in err_class:
                category = err_class[fname]
            else:
                category = "other"

            # gpu — try metadata, then state.log node, then err-header node
            if meta is not None:
                gpu = gpu_label(meta.get("hardware", {}).get("gpus_all", []))
            elif state is not None:
                gpu = node_best_gpu.get(state["node"], f"unknown(node={state['node']})")
            elif fname in err_node:
                gpu = node_best_gpu.get(
                    err_node[fname], f"unknown(node={err_node[fname]})"
                )
            else:
                gpu = "unknown(no-node)"

            entry = {
                "batch": batch,
                "idx": idx,
                "category": category,
                "gpu": gpu,
                "in_state_log": state is not None,
                "runtime_s": (state["end"] - state["start"]).total_seconds()
                if state
                else None,
                "node": state["node"] if state else None,
                "folder_copied": key in finished_meta,
            }
            by_gpu[gpu].append(entry)
            by_batch_gpu[(batch, gpu)].append(entry)

    # 6) Summary helpers
    def fmt_secs(s):
        if s is None:
            return "-"
        h, rem = divmod(int(s), 3600)
        m, s = divmod(rem, 60)
        return f"{h:d}h{m:02d}m{s:02d}s"

    def summarize(entries):
        n = len(entries)
        cats = collections.Counter(e["category"] for e in entries)
        # vram-class crashes are job-side GPU faults; "other"/"noframe"/"traceback"
        # are unidentified or partial-output; preempt/timeout are slurm losses.
        crash = (
            cats["vram"]
            + cats["optix"]
            + cats["other"]
            + cats["noframe"]
            + cats["traceback"]
        )
        runtimes = [e["runtime_s"] for e in entries if e["runtime_s"] is not None]
        ok_runtimes = [
            e["runtime_s"]
            for e in entries
            if e["category"] == "ok" and e["runtime_s"] is not None
        ]
        return {
            "n": n,
            "cats": cats,
            "crash": crash,
            "ok_pct": 100 * cats["ok"] / n if n else 0,
            "preempt_pct": 100 * cats["preempt"] / n if n else 0,
            "timeout_pct": 100 * cats["timeout"] / n if n else 0,
            "vram_pct": 100 * cats["vram"] / n if n else 0,
            "crash_pct": 100 * crash / n if n else 0,
            "median_s": statistics.median(runtimes) if runtimes else None,
            "mean_ok_s": statistics.mean(ok_runtimes) if ok_runtimes else None,
            "total_s": sum(runtimes),
        }

    def print_table(rows, title):
        print(f"\n=== {title} ===")
        header = (
            f"{'GPU':30s}  {'jobs':>5s}  "
            f"{'ok':>5s}  {'preem':>5s}  {'tmout':>5s}  "
            f"{'vram':>4s}  {'noFrm':>5s}  {'tback':>5s}  "
            f"{'optix':>5s}  {'other':>5s}  "
            f"{'ok%':>6s}  {'preem%':>7s}  {'tmout%':>7s}  {'vram%':>6s}  "
            f"{'meanOK':>9s}  {'totalRT':>11s}"
        )
        print(header)
        print("-" * len(header))
        for label, s in rows:
            c = s["cats"]
            print(
                f"{label[:30]:30s}  {s['n']:5d}  "
                f"{c['ok']:5d}  {c['preempt']:5d}  {c['timeout']:5d}  "
                f"{c['vram']:4d}  {c['noframe']:5d}  {c['traceback']:5d}  "
                f"{c['optix']:5d}  {c['other']:5d}  "
                f"{s['ok_pct']:5.1f}%  {s['preempt_pct']:6.1f}%  {s['timeout_pct']:6.1f}%  "
                f"{s['vram_pct']:5.1f}%  "
                f"{fmt_secs(s['mean_ok_s']):>9s}  {fmt_secs(s['total_s']):>11s}"
            )

    # 7) Output
    print(f"finished_root   : {args.finished_root}")
    print(f"renderjobs_dir  : {args.renderjobs_dir}")
    print(f"batches         : {len(batches)}")
    print(f"err files seen  : {len(seen_keys)}")
    print(
        f"with metadata   : {sum(1 for v in finished_meta.values() if v is not None)}"
    )
    print(f"in state.log    : {sum(1 for k in seen_keys if k in state_lookup)}")
    print(f"slurm-filled GPU mappings for {slurm_filled} extra nodes")
    print(
        "categories: ok=metadata.json | preempt='DUE TO PREEMPTION' | "
        "timeout='TIME LIMIT' | vram='out of GPU and shared host memory' | "
        "optix='Failed to build OptiX' | noframe='No frames found' | "
        "tback=Traceback (none above) | other=none of the above"
    )

    overall = sorted(
        ((gpu, summarize(es)) for gpu, es in by_gpu.items()),
        key=lambda x: -x[1]["n"],
    )
    print_table(overall, "By GPU type (all batches under finished-root)")

    if args.per_batch:
        for batch in batches:
            rows = sorted(
                (
                    (g, summarize(es))
                    for (b, g), es in by_batch_gpu.items()
                    if b == batch
                ),
                key=lambda x: -x[1]["n"],
            )
            if rows:
                print_table(rows, f"Batch {batch}")


if __name__ == "__main__":
    main()
