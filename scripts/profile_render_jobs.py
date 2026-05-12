#!/usr/bin/env python3
"""Profile recent infinigen2 render jobs: GPU peak memory, runtime, status.

Builds on audit_render_jobs (classification + GPU mapping) and adds:
  * peak GPU memory parsed from .out (Cycles `Peak:NNN.NNM` lines)
  * peak system memory (`Peak NNN.NNM`, no colon)
  * per-phase generator_times from metadata.json
  * frames saved (max frame index across `Saved: '.../NNNN.png'` lines)

Output is a CSV; load it with pandas:

    df = pd.read_csv("profile.csv")
    df.groupby("gpu")["gpu_peak_mb"].describe()
    df[df.category=="ok"].groupby(["gpu","node"]).agg({"gpu_peak_mb":"max","t_total_s":"median"})

Run on soak. Defaults pick up both pvl-renders and /n/fs/scratch render trees.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_render_jobs as audit

GPU_MEM_RE = re.compile(r"Peak:([0-9.]+)M")
SYS_MEM_RE = re.compile(r"Peak ([0-9.]+)M")
SAVED_FRAME_RE = re.compile(r"Saved: '.*?(\d{4})\.png'")


def parse_out(path):
    """Return (gpu_peak_mb, sys_peak_mb, max_frame_idx) from a .out file."""
    try:
        out = subprocess.check_output(
            ["grep", "-oE", r"Peak:[0-9.]+M|Peak [0-9.]+M|Saved: '[^']+\.png'", path],
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return None, None, None
        out = e.output or ""
    gpu = max((float(m.group(1)) for m in GPU_MEM_RE.finditer(out)), default=None)
    sysm = max((float(m.group(1)) for m in SYS_MEM_RE.finditer(out)), default=None)
    frames = [int(m.group(1)) for m in SAVED_FRAME_RE.finditer(out)]
    return gpu, sysm, max(frames) + 1 if frames else 0


def collect_finished(roots):
    """Walk multiple finished-roots; return (batch, idx) -> (folder, meta)."""
    by_key = {}
    node_to_gpu = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for batch in os.listdir(root):
            bdir = os.path.join(root, batch)
            if not os.path.isdir(bdir):
                continue
            for entry in os.listdir(bdir):
                folder = os.path.join(bdir, entry)
                if not os.path.isdir(folder):
                    continue
                idx = entry.rsplit("_", 1)[-1]
                meta = audit.load_metadata(folder)
                by_key[(batch, idx)] = (folder, meta)
                if meta is not None:
                    hw = meta.get("hardware", {})
                    node = audit.short_node(hw.get("hostname", ""))
                    gpu = audit.gpu_label(hw.get("gpus_all", []))
                    if node:
                        node_to_gpu.setdefault(node, gpu)
    return by_key, node_to_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-glob",
        default="/u/ar8564/projects/infinigen2/outputs/renderjobs/*_state.log",
    )
    ap.add_argument(
        "--finished-roots",
        nargs="+",
        default=[
            "/n/fs/pvl-renders/ar8564/renders",
            "/n/fs/scratch/ar8564/renders",
        ],
    )
    ap.add_argument(
        "--renderjobs-dir", default="/u/ar8564/projects/infinigen2/outputs/renderjobs"
    )
    ap.add_argument(
        "--batch-prefix",
        default=None,
        help="only include batches starting with this prefix (e.g. 2026-05-02)",
    )
    ap.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="cap number of err files processed (most-recent first)",
    )
    ap.add_argument("--out", default="-", help="output CSV path (default stdout)")
    args = ap.parse_args()

    finished, observed_node_gpu = collect_finished(args.finished_roots)

    state_lookup = {}
    for state_log in glob.glob(args.state_glob):
        for e in audit.parse_state_log(state_log):
            state_lookup[(e["batch"], e["idx"])] = e

    err_class = audit.classify_err_files(args.renderjobs_dir)
    err_node = audit.extract_err_nodes(args.renderjobs_dir)

    node_best_gpu = dict(observed_node_gpu)
    for node, gpu in audit.fetch_slurm_node_gpus().items():
        node_best_gpu.setdefault(node, gpu)

    err_pattern = re.compile(r"^(?P<batch>.+)_(?P<idx>\d+)\.err$")
    err_files = []
    for fname in os.listdir(args.renderjobs_dir):
        if not fname.endswith(".err"):
            continue
        m = err_pattern.match(fname)
        if not m:
            continue
        if args.batch_prefix and not m.group("batch").startswith(args.batch_prefix):
            continue
        path = os.path.join(args.renderjobs_dir, fname)
        err_files.append(
            (os.path.getmtime(path), fname, m.group("batch"), m.group("idx"))
        )

    err_files.sort(reverse=True)  # newest first
    if args.max_jobs:
        err_files = err_files[: args.max_jobs]

    rows = []
    for _mtime, fname, batch, idx in err_files:
        key = (batch, idx)
        folder, meta = finished.get(key, (None, None))
        state = state_lookup.get(key)

        if meta is not None:
            category = "ok"
        elif fname in err_class:
            category = err_class[fname]
        else:
            category = "other"

        if meta is not None:
            gpu = audit.gpu_label(meta.get("hardware", {}).get("gpus_all", []))
            node = (
                audit.short_node(meta.get("hardware", {}).get("hostname", "")) or None
            )
        elif state is not None:
            gpu = node_best_gpu.get(state["node"], f"unknown(node={state['node']})")
            node = state["node"]
        elif fname in err_node:
            gpu = node_best_gpu.get(err_node[fname], f"unknown(node={err_node[fname]})")
            node = err_node[fname]
        else:
            gpu, node = "unknown(no-node)", None

        out_path = os.path.join(args.renderjobs_dir, fname[:-4] + ".out")
        gpu_peak, sys_peak, frames_saved = parse_out(out_path)

        gen_times = (meta or {}).get("generator_times", {}) if meta else {}
        t_total = sum(gen_times.values()) if gen_times else None

        runtime_s = (state["end"] - state["start"]).total_seconds() if state else None

        rows.append(
            {
                "batch": batch,
                "idx": idx,
                "category": category,
                "gpu": gpu,
                "node": node,
                "runtime_s": runtime_s,
                "gpu_peak_mb": gpu_peak,
                "sys_peak_mb": sys_peak,
                "frames_saved": frames_saved,
                "t_total_s": t_total,
                "t_left_rgb_s": gen_times.get("render_left_rgb"),
                "t_right_rgb_s": gen_times.get("render_right_rgb"),
                "t_left_gt_s": gen_times.get("render_left_gt"),
                "t_right_gt_s": gen_times.get("render_right_gt"),
                "in_state_log": state is not None,
                "folder": folder,
                "err_path": os.path.join(args.renderjobs_dir, fname),
            }
        )

    fields = list(rows[0].keys()) if rows else []
    out_fh = sys.stdout if args.out == "-" else open(args.out, "w", newline="")
    w = csv.DictWriter(out_fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if args.out != "-":
        out_fh.close()
        print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
