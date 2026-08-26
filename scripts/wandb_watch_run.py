# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import wandb

SLURM_STATES = [
    "PENDING",
    "REQUEUED",
    "RUNNING",
    "SUSPENDED",
    "COMPLETING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "PREEMPTED",
    "NODE_FAIL",
]


def get_cluster_state(jobnamestr: str) -> dict[str, int]:
    if shutil.which("squeue") is None:
        return {}
    result = subprocess.run(
        ["squeue", "--format=%j %T", "--noheader"],
        capture_output=True,
        text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        name, state = parts
        if fnmatch.fnmatch(name, f"*{jobnamestr}*"):
            name = "state/" + state.lower()
            states[name] = states.get(name, 0) + 1
    return states


SLURM_REASONS = [
    ("DUE TO PREEMPTION", "preempted"),
    ("DUE TO TIME LIMIT", "time_limit"),
    ("DUE TO NODE FAILURE", "node_fail"),
    ("oom_kill event", "oom"),
]

DISK_PATTERNS = (
    "No space left on device",
    "Disk quota exceeded",
)

GPU_FAULT_PATTERNS = (
    "Uncorrectable ECC",
    "Illegal address in CUDA",
    "Launch failed in CUDA",
    "Failed to retain CUDA context",
    "Failed to create CUDA context",
    "Failed to load OptiX kernel",
    "CUDA error",
    "NVIDIA-SMI has failed",
)

GPU_OOM_PATTERNS = (
    "out of GPU",
    "System is out of GPU memory",
)


# metadata['hardware']['gpus_all'] reports nvidia-smi's verbose name; collapse
# to a short slug so the wandb section is readable at a glance.
GPU_NAME_TO_SLUG = {
    "NVIDIA GeForce RTX 2080 Ti": "rtx_2080",
    "NVIDIA GeForce RTX 3090": "rtx_3090",
    "NVIDIA RTX A5000": "rtx_a5000",
    "NVIDIA RTX A6000": "rtx_a6000",
    "NVIDIA RTX 6000 Ada Generation": "rtx_6000",
    "NVIDIA A40": "a40",
    "NVIDIA L40": "l40",
    "NVIDIA L40S": "l40s",
    "NVIDIA GeForce GTX 1080 Ti": "gtx_1080",
    "NVIDIA A100-SXM4-80GB": "a100",
    "NVIDIA H100 80GB HBM3": "h100",
}


def gpu_slug(gpu_name: str) -> str:
    if gpu_name in GPU_NAME_TO_SLUG:
        return GPU_NAME_TO_SLUG[gpu_name]
    s = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_")
    return s or "unknown"


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_SLUG_STOPWORDS = {"a", "an", "and", "for", "in", "is", "of", "the", "to", "which"}


def _message_slug(msg: str, n_words: int = 5) -> str:
    """Normalize a log message into a short deduped category slug: quoted strings,
    paths and numbers are stripped so all instances of one message template
    collapse to the same slug."""
    msg = re.sub(r"'[^']*'|\"[^\"]*\"", " ", msg)
    msg = re.sub(r"\S*/\S*", " ", msg)
    words = re.findall(r"[A-Za-z][A-Za-z_]+", msg.lower())
    deduped = []
    for w in words:
        if len(w) >= 2 and w not in deduped:
            deduped.append(w)
    deduped = deduped[:n_words]
    while deduped and deduped[-1] in _SLUG_STOPWORDS:
        deduped.pop()
    return "_".join(deduped)


EXC_LINE_RE = re.compile(
    r"^([A-Za-z_][\w.]*?(?:Error|Exception|Interrupt|Exit))(?::\s*(.*))?$"
)

RAISED_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*\.)?([A-Z]\w+)(?::\s*(.*))?$")


def _classify_message(msg: str) -> str | None:
    if any(p in msg for p in DISK_PATTERNS):
        return "disk_full"
    if any(p in msg for p in GPU_OOM_PATTERNS):
        return "gpu_oom"
    if any(p in msg for p in GPU_FAULT_PATTERNS) or re.search(r"\bXid\b", msg):
        return "gpu_fault"
    if "out of memory" in msg.lower():
        return "oom"
    return None


def _find_exception_line(last_tb: str) -> tuple[str, str] | None:
    """The `SomeError: message` line terminating a traceback: the first
    column-0 line after the header that names an exception class, also catching
    custom classes not named *Error/*Exception (e.g. RejectedScene) by requiring
    the preceding frame line to be indented."""
    lines = last_tb.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(" "):
            continue
        m = EXC_LINE_RE.match(line)
        if m:
            return m.group(1), m.group(2) or ""
        prev_indented = i > 0 and lines[i - 1].startswith(" ")
        m = RAISED_LINE_RE.match(line)
        if m and prev_indented:
            return (m.group(1) or "") + m.group(2), m.group(3) or ""
    return None


def _classify_traceback(last_tb: str) -> str:
    if not last_tb.strip():
        return "truncated_traceback"
    found = _find_exception_line(last_tb)
    if found is None:
        return _classify_message(last_tb) or "unknown_traceback"
    exc_cls, exc_msg = found
    special = _classify_message(exc_msg)
    if special is not None:
        return special
    cls_snake = _camel_to_snake(exc_cls.rsplit(".", 1)[-1])
    slug = _message_slug(re.sub(r"^Error:\s*", "", exc_msg))
    return f"{cls_snake}.{slug}" if slug else cls_snake


def get_error_type(err_lines: list[str]) -> str | None:
    if not err_lines:
        return None
    text = "\n".join(err_lines)
    tail = "\n".join(err_lines[-50:])

    for pattern, reason in SLURM_REASONS:
        if pattern in tail:
            return reason

    if "Traceback (most recent call last):" in text:
        last_tb = text.rsplit("Traceback (most recent call last):", 1)[1]
        return _classify_traceback(last_tb)

    special = _classify_message(tail)
    if special is not None:
        return special
    if "Segmentation fault" in tail or "core dumped" in tail:
        return "segfault"
    if "CANCELLED" in tail:
        return "cancelled"

    return None


WARNING_LINE_RE = re.compile(r"(?:\[([\w.]+)\]\s*)?\[WARNING\]\s*\|?\s*(.*)")


def count_warnings(err_lines: list[str]) -> dict[str, float]:
    """Count WARNING log lines into warn/<category> counts. Category is the
    logger name (when the format includes one) plus a slug of the message
    template, so repeated warnings dedupe into one readable wandb chart each."""
    counts: dict[str, float] = {}
    for line in err_lines:
        m = WARNING_LINE_RE.search(line)
        if m is None:
            continue
        logger_name, msg = m.groups()
        slug = _message_slug(msg)
        if logger_name and slug.startswith(logger_name):
            slug = slug[len(logger_name) :].strip("_")
        category = f"{logger_name}.{slug}" if logger_name else slug
        key = f"warn/{category or 'unlabeled'}"
        counts[key] = counts.get(key, 0) + 1
        counts["warn/total"] = counts.get("warn/total", 0) + 1
    return counts


def get_disk_used(path: Path) -> float | None:
    result = subprocess.run(
        ["df", "--output=pcent", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return int(result.stdout.splitlines()[1].strip().rstrip("%")) / 100.0
    except (ValueError, IndexError):
        return None


def jobid_from_errfile(errfile: str) -> str | None:
    # errfile stem is "<jobname>_<arrayjobid>_<arrayjobid>_<taskid>"
    parts = Path(errfile).stem.rsplit("_", 2)
    if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2].isdigit():
        return None
    return f"{parts[-2]}_{parts[-1]}"


# set to False once we observe jobstats is missing on the host
_jobstats_available = True


def get_jobstats(jobid: str) -> dict[str, float] | None:
    global _jobstats_available
    if not _jobstats_available:
        return None
    try:
        result = subprocess.run(
            ["jobstats", "-j", "-n", jobid],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        _jobstats_available = False
        return None
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    runtime = data.get("total_time") or 0
    nodes = data.get("nodes", {})
    cpu_sec_used = 0.0
    cpu_core_seconds = 0.0
    cpu_mem_used = 0
    cpu_mem_total = 0
    gpu_utils: list[float] = []
    gpu_mem_used = 0
    gpu_mem_total = 0
    for node in nodes.values():
        cpus = node.get("cpus") or 0
        cpu_sec_used += node.get("total_time") or 0.0
        cpu_core_seconds += cpus * runtime
        cpu_mem_used += node.get("used_memory") or 0
        cpu_mem_total += node.get("total_memory") or 0
        gpu_utils.extend((node.get("gpu_utilization") or {}).values())
        gpu_mem_used += sum((node.get("gpu_used_memory") or {}).values())
        gpu_mem_total += sum((node.get("gpu_total_memory") or {}).values())

    metrics: dict[str, float] = {}
    if runtime > 0:
        metrics["jobstats/total_time"] = runtime
    if cpu_core_seconds > 0:
        metrics["jobstats/cpu_util_pct"] = cpu_sec_used / cpu_core_seconds * 100.0
    if cpu_mem_total > 0:
        metrics["jobstats/cpu_mem_used_frac"] = cpu_mem_used / cpu_mem_total
    if gpu_utils:
        metrics["jobstats/gpu_util_pct"] = sum(gpu_utils) / len(gpu_utils)
    if gpu_mem_total > 0:
        metrics["jobstats/gpu_mem_used_frac"] = gpu_mem_used / gpu_mem_total
    return metrics or None


TERMINAL_STATUSES = (
    "preempted",
    "time_limit",
    "node_fail",
    "cancelled",
    "killed",
    "pack_failed",
)


def enrich_completed(
    metadata: dict | None,
    path: str,
    stats: dict[str, float],
    accumulate: dict[str, float],
) -> None:
    accumulate["jobs/completed"] = 1
    if metadata is None:
        return
    for gen, t in metadata.get("generator_times", {}).items():
        stats[f"gentime/{gen}"] = t
    disk_used = get_disk_used(Path(path))
    if disk_used is not None:
        stats["disk_used"] = disk_used
    imgs = metadata.get("exports", {}).get("ExportType.IMAGE", [])
    accumulate["frames_completed"] = len(imgs)


def info_for_event(event: str) -> tuple[dict, dict, dict]:
    fields = event.split()
    start_str, end_str, _node, _gpu, path, errfile = fields[:6]
    status = fields[7] if len(fields) > 7 else None

    start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%S%z")
    end = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%S%z")

    duration = (end - start).total_seconds()
    hours = duration / 3600.0
    errfile_exists = Path(errfile).exists()
    err_lines = Path(errfile).read_text().splitlines() if errfile_exists else []
    error_type = get_error_type(err_lines)

    stats = {"duration": duration}
    accumulate: dict[str, float] = {"gpu_hrs/total": hours}
    for k, v in count_warnings(err_lines).items():
        accumulate[k] = accumulate.get(k, 0) + v

    # only left shards write metadata, so right-shard tasks report without it
    metadata_path = Path(path) / "metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = None

    if metadata is not None:
        gpus_all = metadata.get("hardware", {}).get("gpus_all", [])
        if gpus_all:
            gpu_name = gpus_all[0].split(",")[0].strip()
            accumulate[f"gpu_hrs/{gpu_slug(gpu_name)}"] = hours

    per_event: dict[str, float] = {}
    jobid = jobid_from_errfile(errfile)
    if jobid is not None:
        js = get_jobstats(jobid)
        if js is not None:
            per_event.update(js)

    # Trust the status the job wrote; errfile guessing misreads unreachable outputs
    if status == "preempted" or error_type == "preempted":
        accumulate["jobs/preempted"] = 1
        return stats, accumulate, per_event

    if status == "completed":
        enrich_completed(metadata, path, stats, accumulate)
        return stats, accumulate, per_event

    if status in TERMINAL_STATUSES:
        accumulate["jobs/crashed"] = 1
        accumulate[f"crashreason/{status}"] = 1
        return stats, accumulate, per_event

    if error_type is not None or metadata is None:
        if error_type is not None:
            reason = error_type
        elif not errfile_exists:
            reason = "no_errfile"
        else:
            reason = "no_metadata"
        accumulate["jobs/crashed"] = 1
        accumulate[f"crashreason/{reason}"] = 1
        return stats, accumulate, per_event

    # only a status-less legacy line may reach the success path unrecognized
    if status is not None:
        accumulate["jobs/crashed"] = 1
        accumulate[f"crashreason/unknown_{status}"] = 1
        return stats, accumulate, per_event

    enrich_completed(metadata, path, stats, accumulate)
    return stats, accumulate, per_event


def check_crash_alert(
    recent_outcomes: deque,
    rate_thresh: float,
    run_name: str,
) -> bool:
    if len(recent_outcomes) < recent_outcomes.maxlen:
        return False
    n_crashed = sum(recent_outcomes)
    rate = n_crashed / len(recent_outcomes)
    if rate < rate_thresh:
        return False
    wandb.alert(
        title=f"{run_name} render crash rate high",
        text=(
            f"{n_crashed}/{len(recent_outcomes)} of the last {len(recent_outcomes)} "
            f"jobs crashed (rate {rate:.0%} >= threshold {rate_thresh:.0%})"
        ),
        level=wandb.AlertLevel.WARN,
        wait_duration=timedelta(hours=1),
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobnamestr", type=str)
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--poll", type=int, default=5)
    parser.add_argument("--wandb_project", type=str, default="infinigen2")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="wandb run name; defaults to jobnamestr. Set it per subset when running several watchers",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="Resume an existing wandb run by id so new metrics append to the same chart",
    )
    parser.add_argument(
        "--skip_history",
        action="store_true",
        help="Ignore state.log lines that already exist at startup; only log events appended after launch",
    )
    parser.add_argument(
        "--alert_crash_rate",
        type=float,
        default=0.5,
        help="Send a wandb alert when this fraction of the last --alert_window jobs crashed",
    )
    parser.add_argument("--alert_window", type=int, default=12)
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.name or args.jobnamestr.replace("*", ""),
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
    )

    file_seen_lines: dict[Path, int] = {}
    if args.skip_history:
        for f in args.logdir.glob(f"*{args.jobnamestr}*_state.log"):
            file_seen_lines[f] = len(f.read_text().splitlines())

    recent_outcomes: deque = deque(maxlen=args.alert_window)
    accumulated: dict[str, float] = {
        "jobs/completed": 0,
        "jobs/crashed": 0,
        "jobs/preempted": 0,
        "frames_completed": 0,
        "gpu_hrs/total": 0.0,
    }

    while True:
        state_files = list(args.logdir.glob(f"*{args.jobnamestr}*_state.log"))

        new_events = []
        for f in state_files:
            lines = f.read_text().splitlines()
            new_lines = lines[file_seen_lines.get(f, 0) :]
            file_seen_lines[f] = len(lines)
            new_events.extend(new_lines)

        cluster_state = get_cluster_state(args.jobnamestr)

        all_event_stats: dict[str, list[float]] = {}
        for event in new_events:
            try:
                stats, accumulate, per_event = info_for_event(event)
                for k, v in accumulate.items():
                    accumulated[k] = accumulated.get(k, 0.0) + v
                if accumulate.get("jobs/crashed"):
                    recent_outcomes.append(1)
                elif accumulate.get("jobs/completed"):
                    recent_outcomes.append(0)
                for k, v in stats.items():
                    if isinstance(v, (int, float)):
                        all_event_stats.setdefault(k, []).append(v)
                if per_event:
                    wandb.log(per_event)
                print(f"  event logged: {event[:60]}")
            except Exception as e:
                print("WARNING", e, f"event={event[:60]} is malformed")

        log_data = {**accumulated, **cluster_state}
        for k, vals in all_event_stats.items():
            log_data[k] = sum(vals) / len(vals)

        wandb.log(log_data)

        if new_events and check_crash_alert(
            recent_outcomes, args.alert_crash_rate, args.jobnamestr
        ):
            print(f"  ALERT sent: crash rate over last {len(recent_outcomes)} jobs")

        print(
            f"  completed={accumulated.get('jobs/completed', 0)}"
            f"  frames={accumulated.get('frames_completed', 0)}"
            f"  preempted={accumulated.get('jobs/preempted', 0)}"
            f"  crashed={accumulated.get('jobs/crashed', 0)}"
            f"  cluster={cluster_state}"
        )

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
