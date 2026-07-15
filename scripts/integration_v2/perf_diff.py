# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Compare PR render metrics (Tris, CPU/GPU time) against a baseline run.

Pairs each PR render event with the baseline event of the same
(generator, variant_key), computes the fractional change of each gated metric,
and flags a regression when a metric grows by more than the threshold. Writes a
JSON report plus a markdown summary. Exits 2 if any unapproved regression
remains, 0 otherwise.

This is the perf analogue of pixel_diff.py and shares its approval design: a
regressing render is downgraded from `fail` to `approved` when it is no worse
(within the threshold) than the same render in a run a human already signed off
on. Approvals are an append-only JSONL store next to the render archive, keyed
by (asset, approved run).
"""

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

GATED_METRICS = ("tris", "cpu_time_sec", "gpu_time_sec")
THRESHOLD = 0.05
STORE_NAME = ".perf_approvals.jsonl"


def get_threshold() -> float:
    return float(os.environ.get("PERF_GATE_THRESHOLD", THRESHOLD))


def get_metrics() -> tuple[str, ...]:
    raw = os.environ.get("PERF_GATE_METRICS", "")
    if not raw.strip():
        return GATED_METRICS
    chosen = tuple(m.strip() for m in raw.split(",") if m.strip() in GATED_METRICS)
    return chosen or GATED_METRICS


def store_path(archive_root: Path) -> Path:
    return Path(archive_root) / STORE_NAME


def asset_key(name: str) -> str:
    # Share one key with the parent so a *_preset folds onto its generator's row.
    return name.removesuffix("_preset")


def load_approvals(archive_root: Path) -> list[dict]:
    path = store_path(archive_root)
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def approvals_by_asset(archive_root: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for rec in load_approvals(archive_root):
        grouped.setdefault(asset_key(rec.get("asset", "")), []).append(rec)
    return grouped


def append_approval(archive_root: Path, record: dict) -> None:
    with open(store_path(archive_root), "a") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def rel_to_archive(archive_root: Path, path: Path) -> str:
    path = Path(path).resolve()
    archive_root = Path(archive_root).resolve()
    if path.is_relative_to(archive_root):
        return path.relative_to(archive_root).as_posix()
    return path.as_posix()


def make_approval(
    archive_root: Path,
    asset: str,
    run_path: Path,
    baseline: str = "",
    pr: str = "",
    approver: str = "",
) -> dict:
    return {
        "asset": asset,
        "approved_run": rel_to_archive(archive_root, run_path),
        "baseline": baseline,
        "pr": pr,
        "approver": approver,
        "timestamp": time.time(),
    }


def _events(root: Path):
    for event_file in sorted((root / "render_index" / "events").glob("*.json")):
        yield json.loads(event_file.read_text())


@lru_cache(maxsize=32)
def _index(root_str: str) -> dict:
    idx = {}
    for event in _events(Path(root_str)):
        key = f"{event.get('generator', 'unknown')}\t{event.get('variant_key', 'unknown')}"
        idx[key] = {m: event.get(m) for m in GATED_METRICS}
    return idx


def _regressions(pr_vals: dict, base_vals: dict, metrics, threshold: float) -> dict:
    out = {}
    for metric in metrics:
        pr = pr_vals.get(metric)
        base = base_vals.get(metric)
        if pr is None or base is None or base <= 0:
            continue
        pct = (pr - base) / base
        if pct > threshold:
            out[metric] = {"pr": pr, "base": base, "pct": pct}
    return out


def _match_approval(
    pr_vals: dict, archive_root: Path, recs: list[dict], key: str, metrics, threshold
) -> str | None:
    for rec in recs:
        approved_root = Path(archive_root) / rec.get("approved_run", "")
        approved_vals = _index(str(approved_root.resolve())).get(key)
        if approved_vals is None:
            continue
        if not _regressions(pr_vals, approved_vals, metrics, threshold):
            return rec["approved_run"]
    return None


def _diff_one(
    event: dict,
    base_vals: dict | None,
    archive_root: Path | None,
    recs: list[dict],
    metrics,
    threshold: float,
) -> dict:
    pr_vals = {m: event.get(m) for m in GATED_METRICS}
    key = f"{event.get('generator', 'unknown')}\t{event.get('variant_key', 'unknown')}"
    regressions = {}
    approved_run = None
    if base_vals is None:
        status = "missing_baseline"
    else:
        regressions = _regressions(pr_vals, base_vals, metrics, threshold)
        if not regressions:
            status = "ok"
        elif archive_root is None:
            status = "fail"
        else:
            approved_run = _match_approval(
                pr_vals, archive_root, recs, key, metrics, threshold
            )
            status = "approved" if approved_run else "fail"
    return {
        "asset": event.get("generator", "unknown"),
        "variant": event.get("variant_key", "unknown"),
        "metrics": pr_vals,
        "regressions": regressions,
        "status": status,
        "approved_run": approved_run,
    }


def compare(
    pr_root: Path,
    base_root: Path,
    threshold: float = THRESHOLD,
    metrics=GATED_METRICS,
    archive_root: Path | None = None,
) -> dict:
    grouped = approvals_by_asset(archive_root) if archive_root else {}
    base_idx = _index(str(Path(base_root).resolve()))
    results = []
    for event in _events(pr_root):
        key = f"{event.get('generator', 'unknown')}\t{event.get('variant_key', 'unknown')}"
        recs = grouped.get(asset_key(event.get("generator", "")), [])
        result = _diff_one(
            event, base_idx.get(key), archive_root, recs, metrics, threshold
        )
        results.append(result)
    counts = {k: 0 for k in ("ok", "fail", "approved", "missing_baseline")}
    for r in results:
        counts[r["status"]] += 1
    return {
        "threshold": threshold,
        "metrics": list(metrics),
        "total": len(results),
        "fail_count": counts["fail"],
        "approved_count": counts["approved"],
        "missing_count": counts["missing_baseline"],
        "results": results,
    }


def asset_verdicts(report: dict) -> dict[str, str]:
    failed, approved, seen = set(), set(), set()
    for r in report["results"]:
        key = asset_key(r["asset"])
        seen.add(key)
        if r["status"] == "fail":
            failed.add(key)
        elif r["status"] == "approved":
            approved.add(key)
    return {
        k: "changed" if k in failed else "approved" if k in approved else "unchanged"
        for k in seen
    }


def _worst_summary(report: dict) -> dict[str, str]:
    """asset key -> compact "tris +20%, gpu +8%" of its worst regression per metric."""
    worst: dict[str, dict[str, float]] = {}
    for r in report["results"]:
        key = asset_key(r["asset"])
        for metric, info in r["regressions"].items():
            prev = worst.setdefault(key, {})
            prev[metric] = max(prev.get(metric, 0.0), info["pct"])
    labels = {"tris": "tris", "cpu_time_sec": "cpu", "gpu_time_sec": "gpu"}
    out = {}
    for key, metrics in worst.items():
        parts = [f"{labels[m]} +{pct * 100:.0f}%" for m, pct in metrics.items()]
        out[key] = ", ".join(parts)
    return out


@lru_cache(maxsize=32)
def _cached_report(pr: str, base: str, archive: str, threshold: float, mtime: float):
    return compare(Path(pr), Path(base), threshold, get_metrics(), Path(archive))


def annotate_rows(
    rows: list[dict], run_root: Path, base_root: Path, archive_root: Path, threshold
) -> None:
    store = store_path(archive_root)
    mtime = store.stat().st_mtime if store.exists() else 0.0
    report = _cached_report(
        str(run_root), str(base_root), str(archive_root), threshold, mtime
    )
    verdicts = asset_verdicts(report)
    summary = _worst_summary(report)
    for row in rows:
        verdict = verdicts.get(asset_key(row["asset"]), "unchanged")
        row["perf_regressed"] = verdict in ("changed", "approved")
        row["perf_approved"] = verdict == "approved"
        row["perf_summary"] = summary.get(asset_key(row["asset"]), "")


_LABELS = {"tris": "Tris", "cpu_time_sec": "CPU", "gpu_time_sec": "GPU"}


def write_markdown(report: dict, out: Path):
    metric_names = ", ".join(_LABELS.get(m, m) for m in report["metrics"])
    lines = [
        f"# Perf diff vs baseline ({metric_names}, threshold={report['threshold']:.0%})",
        "",
        f"- compared: {report['total']}",
        f"- regressed: **{report['fail_count']}**",
        f"- approved regressions: {report['approved_count']}",
        f"- missing baseline: {report['missing_count']}",
        "",
    ]
    fails = [r for r in report["results"] if r["status"] == "fail"]
    if fails:
        lines += ["| asset | variant | regression |", "|---|---|---|"]
        for r in _sorted_fails(fails)[:50]:
            lines.append(f"| {r['asset']} | {r['variant']} | {_fmt_regs(r)} |")
        if len(fails) > 50:
            lines.append(f"\n_…{len(fails) - 50} more omitted_")
    out.write_text("\n".join(lines) + "\n")


def _fmt_regs(result: dict) -> str:
    parts = []
    for metric, info in result["regressions"].items():
        parts.append(
            f"{_LABELS.get(metric, metric)} {info['base']:g}→{info['pr']:g} (+{info['pct'] * 100:.0f}%)"
        )
    return "; ".join(parts)


def _sorted_fails(fails: list[dict]) -> list[dict]:
    def worst(result: dict) -> float:
        return max((i["pct"] for i in result["regressions"].values()), default=0.0)

    return sorted(fails, key=worst, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument(
        "--approvals-root",
        type=Path,
        default=None,
        help="Archive root holding the approval store; enables approved-regression downgrade.",
    )
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"baseline {args.baseline} missing; skipping perf diff", file=sys.stderr)
        args.report.write_text(json.dumps({"skipped": True}))
        args.summary.write_text("# Perf diff skipped\n\nNo baseline available.\n")
        return 0

    threshold = get_threshold()
    report = compare(
        args.pr, args.baseline, threshold, get_metrics(), args.approvals_root
    )
    args.report.write_text(json.dumps(report, indent=2))
    write_markdown(report, args.summary)
    print(
        f"perf_diff: {report['fail_count']}/{report['total']} regressed "
        f"> {threshold:.0%} ({report['approved_count']} approved)"
    )
    return 2 if report["fail_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
