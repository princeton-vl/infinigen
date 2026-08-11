# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Compare PR render outputs against a baseline run.

Two subcommands share a common walk over `render_index/events/*.json`, report
envelope, and markdown summary:

- `pixel`: pairs still images by relative path, computes mean squared error in
  [0,1], and flags any pair with MSE > EPS. Missing images (one side has it, the
  other doesn't) are reported but do not fail — crashes are handled by the
  render job's own exit code.
- `perf`: groups render events by generator, takes the median of each gated
  metric across an asset's seeds/variants on each side, and flags a regression
  when the PR's median grows past the baseline's median by more than the
  threshold. The median absorbs a single noisy sample (CI-runner contention,
  GC, cold caches) that would otherwise let one outlier seed stand in for the
  whole asset. Only `tris` is gated by default; the timings are wall-clock under
  GPU contention and are reported but not gated.

Both write a JSON report plus a markdown summary and exit 2 if anything failed,
0 otherwise. The workflow runs both with continue-on-error, so a non-zero exit
surfaces in the PR comment but does not fail the job.
"""

import argparse
import json
import os
import statistics
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def _events(root: Path):
    for event_file in sorted((root / "render_index" / "events").glob("*.json")):
        yield json.loads(event_file.read_text())


def _fail_table(fails: list[dict], header: list[str], row_fn) -> list[str]:
    lines = list(header)
    for r in fails[:50]:
        lines.append(row_fn(r))
    if len(fails) > 50:
        lines.append(f"\n_…{len(fails) - 50} more omitted_")
    return lines


def _has_events(root: Path) -> bool:
    return any(True for _ in (root / "render_index" / "events").glob("*.json"))


def _diff_main(args, compute, write_markdown, skip_msg, skip_summary, done_line) -> int:
    # an existing but eventless baseline would otherwise report 0/0 and read as green
    if not args.baseline.exists() or not _has_events(args.baseline):
        print(skip_msg, file=sys.stderr)
        args.report.write_text(json.dumps({"skipped": True}))
        args.summary.write_text(skip_summary)
        return 0

    report = compute()
    args.report.write_text(json.dumps(report, indent=2))
    write_markdown(report, args.summary)
    print(done_line(report))
    return 2 if report["fail_count"] > 0 else 0


EPS = 1e-3

INFORMATIONAL = "_Informational only — this does not fail the job._"

# The render index also lists videos, which the viewer plays but PIL cannot open.
STILL_SUFFIXES = (".png",)


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# shape_mismatch keeps json valid; float("inf") would serialize as bare `Infinity`
def _mse(pr_root: Path, base_root: Path, rel: str) -> tuple[float | None, str]:
    pr_path = pr_root / rel
    base_path = base_root / rel
    if not pr_path.exists() or not base_path.exists():
        return None, "missing_baseline"
    a = _load(pr_path)
    b = _load(base_path)
    if a.shape != b.shape:
        return None, "shape_mismatch"
    mse = float(np.mean((a - b) ** 2))
    return mse, ("fail" if mse > EPS else "ok")


def _pixel_result(event: dict, pr_root: Path, base_root: Path, rel: str) -> dict:
    mse, status = _mse(pr_root, base_root, rel)
    return {
        "asset": event.get("generator", "unknown"),
        "variant": event.get("variant_key", "unknown"),
        "image": rel,
        "mse": mse,
        "status": status,
    }


def _still_images(event: dict) -> list[str]:
    return [
        rel for rel in event.get("images", []) if rel.lower().endswith(STILL_SUFFIXES)
    ]


def compare_pixel(pr_root: Path, base_root: Path) -> dict:
    results = []
    for event in _events(pr_root):
        for rel in _still_images(event):
            results.append(_pixel_result(event, pr_root, base_root, rel))
    fails = [r for r in results if r["status"] in ("fail", "shape_mismatch")]
    missing = [r for r in results if r["status"] == "missing_baseline"]
    return {
        "eps": EPS,
        "total": len(results),
        "fail_count": len(fails),
        "missing_count": len(missing),
        "results": results,
    }


def _fmt_mse(result: dict) -> str:
    if result["mse"] is None:
        return "shape changed"
    return f"{result['mse']:.4g}"


def write_pixel_markdown(report: dict, out: Path):
    lines = [
        f"# Pixel diff vs baseline (MSE, eps={report['eps']:.0e})",
        "",
        INFORMATIONAL,
        "",
        f"- compared: {report['total']}",
        f"- exceed eps: **{report['fail_count']}**",
        f"- missing baseline: {report['missing_count']}",
        "",
    ]
    fails = [r for r in report["results"] if r["status"] in ("fail", "shape_mismatch")]
    if fails:
        fails = sorted(
            fails, key=lambda r: -(r["mse"] if r["mse"] is not None else 1e9)
        )
        lines += _fail_table(
            fails,
            ["| asset | variant | image | MSE |", "|---|---|---|---|"],
            lambda r: f"| {r['asset']} | {r['variant']} | {r['image']} | {_fmt_mse(r)} |",
        )
    out.write_text("\n".join(lines) + "\n")


def _pixel_main(args) -> int:
    return _diff_main(
        args,
        lambda: compare_pixel(args.pr, args.baseline),
        write_pixel_markdown,
        f"baseline {args.baseline} missing; skipping diff",
        "# Pixel diff skipped\n\nNo baseline available.\n",
        lambda r: f"pixel_diff: {r['fail_count']}/{r['total']} exceed MSE eps={r['eps']:.0e}",
    )


METRICS = ("tris", "cpu_time_sec", "gpu_time_sec")

# times are contended wall-clock, far too noisy to gate; only tris is deterministic
GATED_METRICS = ("tris",)
THRESHOLD = 0.05


def get_threshold() -> float:
    return float(os.environ.get("PERF_GATE_THRESHOLD", THRESHOLD))


def get_metrics() -> tuple[str, ...]:
    raw = os.environ.get("PERF_GATE_METRICS", "")
    if not raw.strip():
        return GATED_METRICS
    chosen = tuple(m.strip() for m in raw.split(",") if m.strip() in METRICS)
    return chosen or GATED_METRICS


def _record_event(samples: dict[str, dict[str, list[float]]], event: dict) -> None:
    bucket = samples.setdefault(event.get("generator", "unknown"), {})
    for metric in METRICS:
        value = event.get(metric)
        if value is None:
            continue
        bucket.setdefault(metric, []).append(value)


def _asset_metric_samples(root: Path) -> dict[str, dict[str, list[float]]]:
    """asset key -> metric -> every sample seen across its seeds/variants/presets."""
    samples: dict[str, dict[str, list[float]]] = {}
    for event in _events(root):
        _record_event(samples, event)
    return samples


def _regressions(
    pr_samples: dict[str, list[float]],
    base_samples: dict[str, list[float]],
    metrics,
    threshold: float,
) -> dict:
    out = {}
    for metric in metrics:
        pr_vals = pr_samples.get(metric)
        base_vals = base_samples.get(metric)
        if not pr_vals or not base_vals:
            continue
        pr = statistics.median(pr_vals)
        base = statistics.median(base_vals)
        if base <= 0:
            continue
        pct = (pr - base) / base
        if pct > threshold:
            out[metric] = {"pr": pr, "base": base, "pct": pct}
    return out


def _asset_result(
    key: str,
    pr_vals: dict[str, list[float]],
    base_vals: dict[str, list[float]] | None,
    metrics: tuple[str, ...],
    threshold: float,
) -> dict:
    if base_vals is None:
        return {
            "asset": key,
            "metrics": {},
            "regressions": {},
            "status": "missing_baseline",
        }
    regressions = _regressions(pr_vals, base_vals, metrics, threshold)
    median_metrics = {m: statistics.median(v) for m, v in pr_vals.items()}
    status = "fail" if regressions else "ok"
    return {
        "asset": key,
        "metrics": median_metrics,
        "regressions": regressions,
        "status": status,
    }


def compare(
    pr_root: Path, base_root: Path, threshold: float = THRESHOLD, metrics=GATED_METRICS
) -> dict:
    pr_samples = _asset_metric_samples(pr_root)
    base_samples = _asset_metric_samples(base_root)
    results = [
        _asset_result(key, pr_samples[key], base_samples.get(key), metrics, threshold)
        for key in sorted(pr_samples)
    ]
    counts = {k: 0 for k in ("ok", "fail", "missing_baseline")}
    for r in results:
        counts[r["status"]] += 1
    return {
        "threshold": threshold,
        "metrics": list(metrics),
        "total": len(results),
        "fail_count": counts["fail"],
        "missing_count": counts["missing_baseline"],
        "results": results,
    }


def asset_verdicts(report: dict) -> dict[str, str]:
    return {
        r["asset"]: "changed" if r["status"] == "fail" else "unchanged"
        for r in report["results"]
    }


def _summary(report: dict) -> dict[str, str]:
    """asset key -> compact "tris +20%, gpu +8%" of its regression, if any."""
    labels = {"tris": "tris", "cpu_time_sec": "cpu", "gpu_time_sec": "gpu"}
    out = {}
    for r in report["results"]:
        if not r["regressions"]:
            continue
        parts = [
            f"{labels[m]} +{info['pct'] * 100:.0f}%"
            for m, info in r["regressions"].items()
        ]
        out[r["asset"]] = ", ".join(parts)
    return out


@lru_cache(maxsize=32)
def _cached_report(pr: str, base: str, threshold: float, metrics: tuple[str, ...]):
    return compare(Path(pr), Path(base), threshold, metrics)


def annotate_rows(
    rows: list[dict], run_root: Path, base_root: Path, threshold: float
) -> None:
    report = _cached_report(str(run_root), str(base_root), threshold, get_metrics())
    verdicts = asset_verdicts(report)
    summary = _summary(report)
    for row in rows:
        row["perf_regressed"] = verdicts.get(row["asset"]) == "changed"
        row["perf_summary"] = summary.get(row["asset"], "")


# not CPU/GPU: gpu_time_sec is exporter wall-clock, cpu_time_sec is the rest of the process
_LABELS = {"tris": "Tris", "cpu_time_sec": "Build", "gpu_time_sec": "Export"}


def _fmt_duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def _fmt_metric_value(metric: str, value: float) -> str:
    if metric in ("cpu_time_sec", "gpu_time_sec"):
        return _fmt_duration(value)
    return f"{value:g}"


def write_perf_markdown(report: dict, out: Path):
    metric_names = ", ".join(_LABELS.get(m, m) for m in report["metrics"])
    lines = [
        f"# Perf diff vs baseline ({metric_names}, threshold={report['threshold']:.0%})",
        "",
        INFORMATIONAL,
        "",
        f"- compared: {report['total']}",
        f"- regressed: **{report['fail_count']}**",
        f"- missing baseline: {report['missing_count']}",
        "",
    ]
    fails = [r for r in report["results"] if r["status"] == "fail"]
    if fails:
        lines += _fail_table(
            _sorted_fails(fails),
            ["| asset | regression |", "|---|---|"],
            lambda r: f"| {r['asset']} | {_fmt_regs(r)} |",
        )
    out.write_text("\n".join(lines) + "\n")


def _fmt_regs(result: dict) -> str:
    parts = []
    for metric, info in result["regressions"].items():
        base_s = _fmt_metric_value(metric, info["base"])
        pr_s = _fmt_metric_value(metric, info["pr"])
        label = _LABELS.get(metric, metric)
        parts.append(f"{label} {base_s}→{pr_s} (+{info['pct'] * 100:.0f}%)")
    return "; ".join(parts)


def _sorted_fails(fails: list[dict]) -> list[dict]:
    def worst(result: dict) -> float:
        return max((i["pct"] for i in result["regressions"].values()), default=0.0)

    return sorted(fails, key=worst, reverse=True)


def _perf_main(args) -> int:
    threshold = get_threshold()
    return _diff_main(
        args,
        lambda: compare(args.pr, args.baseline, threshold, get_metrics()),
        write_perf_markdown,
        f"baseline {args.baseline} missing; skipping perf diff",
        "# Perf diff skipped\n\nNo baseline available.\n",
        lambda r: f"perf_diff: {r['fail_count']}/{r['total']} regressed > {threshold:.0%}",
    )


def _add_common_args(p):
    p.add_argument("--pr", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    pixel = sub.add_parser("pixel", help="per-asset MSE diff")
    _add_common_args(pixel)
    pixel.set_defaults(func=_pixel_main)
    perf = sub.add_parser("perf", help="per-asset perf regression diff")
    _add_common_args(perf)
    perf.set_defaults(func=_perf_main)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
