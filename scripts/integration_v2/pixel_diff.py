"""Compare PR render outputs against a baseline run and report per-asset MSE.

Walks `render_index/events/*.json` in both directories, pairs images by
relative path, computes mean squared error in [0,1], and writes a JSON report
plus a markdown summary. Exits 2 if any paired image has MSE > EPS, 0 otherwise.
Missing images (one side has it, the other doesn't) are reported but do not
fail this check — crashes are handled by the render job's own exit code.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EPS = 1e-3


def _events(root: Path):
    for event_file in sorted((root / "render_index" / "events").glob("*.json")):
        yield json.loads(event_file.read_text())


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _mse(pr_root: Path, base_root: Path, rel: str) -> float | None:
    pr_path = pr_root / rel
    base_path = base_root / rel
    if not pr_path.exists() or not base_path.exists():
        return None
    a = _load(pr_path)
    b = _load(base_path)
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean((a - b) ** 2))


def compare(pr_root: Path, base_root: Path) -> dict:
    results = []
    for event in _events(pr_root):
        asset = event.get("generator", "unknown")
        variant = event.get("variant_key", "unknown")
        for rel in event.get("images", []):
            mse = _mse(pr_root, base_root, rel)
            results.append(
                {
                    "asset": asset,
                    "variant": variant,
                    "image": rel,
                    "mse": mse,
                    "status": (
                        "missing_baseline"
                        if mse is None
                        else "fail"
                        if mse > EPS
                        else "ok"
                    ),
                }
            )
    fails = [r for r in results if r["status"] == "fail"]
    missing = [r for r in results if r["status"] == "missing_baseline"]
    return {
        "eps": EPS,
        "total": len(results),
        "fail_count": len(fails),
        "missing_count": len(missing),
        "results": results,
    }


def write_markdown(report: dict, out: Path):
    lines = [
        f"# Pixel diff vs baseline (MSE, eps={report['eps']:.0e})",
        "",
        f"- compared: {report['total']}",
        f"- exceed eps: **{report['fail_count']}**",
        f"- missing baseline: {report['missing_count']}",
        "",
    ]
    fails = [r for r in report["results"] if r["status"] == "fail"]
    if fails:
        lines += ["| asset | variant | image | MSE |", "|---|---|---|---|"]
        for r in sorted(fails, key=lambda r: -r["mse"])[:50]:
            lines.append(
                f"| {r['asset']} | {r['variant']} | {r['image']} | {r['mse']:.4g} |"
            )
        if len(fails) > 50:
            lines.append(f"\n_…{len(fails) - 50} more omitted_")
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"baseline {args.baseline} missing; skipping diff", file=sys.stderr)
        args.report.write_text(json.dumps({"skipped": True}))
        args.summary.write_text("# Pixel diff skipped\n\nNo baseline available.\n")
        return 0

    report = compare(args.pr, args.baseline)
    args.report.write_text(json.dumps(report, indent=2))
    write_markdown(report, args.summary)
    print(
        f"pixel_diff: {report['fail_count']}/{report['total']} exceed MSE eps={report['eps']:.0e}"
    )
    return 2 if report["fail_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
