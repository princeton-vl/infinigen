#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Turn a raw freeze into a publish-ready tree.

freeze.py emits raw PNG, which keeps the viewer's pixel diff byte-exact against
the render archive but costs a visitor 275 MB across 3204 tiles. This adds a
WebP beside every PNG (275 MB -> 34 MB) and points the comparison HTML at those.

The PNGs are copied through, never replaced or renamed: docs/source/Infinigen2.md
and already-deployed docs HTML deep-link /changes/<version>/.../0000.png.

    freeze.py --out RAW              # on pvlbox, raw PNG
    rsync -a RAW/ <buildhost>:RAW/
    build.py RAW DEST                # here
    rsync -a --delete DEST/ soak:/n/fs/pvl-ifg-pub/www/changes/

Publishing stays a separate act so --delete can be dry-run first.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_PNG_SRC = re.compile(r'(src="[^"]+)\.png"')
_SRC = re.compile(r'src="([^"]+)"')

HTACCESS = """Options -Indexes

<FilesMatch "\\.(webp|png|mp4)$">
  Header set Cache-Control "public, max-age=31536000, immutable"
</FilesMatch>
<FilesMatch "\\.html$">
  Header set Cache-Control "public, max-age=300"
</FilesMatch>
"""

ROBOTS = """User-agent: *
Disallow: /changes/v2.0.0a1/
Disallow: /changes/v2.0.0a2/
"""


def classify(src_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    pngs, htmls, others = [], [], []
    buckets = {".png": pngs, ".html": htmls}
    for path in sorted(src_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(src_root)
            buckets.get(path.suffix, others).append(rel)
    return pngs, htmls, others


def copy_through(src_root: Path, dest: Path, rels: list[Path]) -> int:
    total = 0
    for rel in rels:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rel, dst)
        total += dst.stat().st_size
    return total


def _cwebp(src: Path, dst: Path, flags: list[str]) -> None:
    subprocess.run(
        ["cwebp", "-quiet", *flags, str(src), "-o", str(dst)],
        check=True,
        capture_output=True,
    )


def rgb_psnr(a: Path, b: Path) -> float:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


def encode_webp(src: Path, dst: Path, quality: int, floor: float) -> tuple[int, bool]:
    # -print_psnr is luma-weighted; normal maps carry signal in chroma, which 4:2:0 destroys.
    dst.parent.mkdir(parents=True, exist_ok=True)
    _cwebp(src, dst, ["-q", str(quality)])
    degraded = rgb_psnr(src, dst) < floor
    if degraded:
        _cwebp(src, dst, ["-near_lossless", "60", "-q", "100"])
    return dst.stat().st_size, degraded


def _encode_job(job: tuple[Path, Path, int, float]) -> tuple[int, bool]:
    return encode_webp(*job)


def encode_all(
    src_root: Path, dest: Path, rels: list[Path], quality: int, floor: float
) -> tuple[int, int]:
    jobs = [(src_root / r, dest / r.with_suffix(".webp"), quality, floor) for r in rels]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_encode_job, jobs))
    return sum(size for size, _ in results), sum(1 for _, hit in results if hit)


def rewrite_html(src_root: Path, dest: Path, rels: list[Path]) -> int:
    rewritten = 0
    for rel in rels:
        html = (src_root / rel).read_text()
        new, count = _PNG_SRC.subn(r'\1.webp"', html)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(new)
        rewritten += count
    return rewritten


def check_refs(dest: Path, rels: list[Path]) -> tuple[int, list[str]]:
    refs = [
        (rel, ref) for rel in rels for ref in _SRC.findall((dest / rel).read_text())
    ]
    missing = [
        f"{rel}: {ref}"
        for rel, ref in refs
        if not (dest / rel).parent.joinpath(ref).exists()
    ]
    return len(refs), missing


def write_policy(dest: Path, siteroot: Path) -> None:
    (dest / ".htaccess").write_text(HTACCESS)
    siteroot.mkdir(parents=True, exist_ok=True)
    (siteroot / "robots.txt").write_text(ROBOTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="a raw freeze tree (freeze.py --out)")
    parser.add_argument("dest", type=Path, help="publish-ready tree to write")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="re-encode near-lossless any tile whose RGB PSNR at --quality falls below"
        " this. 32 catches the GT normal passes, ~10%% of tiles, for +8 MB; judged"
        " unnecessary at q90 on 2026-08-04, so off by default.",
    )
    parser.add_argument(
        "--siteroot",
        type=Path,
        help="where to put robots.txt, which belongs at the WWW root shared with"
        " /docs, not under /changes. Defaults to <dest>_siteroot.",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace a non-empty <dest>"
    )
    args = parser.parse_args()

    if shutil.which("cwebp") is None:
        raise SystemExit("cwebp not found; install libwebp (brew install webp)")
    if not (args.src / "index.html").exists():
        raise SystemExit(f"{args.src} has no index.html; not a freeze")

    dest = args.dest
    if dest.exists() and any(dest.iterdir()):
        if not args.force:
            raise SystemExit(f"{dest} is not empty; pass --force to replace it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    siteroot = args.siteroot or dest.with_name(dest.name + "_siteroot")

    pngs, htmls, others = classify(args.src)
    png_bytes = copy_through(args.src, dest, pngs)
    other_bytes = copy_through(args.src, dest, others)
    webp_bytes, degraded = encode_all(args.src, dest, pngs, args.quality, args.floor)
    rewritten = rewrite_html(args.src, dest, htmls)
    write_policy(dest, siteroot)

    checked, missing = check_refs(dest, htmls)
    if missing:
        raise SystemExit("dangling refs:\n" + "\n".join(missing[:20]))

    mb = 1024 * 1024
    print(f"{len(pngs)} png {png_bytes / mb:.0f} MB -> webp {webp_bytes / mb:.0f} MB")
    print(f"{degraded} below {args.floor} dB at q{args.quality}, redone near-lossless")
    print(f"{len(others)} passthrough {other_bytes / mb:.0f} MB")
    print(f"{len(htmls)} html, {rewritten} srcs repointed, {checked} refs all resolve")
    print(f"served per visitor: {(webp_bytes + other_bytes) / mb:.0f} MB")
    print("\npublish with:")
    print(f"  rsync -an --delete -i {dest}/ soak:/n/fs/pvl-ifg-pub/www/changes/")
    print(f"  rsync -a  --delete    {dest}/ soak:/n/fs/pvl-ifg-pub/www/changes/")
    print(f"  rsync -a {siteroot}/ soak:/n/fs/pvl-ifg-pub/www/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
