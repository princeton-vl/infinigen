#!/usr/bin/env python3
# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

"""Freeze the integration viewer into a static site for a fixed set of versions.

The live viewer (compare.py) is a Flask app; the public webspace at
soak:/n/fs/pvl-ifg-pub/www is static-only. This drives the real app via its
test client so the frozen HTML matches the live viewer, rewrites the few
server-dependent URLs (/images, /help) to work from a subfolder, and copies
exactly the media the pages reference. Side-by-side and toggle are the same
page - the viewer switches between them client-side.

Output layout (drop straight into <www>/changes/):
    <out>/index.html                 listing of every comparison present
    <out>/<a>-vs-<b>/index.html      the comparison
    <out>/<a>-vs-<b>/help.html
    <out>/<version>/...              the referenced render media, in archive layout

Each comparison gets its own concrete URL, so freezing a new pair never
overwrites an older one. Media is shared at the site root across comparisons,
which is also what the docs galleries deep-link.
"""

import argparse
import json
import re
import shutil
import tempfile
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import quote

import compare

_IMAGE_REF = re.compile(r'/images/([^"\'\s>]+)')


def parse_version(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        path = spec
        name = Path(spec).name
    return name, Path(path).resolve()


def rewrite_html(html: str) -> str:
    # img/video paths are "<version>/<rel>"; media lives one level up, at the site root.
    html = html.replace('src="/images/', 'src="../')
    html = html.replace('href="/help"', 'href="help.html"')
    return html.replace('href="/select-versions"', 'href="index.html"')


def render_pages(out_dir: Path, versions: list[tuple[str, Path]]) -> set[str]:
    compare.safe_mode = False
    compare.scan_directory = None
    client = compare.app.test_client()

    query = "&".join(f"v={quote(str(p), safe='/')}" for _, p in versions)
    resp = client.get(f"/?{query}")
    if resp.status_code != 200:
        raise SystemExit(f"viewer returned {resp.status_code}")
    html = resp.get_data(as_text=True)
    (out_dir / "index.html").write_text(rewrite_html(html))
    refs = set(_IMAGE_REF.findall(html))

    help_html = client.get("/help").get_data(as_text=True)
    (out_dir / "help.html").write_text(
        help_html.replace('href="/"', 'href="index.html"')
    )
    return refs


def copy_referenced(out_dir: Path, refs: set[str], sources: dict[str, Path]) -> int:
    copied = 0
    for ref in sorted(refs):
        version, _, rel = ref.partition("/")
        src = sources.get(version)
        if src is None or not rel:
            continue
        src_file = src / rel
        if not src_file.exists():
            continue
        dst = out_dir / version / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)
        copied += 1
    return copied


def _symlink_children(dst: Path, src: Path, skip: set[str]) -> None:
    dst.mkdir()
    for child in src.iterdir():
        if child.name not in skip:
            (dst / child.name).symlink_to(child)


def _filter_events(dst_ri: Path, src_ri: Path, excludes: list[str]) -> None:
    _symlink_children(dst_ri, src_ri, {"events"})
    events = dst_ri / "events"
    events.mkdir()
    for event_file in (src_ri / "events").glob("*.json"):
        generator = json.loads(event_file.read_text()).get("generator", "")
        if not any(fnmatch(generator, pat) for pat in excludes):
            (events / event_file.name).symlink_to(event_file)


def alias_version(link: Path, src: Path, tag: str, excludes: list[str]) -> None:
    # Alias under the app's version name; label it with `tag` and drop excluded events.
    skip = {"git_info.toml", "render_index"} if excludes else {"git_info.toml"}
    _symlink_children(link, src, skip)
    original = (
        (src / "git_info.toml").read_text() if (src / "git_info.toml").exists() else ""
    )
    (link / "git_info.toml").write_text(f'{original}\ntag = "{tag}"\n')
    if excludes:
        _filter_events(link / "render_index", src / "render_index", excludes)


def write_listing(out_dir: Path) -> None:
    # Comparison folders carry an index.html; media version folders do not.
    names = sorted(
        d.name for d in out_dir.iterdir() if d.is_dir() and (d / "index.html").exists()
    )
    items = "\n".join(f'<li><a href="{n}/">{n}</a></li>' for n in names)
    (out_dir / "index.html").write_text(
        f"<!doctype html><title>Infinigen visual changes</title>"
        f"<h1>Infinigen visual changes</h1><ul>\n{items}\n</ul>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--version",
        dest="versions",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a version to include; repeat for side-by-side. NAME defaults to the dir basename.",
    )
    parser.add_argument(
        "--exclude",
        dest="excludes",
        action="append",
        default=[],
        metavar="GLOB",
        help="generator-name glob to drop from the viewer (e.g. 'material_*'); repeatable.",
    )
    parser.add_argument(
        "--name",
        help="subfolder for this comparison; defaults to '<a>-vs-<b>' from the version names.",
    )
    args = parser.parse_args()

    versions = [parse_version(spec) for spec in args.versions]
    for name, path in versions:
        if not (path / "render_index" / "events").is_dir():
            raise SystemExit(f"{path} has no render_index/events; not a render archive")

    comparison = args.name or "-vs-".join(name for name, _ in versions)
    pages_dir = args.out / comparison
    pages_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: path for name, path in versions}

    with tempfile.TemporaryDirectory() as tmp:
        aliased = []
        for name, path in versions:
            link = Path(tmp) / name
            alias_version(link, path, name, args.excludes)
            aliased.append((name, link))
        refs = render_pages(pages_dir, aliased)

    total = copy_referenced(args.out, refs, sources)
    write_listing(args.out)
    print(f"Froze {comparison} to {pages_dir} ({total} media files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
