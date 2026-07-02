#!/usr/bin/env python3
"""Publish manifest example renders to the infinigen.cs.princeton.edu/docs webspace.

Takes an integration render archive (the same ``develop_latest`` dir the docs
build reads), selects the per-generator example images via
``collect_manifest_images.collect``, and copies them into the CS-department
project webspace so the URLs the Sphinx docs emit actually resolve.

The docs (see ``docs/source/conf.py``) reference each generator image as::

    <INFINIGEN_DOCS_IMAGE_BASE>/<slug>/images/<name>/<i>.png

with ``INFINIGEN_DOCS_IMAGE_BASE`` defaulting to
``https://infinigen.cs.princeton.edu/docs``. The webspace docroot served at that
URL is ``/n/fs/pvl-ifg-pub/www/docs``, so an image at

    /n/fs/pvl-ifg-pub/www/docs/<slug>/images/<name>/<i>.png

is served as ``https://infinigen.cs.princeton.edu/docs/<slug>/images/<name>/<i>.png``.
This script copies the whole collected ``images/`` tree (including its
``index.json``) into that layout.

The render archive lives on the renderbot runner (pvlbox), but the webspace
mounts on soak, NOT pvlbox, so no single host can run this end to end. Collect
on pvlbox, scp the staged ``images/`` dir to soak, then run this with
``--staging`` there against that staging dir (see the release procedure runbook)::

    # on pvlbox:
    python scripts/docs/collect_manifest_images.py \\
        --renders-root <archive> --output-root /tmp/manifest_images
    # copy /tmp/manifest_images/images -> soak, then on soak:
    python scripts/docs/publish_manifest_images.py --staging /tmp/manifest_images

If a host ever mounts both (it does not today), a bare invocation collects and
places in one shot.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_manifest_images import collect  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "src" / "infinigen2" / "manifest.json"
DEFAULT_RENDERS_ROOT = Path(
    "/home/renderbot/integration_test_archive/renders/integration/develop_latest"
)
DEFAULT_WWW_ROOT = Path("/n/fs/pvl-ifg-pub/www/docs")

# Landing-page demo renders (launch.sh's "DOCS LANDING DEMOS" block), keyed by
# their --output dir name; outside the manifest-keyed images/ tree.
LANDING_DEMOS = {
    "landing-bricks_rand-torus-cycles-0": "bricks_torus.png",
    "landing-fabric_patterned_rand-monkey-cycles-0": "fabric_patterned_monkey.png",
}


def _version_slug() -> str:
    """The per-version path segment; must match ``conf.py:_version_slug``.

    Real ReadTheDocs builds set ``READTHEDOCS_VERSION`` to the exact tag/branch
    slug; otherwise fall back to ``v<__version__>``.
    """

    env = os.environ.get("READTHEDOCS_VERSION") or os.environ.get(
        "INFINIGEN_DOCS_VERSION"
    )
    if env:
        return env
    init = (REPO_ROOT / "src" / "infinigen2" / "__init__.py").read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    version = match.group(1) if match else "unknown"
    return f"v{version}"


def publish(
    staging: Path,
    www_root: Path,
    slug: str,
    dry_run: bool,
) -> None:
    src_images = staging / "images"
    images = sorted(src_images.rglob("*.png"))
    if not images:
        print(
            f"ERROR: no images under {src_images}; nothing to publish.", file=sys.stderr
        )
        sys.exit(1)

    dst_images = www_root / slug / "images"
    generators = len({img.parent for img in images})
    print(f"Publishing {len(images)} images ({generators} generators) -> {dst_images}")
    if dry_run:
        print("(dry run: no files written)")
        return

    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)

    example = images[0].relative_to(src_images)
    print(f"Published under slug '{slug}'.")
    print(
        f"Example URL: https://infinigen.cs.princeton.edu/docs/{slug}/images/{example}"
    )


def publish_landing_demos(
    renders_root: Path,
    www_root: Path,
    slug: str,
    dry_run: bool,
) -> None:
    if not renders_root.is_dir():
        print(f"Skipping landing demos: {renders_root} not found.", file=sys.stderr)
        return

    dst_landing = www_root / slug / "landing"
    for demo_dir, dst_name in LANDING_DEMOS.items():
        pngs = sorted((renders_root / demo_dir).rglob("*.png"))
        if not pngs:
            print(f"Skipping missing landing demo: {demo_dir}", file=sys.stderr)
            continue

        print(f"Publishing landing demo {demo_dir} -> {dst_landing / dst_name}")
        if dry_run:
            continue
        dst_landing.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pngs[0], dst_landing / dst_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="Pre-collected staging dir (contains images/); skips collect.",
    )
    parser.add_argument(
        "--renders-root",
        type=Path,
        default=DEFAULT_RENDERS_ROOT,
        help="Integration render version dir; used only when --staging is absent.",
    )
    parser.add_argument(
        "--www-root",
        type=Path,
        default=DEFAULT_WWW_ROOT,
        help="Webspace docroot served at https://infinigen.cs.princeton.edu/docs.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.json.",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Version path segment (default: env READTHEDOCS_VERSION or v<__version__>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be published without writing to the webspace.",
    )
    args = parser.parse_args()

    slug = args.slug or _version_slug()
    if not args.www_root.is_dir():
        print(
            f"ERROR: webspace root not found: {args.www_root}\n"
            "Run --place on a host that mounts /n/fs/pvl-ifg-pub (soak).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.staging is not None:
        staging = args.staging
    else:
        staging = Path(tempfile.mkdtemp(prefix="manifest_images_"))
        collect(args.renders_root, args.manifest, staging)

    publish(staging, args.www_root, slug, args.dry_run)
    publish_landing_demos(args.renders_root, args.www_root, slug, args.dry_run)
    if args.staging is None:
        shutil.rmtree(staging)


if __name__ == "__main__":
    main()
