# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Jack Nugent

import argparse
import logging
import os
import subprocess
from pathlib import Path

import tomllib
from collection import collect_images_structured, print_collection_summary
from display import build_comparison_data
from flask import Flask, redirect, render_template, request, send_file, session
from werkzeug.utils import safe_join

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "default-key"
)  # replace with env/config in production
if app.secret_key == "default-key":
    print("Warning: flask secret key not set; using default")

available_versions = []
scan_directory = None
default_versions = []
safe_mode = False
allowed_root: Path | None = None


def get_network_ips():
    # Simple/non-portable: ask the shell for all IPs and pick first non-loopback
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            for candidate in result.stdout.split():
                if candidate and not candidate.startswith("127."):
                    return "127.0.0.1", candidate
    except Exception:
        pass

    return "127.0.0.1", "127.0.0.1"


def get_git_info(path: Path) -> dict[str, str | None]:
    info = {"branch": None, "commit": None}
    # Prefer metadata written by launch.sh
    toml_path = path / "git_info.toml"
    if toml_path.exists():
        try:
            data = tomllib.loads(toml_path.read_text())
            for key in info:
                if key in data:
                    info[key] = data[key]
            return info
        except Exception:
            pass

    return info


def scan_versions_directory(parent_dir: Path) -> list[dict]:
    versions = []
    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("."):
            continue

        mtime = item.stat().st_mtime
        git_info = get_git_info(item)

        versions.append(
            {
                "name": item.name,
                "mtime": mtime,
                "branch": git_info["branch"],
                "commit": git_info["commit"],
            }
        )

    versions.sort(key=lambda v: v["mtime"], reverse=True)
    return versions


@app.route("/")
def index():
    versions = request.args.getlist("v")
    stored_versions = session.get("versions", []) or default_versions

    # Path 1: no info specified
    if not versions and not stored_versions:
        if scan_directory:
            return redirect("/select-versions")
        return "No versions provided. Use ?v=path1&v=path2", 400

    # Path 2: paths specified via query params
    if versions:
        version_paths = []
        for v in versions:
            p = Path(v)
            if not p.is_absolute() and scan_directory:
                p = (scan_directory / p).resolve()
            else:
                p = p.resolve()
            version_paths.append(p)
        if safe_mode:
            for p in version_paths:
                if not allowed_root or not p.is_relative_to(allowed_root):
                    return "Path outside allowed directory", 403
        stored_versions = [{"name": p.name, "path": str(p)} for p in version_paths]
        session["versions"] = stored_versions

    # Path 3: load versions previously selected
    elif "versions" not in session and stored_versions:
        # Seed session with defaults so /images works on first load
        session["versions"] = stored_versions

    version_paths = [Path(v["path"]) for v in stored_versions]
    version_names = [p.name for p in version_paths]

    collection_results = []
    for version_path, version_name in zip(version_paths, version_names):
        result = collect_images_structured(version_path, version_name)
        collection_results.append(result)

    print_collection_summary(collection_results)
    rows_data = build_comparison_data(collection_results, version_names)

    mode = request.args.get("mode", "sidebyside")
    template = "mode_toggle.html" if mode == "toggle" else "mode_sidebyside.html"

    all_assets_sets = [set(result.assets.keys()) for result in collection_results]
    assets_match = len(all_assets_sets) <= 1 or all(
        s == all_assets_sets[0] for s in all_assets_sets
    )

    return render_template(
        template,
        rows=rows_data,
        version_names=version_names,
        mode=mode,
        assets_match=assets_match,
    )


@app.route("/select-versions")
def select_versions():
    global available_versions
    if scan_directory is not None:
        print("SCAN DIR", scan_directory)
        available_versions = scan_versions_directory(scan_directory)
    return render_template("select_versions.html", versions=available_versions)


@app.route("/help")
def help_page():
    return render_template("help.html")


@app.route("/load-versions", methods=["POST"])
def load_versions():
    selected = request.form.getlist("versions")
    if not selected:
        return redirect("/select-versions")

    if not scan_directory:
        return "Version selection requires --scan-dir", 403

    version_paths = []
    for v in selected:
        rp = (scan_directory / v).resolve()
        if safe_mode and (not allowed_root or not rp.is_relative_to(allowed_root)):
            return "Path outside allowed directory", 403
        version_paths.append(rp)
    session["versions"] = [
        {"name": p.name, "path": str(p.resolve())} for p in version_paths
    ]

    return redirect("/")


@app.route("/images/<path:filepath>")
def serve_image(filepath):
    parts = filepath.split("/", 1)
    if len(parts) != 2:
        return "Invalid path", 404

    version_name = parts[0]
    rel_path = parts[1]

    stored_versions = session.get("versions", [])
    base_paths = {v["name"]: Path(v["path"]) for v in stored_versions}

    if version_name not in base_paths:
        return "Version not found", 404

    # safe_join prevents ../ traversal relative to the version root
    joined = safe_join(str(base_paths[version_name]), rel_path)
    if not joined:
        return "Invalid path", 404

    abs_path = Path(joined).resolve()
    if safe_mode and (not allowed_root or not abs_path.is_relative_to(allowed_root)):
        return "Path outside allowed directory", 403
    if not abs_path.exists():
        return "Image not found", 404

    return send_file(abs_path)


def main():
    global available_versions, scan_directory, default_versions, safe_mode, allowed_root
    usage = """This is a script to visualize Infinigen Material renders. It can be run with:
1. A directory containing versions (e.g. `outputs/v1`, `outputs/v2`): `python scripts/integration_v2/compare.py --scan-dir outputs`
2. Any number of specific versions folders: `python scripts/integration_v2/compare.py <version1> <version2> ...`
"""

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("version_paths", type=Path, nargs="*")
    parser.add_argument("--port", type=int, default=5000)
    # NOTE: if changing this, look at the message in templates/select_versions.html
    parser.add_argument("--scan-dir", type=Path)
    parser.add_argument(
        "--safe",
        action="store_true",
        help="enforces that all paths are contained in ALLOWED_DIRECTORY env variable",
    )
    args = parser.parse_args()

    safe_mode = args.safe
    allowed_env = os.environ.get("ALLOWED_DIRECTORY")
    allowed_root = Path(allowed_env).resolve() if allowed_env else None
    if safe_mode and not allowed_root:
        raise SystemExit("ALLOWED_DIRECTORY must be set when using --safe")

    if args.scan_dir:
        scan_directory = args.scan_dir.resolve()
        available_versions = scan_versions_directory(scan_directory)
    elif args.version_paths:
        version_paths = [p.resolve() for p in args.version_paths]
        default_versions = [{"name": p.name, "path": str(p)} for p in version_paths]

    localhost, lan_ip = get_network_ips()
    print(f"Local:  http://{localhost}:{args.port}")
    print(f"LAN:    http://{lan_ip}:{args.port}")

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    app.run(host="0.0.0.0", port=args.port, debug=True)


if __name__ == "__main__":
    main()
