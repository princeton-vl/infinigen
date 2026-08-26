import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONF = REPO_ROOT / "docs" / "source" / "conf.py"
LAUNCH = REPO_ROOT / "scripts" / "integration_v2" / "launch.sh"

_OUTPUT_RE = re.compile(r"--output \$OUTPUT_PATH/(\S+)")
_SHELL_SUBS = (
    ("{}", r"[A-Za-z0-9_]+"),
    ("$sn", r"[A-Za-z0-9_]+"),
    ("$CAM_SCENE", "livingroom_rand"),
    ("$disp", r"[A-Za-z0-9_]+"),
    ("$i", r"\d+"),
)


def _load_conf() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_docs_conf", CONF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_dir_patterns() -> list[re.Pattern]:
    patterns = []
    for template in _OUTPUT_RE.findall(LAUNCH.read_text(encoding="utf-8")):
        expr = template
        for token, sub in _SHELL_SUBS:
            expr = expr.replace(token, f"\x00{sub}\x00")
        parts = expr.split("\x00")
        expr = "".join(p if i % 2 else re.escape(p) for i, p in enumerate(parts))
        patterns.append(re.compile(f"^{expr}$"))
    return patterns


def _matches_any(directory: str, patterns: list[re.Pattern]) -> bool:
    return any(p.match(directory) for p in patterns)


def test_docs_images_point_at_directories_launch_sh_renders() -> None:
    """Docs galleries deep-link the integration render archive at
    <base>/<slug>/<category>-<shortname>-<engine>-<seed>/<media>. Those directories are
    created by scripts/integration_v2/launch.sh, so if either side renames a category
    prefix or engine the published images 404 (the #806 class of bug). Keep them in
    lockstep rather than relaxing this test."""
    conf = _load_conf()
    patterns = _launch_dir_patterns()
    for name in conf._IMAGE_COUNTS:
        directory = conf._archive_rel(name, 0).split("/", 1)[0]
        assert _matches_any(directory, patterns), (
            f"{name} points at {directory!r}, which launch.sh never renders"
        )


def test_preset_images_point_at_directory_launch_sh_renders() -> None:
    conf = _load_conf()
    directory = conf._preset_image_url("pkg.mod.demo_preset")
    directory = directory.rsplit(f"/{conf.VERSION_SLUG}/", 1)[-1].split("/", 1)[0]
    assert _matches_any(directory, _launch_dir_patterns())


def test_still_and_trajectory_media_names() -> None:
    conf = _load_conf()
    for name in conf._IMAGE_COUNTS:
        rel = conf._archive_rel(name, 0)
        expected = "image_Camera.mp4" if conf._is_video(name) else "Camera/0000.png"
        assert rel.split("/", 1)[1] == expected


def test_published_urls_are_versioned_webp() -> None:
    """The publish build re-encodes every archive PNG to WebP beside it and the docs
    link the WebP; an unversioned or still-PNG URL means the gallery 404s or serves the
    wrong release."""
    conf = _load_conf()
    assert conf.IMAGE_URL_BASE, "docs images must resolve to a published base URL"
    for name in conf._IMAGE_COUNTS:
        for url in conf._image_urls(name):
            assert url.startswith(f"{conf.IMAGE_URL_BASE}/{conf.VERSION_SLUG}/")
            assert not url.endswith(".png")
            assert url.endswith(".webp") or url.endswith(".mp4")
