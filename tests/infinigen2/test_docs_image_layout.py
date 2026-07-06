from pathlib import Path

CONF = Path(__file__).resolve().parents[2] / "docs" / "source" / "conf.py"


def test_conf_emits_assets_images_layout() -> None:
    """The docs-image publish tool lives out of tree in ~/projects/infinigen_docs_ops
    and mirrors a bundle into <slug>/assets/images/<name>/<i>.png. If conf.py stops
    emitting that exact layout, published images 404 (the #806 class of bug). Update
    collect.py's dest_subpath in lockstep if this fails."""
    src = CONF.read_text(encoding="utf-8")
    assert 'f"images/{name}/{i}.png"' in src
    assert 'f"images/{name}/0.png"' in src
    assert "/assets/{rel}" in src
