import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "integration_v2"
sys.path.insert(0, str(_SCRIPTS))

import launch_andromeda  # noqa: E402


def test_launcher_marks_each_render_slot(monkeypatch, tmp_path):
    launched = []

    class Popen:
        def __init__(self, cmd, env, text):
            launched.append(env["INTEGRATION_SLOT_INDEX"])

        def wait(self):
            return 0

    monkeypatch.setattr(
        sys, "argv", ["launch_andromeda.py", "--output_path", str(tmp_path)]
    )
    monkeypatch.setattr(launch_andromeda, "list_items", lambda *args: [])
    monkeypatch.setattr(launch_andromeda, "resolve_gpu_ids", lambda gpus: ["0", "1"])
    monkeypatch.setattr(launch_andromeda, "render_runner", lambda output_path: "runner")
    monkeypatch.setattr(launch_andromeda.subprocess, "Popen", Popen)
    monkeypatch.setattr(
        launch_andromeda, "failed_render_names", lambda output_path: ([], [])
    )

    assert launch_andromeda.main() == 0
    assert launched == ["0", "1"]
