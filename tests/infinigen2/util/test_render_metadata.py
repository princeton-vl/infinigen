# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import getpass
import json
import socket
from pathlib import Path

from infinigen2.exporters.util.format import ExportType
from infinigen2.util.render_metadata import write_render_metadata


def _write(output: Path, exports: dict) -> dict:
    return write_render_metadata(
        output=output,
        seed=0x1234,
        times={"room": 2.0, "render": 4.0},
        exports=exports,
        build_keys={"room"},
        render_keys={"render"},
        n_frames=2,
    )


def test_metadata_does_not_identify_the_render_machine(tmp_path):
    metadata = _write(tmp_path, {ExportType.CAMERA: [tmp_path / "Cam" / "camera.npz"]})
    text = (tmp_path / "metadata.json").read_text()

    assert "hardware" not in metadata
    assert getpass.getuser() not in text
    assert socket.gethostname() not in text
    assert str(tmp_path) not in text


def test_export_paths_are_relative_to_the_output_folder(tmp_path):
    exports = {
        ExportType.CAMERA: [tmp_path / "Cam" / "camera.npz"],
        ExportType.OBJECT_DATA: [tmp_path / "object-data.npz"],
    }
    metadata = _write(tmp_path, exports)

    assert metadata["exports"][str(ExportType.CAMERA)] == ["Cam/camera.npz"]
    assert metadata["exports"][str(ExportType.OBJECT_DATA)] == ["object-data.npz"]


def test_exports_outside_the_output_folder_keep_only_their_filename(tmp_path):
    outside = tmp_path.parent / "scratch_ar8564" / "depth_0000.npy"
    metadata = _write(tmp_path, {ExportType.DEPTH: [outside]})

    assert metadata["exports"][str(ExportType.DEPTH)] == ["depth_0000.npy"]


def test_metadata_keeps_the_fields_consumers_need(tmp_path):
    metadata = _write(tmp_path, {})
    written = json.loads((tmp_path / "metadata.json").read_text())

    assert written == metadata
    assert metadata["seed"] == "0x1234"
    assert metadata["stats"]["blend_build_sec"] == 2.0
    assert metadata["stats"]["render_sec_per_frame"] == 2.0
    assert metadata["generator_times"] == {"room": 2.0, "render": 4.0}
