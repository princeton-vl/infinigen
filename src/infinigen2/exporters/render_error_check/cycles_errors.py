# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Alexander Raistrick

import logging
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from infinigen2 import context

logger = logging.getLogger(__name__)

CYCLES_ERROR_PATTERNS = (
    "SVM stack",
    "shader graph",
    "shader compile",
    "Error: ",
)


class CyclesShaderError(RuntimeError):
    pass


def _pump_pipe(r: int, replay_fd: int | None, captured: bytearray) -> None:
    while chunk := os.read(r, 4096):
        if replay_fd is not None:
            os.write(replay_fd, chunk)
        captured.extend(chunk)


@contextmanager
def detect_cycles_errors(replay: bool = True) -> Iterator[dict]:
    sys.stdout.flush()
    sys.stderr.flush()

    old1, old2 = os.dup(1), os.dup(2)
    r, w = os.pipe()
    captured = bytearray()
    capture = {"text": ""}

    replay_fd = old1 if replay else None
    t = threading.Thread(target=_pump_pipe, args=(r, replay_fd, captured), daemon=True)
    t.start()

    try:
        os.dup2(w, 1)
        os.dup2(w, 2)
        os.close(w)
        yield capture
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        t.join(timeout=5)
        os.close(old1)
        os.close(old2)
        os.close(r)
        capture["text"] = captured.decode(errors="replace")

    text = capture["text"]
    hits = [
        ln for ln in text.splitlines() if any(p in ln for p in CYCLES_ERROR_PATTERNS)
    ]
    if hits:
        error = CyclesShaderError("\n".join(hits))
        context.raise_or_warn(context.globals.error_mode_cycles_shader, error, logger)
