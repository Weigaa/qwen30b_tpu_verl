#!/usr/bin/env python3
"""Filter known benign runtime noise from long training logs.

The filter is intentionally conservative:
* vLLM Ascend entry-point plugin compatibility failures are removed as a
  traceback block because they are repeatedly emitted before normal runs.
* Bare "path string is NULL" fragments are stripped, but the rest of the line
  is preserved so a real "[ERROR] ..." suffix is still visible.
"""

from __future__ import annotations

import os
import sys


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def main() -> int:
    suppress_noise = _enabled("VERL_SUPPRESS_KNOWN_NOISE", "1")
    suppress_plugins = _enabled("VERL_SUPPRESS_ASCEND_PLUGIN_NOISE", "1")
    strip_path_null = _enabled("VERL_STRIP_ASCEND_PATH_NULL_NOISE", "1")

    suppressing_plugin_traceback = False
    for raw_line in sys.stdin:
        line = raw_line

        if suppress_noise and suppress_plugins:
            if suppressing_plugin_traceback:
                if "[__init__.py:54]" in line:
                    continue
                suppressing_plugin_traceback = False
            if "Failed to load plugin ascend_" in line:
                suppressing_plugin_traceback = True
                continue

        if suppress_noise and strip_path_null:
            line = line.replace("path string is NULL", "")
            if not line.strip():
                continue

        sys.stdout.write(line)
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
