#!/usr/bin/env python3
"""Compatibility entry point for Natural floor2 KV authorization."""

from __future__ import annotations

import sys

from verify_deepseek_kv_cap_run import main


if __name__ == "__main__":
    if "--lifecycle" not in sys.argv:
        sys.argv[1:1] = ["--lifecycle", "natural_f2"]
    raise SystemExit(main())
