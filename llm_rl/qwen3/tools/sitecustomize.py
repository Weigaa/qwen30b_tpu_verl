"""Keep Qwen3 runs isolated from stale global editable framework installs."""

from __future__ import annotations

import os
import sys
from pathlib import Path


repo_root = os.environ.get("QWEN3_LOCAL_IMPORT_ROOT")
if repo_root:
    repo_root = str(Path(repo_root).resolve())
    stale_roots = {
        str(Path("/workspace/vllm").resolve()),
        str(Path("/workspace/vllm-ascend").resolve()),
    }
    editable_prefixes = (
        "__editable___vllm_",
        "__editable___vllm_ascend_",
    )

    def _provider_module(provider: object) -> str:
        return str(
            getattr(provider, "__module__", type(provider).__module__)
        )

    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not _provider_module(finder).startswith(editable_prefixes)
    ]
    sys.path_hooks[:] = [
        hook
        for hook in sys.path_hooks
        if not _provider_module(hook).startswith(editable_prefixes)
    ]
    sys.path[:] = [
        entry
        for entry in sys.path
        if not str(entry).startswith("__editable__.vllm")
        and (
            not entry
            or str(Path(entry).resolve()) not in stale_roots
        )
    ]
    for entry in list(sys.path_importer_cache):
        if str(entry).startswith("__editable__.vllm"):
            sys.path_importer_cache.pop(entry, None)

    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
