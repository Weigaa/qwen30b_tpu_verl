"""Small pyACL compatibility surface backed by the selected CANN runtime.

CANN 8.5 no longer ships the legacy ``acl.so`` Python extension in the
toolkit bundle used by this repository. MindSpeed and vLLM-Ascend only use
three runtime calls, exposed here without loading an extension from another
CANN release.
"""

from . import rt


def get_soc_name() -> str:
    return rt.get_soc_name()


__all__ = ["get_soc_name", "rt"]
