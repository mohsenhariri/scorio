"""Register the Inspect extension when its optional dependencies are installed."""

import sys
from importlib.metadata import PackageNotFoundError, version

if sys.version_info >= (3, 11):
    try:
        _inspect_version = version("inspect-ai")
        version("packaging")
    except PackageNotFoundError:
        pass
    else:
        from packaging.version import Version

        if Version("0.3.288") <= Version(_inspect_version) < Version("0.4"):
            from . import inspect as inspect
