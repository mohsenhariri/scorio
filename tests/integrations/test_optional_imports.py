import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing", ["inspect-ai", "packaging"])
def test_registry_is_inert_without_optional_dependency(missing) -> None:
    script = """
import importlib.metadata
import sys
import scorio
original = importlib.metadata.version
def version(name):
    if name == sys.argv[1]:
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = version
import scorio.integrations._registry
assert 'inspect_ai' not in sys.modules
assert 'scorio.integrations.inspect' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, missing], check=True)


@pytest.mark.parametrize("inspect_version", ["0.3.287", "0.4.0"])
def test_registry_is_inert_for_unsupported_inspect(inspect_version) -> None:
    script = """
import importlib.metadata
import sys
import scorio
original = importlib.metadata.version
def version(name):
    return sys.argv[1] if name == 'inspect-ai' else original(name)
importlib.metadata.version = version
import scorio.integrations._registry
assert 'inspect_ai' not in sys.modules
assert 'scorio.integrations.inspect' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, inspect_version], check=True)


def test_direct_import_explains_missing_optional_dependency() -> None:
    script = """
import importlib.abc
import sys
import scorio
class MissingInspect(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'inspect_ai':
            raise ModuleNotFoundError('blocked for test', name='inspect_ai')
sys.meta_path.insert(0, MissingInspect())
try:
    import scorio.integrations.inspect
except ImportError as exc:
    assert 'scorio[inspect]' in str(exc)
    assert 'Python 3.11' in str(exc)
else:
    raise AssertionError('missing Inspect should raise ImportError')
"""
    subprocess.run([sys.executable, "-c", script], check=True)
