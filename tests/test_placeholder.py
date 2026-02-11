# Todo: integrate Julia and Python tests into a single suite.
from scorio import __version__


def test_placeholder_passes():
    assert True


def test_version_is_non_empty():
    assert isinstance(__version__, str)
    assert __version__.strip() != ""
