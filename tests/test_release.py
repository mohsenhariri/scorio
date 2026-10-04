"""Release metadata and cross-package fixture checks."""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_version_sync_updates_all_package_metadata(tmp_path):
    paths = [
        "scorio/__init__.py",
        "julia/Scorio.jl/Project.toml",
        "julia/Scorio.jl/src/Scorio.jl",
        "julia/Scorio.jl/test/runtests.jl",
        "julia/Scorio.jl/docs/Manifest.toml",
        "docs/conf.py",
        "CITATION.cff",
        "README.md",
        "README_PyPI.md",
        "js/scorio/package.json",
        "js/scorio/package-lock.json",
        "scripts/sync_version.py",
    ]
    for relative in paths:
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    (tmp_path / "VERSION").write_text("9.8.7\n")
    runpy.run_path(str(tmp_path / "scripts/sync_version.py"), run_name="__main__")
    assert '__version__ = "9.8.7"' in (tmp_path / paths[0]).read_text()
    assert 'version = "9.8.7"' in (tmp_path / paths[1]).read_text()
    package = json.loads((tmp_path / "js/scorio/package.json").read_text())
    lock = json.loads((tmp_path / "js/scorio/package-lock.json").read_text())
    assert (
        package["version"]
        == lock["version"]
        == lock["packages"][""]["version"]
        == "9.8.7"
    )
    assert "git@python-v9.8.7" in (tmp_path / "README_PyPI.md").read_text()
    before = {p: (tmp_path / p).read_bytes() for p in paths}
    runpy.run_path(str(tmp_path / "scripts/sync_version.py"), run_name="__main__")
    assert before == {p: (tmp_path / p).read_bytes() for p in paths}


def test_reference_fixtures_are_identical_in_both_ports():
    for filename in ("tailpass.json",):
        assert (ROOT / "js/scorio/test/fixtures" / filename).read_bytes() == (
            ROOT / "julia/Scorio.jl/test/fixtures" / filename
        ).read_bytes()
    for filename in ("R_top_p.npz", "R_greedy.npz"):
        assert (ROOT / "tests/data" / filename).read_bytes() == (
            ROOT / "julia/Scorio.jl/test/fixtures" / filename
        ).read_bytes()
