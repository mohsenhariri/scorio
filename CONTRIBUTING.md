# Contributing to Scorio

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohsenhariri/scorio.git
cd scorio
```

2. Create a virtual environment:

**Using uv:**
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. Install with dev dependencies:

**Using uv:**
```bash
uv pip install -e ".[dev]"
```

**Using pip:**
```bash
pip install -e ".[dev]"
```

### Dependencies

**Runtime:**
- numpy, scipy

**Development:**
- pytest, ruff, mypy, build, twine, sphinx

## Code Style

- Follow PEP 8
- Format and lint with Ruff
- Type check with mypy

```bash
ruff format scorio/
ruff check --fix scorio/
mypy scorio/
```

## Testing

```bash
pytest
```

## Release Process

Use `VERSION` as the package-version source of truth, and keep
`docs/changelog.rst` as the human-written release-notes source of truth.

1. Update `VERSION` to the new version.
2. Run `make sync-version`.
3. Update `docs/changelog.rst` with the user-facing changes for the release.
4. Run the relevant checks:

```bash
make format-check
make test
make pkg-check
make jl-test
make js-test
make -C js typecheck
make -C js pack
```

5. Review and stage the release changes, including new source, tests, and
   package assets, then commit and push them:

```bash
git status --short
git add <release-file-paths>
git commit -m "Prepare vX.Y.Z release"
git push origin main
```

6. Publish the package releases:

```bash
make release-py
make release-jl
make release-js
```

`make release-py` creates the Python GitHub release from the matching
`docs/changelog.rst` version section and triggers PyPI publishing.
`make release-jl` dispatches Julia registration; after registration succeeds,
TagBot creates the Julia tag and GitHub release.
`make release-js` creates the npm GitHub release and triggers npm publishing.

## Docstrings

- Use Google-style docstrings
- Document all public APIs
- Include type hints

## Documentation

Build the Python, Julia, or JavaScript documentation locally:
```bash
make py-docs-build
make jl-docs-build
make -C js install
make js-docs-build
make js-docs-check
```

Run `make js-docs-serve` to preview the TypeDoc site at `http://localhost:4003`.
The JavaScript guides are in `js/scorio/docs/`; API pages are generated from
the public source entry points. `js-docs-check` type-checks and runs guide
examples against the built package, then checks generated local links and
anchors. Generated HTML is ignored by Git.

The documentation workflow builds the JavaScript site on pull requests and
deploys it under `/javascript/` alongside the landing page and Julia docs on
pushes to `main`. npm publishing remains a separate release workflow.

Published docs: [Python](https://scorio.readthedocs.io/),
[Julia](https://mohsenhariri.github.io/scorio/julia/), and
[JavaScript / TypeScript](https://mohsenhariri.github.io/scorio/javascript/).
