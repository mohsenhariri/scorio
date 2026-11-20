# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

## Overview

Scorio is a research toolkit for **Bayesian evaluation and ranking of LLMs** (especially reasoning models under test-time scaling / repeated sampling). It backs three papers (ICLR 2026, ACL 2026, an arXiv preprint on test-time scaling) and ships companion HuggingFace datasets of reasoning traces (`notebooks/datasets/{trace,lite,math,gpqa}`).

The repo contains **three parallel implementations** of the same API, kept in version lockstep via the root `VERSION` file and `scripts/sync_version.py`:

- `scorio/` contains the Python reference implementation, published to PyPI.
- `julia/Scorio.jl/` mirrors the Python `eval`/`rank`/`aggregate`/`sinf`/`utils` API in Julia. The package is registered in the Julia General Registry.
- `js/scorio/` contains the zero-dependency TypeScript port on npm, currently focused on the `eval` APIs.

When changing the public API or fixing a bug in one implementation, check whether the equivalent should also change in the others.

## Common commands

All commands below run from the repo root unless noted; see the root `Makefile` for the authoritative list.

### Python (`scorio/`)

```bash
pip install -e ".[dev]"          # install with dev deps (pytest, ruff, mypy, sphinx...)

make test                        # pytest tests/
pytest tests/eval                # a single test dir
pytest tests/rank -m "not slow"  # rank tests excluding slow ones (make test-rank-py)
pytest tests/rank -m slow        # slow rank tests only (make test-rank-py-slow)
pytest tests/eval/test_bayes.py::test_name -v   # a single test

make format-check                # ruff format --check + ruff check
make format                      # ruff format + ruff check --fix
make lint                        # ruff check + mypy

make py-docs-build               # Sphinx docs -> docs/_build/html
make py-docs-serve                # serve built docs on :4000
```

The `slow` pytest marker is registered in `pyproject.toml` for heavy ranking APIs. Use it on any new test that's expensive (MCMC, MILP solvers, etc.).

### Julia (`julia/Scorio.jl/`)

```bash
make jl-install                  # Pkg.instantiate() for the package + its docs env
make jl-test                     # Pkg.test()
make jl-test-slow                # SCORIO_JL_RUN_SLOW=1 Pkg.test()
make test-eval-jl                # just the eval test file
make test-rank-jl                # just the rank test suite
make jl-docs-build / jl-docs-serve
```

### JavaScript / TypeScript (`js/scorio/`)

```bash
cd js/scorio && npm ci
make js-build                    # tsup -> dist/ (ESM + CJS + d.ts)
make js-test                     # vitest run
npx vitest run <file>            # a single test file (from js/scorio/)
make -C js typecheck             # tsc --noEmit
```

### Release

Version is sourced from the root `VERSION` file; `docs/changelog.rst` is the human-written release-notes source. See `CONTRIBUTING.md` for the full release checklist (`make sync-version`, `make pkg-check`, `make release-py`, `make release-jl`).

## Architecture

### Core data convention

Nearly every module operates on the same shapes, defined in the README and `scorio/eval/_inputs.py`:

- `R`: `M × N` integer matrix, outcomes for `M` questions over `N` trials, categories in `{0, ..., C}`
- `w`: length-`C+1` float vector of rubric weights mapping categories to scores
- `R0` (optional): `M × D` matrix of prior outcomes, same category set as `R`

Read this convention before working with `eval`, `rank`, or `aggregate`. All three modules use these shapes, and the ranking methods construct pairwise or setwise comparisons *from* `R`.

### Scalar evaluation metrics (`scorio/eval/`)

`bayes.py` implements the Bayes@N posterior (mean, std) from `_posterior.py`, optionally combining prior outcomes `R0`. Other metrics (`avg`, `maj`, `pass_at_k`, `gpass`, `max_reward`, `auc`) are built on shared helpers in `_inputs.py` (shape/validation), `_count_score.py`, and `_categorical.py`.

### Model ranking with `eval` and pairwise comparisons (`scorio/rank/`)

`_base.py` and `_types.py` hold shared ranking infrastructure. Method families each live in their own file and wrap a distinct statistical model, documented with paper references in `scorio/rank/README.md`:

- `eval_ranking.py` ranks models by an evaluation metric (`avg`, `bayes`, `pass_at_k`, `g_pass_at_k_tau`, ...).
- `bradley_terry.py` and `pairwise.py` implement the Bradley-Terry family, Elo, TrueSkill and Glicko.
- `voting.py` implements Borda, Copeland, Schulze, ranked pairs, Kemeny-Young and other voting methods.
- `irt.py` implements Rasch/2PL/3PL, MML, dynamic/longitudinal IRT and `mirt`.
- `graph.py`, `rank_centrality.py`, `serial_rank.py` and `hodge_rank.py` implement spectral and graph-theoretic methods.
- `listwise.py` implements Plackett-Luce, Davidson-Luce and related choice models.
- `bayesian.py` implements the Thompson sampling estimator and Bradley-Terry MCMC.
- `priors.py` provides shared prior and regularization helpers for the MAP variants.

Many of these methods raise an exception for cases such as separated data with no finite MLE or a MILP optimum the solver could not prove. Check `scorio/rank/README.md` for the method's failure conditions before changing this behavior.

### Test-time-scaling answer selection (`scorio/aggregate/`, aliased `scorio.agg`)

Picks a final answer among `N` sampled completions per question. `_base.py` defines the shared interface; `confidence.py` derives confidence signals from token log-probs (self-certainty, DeepConf, entropy), `prm.py` aggregates via process reward models, `vote.py`/`best_of.py` implement offline selection (majority vote, weighted vote, Best-of-N, Majority-of-the-Bests), `online.py` implements early-stopping rules, `calibration.py` and `cges.py` support calibration/scoring.

### Rubric-based evaluation pipeline (`scorio/categorical/`)

A higher-level pipeline over `eval.bayes`: `io.py` loads per-completion JSONL files, `thresholds.py` computes corpus-level thresholds, `schemas.py` defines rubric schemas, `evaluate.py` ties these together to classify completions and score models.

### Sequential inference helpers for adaptive stopping and allocation (`scorio/sinf/`)

### Notebooks and datasets (`notebooks/`)

`notebooks/datasets/<trace|lite|math|gpqa>/` each follow the same four-notebook workflow against a companion HuggingFace dataset: load/explore (`<name>.ipynb`) → `eval.ipynb` → `rank.ipynb` → `aggregate.ipynb`, plus a dataset-specific `README.md`. `notebooks/01_bayes_eval.ipynb` is the top-level API walkthrough. `notebooks/datasets/README.md` compares the four datasets.

### `benchmarks/`

`eval_runtime.py` and the checked-in JSON in `benchmarks/baselines/` track performance regressions in the NumPy eval kernels. Update baselines only for intentional performance changes.

### Test fixtures

Use `tests/data/generate_simulation_data.py` to regenerate the outcome-matrix fixtures (`R_greedy.npz`, `R_top_p.npz`) that several tests load.

### `citations/`

`aggregate.bib`, `bayes_at_N.bib` and `rank.bib` are the sources for the citation blocks in `README.md`. When a citation changes, update its bibliography entry and the corresponding README block together.

## Code style notes (Python)

- Keep conventional mathematical names such as `R` and `w`. The Ruff config in `pyproject.toml` ignores `N803`/`N806` to allow them.
- Google-style docstrings, type hints on public APIs (per `CONTRIBUTING.md`).

## CI

On pushes and PRs to `main`, `.github/workflows/python-package.yml` runs `ruff format --check` and `ruff check` on `scorio/`, and `pytest` (`make test`) across Python 3.10–3.13. The local `make lint` target also runs mypy. Fix mypy failures even though they do not block a PR.
