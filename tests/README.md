# Test Data Guide

`scorio` keeps benchmark response tensors in `notebooks/` as `.npz` archives.
Use these files for reproducible local benchmarking and manual experiments with
`scorio.eval` and `scorio.rank`.

## Test layout

- `tests/eval/`: tests for `scorio.eval` APIs
- `tests/rank/`: tests for `scorio.rank` APIs
- `tests/data/`: simulation random-binary `.npz` fixtures for tests (same key/shape
  contract as `notebooks/*.npz`)
  generated with fixed RNG seeds for reproducibility.

## Available data files

- `notebooks/R_top_p.npz`: sampled decoding runs (`N=80` trials per question).
- `notebooks/R_greedy.npz`: greedy decoding runs (`N=1` trial per question).
- `tests/data/R_top_p.npz`: simulation sampled decoding fixture (`N=80`) for CI/tests.
- `tests/data/R_greedy.npz`: simulation greedy decoding fixture (`N=1`) for CI/tests.

Both files contain the same task keys:

- `aime24`
- `aime25`
- `hmmt_feb_2025`
- `brumo_2025`
- `combined` (the four tasks concatenated along the question axis)

## Tensor contract

Each task entry is a binary tensor `R` with shape `(L, M, N)`:

- `L`: number of models
- `M`: number of questions
- `N`: number of independent trials/samples
- `R[l, m, n] in {0, 1}`: model `l` fails/passes question `m` on trial `n`

For this dataset:

- single tasks (`aime24`, `aime25`, `hmmt_feb_2025`, `brumo_2025`) use `M=30`
- `combined` uses `M=120`
- `L=20` in all tasks/files

## Loading pattern

```python
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # when called from tests/eval or tests/rank

# Local tests/CI use simulation fixtures in tests/data
with np.load(ROOT / "tests" / "data" / "R_top_p.npz", allow_pickle=True) as data:
    print(data.files)
    R = data["aime25"]  # shape: (20, 30, 80)
```

To regenerate simulation fixtures:

```bash
python tests/data/generate_simulation_data.py
```

## Using the tensor with `scorio`

- `scorio.rank.*` methods consume the full `(L, M, N)` tensor.
- `scorio.eval.*` methods consume one model slice `R[l]` with shape `(M, N)`.

```python
from scorio import eval, rank

# rank all models
ranks, scores = rank.pass_at_k(R, k=2, return_scores=True)

# evaluate one model
mu, sigma = eval.bayes(R[0])
```
