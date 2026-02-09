# scorio

`scorio` implements the Bayes@N framework introduced in [Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](https://arxiv.org/abs/2510.04265)

[![arXiv](https://img.shields.io/badge/arXiv-2510.04265-b31b1b.svg)](https://arxiv.org/abs/2510.04265)
[![PyPI version](https://img.shields.io/pypi/v/scorio.svg)](https://pypi.org/project/scorio/)
[![Python versions](https://img.shields.io/pypi/pyversions/scorio.svg)](https://pypi.org/project/scorio/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Documentation](https://readthedocs.org/projects/scorio/badge/?version=latest)](https://scorio.readthedocs.io/)

---

## Installation

```bash
pip install scorio
```

Requires Python 3.10+, NumPy, SciPy.

## Data and shape conventions

- Categories: encode outcomes per trial as integers in `{0, ..., C}`.
- Weights: choose rubric weights `w` of length `C+1` (e.g., `[0, 1]` for binary R).
- Shapes: `R` is `M x N`, `R0` is `M x D` (if provided); both must share the same `M` and category set.

## APIs

- `scorio.eval.bayes(R, w, R0=None) -> (mu: float, sigma: float)`
  - `R`: `M x N` int array with entries in `{0, ..., C}`
  - `w`: length `C+1` float array of rubric weights
  - `R0` (optional): `M x D` int array of prior outcomes (same category set as `R`)
  - Returns posterior estimate `mu` of the rubric-weighted performance and its uncertainty `sigma`.

- `scorio.eval.avg(R) -> float`
  - Returns the naive mean of elements in `R`. For binary accuracy, encode incorrect=0, correct=1.


## How to use

```python

import numpy as np
from scorio.eval import bayes

# Outcomes R: shape (M, N) with integer categories in {0, ..., C}
R = np.array([
    [0, 1, 2, 2, 1],   # Item 1, N=5 trials
    [1, 1, 0, 2, 2],   # Item 2, N=5 trials
])

# Rubric weights w: length C+1. Here: 0=incorrect, 1=partial(0.5), 2=correct(1.0)
w = np.array([0.0, 0.5, 1.0])

# Optional prior outcomes R0: shape (M, D). If omitted, D=0.
R0 = np.array([
    [0, 2],
    [1, 2],
])

# With prior (D=2 → T=10)
mu, sigma = bayes(R, w, R0)
print(mu, sigma)      # expected ~ (0.575, 0.084275)

# Without prior (D=0 → T=8)
mu2, sigma2 = bayes(R, w)
print(mu2, sigma2)    # expected ~ (0.5625, 0.091998)

```


## Citing

If you use `scorio` or Bayes@N, please cite:

```
@inproceedings{hariri2026don,
  title={Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation},
  author={Hariri, Mohsen and Samandar, Amirhossein and Hinczewski, Michael and Chaudhary, Vipin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2510.04265}
}
```


## License

MIT License. See the `LICENSE` file for details.


## Support

- Documentation: https://scorio.readthedocs.io/
- Issues and feature requests: https://github.com/mohsenhariri/scorio/issues
