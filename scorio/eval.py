import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import comb


def _as_2d_int_matrix(R: np.ndarray) -> np.ndarray:
    Rm = np.asarray(R, dtype=int)
    if Rm.ndim == 1:
        # treat as single-row
        Rm = Rm.reshape(1, -1)
    elif Rm.ndim != 2:
        raise ValueError("R must be a 1D or 2D array.")
    return Rm


def bayes(
    R: np.ndarray,
    w: np.ndarray,
    R0: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Performance evaluation using the Bayes@N framework.

    References:
        Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2025).
        Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
        *arXiv preprint arXiv:2510.04265*.
        https://arxiv.org/abs/2510.04265

    Args:
        R: :math:`M \\times N` int matrix with entries in :math:`\\{0,\\ldots,C\\}`.
           Row :math:`\\alpha` are the N outcomes for system :math:`\\alpha`.
        w: length :math:`(C+1)` weight vector :math:`(w_0,\\ldots,w_C)` that maps
           category k to score :math:`w_k`.
        R0: optional :math:`M \\times D` int matrix supplying D prior outcomes per row.
             If omitted, :math:`D=0`.

    Returns:
        Tuple[float, float]: :math:`(\\mu, \\sigma)` performance metric estimate and its uncertainty.

    Notation:
        :math:`\\delta_{a,b}` is the Kronecker delta. For each row :math:`\\alpha` and class :math:`k \\in \\{0,\\ldots,C\\}`:

        .. math::

            n_{\\alpha k} &= \\sum_{i=1}^N \\delta_{k, R_{\\alpha i}} \\quad \\text{(counts in R)}

            n^0_{\\alpha k} &= 1 + \\sum_{i=1}^D \\delta_{k, R^0_{\\alpha i}} \\quad \\text{(Dirichlet(+1) prior)}

            \\nu_{\\alpha k} &= n_{\\alpha k} + n^0_{\\alpha k}

        Effective sample size: :math:`T = 1 + C + D + N` (scalar)

    Formula:
        .. math::

            \\mu = w_0 + \\frac{1}{M \\cdot T} \\sum_{\\alpha=1}^M \\sum_{j=0}^C \\nu_{\\alpha j} (w_j - w_0)

        .. math::

            \\sigma = \\sqrt{ \\frac{1}{M^2(T+1)} \\sum_{\\alpha=1}^M \\left[
                \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0)^2
                - \\left( \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0) \\right)^2 \\right] }

    Examples:
        >>> import numpy as np
        >>> R  = np.array([[0, 1, 2, 2, 1],
        ...                [1, 1, 0, 2, 2]])
        >>> w  = np.array([0.0, 0.5, 1.0])
        >>> R0 = np.array([[0, 2],
        ...                [1, 2]])

        With prior (D=2 → T=10):

        >>> mu, sigma = bayes(R, w, R0)
        >>> round(mu, 6), round(sigma, 6)
        (0.575, 0.084275)

        Without prior (D=0 → T=8):

        >>> mu2, sigma2 = bayes(R, w)
        >>> round(mu2, 6), round(sigma2, 6)
        (0.5625, 0.091998)

    """
    R = _as_2d_int_matrix(R)
    w = np.asarray(w, dtype=float)
    M, N = R.shape
    C = w.size - 1

    if R0 is None:
        D = 0
        R0m = np.zeros((M, 0), dtype=int)
    else:
        R0m = np.asarray(R0, dtype=int)
        if R0m.ndim == 1:
            R0m = R0m.reshape(M, -1)
        if R0m.shape[0] != M:
            raise ValueError("R0 must have the same number of rows (M) as R.")
        D = R0m.shape[1]

    # Validate value ranges
    if R.size and (R.min() < 0 or R.max() > C):
        raise ValueError("Entries of R must be integers in [0, C].")
    if R0m.size and (R0m.min() < 0 or R0m.max() > C):
        raise ValueError("Entries of R0 must be integers in [0, C].")

    T = 1 + C + D + N

    def row_bincount(A: np.ndarray, length: int) -> np.ndarray:
        """Count occurrences of 0..length-1 in each row of A."""
        if A.shape[1] == 0:
            return np.zeros((A.shape[0], length), dtype=int)
        out = np.zeros((A.shape[0], length), dtype=int)
        rows = np.repeat(np.arange(A.shape[0]), A.shape[1])
        np.add.at(out, (rows, A.ravel()), 1)
        return out

    # n_{αk} and n^0_{αk}
    n_counts = row_bincount(R, C + 1)
    n0_counts = row_bincount(R0m, C + 1) + 1  # add 1 to every class (Dirichlet prior)

    # ν_{αk} = n_{αk} + n^0_{αk}
    nu = n_counts + n0_counts  # shape: (M, C+1)

    # μ = w0 + (1/(M T)) * Σ_α Σ_j ν_{αj} (w_j - w0)
    delta_w = w - w[0]
    mu = w[0] + (nu @ delta_w).sum() / (M * T)

    # σ = [ (1/(M^2 (T+1))) * Σ_α { Σ_j (ν_{αj}/T)(w_j-w0)^2
    #       - ( Σ_j (ν_{αj}/T)(w_j-w0) )^2 } ]^{1/2}
    nu_over_T = nu / T
    termA = (nu_over_T * (delta_w**2)).sum(axis=1)
    termB = (nu_over_T @ delta_w) ** 2
    sigma = np.sqrt(((termA - termB).sum()) / (M**2 * (T + 1)))

    return float(mu), float(sigma)


def avg(R: np.ndarray) -> float:
    """
    Simple average of all entries in R.

    Computes the arithmetic mean of all entries in the result matrix,

    Args:
        R: :math:`M \\times N` result matrix with entries in :math:`\\{0, 1\\}`.
           Row :math:`\\alpha` are the N outcomes for system :math:`\\alpha`.

    Returns:
        float: The arithmetic mean of all entries in R.

    Notation:
        :math:`R_{\\alpha i}` is the outcome for system :math:`\\alpha` on trial :math:`i`.

    Formula:
        .. math::

            \\text{avg} = \\frac{1}{M \\cdot N} \\sum_{\\alpha=1}^{M} \\sum_{i=1}^{N} R_{\\alpha i}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(avg(R), 6)
        0.7
    """
    R = _as_2d_int_matrix(R)
    return float(np.mean(R))


def pass_at_k(R: np.ndarray, k: int) -> float:
    """
    Unbiased Pass@k estimator.

    Computes the probability that at least one of k randomly selected samples
    is correct, averaged over all M systems.

    References:
        Chen, M., Tworek, J., Jun, H., et al. (2021).
        Evaluating Large Language Models Trained on Code.
        *arXiv preprint arXiv:2107.03374*.
        https://arxiv.org/abs/2107.03374

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for system :math:`\\alpha` passed,
           0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average Pass@k score across all M systems.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

    Formula:
        .. math::

            \\text{Pass@k}_\\alpha = 1 - \\frac{C(N - \\nu_\\alpha, k)}{C(N, k)}

        .. math::

            \\text{Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{Pass@k}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_at_k(R, 1), 6)
        0.7
        >>> round(pass_at_k(R, 2), 6)
        0.95
    """
    R = _as_2d_int_matrix(R)
    _, N = R.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")
    nu = np.sum(R, axis=1)
    denom = comb(N, k)
    vals = 1 - comb(N - nu, k) / denom  # (M,)
    return float(np.mean(vals))


def pass_hat_k(R: np.ndarray, k: int) -> float:
    """
    Pass^k (Pass-hat@k): probability that all k selected trials are correct.

    Computes the probability that k randomly selected samples are ALL correct,
    averaged over all M systems. Also known as G-Pass@k.

    References:
        Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
        τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
        *arXiv preprint arXiv:2406.12045*.
        https://arxiv.org/abs/2406.12045

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for system :math:`\\alpha` passed,
           0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average Pass^k score across all M systems.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

    Formula:
        .. math::

            \\text{Pass}\\hat{\\text{@}}\\text{k}_\\alpha = \\frac{C(\\nu_\\alpha, k)}{C(N, k)}

        .. math::

            \\text{Pass}\\hat{\\text{@}}\\text{k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{Pass}\\hat{\\text{@}}\\text{k}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(pass_hat_k(R, 1), 6)
        0.7
        >>> round(pass_hat_k(R, 2), 6)
        0.45
    """
    R = _as_2d_int_matrix(R)
    _, N = R.shape
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")
    nu = np.sum(R, axis=1)
    denom = comb(N, k)
    vals = comb(nu, k) / denom  # (M,)
    return float(np.mean(vals))


def g_pass_at_k(R: np.ndarray, k: int) -> float:
    """
    Alias for pass_hat_k. See `pass_hat_k` for documentation.

    This function is provided for compatibility with literature that uses
    the G-Pass@k naming convention.
    """
    return pass_hat_k(R, k)


def g_pass_at_k_tao(R: np.ndarray, k: int, tao: float) -> float:
    """
    G-Pass@k_τ: Generalized Pass@k with threshold τ.

    Computes the probability that at least :math:`\\lceil \\tau \\cdot k \\rceil` of k randomly selected
    samples are correct, averaged over all M systems.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2024).
        Are Your LLMs Capable of Stable Reasoning?
        *arXiv preprint arXiv:2412.13147*.
        https://arxiv.org/abs/2412.13147

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for system :math:`\\alpha` passed,
           0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).
        tao: Threshold parameter :math:`\\tau \\in [0, 1]`. Requires at least
             :math:`\\lceil \\tau \\cdot k \\rceil` successes.
             When :math:`\\tau = 0`, equivalent to Pass@k.
             When :math:`\\tau = 1`, equivalent to Pass^k.

    Returns:
        float: The average G-Pass@k_τ score across all M systems.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`C(a, b)` denotes the binomial coefficient :math:`\\binom{a}{b}`.

        :math:`j_0 = \\lceil \\tau \\cdot k \\rceil` is the minimum number of successes required.

    Formula:
        .. math::

            \\text{G-Pass@k}_{\\tau, \\alpha} = \\sum_{j=j_0}^{k} \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}

        .. math::

            \\text{G-Pass@k}_\\tau = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{G-Pass@k}_{\\tau, \\alpha}

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(g_pass_at_k_tao(R, 2, 0.5), 6)
        0.95
        >>> round(g_pass_at_k_tao(R, 2, 1.0), 6)
        0.45
    """
    R = _as_2d_int_matrix(R)
    M, N = R.shape

    if not (0.0 <= tao <= 1.0):
        raise ValueError(f"tao must be in [0, 1]; got {tao}")
    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    # Edge case: if tao -> 0, return pass_at_k(R, k)
    if tao <= 0.0:
        return pass_at_k(R, k)

    nu = np.sum(R, axis=1)
    denom = comb(N, k)

    j0 = int(math.ceil(tao * k))
    if j0 > k:
        return 0.0

    vals = np.zeros(M, dtype=float)
    for j in range(j0, k + 1):
        vals += comb(nu, j) * comb(N - nu, k - j) / denom
    return float(np.mean(vals))


def mg_pass_at_k(R: np.ndarray, k: int) -> float:
    """
    mG-Pass@k: Majority-weighted Generalized Pass@k.

    Measures how much the number of correct samples exceeds the majority
    threshold :math:`\\lceil k/2 \\rceil`, weighted by the hypergeometric distribution.

    References:
        Liu, J., Liu, H., Xiao, L., et al. (2024).
        Are Your LLMs Capable of Stable Reasoning?
        *arXiv preprint arXiv:2412.13147*.
        https://arxiv.org/abs/2412.13147

    Args:
        R: :math:`M \\times N` binary matrix with entries in :math:`\\{0, 1\\}`.
           :math:`R_{\\alpha i} = 1` if trial :math:`i` for system :math:`\\alpha` passed,
           0 otherwise.
        k: Number of samples to select (:math:`1 \\le k \\le N`).

    Returns:
        float: The average mG-Pass@k score across all M systems.

    Notation:
        For each row :math:`\\alpha`:

        .. math::

            \\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}

        :math:`X \\sim \\text{Hypergeometric}(N, \\nu_\\alpha, k)` is the number of successes
        when drawing :math:`k` samples without replacement from a population of size :math:`N`
        containing :math:`\\nu_\\alpha` successes.

        :math:`(\\cdot)_+ = \\max(\\cdot, 0)` denotes the positive part function.

        :math:`m = \\lceil k/2 \\rceil` is the majority threshold.

    Formula:
        .. math::

            \\text{mG-Pass@k}_\\alpha = \\frac{2}{k} \\cdot \\mathbb{E}[(X - m)_+]
            = \\frac{2}{k} \\sum_{j=m+1}^{k} (j - m) \\cdot P(X = j)

        where the probability mass function is:

        .. math::

            P(X = j) = \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}

        The final metric is averaged over all systems:

        .. math::

            \\text{mG-Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{mG-Pass@k}_\\alpha

    Examples:
        >>> import numpy as np
        >>> R = np.array([[0, 1, 1, 0, 1],
        ...               [1, 1, 0, 1, 1]])
        >>> round(mg_pass_at_k(R, 2), 6)
        0.45
        >>> round(mg_pass_at_k(R, 3), 6)
        0.166667
    """
    R = _as_2d_int_matrix(R)
    M, N = R.shape

    if not (1 <= k <= N):
        raise ValueError(f"k must satisfy 1 <= k <= N (N={N}); got k={k}")

    nu = np.sum(R, axis=1)
    denom = comb(N, k)

    majority = int(math.ceil(0.5 * k))
    if majority >= k:
        return 0.0

    vals = np.zeros(M, dtype=float)
    # mG per-question = (2/k) * E[(X - majority)_+], X ~ Hypergeom(N, nu, k)
    for j in range(majority + 1, k + 1):
        pmf = comb(nu, j) * comb(N - nu, k - j) / denom
        vals += (j - majority) * pmf

    vals *= 2.0 / k
    return float(np.mean(vals))


__all__ = [
    "avg",
    "bayes",
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k_tao",
    "mg_pass_at_k",
]
