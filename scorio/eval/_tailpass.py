"""Numerical helpers for TailPass profiles and categorical payoffs."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import binom

from ._posterior import _beta_binomial_probabilities, _log_beta_power_moment

MAX_COUNT_STATES = 20_000
MAX_COVARIANCE_PAIRS = 4_000_000
FloatArray = NDArray[np.float64]


def count_grid(k: int, categories: int) -> NDArray[np.int64]:
    """Enumerate weak compositions, with a bound before allocation."""
    size = math.comb(k + categories - 1, categories - 1)
    if size > MAX_COUNT_STATES:
        raise ValueError(
            f"Exact profile needs {size} count states; the limit is "
            f"{MAX_COUNT_STATES}. Use a smaller k/rubric or moment(lam=1, 2, 4)."
        )
    if categories == 1:
        return np.array([[k]], dtype=np.int64)
    if categories == 2:
        successes = np.arange(k + 1)
        return np.column_stack((k - successes, successes))
    return np.array(
        [
            np.diff((-1, *bars, k + categories - 1)) - 1
            for bars in combinations(range(k + categories - 1), categories - 1)
        ],
        dtype=np.int64,
    )


def log_coefficients(counts: NDArray[np.int64]) -> NDArray[np.float64]:
    """Log multinomial coefficients for a fixed count grid."""
    return gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)


def predictive_probabilities(
    counts: NDArray[np.int64],
    alpha: NDArray[np.float64],
    log_coeff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Dirichlet-multinomial PMF, normalized using rising-factorial logs."""
    k = int(counts[0].sum())
    if alpha.size == 2:
        return _beta_binomial_probabilities(k, float(alpha[1]), float(alpha[0]))
    logs = log_coeff.copy()
    for category, concentration in enumerate(alpha):
        rising = np.r_[0.0, np.cumsum(np.log(concentration + np.arange(k)))]
        logs += rising[counts[:, category]]
    return np.exp(logs - logsumexp(logs))


def conditional_probabilities(
    counts: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    log_coeff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multinomial PMFs for a batch of latent probability vectors."""
    if probabilities.shape[1] == 2:
        return binom.pmf(counts[:, 1], int(counts[0].sum()), probabilities[:, 1, None])
    logs = np.broadcast_to(log_coeff, (len(probabilities), len(counts))).copy()
    for category in range(counts.shape[1]):
        logs += xlogy(counts[:, category], probabilities[:, category, None])
    return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))


def beta_quadrature(
    degree: int, alpha: float, beta: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalized Gauss-Jacobi rule, exact through degree ``2*degree+1``.

    Build the shifted Jacobi recurrence directly on [0, 1]. Squared first
    eigenvector components give probability weights, avoiding the overflowing
    Beta normalization of an unnormalized Jacobi rule. A centered Gram matrix
    under this rule computes polynomial covariance without subtracting two
    nearly equal second moments.
    """
    total = alpha + beta
    n = np.arange(1, degree + 1, dtype=float)
    diagonal = np.empty(degree + 1)
    diagonal[0] = alpha / total
    diagonal[1:] = ((n + alpha) / (2 * n + total)) * (
        (n + total - 1) / (2 * n + total - 1)
    ) + (n / (2 * n + total - 1)) * ((n + beta - 1) / (2 * n + total - 2))
    off_diagonal = np.empty(degree)
    off_diagonal[0] = math.sqrt((alpha / total) * (beta / total) / (total + 1))
    n = n[1:]
    off_diagonal[1:] = np.sqrt(
        (n / (2 * n + total - 2))
        * ((n + alpha - 1) / (2 * n + total - 2))
        * ((n + beta - 1) / (2 * n + total - 1))
        * ((n + total - 2) / (2 * n + total - 3))
    )
    nodes, vectors = eigh_tridiagonal(diagonal, off_diagonal)
    weights = vectors[0] ** 2
    return np.clip(nodes, 0.0, 1.0), weights / weights.sum()


def count_covariance(
    counts: NDArray[np.int64],
    alpha: NDArray[np.float64],
    values: NDArray[np.float64],
    log_coeff: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Posterior covariance of one or more multinomial count payoffs."""
    size = len(counts)
    if size**2 > MAX_COVARIANCE_PAIRS:
        raise ValueError(
            f"Exact covariance needs {size**2} count pairs; the limit is "
            f"{MAX_COVARIANCE_PAIRS}. Use shared posterior draws for covariance."
        )
    k = int(counts[0].sum())
    if alpha.size == 2:
        reflect = alpha[1] > alpha[0]
        a, b = (
            (float(alpha[0]), float(alpha[1]))
            if reflect
            else (float(alpha[1]), float(alpha[0]))
        )
        nodes, weights = beta_quadrature(k, a, b)
        # Threshold payoffs can use a survival function directly, preserving
        # small tail probabilities and exact constant coordinates.
        is_threshold = np.all((values == 0.0) | (values == 1.0)) and np.all(
            np.diff(values, axis=0) >= 0
        )
        if is_threshold:
            cutoffs = np.sum(values == 0.0, axis=0)
            # Cov(1-X, 1-Y) = Cov(X, Y). Integrating failure tails when
            # success is near one preserves tiny but nonzero uncertainty.
            evaluated = binom.sf(
                k - cutoffs if reflect else cutoffs - 1, k, nodes[:, None]
            )
        else:
            coefficients = values[::-1] if reflect else values
            coefficients = coefficients - coefficients[0]
            evaluated = binom.pmf(np.arange(k + 1), k, nodes[:, None]) @ coefficients
        centered = evaluated - weights @ evaluated
        return (centered.T * weights) @ centered

    pmf = predictive_probabilities(counts, alpha, log_coeff)
    centered = values - pmf @ values
    covariance = np.zeros((values.shape[1], values.shape[1]))
    for index, count in enumerate(counts):
        conditional = predictive_probabilities(counts, alpha + count, log_coeff)
        covariance += pmf[index] * np.outer(centered[index], conditional @ centered)
    return (covariance + covariance.T) / 2


def endpoint_moments(
    k: int, alpha: FloatArray, beta: FloatArray, kind: str
) -> tuple[FloatArray, FloatArray]:
    """Stable Beta endpoint moments, including highly concentrated priors."""
    means, variances = np.empty(len(alpha)), np.empty(len(alpha))
    for index, (a, b) in enumerate(zip(alpha, beta, strict=True)):
        if kind == "pass":
            a, b = b, a
        log_mean = _log_beta_power_moment(float(a), float(b), k)
        log_second = _log_beta_power_moment(float(a), float(b), 2 * k)
        # E[p**(2k)] / E[p**k]**2 is a product of
        # 1 + k*b / ((a+j)*(a+b+k+j)). Avoid subtracting almost equal logs.
        indices = np.arange(k, dtype=float)
        with np.errstate(over="ignore", under="ignore"):
            ratios = (k / (a + indices)) * (b / (a + b + k + indices))
        log_ratio = float(np.log1p(ratios).sum())
        means[index] = -math.expm1(log_mean) if kind == "pass" else math.exp(log_mean)
        variances[index] = math.exp(log_second) * -math.expm1(-log_ratio)
    return means, variances


def integer_moment_terms(k: int, power: int) -> list[tuple[float, tuple[int, ...]]]:
    """Raw-moment expansion of the mean of k i.i.d. rubric scores."""
    if power == 1:
        return [(1.0, (1,))]
    if power == 2:
        return [(1.0 / k, (2,)), ((k - 1) / k, (1, 1))]
    terms: list[tuple[float, tuple[int, ...]]] = [(1.0 / k**3, (4,))]
    if k >= 2:
        terms += [(4 * (k - 1) / k**3, (3, 1)), (3 * (k - 1) / k**3, (2, 2))]
    if k >= 3:
        terms.append((6 * ((k - 1) / k) * ((k - 2) / k) / k, (2, 1, 1)))
    if k >= 4:
        terms.append((((k - 1) / k) * ((k - 2) / k) * ((k - 3) / k), (1, 1, 1, 1)))
    return terms


def conditional_integer_moment(
    probabilities: NDArray[np.float64], scores: NDArray[np.float64], k: int, power: int
) -> NDArray[np.float64]:
    """Evaluate the first, second, or fourth moment without count enumeration."""
    output = np.zeros(probabilities.shape[0])
    raw = {p: probabilities @ scores**p for p in range(1, power + 1)}
    for coefficient, factors in integer_moment_terms(k, power):
        term = np.full(len(output), coefficient)
        for factor in factors:
            term *= raw[factor]
        output += term
    return output


def moment_polynomial(
    scores: NDArray[np.float64], k: int, power: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Expand only the low-degree moment polynomial, independently of k."""
    if math.comb(len(scores) + power, power) > MAX_COUNT_STATES:
        raise ValueError("Too many rubric categories for the exact moment expansion")
    coefficients: dict[tuple[int, ...], float] = {}
    zero = (0,) * len(scores)
    for coefficient, factors in integer_moment_terms(k, power):
        terms = {zero: coefficient}
        for factor in factors:
            next_terms: dict[tuple[int, ...], float] = {}
            for powers, value in terms.items():
                for category, score in enumerate(scores):
                    if score == 0.0:
                        continue
                    updated = list(powers)
                    updated[category] += 1
                    key = tuple(updated)
                    next_terms[key] = next_terms.get(key, 0.0) + value * score**factor
            terms = next_terms
        for powers, value in terms.items():
            coefficients[powers] = coefficients.get(powers, 0.0) + value
    if not coefficients:
        return np.zeros((1, len(scores)), dtype=np.int64), np.zeros(1)
    return np.array(list(coefficients), dtype=np.int64), np.array(
        list(coefficients.values())
    )


def monomial_moments(
    powers: NDArray[np.int64], alpha: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Dirichlet monomial expectations using bounded rising-factorial ratios."""
    output = np.ones(powers.shape[:-1])
    used = np.zeros(powers.shape[:-1])
    total = float(alpha.sum())
    for category, concentration in enumerate(alpha):
        for step in range(int(powers[..., category].max())):
            mask = powers[..., category] > step
            output[mask] *= (concentration + step) / (total + used[mask])
            used[mask] += 1
    return output


def polynomial_moments(
    powers: NDArray[np.int64],
    coefficients: NDArray[np.float64],
    alpha: NDArray[np.float64],
) -> tuple[float, float]:
    """Exact first and second moments of a low-degree Dirichlet polynomial."""
    if len(powers) ** 2 > MAX_COVARIANCE_PAIRS:
        raise ValueError(
            "Too many moment terms for exact variance; use posterior draws"
        )
    first = monomial_moments(powers, alpha)
    second = monomial_moments(powers[:, None, :] + powers[None, :, :], alpha)
    mean = float(coefficients @ first)
    variance = float(coefficients @ (second - np.outer(first, first)) @ coefficients)
    return mean, nonnegative_variance(variance)


def nonnegative_variance(value: float) -> float:
    """Clip only variance roundoff, rejecting invalid numerical results."""
    if not math.isfinite(value) or value < -1e-12:
        raise FloatingPointError(f"Invalid TailPass posterior variance: {value}")
    return max(0.0, value)
