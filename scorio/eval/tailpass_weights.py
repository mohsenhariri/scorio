"""Convex threshold weights for binary TailPass utilities.

Entry ``t - 1`` values reaching at least ``t`` successes among ``k`` attempts.
These weights describe preferences, independently of the evaluation prior.
For categorical moment utilities, use ``profile.moment(lam)``: a categorical
bank can attain scores between the binary reporting thresholds.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import betainc, betaincc

from ._count_score import _real_vector
from ._inputs import _finite_positive_scalar, _integral_scalar, validate_latent_k


def _validate_weights(weights: ArrayLike, size: int) -> NDArray[np.float64]:
    values = _real_vector(weights, name="weights")
    if values.shape != (size,):
        raise ValueError(f"weights must be a length-{size} 1D array")
    if np.any(values < 0.0) or not math.isclose(
        float(np.sum(values)), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("weights must be non-negative and sum to one")
    return values


def uniform_weights(k: int) -> NDArray[np.float64]:
    """Return ``1/k`` at every threshold (the binary accuracy utility)."""
    return np.full(validate_latent_k(k), 1.0 / k)


def threshold_weights(k: int, threshold: int) -> NDArray[np.float64]:
    """Assign all value to reaching ``threshold`` successes, from 1 to k."""
    k = validate_latent_k(k)
    threshold = _integral_scalar(threshold, name="threshold")
    if not 1 <= threshold <= k:
        raise ValueError("threshold must satisfy 1 <= threshold <= k")
    weights = np.zeros(k)
    weights[threshold - 1] = 1.0
    return weights


def discovery_weights(k: int) -> NDArray[np.float64]:
    """Assign all value to finding at least one success."""
    return threshold_weights(k, 1)


def stability_weights(k: int) -> NDArray[np.float64]:
    """Assign all value to all k attempts succeeding."""
    return threshold_weights(k, k)


def moment_weights(k: int, lam: float) -> NDArray[np.float64]:
    r"""Return ``(t/k)**lam - ((t-1)/k)**lam`` for finite ``lam > 0``.

    ``lam=1`` gives uniform weights. Smaller exponents emphasize discovery;
    larger exponents emphasize repeatability. The limits at zero and infinity
    are supplied explicitly by ``discovery_weights`` and ``stability_weights``.
    """
    k = validate_latent_k(k)
    lam = _finite_positive_scalar(lam, name="lam")
    t = np.arange(1, k + 1, dtype=float)
    # Avoid subtracting nearly equal powers when lam is small or t is large.
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        weights = np.exp(lam * np.log(t / k)) * -np.expm1(lam * np.log1p(-1.0 / t))
    return weights / weights.sum()


def beta_weights(k: int, theta: float, kappa: float) -> NDArray[np.float64]:
    """Bin Beta threshold mass by mean ``theta`` and concentration ``kappa``.

    Args:
        k: Positive reporting budget.
        theta: Mean fractional threshold, strictly between zero and one.
        kappa: Finite positive concentration around that threshold.
    """
    k = validate_latent_k(k)
    theta = _finite_positive_scalar(theta, name="theta")
    kappa = _finite_positive_scalar(kappa, name="kappa")
    if theta >= 1.0:
        raise ValueError("theta must be in (0, 1)")
    a, b = theta * kappa, (1.0 - theta) * kappa
    if a == 0.0 or b == 0.0:
        raise ValueError("Beta weight parameters must remain positive in float64")
    grid = np.arange(k + 1, dtype=float) / k
    cdf, sf = betainc(a, b, grid), betaincc(a, b, grid)
    weights = np.where(cdf[1:] <= 0.5, np.diff(cdf), -np.diff(sf))
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise FloatingPointError("could not compute finite Beta threshold weights")
    return weights / weights.sum()


def maxent_weights(k: int, mean_threshold: float) -> NDArray[np.float64]:
    """Maximum-entropy threshold weights with a specified mean in (0, 1).

    Integrate the density proportional to ``exp(tau * rho)`` on [0, 1],
    choosing tau to match ``mean_threshold``. A mean of 0.5 gives uniform
    weights. This is the continuous maximum-entropy family in Appendix C.8.
    """
    k = validate_latent_k(k)
    target = _finite_positive_scalar(mean_threshold, name="mean_threshold")
    if target >= 1.0:
        raise ValueError("mean_threshold must be in (0, 1)")
    if target == 0.5:
        return uniform_weights(k)
    distance = min(target, 1.0 - target)

    def lower_mean(tau: float) -> float:
        if tau < 1e-3:
            return 0.5 - tau / 12.0 + tau**3 / 720.0
        return 1.0 / tau - math.exp(-tau) / -math.expm1(-tau)

    upper = 1.0 / distance
    if not math.isfinite(upper):
        # At this scale all representable threshold mass is at an endpoint.
        return discovery_weights(k) if target < 0.5 else stability_weights(k)
    tau = brentq(lambda value: lower_mean(value) - distance, 0.0, upper)
    if tau == 0.0:
        return uniform_weights(k)
    positions = np.arange(k, dtype=float) / k
    weights = np.exp(-tau * positions) * (-math.expm1(-tau / k)) / (-math.expm1(-tau))
    weights /= weights.sum()
    return weights if target < 0.5 else weights[::-1].copy()


def payoff_weights(payoff: ArrayLike) -> NDArray[np.float64]:
    """Convert monotone ``u(0), ..., u(k)`` to marginal threshold weights.

    The payoff must start at zero and end at one. Its length determines k.
    """
    values = _real_vector(payoff, name="payoff")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("payoff must be a 1D array with at least two entries")
    if values[0] != 0.0 or values[-1] != 1.0 or np.any(np.diff(values) < 0.0):
        raise ValueError("payoff must be nondecreasing, start at zero, and end at one")
    return np.diff(values)


__all__ = [
    "uniform_weights",
    "threshold_weights",
    "discovery_weights",
    "stability_weights",
    "moment_weights",
    "beta_weights",
    "maxent_weights",
    "payoff_weights",
]
